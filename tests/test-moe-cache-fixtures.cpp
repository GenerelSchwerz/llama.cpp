#include "test-moe-cache.h"

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

bool grouped_frequency_enabled() {
    const char * value = getenv("GGML_CUDA_MOE_FREQUENCY");
    return value == nullptr || strcmp(value, "0") != 0;
}

cudaStream_t candidate_test_graph_stream(void * data, const ggml_tensor *) {
    return static_cast<cudaStream_t>(data);
}

void CUDART_CB wait_on_host_barrier(void * data) {
    auto * barrier = static_cast<host_barrier *>(data);
    barrier->entered.store(true, std::memory_order_release);
    while (!barrier->released.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

bool candidate_test_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t) {
    return *static_cast<const bool *>(dev->context);
}

candidate_route candidate_top_k_route(
        candidate_test_fixture & fixture,
        int64_t n_experts,
        int64_t n_routes,
        int64_t n_tokens,
        size_t view_offs) {
    candidate_route result;
    const int64_t source_ne[] = {n_experts, n_tokens};
    result.source = fixture.tensor(GGML_TYPE_F32, 2, source_ne);
    result.root = ggml_argsort(fixture.ctx, result.source, GGML_SORT_ORDER_DESC);
    fixture.materialize(result.root);
    result.root->flags |= GGML_TENSOR_FLAG_COMPUTE;
    result.ids = ggml_view_4d(fixture.ctx, result.root, n_routes, n_tokens, 1, 1,
        result.root->nb[1], result.root->nb[2], result.root->nb[3], view_offs);
    fixture.materialize(result.ids);
    result.ids->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

ggml_tensor * candidate_mmid(candidate_test_fixture & fixture, ggml_tensor * weight, ggml_tensor * ids) {
    const int64_t activation_ne[] = {weight->ne[0], 1, ids->ne[1]};
    const int64_t output_ne[] = {weight->ne[1], ids->ne[0], ids->ne[1]};
    ggml_tensor * activation = fixture.tensor(GGML_TYPE_F32, 3, activation_ne);
    ggml_tensor * result = fixture.tensor(GGML_TYPE_F32, 3, output_ne);
    result->op = GGML_OP_MUL_MAT_ID;
    result->src[0] = weight;
    result->src[1] = activation;
    result->src[2] = ids;
    result->flags |= GGML_TENSOR_FLAG_COMPUTE;
    return result;
}

void candidate_set_route_tokens(
        const candidate_route & route,
        std::initializer_list<ggml_tensor *> readers,
        int64_t n_tokens) {
    route.source->ne[1] = n_tokens;
    route.source->nb[2] = route.source->nb[1] * n_tokens;
    route.source->nb[3] = route.source->nb[2];
    route.root->ne[1] = n_tokens;
    route.root->nb[2] = route.root->nb[1] * n_tokens;
    route.root->nb[3] = route.root->nb[2];
    route.ids->ne[1] = n_tokens;
    memcpy(route.ids->nb, route.root->nb, sizeof(route.ids->nb));
    for (ggml_tensor * reader : readers) {
        reader->src[1]->ne[2] = n_tokens;
        reader->src[1]->nb[3] = reader->src[1]->nb[2] * n_tokens;
        reader->ne[2] = n_tokens;
        reader->nb[3] = reader->nb[2] * n_tokens;
    }
}

void candidate_rebuild_graph_uses(ggml_cgraph * graph) {
    ggml_hash_set_reset(&graph->visited_hash_set);
    memset(graph->use_counts, 0, graph->visited_hash_set.size * sizeof(graph->use_counts[0]));
    for (int node_index = 0; node_index < graph->n_nodes; ++node_index) {
        ggml_tensor * node = graph->nodes[node_index];
        CHECK(ggml_hash_find_or_insert(&graph->visited_hash_set, node) != GGML_HASHSET_FULL);
        for (uint32_t src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            if (node->src[src_index] == nullptr) {
                continue;
            }
            const size_t hash_pos = ggml_hash_find_or_insert(&graph->visited_hash_set, node->src[src_index]);
            CHECK(hash_pos != GGML_HASHSET_FULL);
            graph->use_counts[hash_pos]++;
        }
    }
}

void candidate_insert_graph_node(ggml_cgraph * graph, int32_t node_index, ggml_tensor * node) {
    CHECK(graph != nullptr && node != nullptr && node_index >= 0 && node_index <= graph->n_nodes && graph->n_nodes < graph->size);
    memmove(graph->nodes + node_index + 1, graph->nodes + node_index,
        (graph->n_nodes - node_index) * sizeof(graph->nodes[0]));
    graph->nodes[node_index] = node;
    ++graph->n_nodes;
    candidate_rebuild_graph_uses(graph);
}

int32_t candidate_graph_use_count(const ggml_cgraph * graph, const ggml_tensor * tensor) {
    const size_t hash_pos = ggml_hash_find(&graph->visited_hash_set, tensor);
    CHECK(hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, hash_pos));
    return graph->use_counts[hash_pos];
}

void candidate_stamp_execution(
        ggml_cgraph * graph,
        uint32_t domain,
        uint32_t row_semantics,
        uint32_t n_rows,
        uint32_t n_sequences,
        uint64_t source_graph_uid,
        uint32_t flags) {
    CHECK(graph != nullptr);
    if (source_graph_uid == 0) {
        source_graph_uid = ggml_graph_next_uid();
    }
    graph->uid = ggml_graph_next_uid();
    CHECK(source_graph_uid != 0 && graph->uid != 0 && source_graph_uid != graph->uid);
    graph->execution_certificate = {};
    graph->execution_certificate.magic = GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC;
    graph->execution_certificate.abi_version = GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION;
    graph->execution_certificate.struct_size = sizeof(graph->execution_certificate);
    graph->execution_certificate.flags = flags;
    graph->execution_certificate.domain = domain;
    graph->execution_certificate.row_semantics = row_semantics;
    graph->execution_certificate.n_rows = n_rows;
    graph->execution_certificate.n_sequences = n_sequences;
    graph->execution_certificate.owner_namespace = 0x746573742d6d6f65ULL;
    graph->execution_certificate.owner_generation = 1;
    graph->execution_certificate.source_graph_uid = source_graph_uid;
    graph->execution_certificate.split_graph_uid = graph->uid;
}

void candidate_stamp_single_row_execution(ggml_cgraph * graph) {
    for (int32_t node_index = 0; node_index < graph->n_nodes; ++node_index) {
        const ggml_tensor * node = graph->nodes[node_index];
        if (node != nullptr && node->op == GGML_OP_MUL_MAT_ID && node->src[2] != nullptr) {
            if (node->src[2]->ne[1] == 1 && node->src[2]->ne[2] == 1 && node->src[2]->ne[3] == 1) {
                candidate_stamp_execution(graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
                    GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, 1, 1);
            }
            return;
        }
    }
}

ggml_cgraph * candidate_graph(candidate_test_fixture & fixture, std::initializer_list<ggml_tensor *> nodes) {
    ggml_cgraph * graph = ggml_new_graph_custom(fixture.ctx, 32, false);
    CHECK(graph != nullptr);
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
    candidate_rebuild_graph_uses(graph);
    candidate_stamp_single_row_execution(graph);
    return graph;
}

candidate_graph_coverage candidate_certify_graph(
        ggml_cuda_moe_grouped_context & registry,
        ggml_cgraph * graph) {
    candidate_graph_coverage result;
    result.nodes = graph->nodes;
    result.epoch = registry.certify_graph_coverage(graph, &result.mmid_count, &result.mmid_fingerprint);
    CHECK(result.epoch != 0 && result.mmid_fingerprint != 0);
    return result;
}

ggml_backend_moe_candidate_snapshot_v1 candidate_snapshot(
        uint32_t n_slots,
        const ggml_backend_moe_candidate_group_v1 * groups,
        uint32_t n_groups) {
    ggml_backend_moe_candidate_snapshot_v1 result = {};
    result.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_MAGIC;
    result.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V1_VERSION;
    result.struct_size = sizeof(result);
    result.n_slots = n_slots;
    result.groups = groups;
    result.n_groups = n_groups;
    return result;
}

ggml_backend_moe_candidate_snapshot_v2 candidate_snapshot_v2(
        uint32_t n_slots,
        const ggml_backend_moe_candidate_group_v2 * groups,
        uint32_t n_groups,
        const ggml_backend_moe_candidate_tensor_v2 * tensors,
        uint32_t n_tensors) {
    ggml_backend_moe_candidate_snapshot_v2 result = {};
    result.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC;
    result.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION;
    result.struct_size = sizeof(result);
    result.n_slots = n_slots;
    result.groups = groups;
    result.n_groups = n_groups;
    result.tensors = tensors;
    result.n_tensors = n_tensors;
    return result;
}

std::vector<uint8_t> cached_fusion_test_data(const ggml_tensor * tensor, size_t salt) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> values(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = i % 2 == 0 ? 0 : 2;
        }
        memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    std::vector<float> values(ggml_nelements(tensor));
    const bool scale = strstr(tensor->name, ".scale") != nullptr;
    const bool bias = strstr(tensor->name, ".bias") != nullptr;
    for (size_t i = 0; i < values.size(); ++i) {
        const int phase = static_cast<int>((i + 3 * salt) % 17) - 8;
        values[i] = scale ? 0.75f + 0.025f * (phase + 8) : (bias ? 0.01f : 0.035f) * phase;
    }

    if (tensor->type == GGML_TYPE_F32) {
        memcpy(bytes.data(), values.data(), bytes.size());
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row_ref(values.data(), reinterpret_cast<ggml_bf16_t *>(bytes.data()), values.size());
    } else {
        CHECK(ggml_is_quantized(tensor->type));
        const int64_t nrows = ggml_nelements(tensor) / tensor->ne[0];
        CHECK(ggml_quantize_chunk(tensor->type, values.data(), bytes.data(), 0, nrows, tensor->ne[0], nullptr) == bytes.size());
    }
    return bytes;
}

ggml_backend_buffer_type_t pageable_cached_buffer_type() {
    static ggml_backend_buffer_type buft = *ggml_backend_cuda_moe_cached_buffer_type();
    buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t size) {
        auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        if (buffer != nullptr) {
            buffer->buft = ggml_backend_cuda_moe_cached_buffer_type();
        }
        return buffer;
    };
    return &buft;
}

#ifdef __linux__
static void file_mmap_cached_buffer_free(ggml_backend_buffer_t buffer) {
    CHECK(munmap(buffer->context, buffer->size) == 0);
}

static const char * file_mmap_cached_buffer_type_name(ggml_backend_buffer_type_t) {
    return "CUDA_MoE_Cached_File_Mmap_Test";
}

static ggml_backend_buffer_t file_mmap_cached_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t,
        size_t size) {
    char path[] = "/tmp/test-moe-cache-mmap-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        return nullptr;
    }
    const bool sized = ftruncate(fd, static_cast<off_t>(size)) == 0;
    void * data = sized ? mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
    (void) unlink(path);
    (void) close(fd);
    if (data == MAP_FAILED) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_cuda_moe_cached_buffer_from_host_ptr(ggml_backend_cuda_moe_cached_buffer_type(), data, size);
    if (buffer == nullptr) {
        (void) munmap(data, size);
        return nullptr;
    }
    buffer->iface.free_buffer = file_mmap_cached_buffer_free;
    return buffer;
}

static size_t file_mmap_cached_buffer_type_get_alignment(ggml_backend_buffer_type_t) {
    auto * cached = ggml_backend_cuda_moe_cached_buffer_type();
    return cached->iface.get_alignment(cached);
}

static size_t file_mmap_cached_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t,
        const ggml_tensor * tensor) {
    auto * cached = ggml_backend_cuda_moe_cached_buffer_type();
    return cached->iface.get_alloc_size != nullptr ? cached->iface.get_alloc_size(cached, tensor) : ggml_nbytes(tensor);
}

static bool file_mmap_cached_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return false;
}

ggml_backend_buffer_type_t file_mmap_cached_buffer_type() {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ file_mmap_cached_buffer_type_name,
            /* .alloc_buffer     = */ file_mmap_cached_buffer_type_alloc_buffer,
            /* .get_alignment    = */ file_mmap_cached_buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ file_mmap_cached_buffer_type_get_alloc_size,
            /* .is_host          = */ file_mmap_cached_buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buffer_type.device = ggml_backend_cuda_moe_cached_buffer_type()->device;
    return &buffer_type;
}
#endif

active_grouped_dispatch_graph build_active_grouped_dispatch_graph_types(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        const std::array<ggml_type, 3> & types,
        uint32_t layout,
        bool original_direct_down_scale,
        bool original_direct_nvfp4_scales,
        uint32_t n_rows,
        uint32_t n_experts,
        uint32_t n_used,
        uint32_t n_dim,
        const active_grouped_dispatch_graph * shared_banks,
        bool concurrent_stream_fixture,
        bool original_direct_biases,
        uint32_t n_ff,
        bool mapped_host_biases,
        bool lookup_route,
        int router_variant) {
    CHECK(n_rows >= 1 && n_used >= 1 && n_used <= n_experts);
    CHECK(layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP ||
        layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ||
        layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED);
    const uint32_t ff_dim = n_ff != 0 ? n_ff : n_dim;
    CHECK(!concurrent_stream_fixture ||
        (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE && n_rows == 1 && n_used == 1));
    CHECK(!original_direct_biases || (shared_banks == nullptr && !original_direct_down_scale && !original_direct_nvfp4_scales));
    CHECK(!mapped_host_biases || original_direct_biases);
    CHECK(!lookup_route || router_variant < 0);
    const ggml_init_params weight_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 12,
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    const ggml_init_params node_params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    active_grouped_dispatch_graph result;
    result.n_experts = n_experts;
    result.n_used = n_used;
    result.n_rows = n_rows;
    CHECK(shared_banks == nullptr ||
        (shared_banks->n_experts == n_experts && shared_banks->n_used == n_used));
    if (shared_banks == nullptr) {
        result.weights.reset(ggml_init(weight_params));
    }
    result.nodes.reset(ggml_init(node_params));
    CHECK((shared_banks != nullptr || result.weights != nullptr) && result.nodes != nullptr);

    size_t shared_bank_index = 0;
    auto add_bank = [&](ggml_type type, int64_t ne0, int64_t ne1, uint32_t role, const char * name) {
        ggml_tensor * tensor = nullptr;
        if (shared_banks != nullptr) {
            CHECK(shared_bank_index < shared_banks->banks.size() && shared_bank_index < shared_banks->roles.size());
            tensor = shared_banks->banks[shared_bank_index];
            CHECK(tensor != nullptr && tensor->type == type && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
                tensor->ne[2] == n_experts && shared_banks->roles[shared_bank_index] == role);
            ++shared_bank_index;
        } else {
            tensor = ggml_new_tensor_3d(result.weights.get(), type, ne0, ne1, n_experts);
            ggml_set_name(tensor, name);
        }
        result.banks.push_back(tensor);
        result.roles.push_back(role);
        return tensor;
    };
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
        gate_up = add_bank(types[0], n_dim, 2 * ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT,
            "test_active_gate_up_weight");
    } else if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
        gate = add_bank(types[0], n_dim, ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT,
            "test_active_gate_weight");
        up = add_bank(types[1], n_dim, ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            "test_active_up_weight");
    } else {
        up = add_bank(types[0], n_dim, ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            "test_active_up_weight");
    }
    const uint32_t down_type = layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE ? 2 : 1;
    result.down = add_bank(types[down_type], ff_dim,
        n_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT,
        "test_active_down_weight");
    ggml_tensor * gate_bias = nullptr;
    ggml_tensor * up_bias = nullptr;
    ggml_tensor * gate_up_bias = nullptr;
    ggml_tensor * down_bias = nullptr;
    if (original_direct_biases) {
        const auto add_bias = [&](int64_t ne0, uint32_t role, const char * name) {
            ggml_tensor * tensor = ggml_new_tensor_2d(
                mapped_host_biases ? result.weights.get() : result.nodes.get(), GGML_TYPE_F32, ne0, n_experts);
            ggml_set_name(tensor, name);
            result.biases.push_back(tensor);
            result.bias_roles.push_back(role);
            return tensor;
        };
        if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP) {
            gate_up_bias = add_bias(2 * ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS,
                "test_active_gate_up_bias");
        } else if (layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE) {
            gate_bias = add_bias(ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, "test_active_gate_bias");
            up_bias = add_bias(ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, "test_active_up_bias");
        } else {
            up_bias = add_bias(ff_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, "test_active_up_bias");
        }
        down_bias = add_bias(n_dim, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, "test_active_down_bias");
    } else if (shared_banks != nullptr) {
        result.biases = shared_banks->biases;
        result.bias_roles = shared_banks->bias_roles;
        CHECK(result.biases.size() == result.bias_roles.size());
        for (size_t i = 0; i < result.biases.size(); ++i) {
            switch (result.bias_roles[i]) {
                case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS:    gate_bias = result.biases[i]; break;
                case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS:      up_bias = result.biases[i]; break;
                case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS: gate_up_bias = result.biases[i]; break;
                case GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS:    down_bias = result.biases[i]; break;
                default: CHECK(false);
            }
        }
    }
    if (shared_banks != nullptr) {
        result.gate_scale = shared_banks->gate_scale;
        result.up_scale = shared_banks->up_scale;
        result.down_scale = shared_banks->down_scale;
        CHECK((original_direct_nvfp4_scales && result.gate_scale != nullptr && result.up_scale != nullptr && result.down_scale != nullptr) ||
            (!original_direct_nvfp4_scales && result.gate_scale == nullptr && result.up_scale == nullptr &&
                (original_direct_down_scale == (result.down_scale != nullptr))));
    } else if (original_direct_nvfp4_scales) {
        CHECK(layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE &&
            types[0] == GGML_TYPE_NVFP4 && types[1] == GGML_TYPE_NVFP4 && types[2] == GGML_TYPE_NVFP4);
        result.gate_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        result.up_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        result.down_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        ggml_set_name(result.gate_scale, "test.active.gate_scale");
        ggml_set_name(result.up_scale, "test.active.up_scale");
        ggml_set_name(result.down_scale, "test.active.down_scale");
    } else if (original_direct_down_scale) {
        result.down_scale = ggml_new_tensor_1d(result.weights.get(), GGML_TYPE_F32, n_experts);
        ggml_set_name(result.down_scale, "test.active.down_scale");
    }

    result.input = ggml_new_tensor_3d(result.nodes.get(), GGML_TYPE_F32, n_dim, 1, n_rows);
    result.logits = ggml_new_tensor_2d(result.nodes.get(), GGML_TYPE_F32, n_experts, n_rows);
    ggml_set_name(result.input, "test.active.input");
    ggml_set_name(result.logits, "test.active.logits");
    ggml_tensor * mmid_input = result.input;
    ggml_tensor * concurrent_branch = nullptr;
    if (concurrent_stream_fixture) {
        result.concurrent_root = ggml_dup(result.nodes.get(), result.input);
        ggml_set_name(result.concurrent_root, "test.active.attn_norm");
        concurrent_branch = ggml_sqr(result.nodes.get(), result.concurrent_root);
        mmid_input = result.concurrent_root;
    }
    ggml_tensor * router_scores = result.logits;
    ggml_tensor * selected_table = nullptr;
    if (router_variant >= 0) {
        result.router_weights.reset(ggml_init(weight_params));
        CHECK(result.router_weights != nullptr && !concurrent_stream_fixture);
        const uint32_t router_width = router_variant == 2 ? n_dim * 2 : n_dim;
        const uint32_t router_experts = router_variant == 1 || router_variant == 15 ? n_experts * 2 : n_experts;
        auto * router = ggml_new_tensor_2d(result.router_weights.get(), GGML_TYPE_F32, router_width, router_experts);
        auto * norm_scale = ggml_new_tensor_1d(result.router_weights.get(), GGML_TYPE_F32, router_width);
        auto * bias = ggml_new_tensor_1d(result.router_weights.get(), GGML_TYPE_F32, router_experts);
        if (router_variant == 16) {
            selected_table = ggml_new_tensor_2d(result.router_weights.get(), GGML_TYPE_I32, n_used, 16);
        }
        ggml_tensor * lora_a = nullptr;
        ggml_tensor * lora_b = nullptr;
        auto * input = ggml_cont(result.nodes.get(), ggml_view_2d(result.nodes.get(), result.input,
            n_dim, n_rows, result.input->nb[2], 0));
        if (router_variant == 2) {
            input = ggml_concat(result.nodes.get(), input, input, 0);
        }
        if (router_variant != 3 && router_variant != 7) {
            input = ggml_mul(result.nodes.get(), ggml_rms_norm(result.nodes.get(), input, 1e-6f), norm_scale);
        }
        router_scores = ggml_mul_mat(result.nodes.get(), router, input);
        if (router_variant == 6 || router_variant == 7) {
            lora_a = ggml_new_tensor_2d(result.router_weights.get(), GGML_TYPE_F32, router_width, 16);
            lora_b = ggml_new_tensor_2d(result.router_weights.get(), GGML_TYPE_F32, 16, router_experts);
            auto * correction = ggml_mul_mat(result.nodes.get(), lora_b, ggml_mul_mat(result.nodes.get(), lora_a, input));
            router_scores = ggml_add(result.nodes.get(), router_scores, ggml_scale(result.nodes.get(), correction, 0.1f));
        }
        if (router_variant == 4) {
            router_scores = ggml_add(result.nodes.get(), ggml_sigmoid(result.nodes.get(), router_scores), bias);
        }
        if (router_variant == 5) {
            result.router_rows = ggml_new_tensor_1d(result.nodes.get(), GGML_TYPE_I32, n_rows);
            ggml_set_input(result.router_rows);
            router_scores = ggml_get_rows(result.nodes.get(), router_scores, result.router_rows);
        }
        router_scores->flags |= GGML_TENSOR_FLAG_MOE_ROUTER;
        result.router_buffer.reset(ggml_backend_alloc_ctx_tensors(result.router_weights.get(), backend));
        CHECK(result.router_buffer != nullptr);
        ggml_backend_buffer_set_usage(result.router_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> values(router_width * router_experts), scales(router_width, 1.0f), biases(router_experts);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = std::sin(float(i * 13 + 7)) / 8.0f;
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = float((i * 7) % 11) / 17.0f;
        }
        ggml_backend_tensor_set(router, values.data(), 0, ggml_nbytes(router));
        ggml_backend_tensor_set(norm_scale, scales.data(), 0, ggml_nbytes(norm_scale));
        ggml_backend_tensor_set(bias, biases.data(), 0, ggml_nbytes(bias));
        if (selected_table != nullptr) {
            std::vector<int32_t> values(ggml_nelements(selected_table));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = (i * 3) % n_experts;
            }
            ggml_backend_tensor_set(selected_table, values.data(), 0, ggml_nbytes(selected_table));
        }
        for (auto * weight : {lora_a, lora_b}) {
            if (weight != nullptr) {
                const auto bytes = cached_fusion_test_data(weight, 123);
                ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());
            }
        }
    }
    if (!lookup_route) {
        result.ids = ggml_argsort_top_k(result.nodes.get(), router_scores, n_used);
    }
    if (router_variant == 1 || router_variant == 15) {
        result.ids = ggml_cast(result.nodes.get(), ggml_scale(result.nodes.get(),
            ggml_cast(result.nodes.get(), result.ids, GGML_TYPE_F32), 0.5f), GGML_TYPE_I32);
    }
    if (selected_table != nullptr) {
        result.router_rows = ggml_new_tensor_1d(result.nodes.get(), GGML_TYPE_I32, n_rows);
        ggml_set_input(result.router_rows);
        result.ids = ggml_get_rows(result.nodes.get(), selected_table, result.router_rows);
    }
    if (lookup_route) {
        CHECK(shared_banks == nullptr && !concurrent_stream_fixture);
        result.lookup_weights.reset(ggml_init(weight_params));
        result.lookup_table = ggml_new_tensor_2d(result.lookup_weights.get(), GGML_TYPE_I32, n_used, 16);
        result.lookup_buffer.reset(ggml_backend_alloc_ctx_tensors(result.lookup_weights.get(), backend));
        CHECK(result.lookup_buffer != nullptr);
        ggml_backend_buffer_set_usage(result.lookup_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        result.token_ids = ggml_new_tensor_1d(result.nodes.get(), GGML_TYPE_I32, n_rows);
        ggml_set_input(result.token_ids);
        result.ids = ggml_get_rows(result.nodes.get(), result.lookup_table, result.token_ids);
    }
    ggml_set_name(result.ids, "test.active.ids");
    ggml_tensor * hidden = nullptr;
    if (gate_up != nullptr) {
        hidden = ggml_mul_mat_id(result.nodes.get(), gate_up, mmid_input, result.ids);
        result.readers.push_back(hidden);
        if (gate_up_bias != nullptr) {
            hidden = ggml_add_id(result.nodes.get(), hidden, gate_up_bias, result.ids);
        }
        hidden = ggml_dup(result.nodes.get(), hidden);
        hidden = ggml_glu(result.nodes.get(), hidden, GGML_GLU_OP_SWIGLU, false);
    } else if (gate != nullptr) {
        result.gate_output = ggml_mul_mat_id(result.nodes.get(), gate, mmid_input, result.ids);
        result.up_output = ggml_mul_mat_id(result.nodes.get(), up, mmid_input, result.ids);
        result.readers.push_back(result.gate_output);
        result.readers.push_back(result.up_output);
        ggml_tensor * gate_input = result.gate_output;
        ggml_tensor * up_input = result.up_output;
        if (gate_bias != nullptr) {
            gate_input = ggml_add_id(result.nodes.get(), gate_input, gate_bias, result.ids);
            up_input = ggml_add_id(result.nodes.get(), up_input, up_bias, result.ids);
        }
        if (result.gate_scale != nullptr) {
            result.gate_scale_reshape = ggml_reshape_3d(result.nodes.get(), result.gate_scale, 1, n_experts, 1);
            result.gate_scale_repeat = ggml_repeat_4d(result.nodes.get(), result.gate_scale_reshape, 1, n_experts, n_rows, 1);
            result.gate_scale_rows = ggml_get_rows(result.nodes.get(), result.gate_scale_repeat, result.ids);
            gate_input = ggml_mul(result.nodes.get(), result.gate_output, result.gate_scale_rows);
            result.up_scale_reshape = ggml_reshape_3d(result.nodes.get(), result.up_scale, 1, n_experts, 1);
            result.up_scale_repeat = ggml_repeat_4d(result.nodes.get(), result.up_scale_reshape, 1, n_experts, n_rows, 1);
            result.up_scale_rows = ggml_get_rows(result.nodes.get(), result.up_scale_repeat, result.ids);
            up_input = ggml_mul(result.nodes.get(), result.up_output, result.up_scale_rows);
        }
        hidden = ggml_glu_split(result.nodes.get(), gate_input, up_input, GGML_GLU_OP_SWIGLU);
    } else {
        result.up_output = ggml_mul_mat_id(result.nodes.get(), up, mmid_input, result.ids);
        result.readers.push_back(result.up_output);
        ggml_tensor * up_input = result.up_output;
        if (up_bias != nullptr) {
            up_input = ggml_add_id(result.nodes.get(), up_input, up_bias, result.ids);
        }
        hidden = ggml_sqr(result.nodes.get(), up_input);
    }
    result.down_output = ggml_mul_mat_id(result.nodes.get(), result.down, hidden, result.ids);
    result.readers.push_back(result.down_output);
    result.output = result.down_output;
    if (down_bias != nullptr) {
        result.output = ggml_add_id(result.nodes.get(), result.output, down_bias, result.ids);
    }
    if (result.down_scale != nullptr) {
        result.down_scale_reshape = ggml_reshape_3d(result.nodes.get(), result.down_scale, 1, n_experts, 1);
        result.down_scale_repeat = ggml_repeat_4d(result.nodes.get(), result.down_scale_reshape, 1, n_experts, n_rows, 1);
        result.down_scale_rows = ggml_get_rows(result.nodes.get(), result.down_scale_repeat, result.ids);
        result.output = ggml_mul(result.nodes.get(), result.down_output, result.down_scale_rows);
    }
    ggml_set_name(result.output, "test.active.output");
    result.graph = ggml_new_graph_custom(result.nodes.get(), 64, false);
    ggml_build_forward_expand(result.graph, result.output);
    if (concurrent_stream_fixture) {
        ggml_build_forward_expand(result.graph, concurrent_branch);
        CHECK((concurrent_branch->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);
        const auto move_node = [&](ggml_tensor * node, int32_t target) {
            int32_t source = -1;
            for (int32_t index = target; index < result.graph->n_nodes; ++index) {
                if (result.graph->nodes[index] == node) {
                    source = index;
                    break;
                }
            }
            CHECK(source >= target);
            memmove(result.graph->nodes + target + 1, result.graph->nodes + target,
                (source - target) * sizeof(result.graph->nodes[0]));
            result.graph->nodes[target] = node;
        };
        move_node(result.ids->view_src, 0);
        move_node(result.ids, 1);
        CHECK(result.graph->nodes[2] == result.concurrent_root);
        move_node(concurrent_branch, 3);
        candidate_rebuild_graph_uses(result.graph);
    }
    candidate_stamp_execution(result.graph, GGML_GRAPH_EXECUTION_DOMAIN_MAIN,
        GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT, n_rows, n_rows);
    if (shared_banks == nullptr) {
        result.weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(result.weights.get(), weight_buft));
    } else {
        CHECK(shared_bank_index == shared_banks->banks.size());
    }
    result.node_buffer.reset(ggml_backend_alloc_ctx_tensors(result.nodes.get(), backend));
    CHECK((shared_banks != nullptr || result.weight_buffer != nullptr) && result.node_buffer != nullptr);
    if (result.weight_buffer != nullptr) {
        ggml_backend_buffer_set_usage(result.weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if (result.router_rows != nullptr) {
        std::vector<int32_t> rows(n_rows);
        std::iota(rows.begin(), rows.end(), 0);
        ggml_backend_tensor_set(result.router_rows, rows.data(), 0, ggml_nbytes(result.router_rows));
    }
    return result;
}

active_grouped_dispatch_graph build_active_grouped_dispatch_graph(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t weight_buft,
        ggml_type type,
        uint32_t layout,
        bool original_direct_down_scale,
        uint32_t n_rows,
        uint32_t n_experts,
        uint32_t n_used,
        uint32_t n_dim,
        const active_grouped_dispatch_graph * shared_banks,
        bool concurrent_stream_fixture) {
    return build_active_grouped_dispatch_graph_types(
        backend, weight_buft, {type, type, type}, layout, original_direct_down_scale, false,
        n_rows, n_experts, n_used, n_dim, shared_banks, concurrent_stream_fixture, false);
}

static constexpr int32_t active_grouped_route_variants[3][4][2] = {
    {
        {3, 5},
        {5, 2},
        {3, 1},
        {7, 5},
    },
    {
        {2, 6},
        {6, 4},
        {0, 2},
        {4, 1},
    },
    {
        {1, 7},
        {3, 1},
        {7, 4},
        {2, 3},
    },
};

static constexpr int32_t active_grouped_practical_route_variants[3][4][8] = {
    {
        {3, 5, 17, 29, 41, 53, 65, 77},
        {5, 2, 18, 30, 42, 54, 66, 78},
        {3, 1, 19, 31, 43, 55, 67, 79},
        {7, 5, 20, 32, 44, 56, 68, 80},
    },
    {
        {82, 6, 21, 33, 45, 57, 69, 81},
        {6, 4, 22, 34, 46, 58, 70, 83},
        {82, 2, 23, 35, 47, 59, 71, 84},
        {4, 1, 24, 36, 48, 60, 72, 85},
    },
    {
        {9, 87, 25, 37, 49, 61, 73, 86},
        {11, 9, 26, 38, 50, 62, 74, 88},
        {87, 13, 27, 39, 51, 63, 75, 89},
        {15, 11, 28, 40, 52, 64, 76, 90},
    },
};

int32_t active_grouped_route(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant,
        uint32_t row,
        uint32_t route) {
    CHECK(route_variant < 3 && row < graph.n_rows && route < graph.n_used);
    if (graph.n_experts == 8 && (graph.n_used == 1 || graph.n_used == 2)) {
        return active_grouped_route_variants[route_variant][row][route];
    }
    CHECK((graph.n_experts == 128 || graph.n_experts == 256) && graph.n_used == 8);
    if (graph.n_experts == 256) {
        return (64 * route_variant + 8 * row + route) % graph.n_experts;
    }
    if (row < 4) {
        return active_grouped_practical_route_variants[route_variant][row][route];
    }
    return (31 * route_variant + 7 * row + 13 * route) % graph.n_experts;
}

static void set_active_grouped_contract_routes(active_grouped_dispatch_graph & graph) {
    CHECK(graph.ids != nullptr && graph.ids->view_src != nullptr);
    ggml_tensor * sorted = graph.ids->view_src;
    CHECK(sorted->type == GGML_TYPE_I32 && sorted->ne[0] == graph.n_experts && sorted->ne[1] == graph.n_rows);
    std::vector<int32_t> values(ggml_nelements(sorted));
    for (uint32_t row = 0; row < graph.n_rows; ++row) {
        std::vector<bool> used(graph.n_experts, false);
        uint32_t offset = row * graph.n_experts;
        uint32_t index = 0;
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            const int32_t expert = active_grouped_route(graph, 0, row, route);
            CHECK(expert >= 0 && static_cast<uint32_t>(expert) < graph.n_experts && !used[expert]);
            values[offset + index++] = expert;
            used[expert] = true;
        }
        for (uint32_t expert = 0; expert < graph.n_experts; ++expert) {
            if (!used[expert]) {
                values[offset + index++] = expert;
            }
        }
        CHECK(index == graph.n_experts);
    }
    ggml_backend_tensor_set(sorted, values.data(), 0, ggml_nbytes(sorted));
}

static std::vector<int32_t> active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant) {
    CHECK(route_variant < 3);
    std::vector<int32_t> routes(graph.n_rows * graph.n_used);
    for (uint32_t row = 0; row < graph.n_rows; ++row) {
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            routes[row * graph.n_used + route] = active_grouped_route(graph, route_variant, row, route);
        }
    }
    return routes;
}

static std::vector<float> active_grouped_logits(
        const active_grouped_dispatch_graph & graph,
        const std::vector<int32_t> & routes) {
    CHECK(graph.logits != nullptr && graph.logits->ne[0] == graph.n_experts &&
        graph.logits->ne[1] == graph.n_rows && graph.n_rows >= 1 && routes.size() == graph.n_rows * graph.n_used);
    std::vector<float> result(graph.logits->ne[0] * graph.logits->ne[1]);
    for (uint32_t row = 0; row < graph.logits->ne[1]; ++row) {
        std::vector<bool> used(graph.n_experts, false);
        for (uint32_t expert = 0; expert < graph.logits->ne[0]; ++expert) {
            result[row * graph.logits->ne[0] + expert] = -static_cast<float>(expert + 1);
        }
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            const int32_t expert = routes[row * graph.n_used + route];
            CHECK(expert >= 0 && static_cast<uint32_t>(expert) < graph.n_experts && !used[expert]);
            used[expert] = true;
            result[row * graph.logits->ne[0] + expert] = static_cast<float>(graph.n_used - route + 8);
        }
    }
    return result;
}

static std::vector<float> active_grouped_logits(
        const active_grouped_dispatch_graph & graph,
        uint32_t route_variant = 0) {
    return active_grouped_logits(graph, active_grouped_routes(graph, route_variant));
}

void set_active_grouped_dispatch_routes(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        const std::vector<int32_t> & routes) {
    CHECK(!graphs.empty());
    const auto logits = active_grouped_logits(*graphs[0], routes);
    for (auto * graph : graphs) {
        CHECK(graph != nullptr && graph->n_rows == graphs[0]->n_rows && graph->n_used == graphs[0]->n_used &&
            graph->n_experts == graphs[0]->n_experts && ggml_nbytes(graph->logits) == logits.size() * sizeof(float));
        ggml_backend_tensor_set(graph->logits, logits.data(), 0, logits.size() * sizeof(float));
    }
}

void set_active_grouped_dispatch_logits(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        uint32_t route_variant) {
    CHECK(!graphs.empty());
    set_active_grouped_dispatch_routes(graphs, active_grouped_routes(*graphs[0], route_variant));
}

void initialize_active_grouped_dispatch_graphs(
        const std::vector<active_grouped_dispatch_graph *> & graphs,
        const std::vector<std::vector<uint8_t>> * bank_data) {
    CHECK(graphs.size() >= 2);
    const auto logits = graphs[0]->router_weights != nullptr ?
        std::vector<float>(ggml_nelements(graphs[0]->logits)) : active_grouped_logits(*graphs[0]);
    for (auto * graph : graphs) {
        CHECK(graph->banks.size() == graphs[0]->banks.size());
    }
    CHECK(bank_data == nullptr || bank_data->size() == graphs[0]->banks.size());
    for (size_t bank = 0; bank < graphs[0]->banks.size(); ++bank) {
        const auto generated = bank_data == nullptr ? cached_fusion_test_data(graphs[0]->banks[bank], bank + 101) : std::vector<uint8_t>();
        const auto & bytes = bank_data != nullptr ? (*bank_data)[bank] : generated;
        for (auto * graph : graphs) {
            CHECK(graph->banks[bank]->type == graphs[0]->banks[bank]->type);
            CHECK(graph->roles[bank] == graphs[0]->roles[bank]);
            CHECK(ggml_nbytes(graph->banks[bank]) == ggml_nbytes(graphs[0]->banks[bank]));
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                CHECK(graph->banks[bank]->ne[dim] == graphs[0]->banks[bank]->ne[dim]);
                CHECK(graph->banks[bank]->nb[dim] == graphs[0]->banks[bank]->nb[dim]);
            }
            ggml_backend_tensor_set(graph->banks[bank], bytes.data(), 0, bytes.size());
        }
    }
    const std::array<ggml_tensor *, 3> first_scales = {
        graphs[0]->gate_scale, graphs[0]->up_scale, graphs[0]->down_scale,
    };
    for (uint32_t scale_index = 0; scale_index < first_scales.size(); ++scale_index) {
        if (first_scales[scale_index] == nullptr) {
            for (auto * graph : graphs) {
                const std::array<ggml_tensor *, 3> scales = {graph->gate_scale, graph->up_scale, graph->down_scale};
                CHECK(scales[scale_index] == nullptr);
            }
            continue;
        }
        const auto bytes = cached_fusion_test_data(first_scales[scale_index], 149 + scale_index);
        for (auto * graph : graphs) {
            const std::array<ggml_tensor *, 3> scales = {graph->gate_scale, graph->up_scale, graph->down_scale};
            CHECK(scales[scale_index] != nullptr && ggml_nbytes(scales[scale_index]) == bytes.size());
            ggml_backend_tensor_set(scales[scale_index], bytes.data(), 0, bytes.size());
        }
    }
    for (size_t bias_index = 0; bias_index < graphs[0]->biases.size(); ++bias_index) {
        const auto bytes = cached_fusion_test_data(graphs[0]->biases[bias_index], 157 + bias_index);
        for (auto * graph : graphs) {
            CHECK(graph->biases.size() == graphs[0]->biases.size() && graph->bias_roles.size() == graph->biases.size());
            CHECK(graph->bias_roles[bias_index] == graphs[0]->bias_roles[bias_index]);
            CHECK(ggml_nbytes(graph->biases[bias_index]) == bytes.size());
            ggml_backend_tensor_set(graph->biases[bias_index], bytes.data(), 0, bytes.size());
        }
    }
    const auto input = cached_fusion_test_data(graphs[0]->input, 131);
    for (auto * graph : graphs) {
        CHECK(graph->readers.size() == graph->banks.size());
        CHECK(graph->readers.back() == graph->down_output && graph->down_output->src[0] == graph->down);
        int32_t previous_reader = -1;
        for (ggml_tensor * reader : graph->readers) {
            int32_t reader_index = -1;
            for (int32_t node = 0; node < ggml_graph_n_nodes(graph->graph); ++node) {
                if (ggml_graph_node(graph->graph, node) == reader) {
                    reader_index = node;
                    break;
                }
            }
            CHECK(reader_index > previous_reader);
            previous_reader = reader_index;
        }
        CHECK(graph->input->type == graphs[0]->input->type && ggml_nbytes(graph->input) == ggml_nbytes(graphs[0]->input));
        CHECK(graph->logits->type == graphs[0]->logits->type && ggml_nbytes(graph->logits) == ggml_nbytes(graphs[0]->logits));
        CHECK(graph->ids->type == graphs[0]->ids->type && ggml_nbytes(graph->ids) == ggml_nbytes(graphs[0]->ids));
        CHECK(ggml_nbytes(graph->output) == ggml_nbytes(graphs[0]->output));
        ggml_backend_tensor_set(graph->input, input.data(), 0, input.size());
        ggml_backend_tensor_set(graph->logits, logits.data(), 0, logits.size() * sizeof(float));
    }
}

void initialize_active_grouped_dispatch_graph(active_grouped_dispatch_graph & graph, size_t salt) {
    const auto logits = active_grouped_logits(graph);
    for (size_t bank = 0; bank < graph.banks.size(); ++bank) {
        const auto bytes = cached_fusion_test_data(graph.banks[bank], bank + salt);
        ggml_backend_tensor_set(graph.banks[bank], bytes.data(), 0, bytes.size());
    }
    const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
    for (uint32_t scale_index = 0; scale_index < scales.size(); ++scale_index) {
        if (scales[scale_index] != nullptr) {
            const auto bytes = cached_fusion_test_data(scales[scale_index], salt + 48 + scale_index);
            ggml_backend_tensor_set(scales[scale_index], bytes.data(), 0, bytes.size());
        }
    }
    for (size_t bias_index = 0; bias_index < graph.biases.size(); ++bias_index) {
        const auto bytes = cached_fusion_test_data(graph.biases[bias_index], salt + 56 + bias_index);
        ggml_backend_tensor_set(graph.biases[bias_index], bytes.data(), 0, bytes.size());
    }
    const auto input = cached_fusion_test_data(graph.input, salt + 30);
    ggml_backend_tensor_set(graph.input, input.data(), 0, input.size());
    ggml_backend_tensor_set(graph.logits, logits.data(), 0, logits.size() * sizeof(float));
}

void register_active_grouped_dispatch(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t layout,
        uint32_t n_slots,
        bool auxiliary_first) {
    std::array<ggml_backend_moe_candidate_bank_v1, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS> banks = {};
    uint32_t n_banks = 0;
    if (auxiliary_first && graph.down_scale != nullptr) {
        banks[n_banks++] = {graph.down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    }
    for (size_t i = 0; i < graph.banks.size(); ++i) {
        banks[n_banks++] = {graph.banks[i], graph.roles[i], 0};
    }
    for (size_t i = 0; i < graph.biases.size(); ++i) {
        banks[n_banks++] = {graph.biases[i], graph.bias_roles[i], 0};
    }
    if (!auxiliary_first && graph.down_scale != nullptr) {
        banks[n_banks++] = {graph.down_scale, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, 0};
    }
    const ggml_backend_moe_candidate_group_v1 group = {
        banks.data(), n_banks, layout, 0, 0,
    };
    const auto snapshot = candidate_snapshot(n_slots, &group, 1);
    CHECK(ggml_backend_cuda_moe_candidate_replace_v1(backend, &snapshot) == GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED);
}

int32_t replace_active_grouped_nvfp4_dispatch_v2(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        int32_t omitted_scale) {
    CHECK(graph.banks.size() == 3 && graph.gate_scale != nullptr && graph.up_scale != nullptr && graph.down_scale != nullptr);
    const ggml_backend_moe_candidate_group_v2 group = {
        GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE,
        GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY,
        GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_NONE,
        0,
    };
    std::array<ggml_backend_moe_candidate_tensor_v2, 6> tensors = {};
    uint32_t n_tensors = 0;
    for (uint32_t i = 0; i < graph.banks.size(); ++i) {
        tensors[n_tensors++] = {
            graph.banks[i], 0, graph.roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE,
            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0,
        };
    }
    const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
    const std::array<uint32_t, 3> scale_roles = {
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE,
        GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
    };
    for (uint32_t i = 0; i < scales.size(); ++i) {
        if (static_cast<int32_t>(i) == omitted_scale) {
            continue;
        }
        tensors[n_tensors++] = {
            scales[i], 0, scale_roles[i],
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER, 0,
        };
    }
    const auto snapshot = candidate_snapshot_v2(n_slots, &group, 1, tensors.data(), n_tensors);
    return ggml_backend_cuda_moe_candidate_replace_v2(backend, &snapshot);
}

std::vector<float> active_grouped_intermediate_sentinel(active_grouped_dispatch_graph & graph) {
    if (graph.gate_output == nullptr) {
        return {};
    }
    CHECK(graph.up_output != nullptr && ggml_nelements(graph.gate_output) == ggml_nelements(graph.up_output));
    std::vector<float> sentinel(ggml_nelements(graph.gate_output), -12345.25f);
    ggml_backend_tensor_set(graph.gate_output, sentinel.data(), 0, ggml_nbytes(graph.gate_output));
    ggml_backend_tensor_set(graph.up_output, sentinel.data(), 0, ggml_nbytes(graph.up_output));
    return sentinel;
}

void check_active_grouped_intermediates(
        const active_grouped_dispatch_graph & graph,
        const std::vector<float> & sentinel,
        bool skipped) {
    if (graph.gate_output == nullptr) {
        CHECK(sentinel.empty());
        return;
    }
    std::vector<float> gate(ggml_nelements(graph.gate_output));
    std::vector<float> up(ggml_nelements(graph.up_output));
    ggml_backend_tensor_get(graph.gate_output, gate.data(), 0, ggml_nbytes(graph.gate_output));
    ggml_backend_tensor_get(graph.up_output, up.data(), 0, ggml_nbytes(graph.up_output));
    CHECK((gate == sentinel) == skipped);
    CHECK((up == sentinel) == skipped);
}

uint64_t active_grouped_legacy_op_count(ggml_backend_t backend, bool is_decode) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    return ggml_cuda_moe_grouped_context_test_access::legacy_op_count(*context, is_decode);
}

void check_active_grouped_debug_telemetry(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        uint64_t expected_loaded_experts,
        uint64_t expected_calls) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    const auto telemetry = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(telemetry.registered == 1 && telemetry.covered == 1);
    CHECK(telemetry.plan_calls == expected_calls + 1 && telemetry.plan_compiles == 2 &&
        telemetry.plan_reuses == expected_calls - 1);
    CHECK(telemetry.calls == expected_calls && telemetry.ready == expected_calls && telemetry.completed == expected_calls);
    CHECK(telemetry.ready_min == expected_calls && telemetry.ready_max == expected_calls);
    CHECK(telemetry.completed_min == expected_calls && telemetry.completed_max == expected_calls);
    CHECK(telemetry.admitted_banks == expected_calls * graph.banks.size());
    CHECK(telemetry.fallback == 0 && telemetry.rollback == 0);
    CHECK(telemetry.prepare_error == 0 && telemetry.finish_error == 0);
    if (expected_loaded_experts == 0) {
        std::vector<bool> used(graph.n_experts, false);
        for (uint32_t row = 0; row < graph.n_rows; ++row) {
            for (uint32_t route = 0; route < graph.n_used; ++route) {
                const uint32_t expert = active_grouped_route(graph, 0, row, route);
                if (!used[expert]) {
                    used[expert] = true;
                    ++expected_loaded_experts;
                }
            }
        }
    }
    CHECK(telemetry.h2d_banks == expected_loaded_experts * graph.banks.size());
    uint64_t bytes_per_expert = 0;
    for (const ggml_tensor * bank : graph.banks) {
        bytes_per_expert += bank->nb[2];
    }
    CHECK(telemetry.h2d_bytes == expected_loaded_experts * bytes_per_expert);

    const auto reset = ggml_cuda_moe_grouped_context_test_access::take_grouped_debug_telemetry(*context);
    CHECK(reset.registered == 1 && reset.covered == 0 && reset.plan_calls == 0);
    CHECK(reset.calls == 0 && reset.ready == 0 && reset.completed == 0);
    CHECK(reset.ready_min == 0 && reset.ready_max == 0);
    CHECK(reset.completed_min == 0 && reset.completed_max == 0);
    CHECK(reset.admitted_banks == 0 && reset.h2d_banks == 0 && reset.h2d_bytes == 0);
}

void check_active_grouped_legacy_caches(
        ggml_backend_t backend,
        const active_grouped_dispatch_graph & graph,
        bool registered_source,
        bool expect_slot_activity) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    for (ggml_tensor * bank : graph.banks) {
        auto lease = context->acquire_legacy_cache(bank);
        CHECK(lease && lease.get() != nullptr);
        CHECK(lease.acquisition().registered_source == registered_source);
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        ggml_cuda_moe_cache_stats(lease.get(), &hits, &misses, &evictions);
        CHECK((hits + misses > 0) == expect_slot_activity);
    }
}

void check_active_grouped_contract(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint32_t n_slots,
        bool auxiliary_first,
        bool expect_compact_mmvq) {
    auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
    CHECK(context != nullptr);
    set_active_grouped_contract_routes(graph);
    ggml_backend_synchronize(backend);

    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    const auto coverage = candidate_certify_graph(*context, graph.graph);
    CHECK(context->prepare_graph_execution(
        graph.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_COMPILED);
    CHECK(execution.size() == 1);
    if (expect_compact_mmvq) {
        CHECK(graph.n_rows >= 2 && graph.n_rows <= 4);
        for (uint32_t bank_index = 0; bank_index < graph.banks.size(); ++bank_index) {
            const auto capability = ggml_cuda_moe_grouped_context_test_access::graph_bank_capability(*plan, 0, bank_index);
            CHECK(capability.tensor == graph.banks[bank_index] &&
                capability.consumer == GGML_CUDA_MMID_CONSUMER_MMVQ &&
                capability.reason == GGML_CUDA_MMID_CAPABILITY_OK &&
                capability.equivalence_reason == GGML_CUDA_MMID_CAPABILITY_OK &&
                capability.phase == GGML_CUDA_MMID_PHASE_PREFILL &&
                capability.mapping == GGML_CUDA_MMID_MAPPING_DIRECT &&
                capability.n_rows == graph.n_rows && capability.n_routes == graph.n_rows * graph.n_used &&
                capability.n_experts == graph.n_experts && capability.grouped_ne[2] == n_slots &&
                capability.n_slots == n_slots);
        }
    }
    CHECK(context->prepare_graph_execution(
        graph.graph, 801, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNCHANGED, &plan, &execution,
        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint) ==
        GGML_CUDA_MOE_GRAPH_PREPARE_REUSED);
    CHECK(execution.size() == 1);
    if (graph.down_scale != nullptr) {
        ggml_cuda_moe_graph_execution unknown;
        CHECK(context->bind_graph_plan(
            graph.graph, 802, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
        CHECK(unknown.find_group(graph.down_output, nullptr) != nullptr);

        const std::array<std::array<ggml_tensor *, 3>, 3> scale_chains = {{
            {graph.gate_scale_reshape, graph.gate_scale_repeat, graph.gate_scale_rows},
            {graph.up_scale_reshape, graph.up_scale_repeat, graph.up_scale_rows},
            {graph.down_scale_reshape, graph.down_scale_repeat, graph.down_scale_rows},
        }};
        for (const auto & scale_chain : scale_chains) {
            if (scale_chain[0] == nullptr) {
                continue;
            }
            ggml_tensor * saved_scale = scale_chain[0]->src[0];
            scale_chain[0]->src[0] = graph.input;
            CHECK(!context->bind_graph_plan(
                graph.graph, 803, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
            scale_chain[0]->src[0] = saved_scale;

            for (ggml_tensor * auxiliary : scale_chain) {
                int32_t node_index = -1;
                for (int32_t i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
                    if (ggml_graph_node(graph.graph, i) == auxiliary) {
                        node_index = i;
                        break;
                    }
                }
                CHECK(node_index >= 0);
                ggml_tensor * saved_node = graph.graph->nodes[node_index];
                graph.graph->nodes[node_index] = graph.output;
                CHECK(!context->bind_graph_plan(
                    graph.graph, 804, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                    coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                graph.graph->nodes[node_index] = saved_node;
            }

            if (graph.gate_scale_reshape != nullptr) {
                for (ggml_tensor * auxiliary : scale_chain) {
                    const ggml_type saved_type = auxiliary->type;
                    auxiliary->type = GGML_TYPE_BF16;
                    CHECK(!context->bind_graph_plan(
                        graph.graph, 806, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                    auxiliary->type = saved_type;

                    const int64_t saved_ne = auxiliary->ne[3];
                    auxiliary->ne[3]++;
                    CHECK(!context->bind_graph_plan(
                        graph.graph, 807, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                    auxiliary->ne[3] = saved_ne;

                    const size_t saved_nb = auxiliary->nb[3];
                    auxiliary->nb[3]++;
                    CHECK(!context->bind_graph_plan(
                        graph.graph, 808, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
                        coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
                    auxiliary->nb[3] = saved_nb;
                }
            }
        }
        CHECK(context->bind_graph_plan(
            graph.graph, 809, GGML_CUDA_MOE_GRAPH_PROPERTIES_UNKNOWN, *plan, &unknown,
            coverage.epoch, coverage.nodes, coverage.mmid_count, coverage.mmid_fingerprint));
        CHECK(unknown.find_group(graph.down_output, nullptr) != nullptr);
    }
    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(execution.resolve_streams(candidate_test_graph_stream, stream));
    CHECK(context->begin_graph_dispatch(&execution, true));

    ggml_cuda_moe_graph_binding first_binding;
    auto * group = execution.find_group(graph.readers[0], &first_binding);
    CHECK(group != nullptr && group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ARMED);
    CHECK(context->prepare_graph_group(group, first_binding, graph.readers[0], stream) == GGML_CUDA_MOE_GROUPED_DECODE_READY);
    CHECK(group->state == GGML_CUDA_MOE_GRAPH_GROUP_GROUPED_ACTIVE && group->transaction.transaction_token != 0);
    CHECK(group->remapped_ids != nullptr && group->n_slots == n_slots);
    if (graph.gate_scale != nullptr) {
        const std::array<ggml_tensor *, 3> scales = {graph.gate_scale, graph.up_scale, graph.down_scale};
        const std::array<uint32_t, 3> roles = {
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE,
        };
        CHECK(group->n_auxiliary_shadows == scales.size());
        for (uint32_t scale = 0; scale < scales.size(); ++scale) {
            CHECK(group->auxiliary_tensors[scale] == scales[scale] && group->auxiliary_data[scale] != nullptr &&
                group->auxiliary_roles[scale] == roles[scale]);
            CHECK(group->auxiliary_data[scale] ==
                ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(*context, group->key.candidate, scales[scale]));
        }
    } else if (!graph.biases.empty()) {
        CHECK(group->n_auxiliary_shadows == graph.biases.size());
        for (uint32_t bias = 0; bias < graph.biases.size(); ++bias) {
            CHECK(group->auxiliary_tensors[bias] == graph.biases[bias] && group->auxiliary_data[bias] != nullptr &&
                group->auxiliary_roles[bias] == graph.bias_roles[bias]);
            CHECK(group->auxiliary_data[bias] ==
                ggml_cuda_moe_grouped_context_test_access::device_auxiliary_data(
                    *context, group->key.candidate, graph.biases[bias]));
        }
    } else {
        CHECK(group->n_auxiliary_shadows == 0);
    }
    const int32_t * remapped_ids = group->remapped_ids;

    for (size_t i = 0; i < graph.readers.size(); ++i) {
        ggml_cuda_moe_graph_binding binding;
        CHECK(execution.find_group(graph.readers[i], &binding) == group);
        CHECK(binding.bank_index == i + (auxiliary_first ? 1 : 0) && binding.slot_index == i && binding.role == graph.roles[i]);
        CHECK(binding.key.ids.tensor == graph.ids && group->remapped_ids == remapped_ids);
        CHECK(!context->acquire_legacy_cache(graph.banks[i], nullptr, &group->authority));
    }

    ggml_cuda_moe_grouped_resource_info info;
    CHECK(context->get_group_resources(group->transaction.acquisition, &info));
    CHECK(info.transaction_active && info.n_banks == graph.banks.size() && info.n_slots == n_slots);
    for (uint32_t slot_index = 0; slot_index < info.n_banks; ++slot_index) {
        ggml_cuda_moe_grouped_bank_descriptor descriptor;
        CHECK(context->get_group_resource_bank(group->transaction, slot_index, &descriptor));
        CHECK(descriptor.tensor == graph.banks[slot_index] && descriptor.role == graph.roles[slot_index]);
    }
    const auto acquisition = group->transaction.acquisition;
    ggml_cuda_moe_graph_binding last_binding;
    CHECK(execution.find_group(graph.readers.back(), &last_binding) == group);
    CHECK(context->finish_graph_group(group, last_binding, graph.readers.back(), stream));
    CHECK(context->finish_graph_dispatch(&execution));
    CHECK(context->get_group_resources(acquisition, &info) && !info.transaction_active);
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamDestroy(stream));

    ggml_cuda_moe_candidate_group_key key;
    uint64_t clock_bound = 0;
    CHECK(context->find_down_group_key(graph.down, &key));
    CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
    CHECK(clock_bound == graph.n_rows * graph.n_used);
}

std::vector<float> run_active_grouped_dispatch(
        ggml_backend_t backend,
        active_grouped_dispatch_graph & graph,
        uint64_t expected_clock,
        bool f3_skipped,
        bool down_skipped) {
    const auto sentinel = active_grouped_intermediate_sentinel(graph);
    std::vector<float> down_sentinel;
    if (down_skipped) {
        CHECK(graph.output != graph.down_output);
        down_sentinel.resize(ggml_nelements(graph.down_output), -34567.75f);
        ggml_backend_tensor_set(graph.down_output, down_sentinel.data(), 0, ggml_nbytes(graph.down_output));
    }
    CHECK(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    check_active_grouped_intermediates(graph, sentinel, f3_skipped);
    if (down_skipped) {
        std::vector<float> down(ggml_nelements(graph.down_output));
        ggml_backend_tensor_get(graph.down_output, down.data(), 0, ggml_nbytes(graph.down_output));
        CHECK(down == down_sentinel);
    }
    if (expected_clock != 0) {
        auto * context = ggml_cuda_moe_grouped_context_for_test(backend);
        ggml_cuda_moe_candidate_group_key key;
        CHECK(context != nullptr && context->find_down_group_key(graph.down, &key));
        CHECK(ggml_cuda_moe_grouped_context_test_access::has_device_resource(*context, key));
        uint64_t clock_bound = 0;
        CHECK(ggml_cuda_moe_grouped_context_test_access::get_clock_bound(*context, key, &clock_bound));
        CHECK(clock_bound == expected_clock);
        ggml_cuda_moe_grouped_acquisition acquisition;
        ggml_cuda_moe_grouped_resource_info info;
        CHECK(context->acquire_group_resources(key, &acquisition));
        CHECK(context->get_group_resources(acquisition, &info));
        CHECK(!info.transaction_active && info.down == graph.down && info.n_banks == graph.banks.size());
    }
    std::vector<float> output(ggml_nelements(graph.output));
    ggml_backend_tensor_get(graph.output, output.data(), 0, ggml_nbytes(graph.output));
    return output;
}

void check_active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t n_rows,
        const std::vector<int32_t> & routes) {
    CHECK(routes.size() == n_rows * graph.n_used);
    CHECK(graph.ids != nullptr && graph.ids->view_src != nullptr);
    CHECK(graph.ids->type == GGML_TYPE_I32 && graph.ids->ne[0] == graph.n_used && graph.ids->ne[1] == n_rows);
    CHECK(graph.ids->ne[2] == 1 && graph.ids->ne[3] == 1 && graph.ids->nb[0] == sizeof(int32_t));
    CHECK(graph.ids->nb[1] == graph.n_experts * sizeof(int32_t) && graph.ids->nb[1] > graph.ids->ne[0] * graph.ids->nb[0]);
    ggml_tensor * sorted = graph.ids->view_src;
    CHECK(sorted->type == GGML_TYPE_I32 && sorted->ne[0] == graph.n_experts && sorted->ne[1] == n_rows);
    std::vector<int32_t> actual(ggml_nelements(sorted));
    ggml_backend_tensor_get(sorted, actual.data(), 0, ggml_nbytes(sorted));
    const uint32_t row_stride = graph.ids->nb[1] / sizeof(int32_t);
    for (uint32_t row = 0; row < n_rows; ++row) {
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            CHECK(actual[row * row_stride + route] == routes[row * graph.n_used + route]);
        }
    }
}

void check_active_grouped_routes(
        const active_grouped_dispatch_graph & graph,
        uint32_t n_rows,
        uint32_t route_variant) {
    const auto routes = active_grouped_routes(graph, route_variant);
    check_active_grouped_routes(graph, n_rows, routes);
    bool has_repeated_expert = false;
    for (uint32_t row = 0; row < n_rows; ++row) {
        for (uint32_t route = 0; route < graph.n_used; ++route) {
            for (uint32_t previous_row = 0; previous_row < row; ++previous_row) {
                for (uint32_t previous_route = 0; previous_route < graph.n_used; ++previous_route) {
                    has_repeated_expert = has_repeated_expert || routes[row * graph.n_used + route] ==
                        routes[previous_row * graph.n_used + previous_route];
                }
            }
        }
    }
    CHECK(n_rows == 1 || has_repeated_expert || graph.n_experts == 256);
}

void check_active_grouped_exact_output(
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    CHECK(expected.size() == actual.size());
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(expected[i]) && std::isfinite(actual[i]));
        const double difference = expected[i] - actual[i];
        squared_error += difference * difference;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    CHECK(squared_expected > 0.0 && squared_error == 0.0);
}

std::vector<float> active_grouped_tensor_sentinel(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor), -23456.5f);
    ggml_backend_tensor_set(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

std::vector<float> active_grouped_tensor_values(const ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

ggml_cuda_mmid_capability native_mmid_capability(
        int device,
        const ggml_tensor * weight,
        int64_t n_rows,
        ggml_cuda_mmid_mapping mapping) {
    cudaDeviceProp properties;
    CUDA_OK(cudaGetDeviceProperties(&properties, device));
    ggml_cuda_mmid_capability_query query;
    query.source_type = weight->type;
    query.input_type = GGML_TYPE_F32;
    query.output_type = GGML_TYPE_F32;
    memcpy(query.source_ne, weight->ne, sizeof(query.source_ne));
    memcpy(query.source_nb, weight->nb, sizeof(query.source_nb));
    query.n_tokens = n_rows;
    query.n_experts = weight->ne[2];
    query.cc = properties.major * 100 + properties.minor * 10;
    query.warp_size = properties.warpSize;
    query.smpbo = std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin);
    query.phase = n_rows == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    query.mapping = mapping;
    query.use_mmq = ggml_cuda_moe_use_mmq(weight, n_rows);
    return ggml_cuda_mmid_get_capability(query);
}
