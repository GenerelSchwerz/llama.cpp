#include "common.cuh"
#include "moe-fidelity-executor.cuh"
#include "moe-caller-schedule.h"
#include "fattn.cuh"
#include "mmid.cuh"
#include "mmvq.cuh"
#include "moe-reference.cuh"
#include "moe-source-ordinary.cuh"
#include "../moe-fidelity-config.h"
#include "unary.cuh"
#include "../ggml-cpu/moe-fidelity.h"
#include "moe-cache.cuh"
#include "staged-input.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda/atomic>
#endif

struct ggml_cuda_moe_fidelity_graph_allocator::impl {
    std::vector<const ggml_tensor *> originals;
    std::vector<ggml_tensor> tensors;
    std::vector<ggml_tensor *> nodes, leafs;
    std::vector<uint8_t> omitted;
    ggml_cgraph graph = {};
    ggml_gallocr_t allocator = nullptr;
    size_t bytes = 0, metadata = 0, allocator_metadata = 0;
    bool allocated = false;

    ~impl() { ggml_gallocr_free(allocator); }

    size_t index(const ggml_tensor * tensor) const {
        const auto it = std::find(originals.begin(), originals.end(), tensor);
        return it == originals.end() ? SIZE_MAX : size_t(it - originals.begin());
    }

    size_t root(size_t i) const {
        for (size_t depth = 0; i < originals.size() && depth < originals.size(); ++depth) {
            if (!originals[i]->view_src) { return i; }
            i = index(originals[i]->view_src);
        }
        return SIZE_MAX;
    }

    bool measure(ggml_backend_buffer_type_t buft, const ggml_cuda_moe_fidelity_window_query_v1 & query) {
        if (!buft || !query.graph || query.graph->n_nodes <= 0 || !query.graph->nodes ||
                (query.n_regions && !query.regions) || (!query.n_public_outputs && !query.public_output) ||
                (query.n_public_outputs && !query.public_outputs)) { return false; }
        const size_t count = query.graph->n_nodes;
        for (size_t i = 0; i < count; ++i) {
            const auto * tensor = query.graph->nodes[i];
            if (!tensor || index(tensor) != SIZE_MAX) { return false; }
            originals.push_back(tensor);
        }
        for (size_t i = 0; i < originals.size(); ++i) {
            const auto add = [&](const ggml_tensor * tensor) {
                if (tensor && index(tensor) == SIZE_MAX) { originals.push_back(tensor); }
            };
            const auto * tensor = originals[i];
            for (const auto * source : tensor->src) { add(source); }
            add(tensor->view_src);
        }
        if (originals.size() > INT_MAX) { return false; }
        omitted.resize(originals.size(), 0);
        size_t previous = 0;
        for (uint32_t r = 0; r < query.n_regions; ++r) {
            const auto * region = query.regions[r];
            if (!region || region->first_node < previous || region->first_node > region->last_node ||
                    region->last_node >= count || query.graph->nodes[region->last_node] != region->output ||
                    region->output->view_src || !region->activation || !region->ids) { return false; }
            for (const auto * input : {region->activation, region->ids}) {
                const size_t i = index(input);
                if (i == SIZE_MAX || (i < count && i >= region->first_node)) { return false; }
            }
            for (size_t i = region->first_node; i < region->last_node; ++i) {
                if (originals[i]->flags & GGML_TENSOR_FLAG_OUTPUT) { return false; }
                omitted[i] = 1;
            }
            previous = size_t(region->last_node) + 1;
        }
        tensors.resize(originals.size());
        for (size_t i = 0; i < originals.size(); ++i) { tensors[i] = *originals[i]; }
        for (size_t i = 0; i < originals.size(); ++i) {
            auto & tensor = tensors[i];
            const size_t base = root(i);
            if (base == SIZE_MAX || (!omitted[i] && omitted[base])) { return false; }
            for (auto & source : tensor.src) {
                if (!source) { continue; }
                const size_t s = index(source);
                if (s == SIZE_MAX) { return false; }
                source = &tensors[s];
            }
            if (tensor.view_src) {
                size_t offset = 0;
                for (size_t v = i; v != base; v = index(originals[v]->view_src)) {
                    if (originals[v]->view_offs > SIZE_MAX - offset) { return false; }
                    offset += originals[v]->view_offs;
                }
                const size_t span = ggml_nbytes(originals[i]);
                const size_t root_span = ggml_nbytes(originals[base]);
                if (offset > root_span || span > root_span - offset) { return false; }
                tensor.view_src = &tensors[base];
                tensor.view_offs = offset;
                tensor.data = nullptr;
                tensor.buffer = nullptr;
                tensor.extra = nullptr;
            } else if (i < count && tensor.op != GGML_OP_NONE) {
                tensor.data = nullptr;
                tensor.buffer = nullptr;
                tensor.extra = nullptr;
            } else if (!ggml_is_empty(&tensor) && !tensor.data) { return false; }
        }
        for (uint32_t r = 0; r < query.n_regions; ++r) {
            const auto & region = *query.regions[r];
            auto & output = tensors[region.last_node];
            output.op = GGML_OP_NONE;
            std::fill(std::begin(output.src), std::end(output.src), nullptr);
            size_t next = 0;
            for (const auto * input : {region.activation, region.ids}) {
                for (size_t i : {index(input), root(index(input))}) {
                    if (i == SIZE_MAX || omitted[i]) { return false; }
                    if (std::find(output.src, output.src + next, &tensors[i]) == output.src + next) {
                        output.src[next++] = &tensors[i];
                    }
                }
            }
        }
        for (size_t i = 0; i < count; ++i) {
            if (omitted[i]) { continue; }
            auto & tensor = tensors[i];
            for (const auto * source : tensor.src) {
                if (!source) { continue; }
                const size_t s = size_t(source - tensors.data());
                if (omitted[s] || (s < count && s >= i && !(s == i && tensor.op == GGML_OP_CPY))) { return false; }
            }
            if (tensor.view_src && size_t(tensor.view_src - tensors.data()) < count &&
                    size_t(tensor.view_src - tensors.data()) >= i) { return false; }
            if (tensor.flags & GGML_TENSOR_FLAG_OUTPUT) { tensors[root(i)].flags |= GGML_TENSOR_FLAG_OUTPUT; }
            nodes.push_back(&tensor);
        }
        const uint32_t n_outputs = query.n_public_outputs ? query.n_public_outputs : 1;
        for (uint32_t output = 0; output < n_outputs; ++output) {
            const auto * tensor = query.n_public_outputs ? query.public_outputs[output] : query.public_output;
            const size_t i = index(tensor);
            if (i >= count || omitted[i] || !(tensor->flags & GGML_TENSOR_FLAG_OUTPUT)) { return false; }
            const size_t base = root(i);
            if (base >= count || tensors[base].data || omitted[base]) { return false; }
            tensors[base].flags |= GGML_TENSOR_FLAG_OUTPUT;
        }
        for (size_t i = count; i < tensors.size(); ++i) { leafs.push_back(&tensors[i]); }
        graph.n_nodes = int(nodes.size());
        graph.nodes = nodes.data();
        graph.n_leafs = int(leafs.size());
        graph.leafs = leafs.data();
        allocator = ggml_gallocr_new(buft);
        if (!allocator) { return false; }
        ggml_gallocr_reserve_n_size(allocator, &graph, nullptr, nullptr, &bytes);
        allocator_metadata = ggml_gallocr_get_metadata_size(allocator);
        if (allocator_metadata == SIZE_MAX) { return false; }
        metadata = allocator_metadata;
        const auto add = [&](size_t count, size_t item) {
            if (count && item > (SIZE_MAX - metadata) / count) { return false; }
            metadata += count * item;
            return true;
        };
        return add(1, sizeof(*this)) && add(originals.capacity(), sizeof(originals[0])) &&
            add(tensors.capacity(), sizeof(tensors[0])) && add(nodes.capacity(), sizeof(nodes[0])) &&
            add(leafs.capacity(), sizeof(leafs[0])) && add(omitted.capacity(), sizeof(omitted[0]));
    }
};

ggml_cuda_moe_fidelity_graph_allocator::ggml_cuda_moe_fidelity_graph_allocator() = default;
ggml_cuda_moe_fidelity_graph_allocator::~ggml_cuda_moe_fidelity_graph_allocator() = default;

bool ggml_cuda_moe_fidelity_graph_allocator::measure(
        ggml_backend_buffer_type_t buft, const ggml_cuda_moe_fidelity_window_query_v1 & query) {
    if (state && state->allocated) { return false; }
    auto pending = std::make_unique<impl>();
    if (!pending->measure(buft, query)) { return false; }
    state = std::move(pending);
    return true;
}

bool ggml_cuda_moe_fidelity_graph_allocator::allocate() {
    if (!state || state->allocated || !ggml_gallocr_reserve(state->allocator, &state->graph) ||
            ggml_gallocr_get_buffer_size(state->allocator, 0) != state->bytes ||
            !ggml_gallocr_alloc_graph(state->allocator, &state->graph) ||
            ggml_gallocr_get_metadata_size(state->allocator) != state->allocator_metadata) { return false; }
    state->allocated = true;
    return true;
}

void ggml_cuda_moe_fidelity_graph_allocator::reset() { state.reset(); }

const ggml_tensor * ggml_cuda_moe_fidelity_graph_allocator::find(const ggml_tensor * original) const {
    if (!state) { return nullptr; }
    const size_t i = state->index(original);
    return i < state->tensors.size() && !state->omitted[i] ? &state->tensors[i] : nullptr;
}

size_t ggml_cuda_moe_fidelity_graph_allocator::buffer_bytes() const { return state ? state->bytes : 0; }
size_t ggml_cuda_moe_fidelity_graph_allocator::metadata_bytes() const { return state ? state->metadata : 0; }

namespace {

enum terminal_state : uint32_t { window_open, window_canceled, window_committing, window_committed };

struct system_flag {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    uint32_t value = 0;
    uint32_t load() const {
        return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(const_cast<uint32_t &>(value)).load(cuda::memory_order_acquire);
    }
    void store(uint32_t next) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(value).store(next, cuda::memory_order_release);
    }
#else
    std::atomic<uint32_t> value{0};
    uint32_t load() const { return value.load(std::memory_order_acquire); }
    void store(uint32_t next) { value.store(next, std::memory_order_release); }
#endif
};

struct alignas(64) doorbell {
    system_flag ready, done, wake, stop, claimed;
    uint64_t epoch = 0;
    int32_t status = 0;
};

struct reference_route { int32_t expert, slot, reciprocal; };
struct alignas(64) reference_control {
    system_flag plan_ready, transfer_ready, test_validated;
    uint32_t test_transfer_groups = 0;
    uint32_t invalid = 0;
    uint64_t selected_bytes = 0, executed_bytes = 0;
    int32_t groups[2] = {};
};
struct reference_group_view {
    uint64_t * gate, * up, * down;
    int32_t * starts, * tokens, * destinations, * count;
};
static __host__ __device__ size_t reference_group_bytes(uint32_t routes) {
    return (size_t(routes) * 3 * sizeof(uint64_t) + (size_t(routes) * 3 + 1) * sizeof(int32_t) + 7) & ~size_t(7);
}
static __host__ __device__ size_t reference_raw_bytes(uint32_t routes) {
    return (size_t(routes) * sizeof(reference_route) + 7) & ~size_t(7);
}
static size_t reference_control_bytes(uint32_t routes) {
    return sizeof(reference_control) + reference_raw_bytes(routes) + 2 * reference_group_bytes(routes);
}
static __host__ __device__ reference_group_view reference_group(void * memory, uint32_t routes, uint32_t kind) {
    auto * control = static_cast<reference_control *>(memory);
    auto * base = reinterpret_cast<uint8_t *>(control + 1) + reference_raw_bytes(routes) + kind * reference_group_bytes(routes);
    reference_group_view v;
    v.gate = reinterpret_cast<uint64_t *>(base);
    v.up = v.gate + routes;
    v.down = v.up + routes;
    v.starts = reinterpret_cast<int32_t *>(v.down + routes);
    v.tokens = v.starts + routes + 1;
    v.destinations = v.tokens + routes;
    v.count = control->groups + kind;
    return v;
}
struct reference_role { uint32_t source = 0; size_t offset = 0; };

struct window_runtime {
    uint64_t epoch, identity;
    uint32_t active_rows, fault, failed, terminal;
};

struct join_record {
    uint64_t epoch;
    uint64_t imported_bytes;
    uint32_t cpu_terminal;
    uint32_t complete;
};

static bool reserve(size_t & cursor, size_t bytes, size_t & offset) {
    if (cursor > SIZE_MAX - 255) { return false; }
    offset = (cursor + 255) & ~size_t(255);
    if (bytes > SIZE_MAX - offset) { return false; }
    cursor = offset + bytes;
    return true;
}

static bool multiply(size_t first, size_t second, size_t & bytes) {
    if (first && second > SIZE_MAX / first) { return false; }
    bytes = first * second;
    return true;
}

static uint64_t fingerprint(uint64_t hash, const void * data, size_t bytes) {
    const auto * values = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) { hash = (hash ^ values[i]) * 1099511628211ull; }
    return hash;
}

static bool certificate_valid(const ggml_graph_execution_certificate & c) {
    if (c.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC || c.abi_version != GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION ||
            c.struct_size != sizeof(c) ||
            c.domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN || !c.owner_namespace || !c.owner_generation ||
            !c.source_graph_uid || !c.split_graph_uid || c.source_graph_uid == c.split_graph_uid || c.n_sequences != 1) { return false; }
    for (const auto value : c.reserved) { if (value) { return false; } }
    return (c.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT && c.n_rows == 1 &&
            c.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE) ||
        (c.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE && c.n_rows > c.n_sequences &&
            c.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED);
}

static bool compact_mmvq(const ggml_tensor * node, uint32_t banks) {
    const auto * weight = node->src[0];
    const auto * input = node->src[1];
    const auto device = ggml_cuda_get_device();
    const auto & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count) { return false; }
    ggml_cuda_mmid_capability_query q;
    q.source_type = weight->type;
    q.input_type = input->type;
    q.output_type = node->type;
    memcpy(q.source_ne, weight->ne, sizeof(q.source_ne));
    memcpy(q.source_nb, weight->nb, sizeof(q.source_nb));
    q.n_tokens = input->ne[2];
    q.n_experts = weight->ne[2];
    q.phase = q.n_tokens == 1 ? GGML_CUDA_MMID_PHASE_DECODE : GGML_CUDA_MMID_PHASE_PREFILL;
    q.use_mmq = ggml_cuda_moe_use_mmq(weight, q.n_tokens);
    q.cc = info.devices[device].cc;
    q.warp_size = info.devices[device].warp_size;
    q.smpbo = info.devices[device].smpbo;
    return ggml_cuda_mmid_can_use_compact_mmvq(q, banks);
}

struct gpu_body {
    std::vector<ggml_tensor> tensors;
    std::vector<ggml_tensor *> nodes;
    std::vector<size_t> offsets;
    std::vector<const ggml_tensor *> originals;

    ggml_tensor * find(const ggml_tensor * original) {
        const auto it = std::find(originals.begin(), originals.end(), original);
        return it == originals.end() ? nullptr : &tensors[it - originals.begin()];
    }

    bool compile(ggml_backend_t backend, const ggml_backend_moe_cpu_region_query_v1 & query,
            uint32_t routes, uint32_t banks, bool resident, size_t & bytes, size_t & workspace) {
        if (query.n_dynamic_inputs != 2 || !routes || routes > INT32_MAX || !banks) { return false; }
        originals.assign(query.dynamic_inputs, query.dynamic_inputs + query.n_dynamic_inputs);
        for (uint32_t i = 0; i < query.n_sources; ++i) { originals.push_back(query.sources[i].tensor); }
        originals.insert(originals.end(), query.body_nodes, query.body_nodes + query.n_body_nodes);
        tensors.resize(originals.size());
        offsets.resize(originals.size(), SIZE_MAX);
        nodes.reserve(query.n_body_nodes);
        for (size_t i = 0; i < originals.size(); ++i) {
            const auto * original = originals[i];
            auto & tensor = tensors[i];
            tensor = *original;
            tensor.extra = nullptr;
            tensor.data = nullptr;
            for (int src = 0; src < GGML_MAX_SRC; ++src) { tensor.src[src] = find(original->src[src]); }
            tensor.view_src = find(original->view_src);
            if (i >= query.n_dynamic_inputs && i < query.n_dynamic_inputs + query.n_sources) {
                if (tensor.ne[2] <= 0 || !ggml_is_contiguous(&tensor) || tensor.nb[2] > SIZE_MAX / banks) { return false; }
                tensor.ne[2] = banks;
                tensor.nb[3] = tensor.nb[2] * banks;
                if (!resident) {
                    const auto buft = ggml_backend_get_default_buffer_type(backend);
                    if (!reserve(bytes, ggml_backend_buft_get_alloc_size(buft, &tensor), offsets[i])) { return false; }
                }
            } else {
                if (i == 1) {
                    tensor.ne[0] = routes;
                    tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
                    tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = size_t(routes) * sizeof(int32_t);
                } else {
                    if (tensor.type != GGML_TYPE_F32 || tensor.nb[1] > SIZE_MAX / routes) { return false; }
                    tensor.ne[1] = routes;
                    tensor.ne[2] = tensor.ne[3] = 1;
                    tensor.nb[2] = tensor.nb[3] = tensor.nb[1] * routes;
                }
                if (!tensor.view_src && !reserve(bytes, ggml_nbytes(&tensor), offsets[i])) { return false; }
                if (i >= query.n_dynamic_inputs + query.n_sources) {
                    if (!ggml_backend_supports_op(backend, &tensor)) { return false; }
                    if (tensor.op == GGML_OP_MUL_MAT_ID) {
                        if (!compact_mmvq(original, banks) ||
                                tensor.src[1]->ne[0] <= 0 || tensor.src[1]->ne[0] > INT64_MAX - MATRIX_ROW_PADDING) { return false; }
                        size_t needed;
                        const size_t blocks = GGML_PAD(tensor.src[1]->ne[0], MATRIX_ROW_PADDING) / QK8_1;
                        if (!multiply(blocks, sizeof(block_q8_1), needed) || !multiply(needed, routes, needed) || needed > SIZE_MAX - 255) { return false; }
                        workspace = std::max(workspace, (needed + 255) & ~size_t(255));
                    } else if (tensor.op != GGML_OP_GLU && tensor.op != GGML_OP_VIEW) { return false; }
                    nodes.push_back(&tensor);
                }
            }
        }
        return nodes.size() == query.n_body_nodes;
    }

    void bind(ggml_backend_buffer_t arena) {
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(arena));
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto & tensor = tensors[i];
            tensor.buffer = arena;
            tensor.data = tensor.view_src ? static_cast<uint8_t *>(tensor.view_src->data) + tensor.view_offs :
                offsets[i] == SIZE_MAX ? nullptr : base + offsets[i];
        }
    }
};

struct layer {
    ggml_backend_moe_hybrid_region_v1 descriptor = {};
    const ggml_backend_moe_cpu_region_query_v1 * cpu_query = nullptr;
    ggml_backend_moe_cpu_prepared_region_v1_t cpu_region = 0;
    ggml_backend_moe_cpu_prepared_requirements_v1 cpu_requirements = {};
    ggml_cuda_moe_candidate_group_key key;
    ggml_cuda_moe_hybrid_rows_layout rows;
    ggml_cuda_moe_graph_group_dispatch * group = nullptr;
    gpu_body resident, transfer;
    ggml_moe_reference_gpu_layout native = {};
    reference_role roles[3];
    ggml_cuda_moe_hybrid_reference_view owner_view = {};
    size_t reference_offset = 0, reference_device_offset = 0, selected_bytes = 0;
    std::vector<size_t> selected_offsets;
    std::vector<int32_t> distinct, misses, distinct_slots;
    std::vector<uint32_t> route_weights, reference_counts;
    uint32_t cpu_group_max = 0;
    std::vector<ggml_cuda_moe_grouped_bank_descriptor> source_banks;
    std::vector<uint32_t> bank_indices;
    std::vector<ggml_cuda_moe_hybrid_rows_source> sources;
    size_t rows_offset = 0, packet_offset = 0, runtime_offset = 0, status_offset = 0;
    size_t host_rows_offset = 0, host_input_offset = 0, host_output_offset = 0, host_bell_offset = 0;
    size_t join_offset = 0, host_join_offset = 0, host_completion_offset = 0;
    std::vector<int32_t> cpu_ids;
    std::vector<uint32_t> cpu_rows, cpu_scatter;
    uint32_t first = 0, end = 0;
    uint64_t cpu_graph_uid = 0, cpu_graph_generation = 0, cpu_source_generation = 0;
};

struct compiled_window {
    struct ordinary_mul_mat {
        uint32_t index;
        ggml_cuda_moe_fidelity_mul_mat_resources resources;
    };
    struct ordinary_attention {
        uint32_t index;
        ggml_cuda_fattn_resources resources;
    };

    ggml_cuda_moe_fidelity_window_query_v1 query = {};
    ggml_cuda_moe_fidelity_window_state_v1 state = {};
    std::vector<layer> layers;
    std::vector<ggml_tensor> tensors;
    std::vector<ggml_tensor *> nodes;
    std::vector<const ggml_tensor *> originals;
    std::vector<ggml_tensor *> public_outputs;
    std::vector<ordinary_mul_mat> ordinary_matmuls;
    std::vector<ordinary_attention> ordinary_attentions;
    std::vector<ggml_cuda_moe_source_norm> ordinary_norms;
    ggml_cuda_moe_fidelity_graph_allocator allocation;
    size_t arena_bytes = 0, pinned_bytes = 0, workspace = 0, workspace_offset = 0;
    size_t cublas_workspace = 0, cublas_workspace_offset = 0;
    size_t staged_input_bytes = 0, staged_input_offset = 0;
    size_t runtime_offset = 0, host_runtime_offset = 0, host_result_offset = 0;
    uint32_t row_capacity = 0;
    bool reference = false;
    bool source_ordinary = false;
    uint32_t pcie_num = 0;
    size_t selected_bytes = 0, selected_offset = 0, native_scratch_bytes = 0, native_scratch_offset = 0;
    size_t source_quant_bytes = 0, source_quant_offset = 0;

    ggml_tensor * find(const ggml_tensor * original) {
        const auto it = std::find(originals.begin(), originals.end(), original);
        return it == originals.end() ? nullptr : &tensors[it - originals.begin()];
    }

    bool validate_matmul_resources(int device) const {
        for (const auto & item : ordinary_matmuls) {
            ggml_cuda_moe_fidelity_mul_mat_resources current;
            const auto & expected = item.resources;
            if (!ggml_cuda_moe_fidelity_mul_mat_requirements(device, nodes[item.index], current) ||
                    current.dispatch != expected.dispatch || current.compute_type != expected.compute_type ||
                    current.pool_bytes != expected.pool_bytes || current.cublas_workspace_bytes != expected.cublas_workspace_bytes) {
                fprintf(stderr, "moe-fidelity: bound MUL_MAT resources changed at node %u\n", item.index);
                return false;
            }
        }
        return true;
    }

    bool validate_attention_resources(ggml_backend_cuda_context & context) const {
        for (const auto & item : ordinary_attentions) {
            ggml_cuda_fattn_resources current;
            if (!ggml_cuda_flash_attn_ext_prepare_resources(context, nodes[item.index], current) ||
                    current.pool_bytes != item.resources.pool_bytes || current.identity != item.resources.identity) {
                fprintf(stderr, "moe-fidelity: bound FLASH_ATTN_EXT resources changed at node %u\n", item.index);
                return false;
            }
        }
        return true;
    }

    bool compile_reference(layer & l) {
        const auto & q = *l.descriptor.query;
        if (q.n_live_outputs != 1 || !q.live_outputs || q.live_outputs[0]->op != GGML_OP_MUL_MAT_ID) { return false; }
        const auto * down = q.live_outputs[0];
        const auto * glu = down->src[1];
        if (!glu || glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) { return false; }
        const ggml_tensor * values[] = {glu->src[0], glu->src[1], down};
        uint32_t widths[3], rows[3];
        ggml_type types[3];
        size_t strides[3];
        for (uint32_t role = 0; role < 3; ++role) {
            const auto * value = values[role];
            if (!value || value->ne[0] <= 0 || value->ne[0] > INT_MAX) { return false; }
            rows[role] = uint32_t(value->ne[0]);
            size_t row_offset = 0;
            if (value->op == GGML_OP_VIEW) {
                if (role == 2 || value->view_offs % sizeof(float) || value->nb[0] != sizeof(float) ||
                        value->view_src != value->src[0]) { return false; }
                row_offset = value->view_offs / sizeof(float);
                value = value->view_src;
            }
            if (!value || value->op != GGML_OP_MUL_MAT_ID || value->src[2] != q.ids ||
                    (role < 2 && value->src[1] != q.activation)) { return false; }
            const auto * weight = value->src[0];
            uint32_t source = 0;
            for (; source < q.n_sources; ++source) { if (q.sources[source].tensor == weight) { break; } }
            if (source == q.n_sources || !weight || weight->ne[0] <= 0 || weight->ne[0] > INT_MAX ||
                    weight->ne[2] != l.rows.geometry.expert_count || weight->ne[3] != 1 ||
                    row_offset > uint64_t(weight->ne[1]) || rows[role] > uint64_t(weight->ne[1]) - row_offset ||
                    weight->nb[1] < ggml_row_size(weight->type, weight->ne[0]) ||
                    row_offset > SIZE_MAX / weight->nb[1] ||
                    (row_offset + rows[role] - 1) > SIZE_MAX / weight->nb[1] ||
                    (row_offset + rows[role] - 1) * weight->nb[1] > l.sources[source].expert_bytes ||
                    ggml_row_size(weight->type, weight->ne[0]) > l.sources[source].expert_bytes - (row_offset + rows[role] - 1) * weight->nb[1]) { return false; }
            widths[role] = uint32_t(weight->ne[0]); types[role] = weight->type; strides[role] = weight->nb[1];
            l.roles[role] = {source, row_offset * weight->nb[1]};
        }
        if (q.activation->nb[0] != sizeof(float) || q.activation->nb[2] != size_t(widths[0]) * sizeof(float) ||
                l.rows.geometry.weight_capacity > 65535 || l.rows.geometry.route_capacity > INT_MAX) { return false; }
        if (widths[0] != widths[1] || types[0] != types[1] || rows[0] != rows[1] || widths[2] != rows[0] ||
                widths[0] != l.rows.input_width || rows[2] != l.rows.output_width) { return false; }
        l.native = {int(types[0]), int(types[2]), int(widths[0]), int(rows[0]), int(rows[2]), strides[0], strides[1], strides[2]};
        if (!ggml_moe_reference_gpu_supported(l.native)) { return false; }
        l.selected_offsets.resize(l.sources.size());
        for (size_t source = 0; source < l.sources.size(); ++source) {
            size_t bytes;
            if (!multiply(l.rows.transfer_capacity, l.sources[source].expert_bytes, bytes) ||
                    !reserve(l.selected_bytes, bytes, l.selected_offsets[source])) { return false; }
        }
        selected_bytes = std::max(selected_bytes, l.selected_bytes);
        const size_t scratch = ggml_moe_reference_gpu_scratch(l.rows.geometry.route_capacity, l.native.n_ff);
        if (scratch == SIZE_MAX) { return false; }
        native_scratch_bytes = std::max(native_scratch_bytes, scratch);
        size_t quant;
        if (!multiply(l.rows.geometry.row_capacity, size_t(l.native.n_embd / 32) * sizeof(block_q8_1), quant)) { return false; }
        source_quant_bytes = std::max(source_quant_bytes, quant);
        if (!reserve(pinned_bytes, reference_control_bytes(l.rows.geometry.route_capacity), l.reference_offset) ||
                !reserve(arena_bytes, reference_control_bytes(l.rows.geometry.route_capacity), l.reference_device_offset)) { return false; }
        l.distinct.reserve(l.rows.geometry.weight_capacity);
        l.misses.reserve(l.rows.geometry.weight_capacity);
        l.distinct_slots.reserve(l.rows.geometry.weight_capacity);
        l.route_weights.resize(l.rows.geometry.route_capacity);
        l.reference_counts.resize(l.rows.geometry.weight_capacity);
        return true;
    }

    bool compile(ggml_backend_t backend, const ggml_cuda_moe_fidelity_window_query_v1 * input) {
        if (!backend || !input || input->struct_size != sizeof(*input) || !input->graph || !input->graph->uid ||
                !input->regions || !input->n_regions || !input->source_owner || !input->cpu_api ||
                !input->cpu_requirements_api || input->cpu_requirements_api->struct_size != sizeof(*input->cpu_requirements_api) ||
                input->cpu_requirements_api->abi_version != 1 || !input->cpu_requirements_api->service || !input->cpu_requirements_api->region ||
                (!input->n_public_outputs && !input->public_output) ||
                (input->n_public_outputs && !input->public_outputs) ||
                !input->graph_owner || !input->retain_graph || !input->release_graph || !input->arena_generation || input->admission_quota != 0 ||
                !input->n_threads || input->admission_quota > input->gpu_miss_quota ||
                input->staging_tile_bytes > SIZE_MAX || input->no_host_alias > 1 || input->graph->n_nodes <= 0 ||
                input->n_regions > uint32_t(input->graph->n_nodes) ||
                !certificate_valid(input->certificate) || input->certificate.split_graph_uid != input->graph->uid ||
                memcmp(&input->certificate, &input->graph->execution_certificate, sizeof(input->certificate)) != 0) { return false; }
        const auto & selection = ggml_moe_fidelity_selection();
        if (!selection.valid || (selection.reference && input->no_host_alias)) { return false; }
        reference = selection.reference;
        source_ordinary = selection.source_pool;
        pcie_num = selection.pcie_num;
        query = *input;
        if (query.n_public_outputs) {
            public_outputs.assign(query.public_outputs, query.public_outputs + query.n_public_outputs);
        } else {
            public_outputs.push_back(query.public_output);
        }
        query.public_output = public_outputs[0];
        query.n_public_outputs = public_outputs.size();
        query.public_outputs = public_outputs.data();
        row_capacity = query.certificate.n_rows;
        auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
        if (!context || !context->moe_grouped_context) { return false; }
        auto & owner = *context->moe_grouped_context;
        state.identity = fingerprint(1469598103934665603ull, &query.certificate, sizeof(query.certificate));
        state.identity = fingerprint(state.identity, &query.graph->uid, sizeof(query.graph->uid));
        state.identity = fingerprint(state.identity, &reference, sizeof(reference));
        if (source_ordinary) {
            const uint32_t ordinary_version = 1;
            state.identity = fingerprint(state.identity, &ordinary_version, sizeof(ordinary_version));
        }
        state.identity = fingerprint(state.identity, &pcie_num, sizeof(pcie_num));
        if (!reserve(arena_bytes, sizeof(window_runtime), runtime_offset) ||
                !reserve(pinned_bytes, sizeof(window_runtime), host_runtime_offset) ||
                !reserve(pinned_bytes, sizeof(window_runtime), host_result_offset)) { return false; }
        state.storage.device_runtime_bytes = arena_bytes;
        state.storage.pinned_runtime_bytes = pinned_bytes;
        layers.resize(query.n_regions);
        uint32_t previous = 0;
        size_t staging_tile_bytes = 0;
        for (uint32_t i = 0; i < query.n_regions; ++i) {
            auto & l = layers[i];
            const auto * d = query.regions[i];
            ggml_cuda_moe_candidate_group_info group;
            if (!d || d->struct_size != sizeof(*d) || !d->query || !d->activation || !d->ids || !d->output ||
                    d->n_cpu_batch_queries != 1 || !d->cpu_batch_queries || !d->cpu_batch_queries[0] ||
                    d->geometry.row_capacity != row_capacity || d->geometry.route_capacity == 0 ||
                    !owner.find_down_group_key(d->down, &l.key) || !owner.get_group(l.key, &group) ||
                    !group.n_slots || memcmp(&d->certificate, &query.certificate, sizeof(query.certificate)) != 0 ||
                    d->source_graph_uid != query.certificate.source_graph_uid || d->split_graph_uid != query.certificate.split_graph_uid ||
                    d->owner_generation != query.certificate.owner_generation ||
                    ggml_backend_moe_hybrid_validate_buckets_v1(d) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return false; }
            l.descriptor = *d;
            l.cpu_query = d->cpu_batch_queries[0];
            l.cpu_graph_uid = l.cpu_query->graph_uid;
            l.cpu_graph_generation = l.cpu_query->graph_generation;
            l.cpu_source_generation = l.cpu_query->source_generation;
            if (l.cpu_query->n_threads != query.n_threads || l.cpu_query->n_lanes != 1) { return false; }
            auto begin = std::find(query.graph->nodes, query.graph->nodes + query.graph->n_nodes, d->output);
            if (begin == query.graph->nodes + query.graph->n_nodes) { return false; }
            l.end = uint32_t(begin - query.graph->nodes) + 1;
            if (l.end < d->query->n_body_nodes) { return false; }
            l.first = l.end - d->query->n_body_nodes;
            if (l.first < previous || l.first != d->first_node || l.end != uint64_t(d->last_node) + 1) { return false; }
            for (uint32_t node = 0; node < d->query->n_body_nodes; ++node) {
                const auto * original = query.graph->nodes[l.first + node];
                const auto * certified = d->query->body_nodes[node];
                if (!certified || original->op != certified->op || original->type != certified->type ||
                        original->ne[0] != certified->ne[0] ||
                        memcmp(original->op_params, certified->op_params, sizeof(original->op_params))) { return false; }
            }
            previous = l.end;
            const auto routes = d->geometry.route_capacity;
            const auto transfers = reference ? d->geometry.weight_capacity : std::max(1u, std::min(query.gpu_miss_quota, d->geometry.weight_capacity));
            std::vector<uint8_t> accessible(d->query->n_sources, 0);
            ggml_cuda_moe_hybrid_rows_query rq;
            rq.region = d;
            rq.certificate = d->certificate;
            rq.plan_capacity = routes;
            rq.slot_capacity = group.n_slots;
            rq.transfer_capacity = transfers;
            rq.workspace_generation = query.arena_generation;
            rq.resource_fingerprint = owner.state().generation;
            rq.device_alignment = rq.host_alignment = 256;
            rq.source_device_accessible = accessible.data();
            rq.n_sources = accessible.size();
            rq.external_publish_storage = true;
            rq.external_transfer_storage = true;
            if (!ggml_cuda_moe_hybrid_rows_measure(rq, l.rows, l.sources)) { return false; }
            if (l.rows.publish_offset != SIZE_MAX || !l.rows.external_publish_storage || !l.rows.external_transfer_storage ||
                    std::any_of(l.sources.begin(), l.sources.end(), [](const auto & source) { return source.device_offset != SIZE_MAX; })) { return false; }
            for (const auto & source : l.sources) {
                if (source.role != GGML_CUDA_MOE_HYBRID_SOURCE_MMID_WEIGHT) { return false; }
            }
            staging_tile_bytes = std::max(staging_tile_bytes, l.rows.staging_tile_bytes);
            size_t packet_bytes, control_bytes;
            if (!multiply(routes, GGML_CUDA_MOE_HYBRID_PACKET_ARRAYS * sizeof(int32_t), packet_bytes) ||
                    packet_bytes > SIZE_MAX - sizeof(ggml_cuda_moe_hybrid_selection)) { return false; }
            packet_bytes += sizeof(ggml_cuda_moe_hybrid_selection);
            control_bytes = l.rows.pinned_bytes;
            const size_t device_rows_begin = arena_bytes;
            const size_t pinned_rows_begin = pinned_bytes;
            if (!reserve(arena_bytes, l.rows.device_bytes, l.rows_offset) ||
                    !reserve(arena_bytes, packet_bytes, l.packet_offset) ||
                    !reserve(arena_bytes, sizeof(ggml_cuda_moe_hybrid_runtime), l.runtime_offset) ||
                    !reserve(arena_bytes, sizeof(uint32_t), l.status_offset) ||
                    !reserve(arena_bytes, sizeof(join_record), l.join_offset) ||
                    !reserve(pinned_bytes, control_bytes, l.host_rows_offset) ||
                    !reserve(pinned_bytes, sizeof(doorbell), l.host_bell_offset) ||
                    !reserve(pinned_bytes, l.rows.control_bytes, l.host_completion_offset) ||
                    !reserve(pinned_bytes, sizeof(join_record), l.host_join_offset)) { return false; }
            state.storage.device_rows_bytes += arena_bytes - device_rows_begin;
            state.storage.pinned_rows_bytes += pinned_bytes - pinned_rows_begin;
            const size_t reference_pinned_begin = pinned_bytes;
            const size_t reference_device_begin = arena_bytes;
            if (reference) {
                if (!compile_reference(l)) { return false; }
                state.storage.pinned_rows_bytes += pinned_bytes - reference_pinned_begin;
                state.storage.device_rows_bytes += arena_bytes - reference_device_begin;
            } else {
                const size_t resident_begin = arena_bytes;
                if (!l.resident.compile(backend, *d->query, routes, group.n_slots, true, arena_bytes, workspace)) { return false; }
                state.storage.device_resident_body_bytes += arena_bytes - resident_begin;
                const size_t transfer_begin = arena_bytes;
                if (!l.transfer.compile(backend, *d->query, routes, transfers, false, arena_bytes, workspace)) { return false; }
                state.storage.device_transfer_body_bytes += arena_bytes - transfer_begin;
            }
            l.host_input_offset = l.host_rows_offset + l.rows.host_input_offset;
            l.host_output_offset = l.host_rows_offset + l.rows.host_output_offset;
            l.cpu_ids.resize(routes);
            l.cpu_rows.resize(routes);
            l.cpu_scatter.resize(routes);
            l.source_banks.resize(d->query->n_sources);
            l.bank_indices.resize(d->query->n_sources);
            state.identity = fingerprint(state.identity, &l.rows.identity, sizeof(l.rows.identity));
            if (reference) {
                const uint64_t layout[] = {1, reference_control_bytes(routes), l.selected_bytes, uint64_t(l.native.n_embd), uint64_t(l.native.n_ff), uint64_t(l.native.n_out)};
                state.identity = fingerprint(state.identity, layout, sizeof(layout));
                for (const auto role : l.roles) {
                    state.identity = fingerprint(state.identity, &role.source, sizeof(role.source));
                    state.identity = fingerprint(state.identity, &role.offset, sizeof(role.offset));
                }
                for (const auto offset : l.selected_offsets) { state.identity = fingerprint(state.identity, &offset, sizeof(offset)); }
            }
        }
        if (reference) {
            // The compute chain joins selected copies and all GPU consumers before the next layer.
            const size_t begin = arena_bytes;
            if (!reserve(arena_bytes, selected_bytes, selected_offset) ||
                    !reserve(arena_bytes, native_scratch_bytes, native_scratch_offset) ||
                    !reserve(arena_bytes, source_quant_bytes, source_quant_offset)) { return false; }
            state.storage.device_transfer_body_bytes += arena_bytes - begin;
        }
        if (query.staging_tile_bytes && query.staging_tile_bytes != staging_tile_bytes) { return false; }
        query.staging_tile_bytes = staging_tile_bytes;
        originals.assign(query.graph->nodes, query.graph->nodes + query.graph->n_nodes);
        for (size_t i = 0; i < originals.size(); ++i) {
            if (!originals[i]) { return false; }
            const auto add = [&](const ggml_tensor * tensor) {
                if (tensor && std::find(originals.begin(), originals.end(), tensor) == originals.end()) { originals.push_back(tensor); }
            };
            const auto * tensor = originals[i];
            for (const auto * source : tensor->src) { add(source); }
            add(tensor->view_src);
        }
        tensors.resize(originals.size());
        nodes.resize(query.graph->n_nodes);
        for (size_t i = 0; i < originals.size(); ++i) { tensors[i] = *originals[i]; }
        for (size_t i = 0; i < originals.size(); ++i) {
            auto & tensor = tensors[i];
            for (auto & source : tensor.src) { source = find(source); }
            tensor.view_src = find(tensor.view_src);
            if (i >= size_t(query.graph->n_nodes)) { continue; }
            nodes[i] = &tensor;
            bool body = false;
            for (const auto & l : layers) { body |= i >= l.first && i < l.end; }
            if (!body) {
                const auto * weight = tensor.op == GGML_OP_MUL_MAT_ID ? tensor.src[0] : nullptr;
                if ((weight && weight->buffer && ggml_backend_buft_is_cuda_moe_cached(weight->buffer->buft)) ||
                        !ggml_backend_supports_op(backend, &tensor)) { return false; }
                if (tensor.op == GGML_OP_ARGSORT && (tensor.ne[0] > 1024 ||
                        size_t(1024) * sizeof(int) > ggml_cuda_info().devices[context->device].smpb)) { return false; }
                if (i >= layers[0].end && (tensor.flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_cuda_staged_input_supports(&tensor)) {
                    staged_input_bytes = std::max(staged_input_bytes, ggml_nbytes(&tensor));
                }
                if (tensor.op == GGML_OP_MUL_MAT && !ggml_is_empty(&tensor) && (tensor.flags & GGML_TENSOR_FLAG_COMPUTE)) {
                    ggml_cuda_moe_fidelity_mul_mat_resources resources;
                    if (!ggml_cuda_moe_fidelity_mul_mat_requirements(context->device, &tensor, resources)) {
                        fprintf(stderr, "moe-fidelity: MUL_MAT resource capability rejected at node %zu (%s)\n", i, tensor.name);
                        return false;
                    }
                    workspace = std::max(workspace, resources.pool_bytes);
                    cublas_workspace = std::max(cublas_workspace, resources.cublas_workspace_bytes);
                    ordinary_matmuls.push_back({uint32_t(i), resources});
                    state.identity = fingerprint(state.identity, &resources.dispatch, sizeof(resources.dispatch));
                    state.identity = fingerprint(state.identity, &resources.compute_type, sizeof(resources.compute_type));
                    state.identity = fingerprint(state.identity, &resources.pool_bytes, sizeof(resources.pool_bytes));
                    state.identity = fingerprint(state.identity, &resources.cublas_workspace_bytes, sizeof(resources.cublas_workspace_bytes));
                }
                if (tensor.op == GGML_OP_FLASH_ATTN_EXT && !ggml_is_empty(&tensor) && (tensor.flags & GGML_TENSOR_FLAG_COMPUTE)) {
                    ggml_cuda_fattn_resources resources;
                    if (!ggml_cuda_flash_attn_ext_prepare_resources(*context, &tensor, resources)) {
                        fprintf(stderr, "moe-fidelity: FLASH_ATTN_EXT resource capability rejected at node %zu (%s)\n", i, tensor.name);
                        return false;
                    }
                    workspace = std::max(workspace, resources.pool_bytes);
                    ordinary_attentions.push_back({uint32_t(i), resources});
                    state.identity = fingerprint(state.identity, &resources.pool_bytes, sizeof(resources.pool_bytes));
                    state.identity = fingerprint(state.identity, &resources.identity, sizeof(resources.identity));
                }
            }
            const auto * original = originals[i];
            state.identity = fingerprint(state.identity, &original->op, sizeof(original->op));
            state.identity = fingerprint(state.identity, &original->type, sizeof(original->type));
            state.identity = fingerprint(state.identity, original->ne, sizeof(original->ne));
            state.identity = fingerprint(state.identity, original->nb, sizeof(original->nb));
            state.identity = fingerprint(state.identity, original->op_params, sizeof(original->op_params));
            state.identity = fingerprint(state.identity, &original->flags, sizeof(original->flags));
            state.identity = fingerprint(state.identity, &original->view_offs, sizeof(original->view_offs));
            const uintptr_t address = reinterpret_cast<uintptr_t>(original->data);
            state.identity = fingerprint(state.identity, &address, sizeof(address));
        }
        for (size_t i = 0; i < public_outputs.size(); ++i) {
            const auto * output = public_outputs[i];
            if (!output || !ggml_is_contiguous(output) || !(output->flags & GGML_TENSOR_FLAG_OUTPUT) || output->op == GGML_OP_NONE ||
                    !find(output) || std::find(query.graph->nodes, query.graph->nodes + query.graph->n_nodes, output) == query.graph->nodes + query.graph->n_nodes ||
                    std::find(public_outputs.begin(), public_outputs.begin() + i, output) != public_outputs.begin() + i) { return false; }
        }
        if (source_ordinary) {
            for (uint32_t i = 0; i + 1 < nodes.size(); ++i) {
                auto * norm = nodes[i];
                auto * output = nodes[i + 1];
                if (norm->op != GGML_OP_RMS_NORM || output->op != GGML_OP_MUL) { continue; }
                if (output->src[0] != norm && output->src[1] != norm) { continue; }
                bool body = false;
                for (const auto & l : layers) { body |= i < l.end && i + 1 >= l.first; }
                if (body || std::find(public_outputs.begin(), public_outputs.end(), originals[i]) != public_outputs.end()) { continue; }
                auto * gamma = output->src[output->src[0] == norm ? 1 : 0];
                ggml_cuda_moe_source_norm descriptor;
                const auto status = ggml_cuda_moe_source_norm_prepare(query.graph, i, norm, norm->src[0], gamma, output, descriptor);
                if (status == GGML_CUDA_MOE_SOURCE_NORM_INVALID) { return false; }
                if (status != GGML_CUDA_MOE_SOURCE_NORM_READY) { continue; }
                ordinary_norms.push_back(descriptor);
                state.identity = fingerprint(state.identity, &i, sizeof(i));
            }
        }
        if (!allocation.measure(ggml_backend_get_default_buffer_type(backend), query)) { return false; }
        state.planned_clone_bytes = allocation.buffer_bytes();
        state.storage.device_clone_bytes = allocation.buffer_bytes();
        const size_t softmax_workspace = 2 * ((size_t(ggml_cuda_info().devices[context->device].nsm) * sizeof(float) * sizeof(float) + 255) & ~size_t(255));
        workspace = std::max(workspace, softmax_workspace);
        const size_t workspace_begin = arena_bytes;
        if (!reserve(arena_bytes, workspace, workspace_offset)) { return false; }
        state.storage.device_workspace_bytes = arena_bytes - workspace_begin;
        const size_t cublas_begin = arena_bytes;
        if (cublas_workspace && !reserve(arena_bytes, cublas_workspace, cublas_workspace_offset)) { return false; }
        state.storage.device_cublas_bytes = arena_bytes - cublas_begin;
        const size_t staged_begin = arena_bytes;
        if (staged_input_bytes && !reserve(arena_bytes, staged_input_bytes, staged_input_offset)) { return false; }
        state.storage.device_staged_input_bytes = arena_bytes - staged_begin;
        state.segments = query.n_regions + 1;
        if (allocation.buffer_bytes() > SIZE_MAX - arena_bytes) { return false; }
        state.storage.device_bytes = arena_bytes + allocation.buffer_bytes();
        state.storage.pinned_bytes = pinned_bytes;
        const uint64_t completion_layout[] = {4, selected_bytes, native_scratch_bytes, source_quant_bytes, sizeof(doorbell), sizeof(join_record), arena_bytes, pinned_bytes,
            staged_input_offset, staged_input_bytes};
        state.identity = fingerprint(state.identity, completion_layout, sizeof(completion_layout));
        if (state.storage.device_runtime_bytes + state.storage.device_rows_bytes +
                state.storage.device_resident_body_bytes + state.storage.device_transfer_body_bytes +
                state.storage.device_clone_bytes + state.storage.device_workspace_bytes + state.storage.device_cublas_bytes +
                state.storage.device_staged_input_bytes != state.storage.device_bytes ||
                state.storage.pinned_runtime_bytes + state.storage.pinned_rows_bytes != state.storage.pinned_bytes) { return false; }
        state.storage.metadata_bytes = sizeof(*this) + layers.capacity() * sizeof(layer) +
            tensors.capacity() * sizeof(ggml_tensor) + nodes.capacity() * sizeof(ggml_tensor *) +
            originals.capacity() * sizeof(ggml_tensor *) + public_outputs.capacity() * sizeof(ggml_tensor *);
        size_t ordinary_metadata;
        if (!multiply(ordinary_matmuls.capacity(), sizeof(ordinary_mul_mat), ordinary_metadata) ||
                ordinary_metadata > SIZE_MAX - state.storage.metadata_bytes) { return false; }
        state.storage.metadata_bytes += ordinary_metadata;
        if (!multiply(ordinary_attentions.capacity(), sizeof(ordinary_attention), ordinary_metadata) ||
                ordinary_metadata > SIZE_MAX - state.storage.metadata_bytes) { return false; }
        state.storage.metadata_bytes += ordinary_metadata;
        if (!multiply(ordinary_norms.capacity(), sizeof(ggml_cuda_moe_source_norm), ordinary_metadata) ||
                ordinary_metadata > SIZE_MAX - state.storage.metadata_bytes) { return false; }
        state.storage.metadata_bytes += ordinary_metadata;
        if (allocation.metadata_bytes() > SIZE_MAX - state.storage.metadata_bytes) { return false; }
        state.storage.metadata_bytes += allocation.metadata_bytes();
        for (const auto & l : layers) {
            for (const auto * body : {&l.resident, &l.transfer}) {
                state.storage.metadata_bytes += body->tensors.capacity() * sizeof(ggml_tensor) + body->nodes.capacity() * sizeof(ggml_tensor *) +
                    body->originals.capacity() * sizeof(ggml_tensor *) + body->offsets.capacity() * sizeof(size_t);
            }
            state.storage.metadata_bytes += l.cpu_ids.capacity() * sizeof(int32_t) +
                (l.cpu_rows.capacity() + l.cpu_scatter.capacity() + l.bank_indices.capacity()) * sizeof(uint32_t) +
                l.sources.capacity() * sizeof(ggml_cuda_moe_hybrid_rows_source) +
                l.source_banks.capacity() * sizeof(ggml_cuda_moe_grouped_bank_descriptor) +
                l.selected_offsets.capacity() * sizeof(size_t) +
                (l.distinct.capacity() + l.misses.capacity() + l.distinct_slots.capacity()) * sizeof(int32_t) +
                (l.route_weights.capacity() + l.reference_counts.capacity()) * sizeof(uint32_t);
        }
        return true;
    }
};


struct bounded_pool : ggml_cuda_pool {
    uint8_t * data;
    size_t bytes, used = 0;
    bounded_pool(uint8_t * data, size_t bytes) : data(data), bytes(bytes) {}
    void * alloc(size_t count, size_t * actual) override {
        if (count > SIZE_MAX - 255 || ((count + 255) & ~size_t(255)) > bytes - used) {
            if (getenv("GGML_MOE_FIDELITY_TRACE")) {
                fprintf(stderr, "moe-fidelity: pool capacity exceeded request=%zu used=%zu capacity=%zu\n", count, used, bytes);
                fflush(stderr);
            }
            throw std::bad_alloc();
        }
        *actual = (count + 255) & ~size_t(255);
        auto * result = data + used;
        used += *actual;
        return result;
    }
    void free(void * pointer, size_t count) override {
        GGML_ASSERT(count <= used && pointer == data + used - count);
        used -= count;
    }
};

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static __device__ uint32_t acquire_flag(const system_flag * flag) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 600
    return cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(const_cast<uint32_t &>(flag->value)).load(cuda::memory_order_acquire);
#else
    GGML_UNUSED(flag);
    return 0;
#endif
}

static __global__ void wait_cpu(const doorbell * bell, window_runtime * runtime) {
    while (acquire_flag(&bell->wake) == 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
    if (acquire_flag(&bell->stop)) { runtime->failed = 1; }
}

static __device__ void release_flag(system_flag * flag) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 600
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(flag->value).store(1, cuda::memory_order_release);
#else
    GGML_UNUSED(flag);
#endif
}

static __global__ void reference_input(const float * input, float * mapped, size_t values, const window_runtime * runtime) {
    if (runtime->failed) { return; }
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < values; i += size_t(gridDim.x) * blockDim.x) { mapped[i] = input[i]; }
}
static __global__ void reference_publish(ggml_cuda_moe_hybrid_rows_layout layout,
        const int32_t * ids, ggml_cuda_moe_hybrid_reference_view owner,
        reference_control * control, doorbell * bell, const window_runtime * runtime) {
    auto * raw = reinterpret_cast<reference_route *>(control + 1);
    if (threadIdx.x == 0) { control->invalid = runtime->failed; }
    __syncthreads();
    const uint32_t count = runtime->active_rows * layout.geometry.routes_per_row;
    for (uint32_t route = threadIdx.x; route < count; route += blockDim.x) {
        const int32_t expert = ids[size_t(route / layout.geometry.routes_per_row) * layout.ids_row_stride + route % layout.geometry.routes_per_row];
        int32_t slot = -1, reciprocal = -1;
        bool valid = expert >= 0 && uint32_t(expert) < owner.n_experts;
        if (valid) {
            slot = owner.slot_for_expert[expert];
            valid = slot >= -1 && (slot < 0 || uint32_t(slot) < owner.n_slots);
            if (slot >= 0 && valid) { reciprocal = owner.expert_for_slot[slot]; valid = reciprocal == expert; }
        }
        raw[route] = {expert, slot, reciprocal};
        if (!valid) { atomicExch(&control->invalid, 1u); }
    }
    __syncthreads();
    if (threadIdx.x == 0) { release_flag(&bell->ready); }
}
static __global__ void reference_wait_plan(reference_control * control, const doorbell * bell, window_runtime * runtime) {
    while (!acquire_flag(&control->plan_ready) && !acquire_flag(&bell->stop)) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
    if (acquire_flag(&bell->stop)) { runtime->failed = 1; }
}
static __global__ void reference_import(ggml_cuda_moe_hybrid_rows_layout layout,
        ggml_cuda_moe_hybrid_rows_view packet, const uint8_t * mapped, size_t bytes,
        const reference_control * control, reference_control * private_control, size_t group_bytes,
        const doorbell * bell, window_runtime * runtime) {
    __shared__ bool valid;
    if (threadIdx.x == 0) { valid = acquire_flag(&control->plan_ready) && !acquire_flag(&bell->stop) && !runtime->failed; }
    __syncthreads();
    if (!valid) {
        if (threadIdx.x == 0) {
            *packet.ticket = {};
            packet.ticket->epoch = runtime->epoch;
            packet.ticket->identity = layout.identity;
            packet.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
            private_control->groups[0] = private_control->groups[1] = 0;
            private_control->executed_bytes = 0;
        }
        for (uint32_t i = threadIdx.x; i < layout.geometry.row_capacity + 2; i += blockDim.x) { packet.expected_routes[i] = 0; }
        return;
    }
    auto * output = reinterpret_cast<uint8_t *>(packet.ticket);
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) { output[i] = mapped[i]; }
    auto * target = reinterpret_cast<uint8_t *>(private_control);
    const auto * group_source = reinterpret_cast<const uint8_t *>(control);
    for (size_t i = sizeof(reference_control) + threadIdx.x; i < group_bytes; i += blockDim.x) { target[i] = group_source[i]; }
    __syncthreads();
    if (threadIdx.x == 0) {
        private_control->groups[0] = control->groups[0]; private_control->groups[1] = control->groups[1];
        private_control->executed_bytes = 0;
    }
}
struct reference_role_binding { uint64_t resident, selected; size_t resident_stride, selected_stride, offset; };
static __global__ void reference_validated(ggml_cuda_moe_hybrid_rows_layout layout,
        ggml_cuda_moe_hybrid_rows_view packet,
        reference_control * control, reference_control * private_control,
        reference_role_binding gate, reference_role_binding up, reference_role_binding down, bool eligible, window_runtime * runtime) {
    bool valid = packet.ticket->status == 0 && !runtime->failed && acquire_flag(&control->plan_ready) && (eligible || packet.ticket->transfer_count == 0);
    const reference_role_binding roles[] = {gate, up, down};
    for (uint32_t kind = 0; valid && kind < 2; ++kind) {
        const auto group = reference_group(private_control, layout.geometry.route_capacity, kind);
        uint32_t groups = 0, entries = 0;
        if (*group.count < 0 || uint32_t(*group.count) > packet.ticket->weight_count || group.starts[0] != 0) { valid = false; break; }
        for (uint32_t weight = 0; valid && weight < packet.ticket->weight_count; ++weight) {
            if (packet.weight_classes[weight] != kind) { continue; }
            if (groups >= uint32_t(*group.count) || group.starts[groups] != int32_t(entries)) { valid = false; break; }
            const uint64_t pointers[] = {group.gate[groups], group.up[groups], group.down[groups]};
            for (uint32_t role = 0; role < 3; ++role) {
                const auto binding = roles[role];
                const uint64_t base = kind == 0 ? binding.resident : binding.selected;
                const size_t stride = kind == 0 ? binding.resident_stride : binding.selected_stride;
                if (pointers[role] != base + size_t(packet.weight_storage[weight]) * stride + binding.offset) { valid = false; }
            }
            for (uint32_t route = 0; valid && route < packet.ticket->route_count; ++route) {
                if (packet.routes[route].weight_index != weight) { continue; }
                if (entries >= layout.geometry.route_capacity || group.tokens[entries] != int32_t(packet.routes[route].source_row) ||
                        group.destinations[entries] != int32_t(packet.route_lanes[route])) { valid = false; break; }
                ++entries;
            }
            if (group.starts[++groups] != int32_t(entries)) { valid = false; }
        }
        if (groups != uint32_t(*group.count) || entries != packet.expected_routes[kind]) { valid = false; }
    }
    if (!valid) {
        if (!packet.ticket->status) { packet.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; }
        private_control->groups[0] = private_control->groups[1] = 0;
        runtime->failed = 1;
    }
    if (runtime->fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_B) {
        control->test_transfer_groups = private_control->groups[1];
        release_flag(&control->test_validated);
    }

}
// Flatten selected experts into uint4 words.
static __global__ void reference_copy(ggml_cuda_moe_hybrid_rows_view packet, const int32_t * experts,
        const uint8_t * source, size_t expert_stride, size_t bytes, uint8_t * output,
        reference_control * control, const window_runtime * runtime) {
    if (runtime->failed || packet.ticket->status) { return; }
    const size_t per = bytes / sizeof(uint4), total = size_t(packet.ticket->transfer_count) * per;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const size_t expert = i / per, off = i - expert * per;
        reinterpret_cast<uint4 *>(output)[i] = reinterpret_cast<const uint4 *>(source + size_t(experts[expert]) * expert_stride)[off];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) { control->executed_bytes += uint64_t(packet.ticket->transfer_count) * bytes; }
}
static __global__ void reference_wait_sources(const reference_control * control, reference_control * private_control,
        ggml_cuda_moe_hybrid_rows_view packet, const doorbell * bell, window_runtime * runtime) {
    while (!acquire_flag(&control->transfer_ready) && !acquire_flag(&bell->stop)) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
    if (acquire_flag(&bell->stop) || runtime->failed) {
        runtime->failed = 1; private_control->groups[1] = 0;
        if (!packet.ticket->status) { packet.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED; }
    }
}
static __global__ void reference_complete(ggml_cuda_moe_hybrid_rows_view packet, uint32_t kind) {
    const uint32_t count = packet.expected_routes[kind];
    if (count) { packet.producers[kind] = {packet.ticket->epoch, packet.ticket->status ? 0u : count, int32_t(packet.ticket->status), 1}; }
}

static __global__ void probe_mapped_protocol(doorbell * bell, uint64_t identity) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 600
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(bell->ready.value).store(1, cuda::memory_order_release);
    while (acquire_flag(&bell->done) == 0 && acquire_flag(&bell->stop) == 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
    const bool valid = acquire_flag(&bell->done) == 1 && bell->epoch == identity && bell->status == 0;
    bell->epoch = ~identity;
    bell->status = valid ? 0 : GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>(bell->wake.value).store(1, cuda::memory_order_release);
#else
    GGML_UNUSED(bell);
    GGML_UNUSED(identity);
#endif
}

static __global__ void accept_cpu(ggml_cuda_moe_hybrid_rows_view rows, window_runtime * runtime,
        const doorbell * bell, join_record * record) {
    *record = {runtime->epoch, 0, bell ? acquire_flag(&bell->done) : 1, 0};
    const bool stopped = bell && acquire_flag(&bell->stop);
    const bool accepted = record->cpu_terminal && !stopped &&
        (!bell || (bell->epoch == runtime->epoch && bell->status == 0));
    if (!accepted || runtime->failed) {
        runtime->failed = 1;
        if (!rows.ticket->status) { rows.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED; }
    }
}

static __global__ void import_cpu(ggml_cuda_moe_hybrid_rows_layout layout, ggml_cuda_moe_hybrid_rows_view rows,
        const ggml_cuda_moe_hybrid_producer_result * producers, const float * source, float * output, join_record * record) {
    const auto & ticket = *rows.ticket;
    if (ticket.status || !record->cpu_terminal || ticket.route_count > layout.geometry.route_capacity) { return; }
    const size_t values = size_t(ticket.route_count) * layout.output_width;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < values; i += size_t(gridDim.x) * blockDim.x) {
        const auto weight = rows.routes[i / layout.output_width].weight_index;
        if (weight < layout.geometry.weight_capacity && rows.weight_classes[weight] == GGML_CUDA_MOE_HYBRID_CPU) {
            output[i] = source[i];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint32_t row = 0; row < layout.geometry.row_capacity; ++row) { rows.producers[row + 2] = producers[row]; }
        record->imported_bytes = size_t(ticket.cpu_routes) * layout.output_width * sizeof(float) +
            size_t(layout.geometry.row_capacity) * sizeof(*producers);
    }
}

static __global__ void producer_status(
        ggml_cuda_moe_hybrid_rows_view rows, const window_runtime * runtime, uint32_t * bounded_status) {
    const uint32_t fault = runtime->fault;
    if (fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_GPU) { *bounded_status = 1; }
    if (*bounded_status != 0 && rows.ticket->status == 0) {
        rows.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }
    if (fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TRANSFER) {
        rows.producers[1].status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
        if (rows.ticket->status == 0) { rows.ticket->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
    }
}

static __global__ void layer_terminal(ggml_cuda_moe_hybrid_rows_view rows, window_runtime * runtime, join_record * record) {
    if (!rows.ticket->accepted || rows.ticket->status) { runtime->failed = 1; }
    record->complete = 1;
}

#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
static __global__ void continue_window(const window_runtime * runtime, cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle, runtime->failed == 0);
}
#endif

struct window_session;
struct tile_job {
    window_session * session = nullptr;
    uint32_t layer_index = 0, source_index = 0, transfer_index = 0;
};
struct layer_capture {
    window_session * session = nullptr;
    uint32_t index = 0;
    cudaEvent_t plan = nullptr, transfer = nullptr;
    std::vector<tile_job> tiles;
};

struct window_session {
    compiled_window compiled;
    ggml_backend_t backend = nullptr;
    ggml_backend_cuda_context * context = nullptr;
    std::unique_ptr<ggml_backend_cuda_context> compute;
    ggml_backend_buffer_t arena = nullptr;
    uint8_t * host = nullptr, * alias = nullptr;
    ggml_backend_reg_t cpu_module = nullptr;
    ggml_backend_moe_cpu_service_v1_t cpu_service = nullptr;
    ggml_backend_moe_source_owner_v1 source_owner = {};
    bool graph_retained = false, dispatch = false, closed = false;
    bool trace_emission = getenv("GGML_MOE_FIDELITY_TRACE") != nullptr;
    std::atomic<bool> resources_live{false};
    uint32_t arm = GGML_CUDA_MOE_FIDELITY_SEGMENTED;
    uint64_t workspace_generation = 0, resource_identity = 0, last_epoch = 0;
    uint64_t source_norm_capture_launches = 0;
    cudaStream_t stream = nullptr, io = nullptr;
    std::shared_ptr<ggml_cuda_moe_graph_plan> plan;
    ggml_cuda_moe_graph_execution execution;
    std::vector<std::shared_ptr<void>> resource_leases;
    std::vector<layer_capture> captures;
    std::vector<cudaGraph_t> graphs;
    std::vector<cudaGraphExec_t> executables;
    cudaGraph_t probe_graph = nullptr;
    cudaGraphExec_t probe_executable = nullptr;
    std::mutex mutex;
    std::condition_variable condition;
    std::thread worker, supervisor;
    bool stopping = false, armed = false, working = false;
    const bool caller_schedule = ggml_moe_fidelity_selection().source_pool;
    bool caller_active = false;
    std::thread::id caller_id;
    uint32_t external_drains = 0;
    uint64_t caller_replays = 0, caller_jobs = 0, caller_progress_probes = 0;
    uint64_t deadline = 0, cpu_ns = 0;
    std::atomic<bool> canceled{false};
    ggml_backend_moe_cpu_test_hook_v1_t test_hook = nullptr;
    void * test_hook_data = nullptr;
    uint32_t cpu_layer = 0, test_phase = 0;
    bool test_hook_active = false;

    uint8_t * data() const { return static_cast<uint8_t *>(ggml_backend_buffer_get_base(arena)); }
    window_runtime * runtime() const { return reinterpret_cast<window_runtime *>(data() + compiled.runtime_offset); }
    window_runtime * host_runtime() const { return reinterpret_cast<window_runtime *>(host + compiled.host_runtime_offset); }
    window_runtime * host_result() const { return reinterpret_cast<window_runtime *>(host + compiled.host_result_offset); }
    doorbell * bell(uint32_t index) const { return reinterpret_cast<doorbell *>(host + compiled.layers[index].host_bell_offset); }
    ggml_cuda_moe_hybrid_rows_view rows(uint32_t index, bool cpu = false) const {
        const auto & l = compiled.layers[index];
        return ggml_cuda_moe_hybrid_rows_packet(cpu ? host + l.host_rows_offset + l.rows.host_control_offset :
            data() + l.rows_offset + l.rows.control_offset, l.rows.geometry);
    }
    ggml_cuda_moe_hybrid_rows_view completion(uint32_t index) const {
        const auto & l = compiled.layers[index];
        return ggml_cuda_moe_hybrid_rows_packet(host + l.host_completion_offset, l.rows.geometry);
    }
    join_record * join(uint32_t index, bool cpu = false) const {
        const auto & l = compiled.layers[index];
        return reinterpret_cast<join_record *>(cpu ? host + l.host_join_offset : data() + l.join_offset);
    }
    void * row_plan(const layer & l) const { return data() + l.rows_offset + l.rows.plan_offset; }

    reference_control * reference_device(uint32_t index) const {
        return reinterpret_cast<reference_control *>(data() + compiled.layers[index].reference_device_offset);
    }
    reference_control * reference(uint32_t index, bool mapped = false) const {
        return reinterpret_cast<reference_control *>((mapped ? alias : host) + compiled.layers[index].reference_offset);
    }
    static void cpu_test_hook(void * opaque, uint32_t phase) {
        auto & self = *static_cast<window_session *>(opaque);
        {
            std::lock_guard<std::mutex> lock(self.mutex);
            self.test_phase = phase;
            self.test_hook_active = true;
        }
        try {
            self.test_hook(self.test_hook_data, phase);
        } catch (...) {
            std::lock_guard<std::mutex> lock(self.mutex);
            self.test_hook_active = false;
            throw;
        }
        std::lock_guard<std::mutex> lock(self.mutex);
        self.test_hook_active = false;
    }

    bool request_stop() {
        canceled.store(true);
        if (host) {
            for (uint32_t i = 0; i < compiled.layers.size(); ++i) {
                bell(i)->stop.store(1);
                bell(i)->wake.store(1);
            }
        }
        const bool ok = !cpu_service || !last_epoch || compiled.query.cpu_api->cancel(cpu_service, last_epoch) == 0;
        condition.notify_all();
        return ok;
    }

    ggml_backend_moe_cpu_service_config_v1 cpu_config(uint64_t prepared_payload_limit) {
        source_owner = *compiled.query.source_owner;
        ggml_backend_moe_cpu_service_config_v1 config = {};
        config.struct_size = sizeof(config);
        config.abi_version = 1;
        config.source_owner = &source_owner;
        config.n_threads = compiled.query.n_threads;
        config.n_lanes = 1;
        config.max_regions = compiled.layers.size();
        config.flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
            GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
        config.prepared_payload_limit = prepared_payload_limit;
        return config;
    }

    int32_t measure_cpu(uint64_t prepared_payload_limit) {
        const auto config = cpu_config(prepared_payload_limit);
        auto * api = compiled.query.cpu_requirements_api;
        uint64_t total = 0;
        int32_t status = api->service(&config, &total);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
        for (auto & l : compiled.layers) {
            l.cpu_requirements.struct_size = sizeof(l.cpu_requirements);
            l.cpu_requirements.abi_version = 1;
            status = api->region(&config, l.cpu_query, &l.cpu_requirements);
            if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
            if (l.cpu_requirements.lane_context_bytes || l.cpu_requirements.execution.graph_work_bytes) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
            if (l.cpu_requirements.prepared_payload_bytes > UINT64_MAX - total) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
            }
            total += l.cpu_requirements.prepared_payload_bytes;
        }
        if (prepared_payload_limit && total > prepared_payload_limit) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
        compiled.state.storage.cpu_bytes = total;
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    }

    int32_t prepare_cpu(uint64_t prepared_payload_limit) {
        cpu_module = ggml_backend_moe_cpu_module_acquire_v1();
        if (!cpu_module) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
        const auto get = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
            ggml_backend_reg_get_proc_address(cpu_module, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME));
        if (!get || get() != compiled.query.cpu_api) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
        const auto config = cpu_config(prepared_payload_limit);
        int32_t status = compiled.query.cpu_api->create(&config, &cpu_service);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
        for (auto & l : compiled.layers) {
            l.cpu_requirements.struct_size = sizeof(l.cpu_requirements);
            l.cpu_requirements.abi_version = 1;
            status = compiled.query.cpu_api->prepare(cpu_service, l.cpu_query, &l.cpu_requirements, &l.cpu_region);
            if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
            if (l.cpu_requirements.lane_context_bytes || l.cpu_requirements.execution.graph_work_bytes) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
        }
        ggml_backend_moe_cpu_service_state_v1 state = {};
        state.struct_size = sizeof(state);
        status = compiled.query.cpu_api->state(cpu_service, &state);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
        if (state.prepared_payload_bytes != compiled.state.storage.cpu_bytes) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    }

    bool prepare_owner() {
        auto & owner = *context->moe_grouped_context;
        uint32_t mmids = 0;
        uint64_t fingerprint = 0;
        const uint64_t coverage = owner.certify_graph_coverage(compiled.query.graph, &mmids, &fingerprint);
        if (!coverage || !mmids || !fingerprint || owner.prepare_graph_execution(compiled.query.graph,
                compiled.query.graph->uid, GGML_CUDA_MOE_GRAPH_PROPERTIES_CHANGED, &plan, &execution,
                coverage, compiled.query.graph->nodes, mmids, fingerprint) == GGML_CUDA_MOE_GRAPH_PREPARE_UNAVAILABLE ||
                !execution.resolve_streams([](void * opaque, const ggml_tensor *) {
                    return static_cast<window_session *>(opaque)->stream;
                }, this)) { return false; }
        return true;
    }

    bool begin_dispatch() {
        auto & owner = *context->moe_grouped_context;
        if (dispatch || !owner.begin_graph_dispatch(&execution, GGML_CUDA_MOE_GRAPH_DISPATCH_DIRECT)) { return false; }
        dispatch = true;
        for (auto & l : compiled.layers) {
            ggml_cuda_moe_graph_binding binding;
            l.group = execution.find_group(compiled.query.graph->nodes[l.first], &binding);
            auto packet = ggml_cuda_moe_hybrid_packet(data() + l.packet_offset, l.rows.geometry.route_capacity);
            ggml_cuda_moe_hybrid_runtime initial = {};
            if (!l.group || !owner.prepare_hybrid_group(l.group, stream, packet.header, packet.slots, packet.residents,
                    0, packet, compiled.query.gpu_miss_quota, last_epoch, &initial,
                    l.rows.geometry.route_capacity, std::max(1u, l.rows.transfer_capacity))) { return false; }
            if (compiled.reference) {
                ggml_cuda_moe_hybrid_reference_view fresh;
                if (!owner.get_hybrid_reference_view(*l.group, &fresh) || fresh.n_experts != l.rows.geometry.expert_count || fresh.n_slots != l.rows.slot_capacity ||
                        (l.owner_view.resource_identity && (fresh.resource_identity != l.owner_view.resource_identity ||
                            fresh.n_experts != l.owner_view.n_experts || fresh.n_slots != l.owner_view.n_slots ||
                            fresh.slot_for_expert != l.owner_view.slot_for_expert || fresh.expert_for_slot != l.owner_view.expert_for_slot))) { return false; }
                l.owner_view = fresh;
            }
            uint32_t mask = 0;
            for (uint32_t i = 0; i < l.source_banks.size(); ++i) {
                const auto & source = l.descriptor.query->sources[i];
                ggml_cuda_moe_grouped_bank_descriptor fresh;
                uint32_t bank = 0;
                for (; bank < l.group->key.n_banks; ++bank) {
                    if (!owner.get_group_resource_bank(l.group->transaction, bank, &fresh)) { return false; }
                    if (fresh.tensor == source.witness) { break; }
                }
                const auto & found = fresh;
                if (bank == l.group->key.n_banks || (mask & (1u << bank)) || found.source_data != source.data ||
                        found.expert_stride != source.expert_stride || found.byte_extent != source.bytes ||
                        found.type != uint32_t(source.tensor->type) || !l.group->bank_data[bank]) { return false; }
                if (compiled.reference && (found.encoding != GGML_CUDA_MOE_CANDIDATE_ENCODING_PLAIN ||
                        found.movement != GGML_CUDA_MOE_CANDIDATE_MOVEMENT_SLOT_BOUND ||
                        memcmp(found.ne, source.tensor->ne, sizeof(found.ne)) || memcmp(found.nb, source.tensor->nb, sizeof(found.nb)))) { return false; }
                mask |= 1u << bank;
                l.bank_indices[i] = bank;
                if (compiled.reference) {
                    const auto & previous = l.source_banks[i];
                    if (previous.tensor && (previous.source_data != fresh.source_data || previous.source_device_alias != fresh.source_device_alias ||
                            previous.buffer != fresh.buffer || previous.buft != fresh.buft || previous.buffer_base != fresh.buffer_base ||
                            previous.buffer_size != fresh.buffer_size || previous.data_offset != fresh.data_offset ||
                            previous.byte_extent != fresh.byte_extent || previous.expert_stride != fresh.expert_stride || previous.alignment != fresh.alignment ||
                            previous.role != fresh.role || previous.type != fresh.type || previous.encoding != fresh.encoding ||
                            previous.movement != fresh.movement || previous.index_modes != fresh.index_modes || previous.source_path != fresh.source_path ||
                            memcmp(previous.ne, fresh.ne, sizeof(fresh.ne)) || memcmp(previous.nb, fresh.nb, sizeof(fresh.nb)))) { return false; }
                } else {
                    auto * tensor = l.resident.find(source.tensor);
                    if (!tensor) { return false; }
                    tensor->data = const_cast<void *>(l.group->bank_data[bank]);
                }
                l.source_banks[i] = fresh;
            }
            if (mask != (1u << l.group->key.n_banks) - 1) { return false; }
        }
        uint64_t current = 0;
        if (!owner.graph_resource_fingerprint(execution, stream, &current, resource_identity ? nullptr : &resource_leases) ||
                (resource_identity && current != resource_identity)) { return false; }
        resource_identity = current;
        {
            std::lock_guard<std::mutex> lock(mutex);
            compiled.state.source_banks = compiled.state.mapped_source_banks = 0;
            for (const auto & l : compiled.layers) {
                for (const auto & source : l.source_banks) {
                    ++compiled.state.source_banks;
                    compiled.state.mapped_source_banks += source.source_device_alias != nullptr;
                }
            }
        }
        return true;
    }

    bool finish_dispatch() {
        if (!dispatch) { return true; }
        auto & owner = *context->moe_grouped_context;
        for (auto & l : compiled.layers) {
            if (l.group && l.group->hybrid_admission_started &&
                    !owner.finish_hybrid_admission(*l.group, false, true)) { return false; }
        }
        if (!owner.finish_graph_dispatch(&execution)) { return false; }
        dispatch = false;
        for (auto & l : compiled.layers) { l.group = nullptr; }
        return true;
    }

    static void ready_callback(void * opaque) {
        auto & capture = *static_cast<layer_capture *>(opaque);
        auto & session = *capture.session;
        auto * b = session.bell(capture.index);
        b->ready.store(1);
    }

    static void fill_tile(void * opaque) {
        const auto & job = *static_cast<tile_job *>(opaque);
        auto & session = *job.session;
        const auto & l = session.compiled.layers[job.layer_index];
        const auto packet = session.rows(job.layer_index, true);
        const auto & source = l.source_banks[job.source_index];
        auto * tile = session.host + l.host_rows_offset + l.rows.host_staging_offsets[job.transfer_index % 2];
        memset(tile, 0, l.sources[job.source_index].expert_bytes);
        if (packet.ticket->status || session.canceled.load() ||
                session.host_runtime()->fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TRANSFER) { return; }
        for (uint32_t weight = 0; weight < packet.ticket->weight_count; ++weight) {
            if (packet.weight_classes[weight] == GGML_CUDA_MOE_HYBRID_TRANSFER &&
                    packet.weight_storage[weight] == job.transfer_index) {
                const int32_t expert = packet.weight_experts[weight];
                if (expert >= 0 && uint32_t(expert) < l.rows.geometry.expert_count) {
                    memcpy(tile, static_cast<const uint8_t *>(source.source_data) + size_t(expert) * source.expert_stride,
                        l.sources[job.source_index].expert_bytes);
                }
                return;
            }
        }
    }

    void plan_reference(uint32_t index) {
        auto & l = compiled.layers[index];
        if (host_runtime()->fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_A) { request_stop(); }
        auto * control = reference(index);
        auto packet = rows(index, true);
        auto & ticket = *packet.ticket;
        const auto * raw = reinterpret_cast<const reference_route *>(control + 1);
        const auto runtime = *packet.runtime;
        ticket = {};
        ticket.epoch = runtime.epoch; ticket.identity = l.rows.identity; ticket.active_rows = runtime.active_rows;
        const uint64_t route_count = uint64_t(runtime.active_rows) * l.rows.geometry.routes_per_row;
        bool valid = !control->invalid && !canceled.load() && runtime.epoch == host_runtime()->epoch && runtime.epoch == bell(index)->epoch &&
            runtime.identity == l.rows.identity && runtime.active_rows > 0 && runtime.active_rows <= l.rows.geometry.row_capacity &&
            runtime.n_sequences == compiled.query.certificate.n_sequences && runtime.corrupt_route == 0 &&
            (l.rows.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ? runtime.active_rows == 1 && runtime.n_sequences == 1 :
                l.rows.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE && runtime.active_rows > runtime.n_sequences) &&
            route_count <= l.rows.geometry.route_capacity;
        ticket.route_count = valid ? uint32_t(route_count) : 0;
        l.distinct.clear(); l.distinct_slots.clear(); l.misses.clear();
        std::fill(l.reference_counts.begin(), l.reference_counts.end(), 0u);
        l.cpu_group_max = 0;
        for (uint32_t route = 0; valid && route < ticket.route_count; ++route) {
            const auto record = raw[route];
            if (record.expert < 0 || uint32_t(record.expert) >= l.rows.geometry.expert_count || record.slot < -1 ||
                    (record.slot >= 0 && (uint32_t(record.slot) >= l.rows.slot_capacity || record.reciprocal != record.expert))) { valid = false; break; }
            const auto found = std::find(l.distinct.begin(), l.distinct.end(), record.expert);
            uint32_t weight = uint32_t(found - l.distinct.begin());
            if (found == l.distinct.end()) {
                if (l.distinct.size() >= l.rows.geometry.weight_capacity) { valid = false; break; }
                l.distinct.push_back(record.expert); l.distinct_slots.push_back(record.slot);
                packet.weight_experts[weight] = record.expert;
                packet.weight_classes[weight] = record.slot >= 0 ? GGML_CUDA_MOE_HYBRID_RESIDENT : GGML_CUDA_MOE_HYBRID_CPU;
                packet.weight_storage[weight] = record.slot >= 0 ? uint32_t(record.slot) : UINT32_MAX;
                if (record.slot < 0) { l.misses.push_back(int32_t(weight)); }
            } else if (l.distinct_slots[weight] != record.slot) { valid = false; break; }
            l.route_weights[route] = weight;
            ++l.reference_counts[weight];
        }
        ticket.weight_count = uint32_t(l.distinct.size());
        bool eligible = true;
        for (uint32_t source = 0; source < l.source_banks.size(); ++source) {
            const auto & bank = l.source_banks[source];
            eligible &= bank.source_device_alias != nullptr && reinterpret_cast<uintptr_t>(bank.source_device_alias) % 16 == 0 &&
                bank.expert_stride % 16 == 0 && l.sources[source].expert_bytes % 16 == 0;
        }
        const size_t selected = l.misses.size() * compiled.pcie_num / 256;
        control->selected_bytes = 0;
        // Rank all misses first; an ineligible selected expert stays on CPU without backfill.
        for (size_t i = l.misses.size() - selected; valid && i < l.misses.size(); ++i) {
            if (!eligible) { continue; }
            const auto weight = uint32_t(l.misses[i]);
            packet.weight_classes[weight] = GGML_CUDA_MOE_HYBRID_TRANSFER;
            packet.weight_storage[weight] = ticket.transfer_count++;
            for (const auto & source : l.sources) { control->selected_bytes += source.expert_bytes; }
        }
        std::fill_n(packet.expected_routes, l.rows.geometry.row_capacity + 2, 0u);
        std::fill_n(packet.cpu_routes, l.rows.geometry.route_capacity, UINT32_MAX);
        std::fill_n(packet.route_lanes, l.rows.geometry.route_capacity, UINT32_MAX);
        std::fill_n(packet.gpu_routes, l.rows.geometry.route_capacity, UINT32_MAX);
        for (uint32_t route = 0; valid && route < ticket.route_count; ++route) {
            const uint32_t row = route / l.rows.geometry.routes_per_row, weight = l.route_weights[route];
            packet.routes[route] = {row, route % l.rows.geometry.routes_per_row, weight, route};
            const auto kind = packet.weight_classes[weight];
            if (kind == GGML_CUDA_MOE_HYBRID_CPU) {
                packet.cpu_routes[size_t(row) * l.rows.geometry.routes_per_row + packet.expected_routes[row + 2]++] = route;
                ++ticket.cpu_routes;
            } else { ++packet.expected_routes[kind]; }
        }
        for (uint32_t weight = 0; weight < ticket.weight_count; ++weight) {
            if (packet.weight_classes[weight] == GGML_CUDA_MOE_HYBRID_CPU) { l.cpu_group_max = std::max(l.cpu_group_max, l.reference_counts[weight]); }
        }
        ticket.resident_lanes = packet.expected_routes[0]; ticket.transfer_lanes = packet.expected_routes[1];
        ticket.gpu_lanes = ticket.resident_lanes + ticket.transfer_lanes;
        uint32_t lane[2] = {0, ticket.resident_lanes};
        for (uint32_t route = 0; valid && route < ticket.route_count; ++route) {
            const auto kind = packet.weight_classes[l.route_weights[route]];
            if (kind < 2) { packet.gpu_routes[lane[kind]] = route; packet.route_lanes[route] = lane[kind]++; }
        }
        for (uint32_t producer = 0; producer < l.rows.geometry.row_capacity + 2; ++producer) { ticket.expected_producers += packet.expected_routes[producer] != 0; }
        for (uint32_t kind = 0; kind < 2; ++kind) {
            auto group = reference_group(control, l.rows.geometry.route_capacity, kind);
            uint32_t n_groups = 0, n_entries = 0;
            group.starts[0] = 0;
            for (uint32_t weight = 0; valid && weight < ticket.weight_count; ++weight) {
                if (packet.weight_classes[weight] != kind) { continue; }
                uint64_t * pointers[] = {group.gate, group.up, group.down};
                for (uint32_t role = 0; role < 3; ++role) {
                    const auto binding = l.roles[role];
                    const auto source = binding.source;
                    const auto base = kind == 0 ? reinterpret_cast<uint64_t>(l.group->bank_data[l.bank_indices[source]]) :
                        reinterpret_cast<uint64_t>(data() + compiled.selected_offset + l.selected_offsets[source]);
                    const auto stride = kind == 0 ? l.source_banks[source].expert_stride : l.sources[source].expert_bytes;
                    pointers[role][n_groups] = base + size_t(packet.weight_storage[weight]) * stride + binding.offset;
                }
                for (uint32_t route = 0; route < ticket.route_count; ++route) {
                    if (l.route_weights[route] != weight) { continue; }
                    group.tokens[n_entries] = packet.routes[route].source_row;
                    group.destinations[n_entries++] = packet.route_lanes[route];
                }
                group.starts[++n_groups] = n_entries;
            }
            *group.count = n_groups;
        }
        if (!valid) { ticket.status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; }
        if (valid) {
            const uint32_t kind = control->groups[0] ? 0u : 1u;
            auto group = reference_group(control, l.rows.geometry.route_capacity, kind);
            switch (host_runtime()->fault) {
                case GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_COUNT: *group.count = int32_t(l.rows.geometry.route_capacity + 1); break;
                case GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_START: group.starts[0] = 1; break;
                case GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_TOKEN: group.tokens[0] = int32_t(l.rows.geometry.row_capacity); break;
                case GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_DESTINATION: group.destinations[0] = int32_t(l.rows.geometry.route_capacity); break;
                case GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_GROUP_POINTER: group.gate[0] = 1; break;
                default: break;
            }
        }
        control->plan_ready.store(1);
        if (host_runtime()->fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_B) {
            while (!control->test_validated.load() && !canceled.load()) { std::this_thread::yield(); }
            if (control->test_validated.load() && control->test_transfer_groups > 0) {
                std::lock_guard<std::mutex> lock(mutex);
                ++compiled.state.reference_before_b_guards;
            }
            request_stop();
        }
        control->transfer_ready.store(1);
    }

    void run_cpu(uint32_t index) {
        auto & l = compiled.layers[index];
        auto packet = rows(index, true);
        const auto & ticket = *packet.ticket;
        int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
        uint32_t count = 0;
        if (canceled.load() || ticket.status || ticket.epoch != host_runtime()->epoch || ticket.identity != l.rows.identity ||
                ticket.active_rows > l.rows.geometry.row_capacity || ticket.route_count > l.rows.geometry.route_capacity ||
                ticket.weight_count > l.rows.geometry.weight_capacity || ticket.cpu_routes > l.rows.geometry.route_capacity) {
            status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        for (uint32_t row = 0; status == 0 && row < ticket.active_rows; ++row) {
            const uint32_t routes = packet.expected_routes[row + 2];
            if (routes > l.rows.geometry.routes_per_row) { status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; break; }
            for (uint32_t i = 0; i < routes; ++i) {
                const uint32_t route = packet.cpu_routes[row * l.rows.geometry.routes_per_row + i];
                if (route >= ticket.route_count) { status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; break; }
                const auto record = packet.routes[route];
                if (record.weight_index >= ticket.weight_count || record.source_row != row || record.scatter_destination != route ||
                        packet.weight_classes[record.weight_index] != GGML_CUDA_MOE_HYBRID_CPU) {
                    status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; break;
                }
                l.cpu_ids[count] = packet.weight_experts[record.weight_index];
                l.cpu_rows[count] = row;
                l.cpu_scatter[count++] = route;
            }
        }
        if (count != ticket.cpu_routes || host_runtime()->fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CPU) {
            status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
        }
        if (!status && count) {
            ggml_backend_moe_cpu_region_binding_v1 binding{sizeof(binding), count, count,
                l.cpu_ids.data(), l.cpu_rows.data(), l.cpu_scatter.data()};
            ggml_backend_moe_cpu_dynamic_input_v1 inputs[2] = {{host + l.host_input_offset,
                l.rows.host_input_bytes, l.descriptor.activation->nb[2]}, {}};
            ggml_backend_moe_cpu_output_v1 output{host + l.host_output_offset,
                l.rows.private_output_bytes, l.descriptor.output->nb[1]};
            ggml_backend_moe_cpu_execute_v1 job = {};
            job.struct_size = sizeof(job);
            job.epoch = ticket.epoch;
            job.graph_uid = l.cpu_graph_uid;
            job.graph_generation = l.cpu_graph_generation;
            job.source_generation = l.cpu_source_generation;
            job.binding = &binding;
            job.dynamic_inputs = inputs;
            job.n_dynamic_inputs = 2;
            job.outputs = &output;
            job.n_outputs = 1;
            ggml_backend_moe_cpu_execute_result_v1 result = {};
            result.struct_size = sizeof(result);
            const auto started = now_ns();
            status = compiled.query.cpu_api->execute(cpu_service, l.cpu_region, &job, &result);
            {
                std::lock_guard<std::mutex> lock(mutex);
                cpu_ns += now_ns() - started;
                ++compiled.state.cpu_calls;
                if (compiled.reference && status == 0) {
                    compiled.state.cpu_max_group_entries = std::max(compiled.state.cpu_max_group_entries, uint64_t(l.cpu_group_max));
                }
            }
            if (!status && (result.flags != GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED ||
                    result.epoch != ticket.epoch || result.published_routes != count || result.published_outputs != 1)) {
                status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
            }
        }
        for (uint32_t row = 0; row < ticket.active_rows && row < l.rows.geometry.row_capacity; ++row) {
            const uint32_t routes = packet.expected_routes[row + 2];
            packet.producers[row + 2] = {ticket.epoch, status == 0 ? routes : 0, status, routes != 0};
        }
        bell(index)->status = status;
    }

    void worker_main() noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping) {
            condition.wait(lock, [&] { return stopping || armed; });
            if (stopping) { break; }
            working = true;
            lock.unlock();
            for (uint32_t i = 0; i < compiled.layers.size(); ++i) {
                auto * b = bell(i);
                while (!b->ready.load() && !canceled.load()) { std::this_thread::yield(); }
                lock.lock();
                const bool claimed = !canceled.load() && b->ready.load();
                if (claimed) { b->claimed.store(1); cpu_layer = i; }
                lock.unlock();
                if (claimed) {
                    try { if (compiled.reference) { plan_reference(i); } run_cpu(i); } catch (...) { b->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
                } else {
                    b->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
                }
                lock.lock();
                b->done.store(1);
                b->wake.store(1);
                condition.notify_all();
                lock.unlock();
            }
            lock.lock();
            working = false;
            armed = false;
            condition.notify_all();
        }
    }

    static bool caller_launch(void * opaque, uint32_t segment) {
        auto & self = *static_cast<window_session *>(opaque);
        const bool ok = cudaGraphLaunch(self.executables[segment], self.stream) == cudaSuccess;
        std::lock_guard<std::mutex> lock(self.mutex);
        ++self.compiled.state.launches;
        return ok;
    }

    static bool caller_ready(void * opaque, uint32_t index) {
        return static_cast<window_session *>(opaque)->bell(index)->ready.load() != 0;
    }

    static bool caller_canceled(void * opaque) {
        return static_cast<window_session *>(opaque)->canceled.load();
    }

    static ggml_moe_caller_progress caller_progress(void * opaque) {
        auto & self = *static_cast<window_session *>(opaque);
        const auto main_status = cudaStreamQuery(self.stream);
        const auto io_status = cudaStreamQuery(self.io);
        if ((main_status != cudaSuccess && main_status != cudaErrorNotReady) ||
                (io_status != cudaSuccess && io_status != cudaErrorNotReady)) { return ggml_moe_caller_progress::failed; }
        return main_status == cudaSuccess && io_status == cudaSuccess ?
            ggml_moe_caller_progress::complete : ggml_moe_caller_progress::pending;
    }

    static bool caller_service(void * opaque, uint32_t index) {
        auto & self = *static_cast<window_session *>(opaque);
        auto * b = self.bell(index);
        {
            std::lock_guard<std::mutex> lock(self.mutex);
            if (!self.caller_active || self.caller_id != std::this_thread::get_id() || self.canceled.load() || !b->ready.load()) { return false; }
            b->claimed.store(1);
            self.cpu_layer = index;
        }
        try {
            if (self.compiled.reference) { self.plan_reference(index); }
            self.run_cpu(index);
        } catch (...) { b->status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
        std::lock_guard<std::mutex> lock(self.mutex);
        b->done.store(1);
        b->wake.store(1);
        self.condition.notify_all();
        return b->status == 0;
    }

    void supervisor_main() noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping) {
            condition.wait(lock, [&] { return stopping || armed; });
            if (stopping) { break; }
            const auto expiry = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(deadline));
            const auto supervised_epoch = last_epoch;
            if (condition.wait_until(lock, expiry, [&] { return stopping || !armed || (caller_schedule && last_epoch != supervised_epoch); })) { continue; }
            request_stop();
            condition.wait(lock, [&] { return stopping || !armed || (caller_schedule && last_epoch != supervised_epoch); });
        }
    }

    void trace_node(const char * path, uint32_t index, const ggml_tensor * node, uint32_t lanes = 0) const {
        if (!trace_emission) { return; }
        fprintf(stderr, "moe-fidelity: emit path=%s index=%u op=%s name=%s lanes=%u stream=%p\n",
            path, index, ggml_op_name(node->op), node->name, lanes, static_cast<void *>(stream));
        for (int source = -1; source < GGML_MAX_SRC; ++source) {
            const auto * tensor = source < 0 ? node : node->src[source];
            if (!tensor) { continue; }
            fprintf(stderr, "moe-fidelity: tensor source=%d type=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu data=%p view=%p offset=%zu\n",
                source, ggml_type_name(tensor->type), (long long) tensor->ne[0], (long long) tensor->ne[1],
                (long long) tensor->ne[2], (long long) tensor->ne[3], tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
                tensor->data, static_cast<void *>(tensor->view_src), tensor->view_offs);
        }
        fflush(stderr);
    }

    bool emit_range(uint32_t first, uint32_t end) {
        for (uint32_t i = first; i < end; ++i) {
            if (i + 1 < end) {
                const auto found = std::find_if(compiled.ordinary_norms.begin(), compiled.ordinary_norms.end(),
                    [&](const auto & item) { return item.index == i && item.bound; });
                if (found != compiled.ordinary_norms.end()) {
                    trace_node("source-weighted-rms", i, compiled.nodes[i]);
                    trace_node("source-weighted-rms-output", i + 1, compiled.nodes[i + 1]);
                    if (!ggml_cuda_moe_source_norm_emit(*compute, *found)) { return false; }
                    ++source_norm_capture_launches;
                    ++i;
                    continue;
                }
            }
            trace_node("ordinary", i, compiled.nodes[i]);
            if (!ggml_cuda_moe_fidelity_emit_node(*compute, compiled.query.graph, compiled.query.graph->uid, i, compiled.nodes[i])) { return false; }
        }
        return true;
    }

    bool emit_body(layer & l, gpu_body & body, uint32_t kind) {
        const uint32_t index = &l - compiled.layers.data();
        const auto packet = rows(index);
        const auto * packed = reinterpret_cast<float *>(data() + l.rows_offset + l.rows.packed_input_offset);
        if (!ggml_cuda_moe_hybrid_rows_gpu_binding(l.rows, packet, kind, packed,
                static_cast<float *>(body.tensors[0].data), static_cast<int32_t *>(body.tensors[1].data), stream)) { return false; }
        const uint32_t * count = kind == GGML_CUDA_MOE_HYBRID_RESIDENT ? &packet.ticket->resident_lanes : &packet.ticket->transfer_lanes;
        auto * status = reinterpret_cast<uint32_t *>(data() + l.status_offset);
        for (auto * node : body.nodes) {
            if (node->op == GGML_OP_VIEW) { continue; }
            trace_node(kind == GGML_CUDA_MOE_HYBRID_RESIDENT ? "resident" : "transfer", index, node, l.rows.geometry.route_capacity);
            if (node->op == GGML_OP_MUL_MAT_ID) {
                if (cudaMemsetAsync(node->data, 0, ggml_nbytes(node), stream) != cudaSuccess ||
                        !ggml_cuda_mul_mat_vec_q_bounded(*compute, node->src[0], node->src[1], node->src[2], node, count, status)) { return false; }
            } else if (node->op == GGML_OP_GLU) {
                const auto op = ggml_get_glu_op(node);
                if (op == GGML_GLU_OP_SWIGLU) { ggml_cuda_op_swiglu(*compute, node); }
                else if (op == GGML_GLU_OP_GEGLU) { ggml_cuda_op_geglu(*compute, node); }
                else { return false; }
            } else { return false; }
        }
        return ggml_cuda_moe_hybrid_rows_gpu_complete(l.rows, packet, kind,
            static_cast<float *>(body.nodes.back()->data),
            reinterpret_cast<float *>(data() + l.rows_offset + l.rows.gpu_output_offset), stream);
    }

    bool emit_reference_start(uint32_t index) {
        auto & l = compiled.layers[index];
        const auto packet = rows(index);
        const auto cpu = rows(index, true);
        const auto mapped_packet = ggml_cuda_moe_hybrid_rows_packet(alias + l.host_rows_offset + l.rows.host_control_offset, l.rows.geometry);
        auto * control = reference(index, true);
        auto * private_control = reference_device(index);
        auto * mapped_bell = reinterpret_cast<doorbell *>(alias + l.host_bell_offset);
        const auto * input = static_cast<const float *>(compiled.find(l.descriptor.activation)->data);
        const auto * ids = static_cast<const int32_t *>(compiled.find(l.descriptor.ids)->data);
        if (cudaMemcpyAsync(packet.runtime, cpu.runtime, sizeof(*cpu.runtime), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                cudaMemsetAsync(packet.producers, 0, (l.rows.geometry.row_capacity + 2) * sizeof(*packet.producers), stream) != cudaSuccess ||
                cudaMemsetAsync(data() + l.status_offset, 0, sizeof(uint32_t), stream) != cudaSuccess) { return false; }
        const size_t values = l.rows.host_input_bytes / sizeof(float);
        reference_input<<<unsigned(std::min<size_t>(65535, (values + 255) / 256)), 256, 0, stream>>>(input,
            reinterpret_cast<float *>(alias + l.host_input_offset), values, runtime());
        reference_publish<<<1, 256, 0, stream>>>(l.rows, ids, l.owner_view, control, mapped_bell, runtime());
        if (!ggml_moe_reference_gpu_quantize(input, data() + compiled.source_quant_offset,
                size_t(l.rows.geometry.row_capacity) * l.native.n_embd, stream)) { return false; }
        reference_wait_plan<<<1, 1, 0, stream>>>(control, mapped_bell, runtime());
        const size_t plan_bytes = reinterpret_cast<uint8_t *>(cpu.producers) - reinterpret_cast<uint8_t *>(cpu.ticket);
        reference_import<<<1, 256, 0, stream>>>(l.rows, packet, reinterpret_cast<const uint8_t *>(mapped_packet.ticket), plan_bytes,
            control, private_control, reference_control_bytes(l.rows.geometry.route_capacity), mapped_bell, runtime());
        if (!context->moe_grouped_context->import_hybrid_rows_plan(*l.group, l.rows, row_plan(l), packet, ids, &runtime()->failed)) { return false; }
        bool eligible = true;
        for (uint32_t source = 0; source < l.sources.size(); ++source) {
            const auto & bank = l.source_banks[source];
            eligible &= bank.source_device_alias && reinterpret_cast<uintptr_t>(bank.source_device_alias) % 16 == 0 &&
                bank.expert_stride % 16 == 0 && l.sources[source].expert_bytes % 16 == 0;
        }
        reference_role_binding bindings[3];
        for (uint32_t role = 0; role < 3; ++role) {
            const auto binding = l.roles[role];
            const uint32_t source = binding.source;
            bindings[role] = {reinterpret_cast<uint64_t>(l.group->bank_data[l.bank_indices[source]]),
                reinterpret_cast<uint64_t>(data() + compiled.selected_offset + l.selected_offsets[source]),
                size_t(l.source_banks[source].expert_stride), l.sources[source].expert_bytes, binding.offset};
        }
        reference_validated<<<1, 1, 0, stream>>>(l.rows, packet, control, private_control, bindings[0], bindings[1], bindings[2], eligible, runtime());
        for (uint32_t kind = 0; kind < 2; ++kind) {
            if (kind) {
                reference_wait_sources<<<1, 1, 0, stream>>>(control, private_control, packet, mapped_bell, runtime());
                for (uint32_t source = 0; source < l.sources.size(); ++source) {
                    if (!l.source_banks[source].source_device_alias) { continue; }
                    const auto bytes = l.sources[source].expert_bytes;
                    reference_copy<<<48 * 8, 256, 0, stream>>>(packet,
                        ggml_cuda_moe_hybrid_rows_transfer_experts(row_plan(l), l.rows.plan_capacity),
                        static_cast<const uint8_t *>(l.source_banks[source].source_device_alias), l.source_banks[source].expert_stride, bytes,
                        data() + compiled.selected_offset + l.selected_offsets[source], private_control, runtime());
                }
            }
            const auto view = reference_group(private_control, l.rows.geometry.route_capacity, kind);
            const ggml_moe_reference_gpu_group group{view.gate, view.up, view.down, view.starts, view.count, view.tokens, view.destinations};
            if (!ggml_moe_reference_gpu_execute(l.native, group, l.rows.geometry.weight_capacity, l.rows.geometry.route_capacity,
                    data() + compiled.source_quant_offset, data() + compiled.native_scratch_offset,
                    reinterpret_cast<float *>(data() + l.rows_offset + l.rows.gpu_output_offset), stream)) { return false; }
            reference_complete<<<1, 1, 0, stream>>>(packet, kind);
        }
        wait_cpu<<<1, 1, 0, stream>>>(mapped_bell, runtime());
        return cudaGetLastError() == cudaSuccess;
    }

    bool emit_start(uint32_t index) {
        if (compiled.reference) { return emit_reference_start(index); }
        auto & l = compiled.layers[index];
        auto & capture = captures[index];
        const auto packet = rows(index);
        const auto cpu = rows(index, true);
        auto * packed = reinterpret_cast<float *>(data() + l.rows_offset + l.rows.packed_input_offset);
        if (cudaMemcpyAsync(packet.runtime, cpu.runtime, sizeof(*cpu.runtime), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                cudaMemsetAsync(packet.producers, 0, (l.rows.geometry.row_capacity + 2) * sizeof(*packet.producers), stream) != cudaSuccess ||
                cudaMemsetAsync(data() + l.status_offset, 0, sizeof(uint32_t), stream) != cudaSuccess ||
                !context->moe_grouped_context->plan_hybrid_rows(*l.group, l.rows, row_plan(l), packet, &runtime()->failed,
                    static_cast<const int32_t *>(compiled.find(l.descriptor.ids)->data)) ||
                !ggml_cuda_moe_hybrid_rows_pack(l.rows, packet, static_cast<const float *>(compiled.find(l.descriptor.activation)->data), packed, stream) ||
                cudaEventRecord(capture.plan, stream) != cudaSuccess || cudaStreamWaitEvent(io, capture.plan, 0) != cudaSuccess ||
                cudaMemcpyAsync(cpu.ticket, packet.ticket, l.rows.control_bytes, cudaMemcpyDeviceToHost, io) != cudaSuccess ||
                cudaMemcpyAsync(host + l.host_input_offset, compiled.find(l.descriptor.activation)->data,
                    l.rows.host_input_bytes, cudaMemcpyDeviceToHost, io) != cudaSuccess ||
                cudaLaunchHostFunc(io, ready_callback, &capture) != cudaSuccess || !emit_body(l, l.resident, GGML_CUDA_MOE_HYBRID_RESIDENT)) { return false; }
        for (auto & tile : capture.tiles) {
            const auto & source = l.sources[tile.source_index];
            auto * output = l.transfer.find(l.descriptor.query->sources[tile.source_index].tensor);
            if (!output || cudaLaunchHostFunc(io, fill_tile, &tile) != cudaSuccess ||
                    cudaMemcpyAsync(static_cast<uint8_t *>(output->data) + size_t(tile.transfer_index) * source.expert_bytes,
                        host + l.host_rows_offset + l.rows.host_staging_offsets[tile.transfer_index % 2], source.expert_bytes,
                        cudaMemcpyHostToDevice, io) != cudaSuccess) { return false; }
        }
        if (cudaEventRecord(capture.transfer, io) != cudaSuccess || cudaStreamWaitEvent(stream, capture.transfer, 0) != cudaSuccess ||
                !emit_body(l, l.transfer, GGML_CUDA_MOE_HYBRID_TRANSFER)) { return false; }
        if (arm == GGML_CUDA_MOE_FIDELITY_POLL) {
            auto * mapped = reinterpret_cast<doorbell *>(alias + l.host_bell_offset);
            wait_cpu<<<1, 1, 0, stream>>>(mapped, runtime());
        }
        return cudaGetLastError() == cudaSuccess;
    }

    bool emit_join(uint32_t index) {
        auto & l = compiled.layers[index];
        const auto packet = rows(index);
        const auto cpu = rows(index, true);
        producer_status<<<1, 1, 0, stream>>>(packet, runtime(), reinterpret_cast<uint32_t *>(data() + l.status_offset));
        const auto * mapped = arm == GGML_CUDA_MOE_FIDELITY_POLL ? reinterpret_cast<doorbell *>(alias + l.host_bell_offset) : nullptr;
        accept_cpu<<<1, 1, 0, stream>>>(packet, runtime(), mapped, join(index));
        if (mapped) {
            const auto mapped_rows = ggml_cuda_moe_hybrid_rows_packet(alias + l.host_rows_offset + l.rows.host_control_offset, l.rows.geometry);
            const size_t values = l.rows.private_output_bytes / sizeof(float);
            const uint32_t blocks = uint32_t(std::min<size_t>(65535, (values + 255) / 256));
            import_cpu<<<blocks, 256, 0, stream>>>(l.rows, packet, mapped_rows.producers + 2,
                reinterpret_cast<const float *>(alias + l.host_output_offset),
                reinterpret_cast<float *>(data() + l.rows_offset + l.rows.cpu_output_offset), join(index));
        } else if (cudaMemcpyAsync(packet.producers + 2, cpu.producers + 2,
                l.rows.geometry.row_capacity * sizeof(*packet.producers), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                cudaMemcpyAsync(data() + l.rows_offset + l.rows.cpu_output_offset, host + l.host_output_offset,
                    l.rows.private_output_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) { return false; }
        if (!ggml_cuda_moe_hybrid_rows_commit(l.rows, row_plan(l), packet,
                    reinterpret_cast<float *>(data() + l.rows_offset + l.rows.gpu_output_offset),
                    reinterpret_cast<float *>(data() + l.rows_offset + l.rows.cpu_output_offset),
                    static_cast<float *>(compiled.find(l.descriptor.output)->data), stream)) { return false; }
        layer_terminal<<<1, 1, 0, stream>>>(packet, runtime(), join(index));
        if (compiled.reference && cudaMemcpyAsync(&reference(index)->executed_bytes, &reference_device(index)->executed_bytes,
                sizeof(uint64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess) { return false; }
        return cudaMemcpyAsync(completion(index).ticket, packet.ticket, l.rows.control_bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaMemcpyAsync(join(index, true), join(index), sizeof(join_record), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    }

#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
    bool capture_status(cudaError_t status, const char * phase, uint32_t first, uint32_t end) const {
        if (status == cudaSuccess) { return true; }
        fprintf(stderr, "moe-fidelity-capture-failure: graph_uid=%llu arm=%u phase=%s first=%u end=%u status=%d error=%s\n",
            (unsigned long long) compiled.query.graph->uid, arm, phase, first, end, int(status), cudaGetErrorString(status));
        fflush(stderr);
        return false;
    }

    template<typename F>
    bool capture_into(cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies,
            const char * phase, uint32_t first, uint32_t end, F emit) {
        const auto status = ggml_cuda_moe_fidelity_capture_to_graph(graph, stream, dependencies, [](void * opaque) {
            return (*static_cast<F *>(opaque))() ? cudaSuccess : cudaErrorNotSupported;
        }, &emit);
        return capture_status(status, phase, first, end);
    }

    bool capture_layer(cudaGraph_t graph, std::vector<cudaGraphNode_t> & dependencies, uint32_t index, bool start) {
        const auto first = compiled.layers[index].end;
        const auto end = index + 1 == compiled.layers.size() ? uint32_t(compiled.nodes.size()) : compiled.layers[index + 1].first;
        cudaGraphConditionalHandle handle;
        if (!capture_status(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault),
                    "conditional_handle", first, end) ||
                !capture_into(graph, dependencies, "producer_join", compiled.layers[index].first, first, [&] {
                    if ((start && !emit_start(index)) || !emit_join(index)) { return false; }
                    continue_window<<<1, 1, 0, stream>>>(runtime(), handle);
                    return cudaGetLastError() == cudaSuccess;
                })) { return false; }
        uint32_t cursor = first;
        ggml_tensor * staged = nullptr;
        for (;;) {
            uint32_t next = cursor;
            while (next < end && (!(compiled.nodes[next]->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                    !ggml_cuda_staged_input_supports(compiled.nodes[next]))) { ++next; }
            cudaGraphNode_t conditional = nullptr;
            cudaGraph_t body = nullptr;
            if (!capture_status(ggml_cuda_moe_fidelity_add_if(graph, handle, dependencies, conditional, body),
                    "conditional_node", cursor, next)) { return false; }
            std::vector<cudaGraphNode_t> body_dependencies;
            if (!capture_into(body, body_dependencies, "ordinary_body", cursor, next, [&] {
                    if (staged && cudaMemcpyAsync(staged->data, data() + compiled.staged_input_offset,
                            ggml_nbytes(staged), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) { return false; }
                    return emit_range(cursor, next);
                })) { return false; }
            if (!ggml_cuda_moe_fidelity_body_supported(body)) {
                fprintf(stderr, "moe-fidelity-capture-failure: graph_uid=%llu arm=%u phase=body_validation first=%u end=%u\n",
                    (unsigned long long) compiled.query.graph->uid, arm, cursor, next);
                if (const char * path = getenv("GGML_MOE_HYBRID_REJECTED_GRAPH")) {
                    const auto dumped = cudaGraphDebugDotPrint(body, path, cudaGraphDebugDotFlagsVerbose);
                    fprintf(stderr, "moe-fidelity: rejected body graph path=%s status=%s\n", path, cudaGetErrorString(dumped));
                }
                fflush(stderr);
                return false;
            }
            dependencies.assign(1, conditional);
            if (next == end) { return true; }
            staged = compiled.nodes[next];
            if (!staged->data || !ggml_nbytes(staged) || ggml_nbytes(staged) > compiled.staged_input_bytes ||
                    !capture_status(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault),
                        "input_handle", next, next + 1)) { return false; }
            if (!capture_into(graph, dependencies, "input_transport", next, next + 1, [&] {
                    auto transport = *staged;
                    transport.data = data() + compiled.staged_input_offset;
                    transport.buffer = arena;
                    trace_node("staged_input", next, &transport);
                    if (!ggml_cuda_moe_fidelity_emit_node(*compute, compiled.query.graph, compiled.query.graph->uid, next, &transport)) { return false; }
                    continue_window<<<1, 1, 0, stream>>>(runtime(), handle);
                    return cudaGetLastError() == cudaSuccess;
                })) { return false; }
            cursor = next + 1;
        }
    }
#endif

    bool capture_graphs() {
#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
        struct capture_guard {
            bool acquired = ggml_cuda_moe_fidelity_capture_enter();
            ~capture_guard() { release(); }
            void release() {
                if (acquired) {
                    ggml_cuda_moe_fidelity_capture_leave();
                    acquired = false;
                }
            }
        };
        const uint32_t count = arm == GGML_CUDA_MOE_FIDELITY_POLL ? 1 : compiled.layers.size() + 1;
        graphs.resize(count, nullptr);
        executables.resize(count, nullptr);
        for (uint32_t segment = 0; segment < count; ++segment) {
            if (trace_emission) {
                fprintf(stderr, "moe-fidelity: capture segment=%u/%u arm=%u stream=%p io=%p\n",
                    segment, count, arm, static_cast<void *>(stream), static_cast<void *>(io));
                fflush(stderr);
            }
            capture_guard guard;
            if (!guard.acquired || !capture_status(cudaGraphCreate(&graphs[segment], 0), "graph_create", segment, count)) { return false; }
            const auto graph = graphs[segment];
            std::vector<cudaGraphNode_t> dependencies;
            try {
                if (segment == 0 && !capture_into(graph, dependencies, "prefix", 0, compiled.layers[0].first, [&] {
                        return cudaMemcpyAsync(runtime(), host_runtime(), sizeof(window_runtime), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                            emit_range(0, compiled.layers[0].first);
                    })) { return false; }
                if (arm == GGML_CUDA_MOE_FIDELITY_POLL) {
                    for (uint32_t i = 0; i < compiled.layers.size(); ++i) {
                        if (!capture_layer(graph, dependencies, i, true)) { return false; }
                    }
                } else {
                    if (segment && !capture_layer(graph, dependencies, segment - 1, false)) { return false; }
                    if (segment < compiled.layers.size() && !capture_into(graph, dependencies, "producer_start",
                            compiled.layers[segment].first, compiled.layers[segment].end, [&] { return emit_start(segment); })) { return false; }
                }
                if ((arm == GGML_CUDA_MOE_FIDELITY_POLL || segment == compiled.layers.size()) &&
                        !capture_into(graph, dependencies, "window_result", compiled.nodes.size(), compiled.nodes.size(), [&] {
                            return cudaMemcpyAsync(host_result(), runtime(), sizeof(window_runtime), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
                        })) { return false; }
            } catch (...) {
                fprintf(stderr, "moe-fidelity-capture-failure: graph_uid=%llu arm=%u phase=emission_exception segment=%u\n",
                    (unsigned long long) compiled.query.graph->uid, arm, segment);
                fflush(stderr);
                (void) cudaGetLastError();
                return false;
            }
            guard.release();
            if (!capture_status(cudaGraphInstantiate(&executables[segment], graph, nullptr, nullptr, 0),
                    "instantiate", segment, count)) { return false; }
            if (caller_schedule && (!capture_status(cudaGraphUpload(executables[segment], stream), "upload", segment, count) ||
                    !wait_stream(stream, now_ns() + 5'000'000'000ull))) { return false; }
            ++compiled.state.captures;
        }
        compiled.state.segments = count;
        if (compiled.source_ordinary) {
            fprintf(stderr, "moe-fidelity: ordinary=source-weighted-rms uid=%llu captured_launches=%llu\n",
                (unsigned long long) compiled.query.graph->uid, (unsigned long long) source_norm_capture_launches);
        }
        return true;
#else
        return false;
#endif
    }

    bool probe_protocol() {
        if (arm != GGML_CUDA_MOE_FIDELITY_POLL) { return true; }
        auto * b = bell(0);
        b->epoch = 0;
        b->status = 0;
        b->ready.store(0);
        b->done.store(0);
        b->wake.store(0);
        b->stop.store(0);
        if (!ggml_cuda_moe_fidelity_capture_enter()) { return false; }
        auto status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (status == cudaSuccess) {
            probe_mapped_protocol<<<1, 1, 0, stream>>>(reinterpret_cast<doorbell *>(alias + compiled.layers[0].host_bell_offset), compiled.state.identity);
            const auto launched = cudaGetLastError();
            status = cudaStreamEndCapture(stream, &probe_graph);
            if (launched != cudaSuccess) { status = launched; }
        }
        ggml_cuda_moe_fidelity_capture_leave();
        if (status != cudaSuccess || cudaGraphInstantiate(&probe_executable, probe_graph, nullptr, nullptr, 0) != cudaSuccess ||
                cudaGraphLaunch(probe_executable, stream) != cudaSuccess) { return false; }
        ++compiled.state.protocol_probes;
        const uint64_t expiry = now_ns() + 5'000'000'000ull;
        status = cudaStreamQuery(stream);
        if (status != cudaSuccess && status != cudaErrorNotReady) { return false; }
        const auto wait = [&](const system_flag & flag) {
            while (!flag.load()) {
                if (now_ns() >= expiry) { return false; }
                std::this_thread::yield();
            }
            return true;
        };
        if (!wait(b->ready)) { return false; }
        b->epoch = compiled.state.identity;
        b->status = 0;
        b->done.store(1);
        if (!wait(b->wake) || b->epoch != ~compiled.state.identity || b->status != 0 || !wait_stream(stream, expiry)) { return false; }
        if (cudaGraphExecDestroy(probe_executable) != cudaSuccess) { return false; }
        probe_executable = nullptr;
        if (cudaGraphDestroy(probe_graph) != cudaSuccess) { return false; }
        probe_graph = nullptr;
        b->ready.store(0);
        b->done.store(0);
        b->wake.store(0);
        return true;
    }

    bool allocate() try {
        if (!std::atomic<uint32_t>::is_always_lock_free || sizeof(std::atomic<uint32_t>) != sizeof(uint32_t)) { return false; }
        if (trace_emission) {
            const auto & storage = compiled.state.storage;
            fprintf(stderr, "moe-fidelity: allocate device=%llu pinned=%llu cpu=%llu metadata=%llu clone=%llu planned_clone=%llu workspace=%llu cublas=%llu staged_input=%llu regions=%zu rows=%u\n",
                (unsigned long long) storage.device_bytes, (unsigned long long) storage.pinned_bytes,
                (unsigned long long) storage.cpu_bytes, (unsigned long long) storage.metadata_bytes,
                (unsigned long long) storage.device_clone_bytes, (unsigned long long) compiled.state.planned_clone_bytes,
                (unsigned long long) storage.device_workspace_bytes, (unsigned long long) storage.device_cublas_bytes,
                (unsigned long long) storage.device_staged_input_bytes,
                compiled.layers.size(), compiled.query.certificate.n_rows);
            fflush(stderr);
        }
        ggml_cuda_set_device(context->device);
        compute = std::make_unique<ggml_backend_cuda_context>(context->device);
        compute->borrowed_stream = context->stream();
        compute->streams[context->device][0] = context->stream();
        stream = compute->stream();
        if (cudaStreamCreateWithFlags(&io, cudaStreamNonBlocking) != cudaSuccess) { return false; }
        arena = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), compiled.arena_bytes);
        if (!arena || cudaHostAlloc(&host, compiled.pinned_bytes, cudaHostAllocMapped) != cudaSuccess) { return false; }
        memset(host, 0, compiled.pinned_bytes);
        for (uint32_t i = 0; i < compiled.layers.size(); ++i) { new (bell(i)) doorbell(); }
        if (!compiled.query.no_host_alias && cudaHostGetDevicePointer(&alias, host, 0) != cudaSuccess) {
            alias = nullptr;
            (void) cudaGetLastError();
        }
        if (arm == GGML_CUDA_MOE_FIDELITY_POLL && !alias) { return false; }
        compiled.state.mapped_alias = alias != nullptr;
        resources_live = true;
        if (cudaMemsetAsync(data(), 0, compiled.arena_bytes, stream) != cudaSuccess) { return false; }
        compute->pools[context->device][0] = std::make_unique<bounded_pool>(data() + compiled.workspace_offset, compiled.workspace);
        if (!compiled.allocation.allocate()) { return false; }
        for (size_t i = 0; i < compiled.tensors.size(); ++i) {
            auto & t = compiled.tensors[i];
            const auto * storage = compiled.allocation.find(compiled.originals[i]);
            t.data = storage ? storage->data : nullptr;
            t.buffer = storage ? storage->buffer : nullptr;
            t.extra = storage ? storage->extra : nullptr;
        }
        size_t norms_bound = 0;
        for (auto & norm : compiled.ordinary_norms) {
            const auto status = ggml_cuda_moe_source_norm_bind(norm, context->device);
            if (status == GGML_CUDA_MOE_SOURCE_NORM_INVALID) { return false; }
            norms_bound += status == GGML_CUDA_MOE_SOURCE_NORM_READY;
            if (trace_emission) {
                fprintf(stderr, "moe-fidelity: ordinary-bind index=%u status=%d input_buffer=%p gamma_buffer=%p output_buffer=%p\n",
                    norm.index, int(status), static_cast<void *>(norm.input->buffer), static_cast<void *>(norm.gamma->buffer),
                    static_cast<void *>(norm.output->buffer));
            }
        }
        if (compiled.source_ordinary) {
            fprintf(stderr, "moe-fidelity: ordinary=source-weighted-rms uid=%llu candidates=%zu selected=%zu unsupported=%zu workspace_bytes=0\n",
                (unsigned long long) compiled.query.graph->uid, compiled.ordinary_norms.size(), norms_bound,
                compiled.ordinary_norms.size() - norms_bound);
        }
        if (!compiled.validate_matmul_resources(context->device) || !compiled.validate_attention_resources(*compute) ||
                (compiled.cublas_workspace && !ggml_cuda_moe_fidelity_prepare_cublas(*compute,
                    data() + compiled.cublas_workspace_offset, compiled.cublas_workspace))) { return false; }
        captures.resize(compiled.layers.size());
        for (uint32_t i = 0; i < compiled.layers.size(); ++i) {
            auto & l = compiled.layers[i];
            if (!compiled.reference) { l.resident.bind(arena); l.transfer.bind(arena); }
            auto & capture = captures[i];
            capture.session = this;
            capture.index = i;
            if (cudaEventCreateWithFlags(&capture.plan, cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&capture.transfer, cudaEventDisableTiming) != cudaSuccess) { return false; }
            if (!compiled.reference) {
                capture.tiles.reserve(l.sources.size() * l.rows.transfer_capacity);
                for (uint32_t bank = 0; bank < l.sources.size(); ++bank) {
                    for (uint32_t transfer = 0; transfer < l.rows.transfer_capacity; ++transfer) {
                        capture.tiles.push_back({this, i, bank, transfer});
                    }
                }
            }
        }
        if (!prepare_owner()) { return false; }
        if (!begin_dispatch()) { return false; }
        if (cudaStreamSynchronize(stream) != cudaSuccess) { return false; }
        if (!probe_protocol()) {
            fprintf(stderr, "moe-fidelity: mapped acquire/release protocol unavailable\n");
            return false;
        }
        if (!capture_graphs()) { return false; }
        if (!finish_dispatch()) { return false; }
        resources_live = false;
        workspace_generation = context->workspace_generation;
        try {
            if (!caller_schedule) { worker = std::thread([this] { worker_main(); }); }
            supervisor = std::thread([this] { supervisor_main(); });
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
                condition.notify_all();
            }
            if (supervisor.joinable()) { supervisor.join(); }
            if (worker.joinable()) { worker.join(); }
            return false;
        }
        if (getenv("GGML_MOE_HYBRID_LOG")) {
            fprintf(stderr, "moe-fidelity-caller: caller_schedule=%u coordinator=%u supervisor=%u graph_uploads=%llu\n",
                unsigned(caller_schedule), unsigned(worker.joinable()), unsigned(supervisor.joinable()),
                (unsigned long long) (caller_schedule ? compiled.state.captures : 0));
        }
        return true;
    } catch (...) { return false; }

    bool wait_stream(cudaStream_t target, uint64_t expiry) {
        for (;;) {
            const auto status = cudaStreamQuery(target);
            if (status == cudaSuccess) { return true; }
            if (status != cudaErrorNotReady || now_ns() >= expiry) { return false; }
            std::this_thread::yield();
        }
    }

    bool drain(bool internal_caller = false) {
        const uint64_t expiry = now_ns() + 5'000'000'000ull;
        struct drain_guard {
            window_session * session = nullptr;
            ~drain_guard() {
                if (!session) { return; }
                std::lock_guard<std::mutex> lock(session->mutex);
                --session->external_drains;
                session->condition.notify_all();
            }
        } guard;
        if (caller_schedule) {
            std::unique_lock<std::mutex> lock(mutex);
            const bool owns_caller = caller_active && caller_id == std::this_thread::get_id();
            if (internal_caller != owns_caller) { return false; }
            if (!internal_caller) {
                ++external_drains;
                guard.session = this;
                if (!condition.wait_until(lock, std::chrono::steady_clock::time_point(std::chrono::nanoseconds(expiry)),
                        [&] { return !caller_active; })) { return false; }
            }
        }
        if (resources_live && ((!wait_stream(stream, expiry)) || !wait_stream(io, expiry))) { return false; }
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!condition.wait_until(lock, std::chrono::steady_clock::time_point(std::chrono::nanoseconds(expiry)),
                    [&] { return !armed && !working; })) { return false; }
        }
        if (cpu_service) {
            ggml_backend_moe_cpu_service_state_v1 state = {};
            state.struct_size = sizeof(state);
            if (compiled.query.cpu_api->state(cpu_service, &state) != 0 || state.active_jobs != 0 ||
                    compiled.query.cpu_api->drain(cpu_service) != 0) { return false; }
        }
        resources_live = false;
        { std::lock_guard<std::mutex> lock(mutex); ++compiled.state.finite_drains; }
        return true;
    }

    void poison() {
        std::lock_guard<std::mutex> lock(mutex);
        closed = true;
        request_stop();
    }

    int32_t replay(const ggml_cuda_moe_fidelity_window_replay_v1 & input, ggml_cuda_moe_fidelity_sample_v1 & sample) {
        sample = {};
        struct caller_guard {
            window_session * session = nullptr;
            ~caller_guard() {
                if (!session) { return; }
                std::lock_guard<std::mutex> lock(session->mutex);
                session->caller_active = false;
                session->caller_id = {};
                session->condition.notify_all();
            }
        } guard;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const char * invalid = closed ? "closed" : caller_active ? "caller_active" : external_drains ? "external_drain" : resources_live ? "resources_live" :
                input.epoch <= last_epoch || !input.epoch ? "epoch" :
                input.identity != compiled.state.identity ? "identity" :
                input.active_rows > compiled.row_capacity ? "row_capacity" :
                input.fault > (compiled.reference ? GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_REFERENCE_BEFORE_B : GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TEARDOWN) ? "fault" :
                context->workspace_generation != workspace_generation ? "workspace_generation" :
                (compiled.query.certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ?
                    input.active_rows != 1 : input.active_rows <= compiled.query.certificate.n_sequences) ? "row_contract" :
                memcmp(&compiled.query.graph->execution_certificate, &compiled.query.certificate, sizeof(compiled.query.certificate)) ? "certificate" : nullptr;
            if (invalid) {
                fprintf(stderr, "moe-fidelity-replay-reject: reason=%s graph_uid=%llu epoch=%llu last_epoch=%llu identity=%llu expected_identity=%llu"
                    " rows=%u capacity=%u row_semantics=%u sequences=%u fault=%u workspace_generation=%llu expected_workspace_generation=%llu\n",
                    invalid, (unsigned long long) compiled.query.graph->uid, (unsigned long long) input.epoch, (unsigned long long) last_epoch,
                    (unsigned long long) input.identity, (unsigned long long) compiled.state.identity,
                    input.active_rows, compiled.row_capacity, compiled.query.certificate.row_semantics, compiled.query.certificate.n_sequences, input.fault,
                    (unsigned long long) context->workspace_generation, (unsigned long long) workspace_generation);
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
            if (caller_schedule) {
                caller_active = true;
                caller_id = std::this_thread::get_id();
                guard.session = this;
            }
            last_epoch = input.epoch;
            canceled.store(false);
            cpu_ns = 0;
            *host_runtime() = {input.epoch, input.identity, input.active_rows, input.fault, 0, window_open};
            *host_result() = {};
            for (uint32_t i = 0; i < compiled.layers.size(); ++i) {
                auto & l = compiled.layers[i];
                auto * b = bell(i);
                b->ready.store(0);
                b->done.store(0);
                b->wake.store(0);
                b->stop.store(0);
                b->claimed.store(0);
                b->epoch = input.epoch;
                b->status = 0;
                if (compiled.reference) {
                    auto * control = reference(i);
                    control->plan_ready.store(0); control->transfer_ready.store(0); control->test_validated.store(0);
                    control->test_transfer_groups = 0;
                    control->groups[0] = control->groups[1] = 0; control->invalid = 0; control->selected_bytes = control->executed_bytes = 0;
                }
                const auto cpu = rows(i, true);
                std::fill_n(reinterpret_cast<uint8_t *>(cpu.ticket), l.rows.control_bytes, uint8_t{0});
                std::fill_n(reinterpret_cast<uint8_t *>(completion(i).ticket), l.rows.control_bytes, uint8_t{0});
                *join(i, true) = {};
                *cpu.runtime = {input.epoch, l.rows.identity, 0, input.active_rows, compiled.query.certificate.n_sequences,
                    compiled.reference ? l.rows.transfer_capacity : std::min(compiled.query.gpu_miss_quota, l.rows.transfer_capacity),
                    input.fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_STALE_PACKET ? 1u : 0u};
                memset(host + l.host_output_offset, 0, l.rows.private_output_bytes);
            }
        }
        ggml_cuda_set_device(context->device);
        if (!begin_dispatch()) { canceled.store(true); (void) finish_dispatch(); return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
        const auto start = now_ns();
        {
            std::lock_guard<std::mutex> lock(mutex);
            deadline = start + 2'000'000'000ull;
            armed = true;
            if (caller_schedule) { working = true; }
            condition.notify_all();
        }
        resources_live = true;
        bool ok = true;
        if (caller_schedule) {
            const ggml_moe_caller_schedule schedule{this, uint32_t(compiled.layers.size()),
                arm == GGML_CUDA_MOE_FIDELITY_SEGMENTED, deadline,
                caller_launch, caller_ready, caller_canceled, caller_progress, caller_service};
            ggml_moe_caller_sample observed;
            const auto status = ggml_moe_caller_run(schedule, observed);
            ok = status == ggml_moe_caller_status::complete;
            sample.launch_ns += observed.launch_ns;
            sample.observation_ns += observed.observation_ns;
            std::lock_guard<std::mutex> lock(mutex);
            ++caller_replays;
            caller_jobs += observed.jobs;
            caller_progress_probes += observed.progress_probes;
            if (!ok) {
                fprintf(stderr, "moe-fidelity-caller-failure: epoch=%llu status=%u jobs=%llu probes=%llu\n",
                    (unsigned long long) input.epoch, unsigned(status),
                    (unsigned long long) observed.jobs, (unsigned long long) observed.progress_probes);
                request_stop();
            }
            working = false;
            armed = false;
            condition.notify_all();
        } else for (uint32_t segment = 0; segment < executables.size(); ++segment) {
            if (segment) {
                const auto before = now_ns();
                std::unique_lock<std::mutex> lock(mutex);
                ok = condition.wait_until(lock, std::chrono::steady_clock::time_point(std::chrono::nanoseconds(deadline)),
                    [&] { return canceled.load() || bell(segment - 1)->done.load(); });
                sample.continuation_ns += now_ns() - before;
                if (!ok || canceled.load() || !bell(segment - 1)->done.load() || bell(segment - 1)->status != 0) { ok = false; break; }
            }
            const auto before = now_ns();
            ok = cudaGraphLaunch(executables[segment], stream) == cudaSuccess;
            sample.launch_ns += now_ns() - before;
            { std::lock_guard<std::mutex> lock(mutex); ++compiled.state.launches; }
            if (!ok) { break; }
        }
        if (!ok || input.fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TEARDOWN) {
            std::lock_guard<std::mutex> lock(mutex);
            request_stop();
        }
        const auto observed = now_ns();
        const bool drained = drain(caller_schedule);
        sample.observation_ns += now_ns() - observed;
        sample.cpu_service_ns = drained ? cpu_ns : 0;
        if (!drained) { poison(); return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
        ok &= !canceled.load() && host_result()->epoch == input.epoch && !host_result()->failed;
        for (uint32_t i = 0; i < compiled.layers.size(); ++i) {
            const auto & l = compiled.layers[i];
            const auto packet = completion(i);
            const auto & ticket = *packet.ticket;
            std::lock_guard<std::mutex> lock(mutex);
            ok &= ticket.epoch == input.epoch && ticket.identity == l.rows.identity && ticket.accepted && !ticket.status;
            compiled.state.cpu_routes += ticket.cpu_routes;
            compiled.state.resident_routes += ticket.resident_lanes;
            compiled.state.transfer_routes += ticket.transfer_lanes;
            compiled.state.producers_expected += ticket.expected_producers;
            for (uint32_t producer = 0; producer < l.rows.geometry.row_capacity + 2; ++producer) {
                compiled.state.producers_completed += packet.expected_routes[producer] && packet.producers[producer].complete &&
                    packet.producers[producer].epoch == input.epoch;
            }
            for (const auto & source : l.sources) { compiled.state.transfer_bytes += source.expert_bytes * ticket.transfer_count; }
            if (compiled.reference) {
                compiled.state.executed_selected_bytes += reference(i)->executed_bytes;
                if (ticket.accepted) { ok &= reference(i)->executed_bytes == reference(i)->selected_bytes; }
            } else {
                compiled.state.planning_callbacks += 1;
                compiled.state.transfer_callbacks += l.sources.size() * l.rows.transfer_capacity;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (input.fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_BEFORE_COMMIT) { canceled.store(true); }
            ok &= !canceled.load();
            compiled.state.terminal = ok ? window_committing : window_canceled;
            compiled.state.published = 0;
            if (input.fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_CANCEL_AFTER_COMMIT && ok) { canceled.store(true); }
        }
        if (ok) {
            for (auto * output : compiled.public_outputs) {
                auto * private_output = compiled.find(output);
                ok = ok && private_output && output->data != private_output->data && cudaMemcpyAsync(output->data, private_output->data,
                    ggml_nbytes(output), cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
            }
            if (!ok || !wait_stream(stream, now_ns() + 5'000'000'000ull)) {
                resources_live = true;
                poison();
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
            }
            std::lock_guard<std::mutex> lock(mutex);
            compiled.state.terminal = window_committed;
            compiled.state.published = 1;
            ++compiled.state.accepted;
        } else {
            std::lock_guard<std::mutex> lock(mutex);
            ++compiled.state.rejected;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++compiled.state.epochs;
            if (input.fault == GGML_CUDA_MOE_FIDELITY_WINDOW_FAULT_TEARDOWN) { closed = true; }
        }
        if (!finish_dispatch()) { poison(); return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
        sample.whole_ns = now_ns() - start;
        return ok ? GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK : GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }

    bool close() {
        std::lock_guard<std::mutex> lock(mutex);
        closed = true;
        return request_stop();
    }

    bool release() noexcept try {
        const bool closed_ok = close();
        if (!drain()) { return false; }
        if (cpu_service && compiled.query.cpu_api->close(cpu_service) != 0) { return false; }
        const bool dispatch_ok = finish_dispatch();
        if (!closed_ok || !dispatch_ok) { return false; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            condition.notify_all();
        }
        if (getenv("GGML_MOE_HYBRID_LOG")) {
            fprintf(stderr, "moe-fidelity-caller-summary: caller_schedule=%u coordinator=%u replays=%llu jobs=%llu progress_probes=%llu\n",
                unsigned(caller_schedule), unsigned(worker.joinable()), (unsigned long long) caller_replays,
                (unsigned long long) caller_jobs, (unsigned long long) caller_progress_probes);
        }
        if (supervisor.joinable()) { supervisor.join(); }
        if (worker.joinable()) { worker.join(); }
        if (probe_executable) { if (cudaGraphExecDestroy(probe_executable) != cudaSuccess) { return false; } probe_executable = nullptr; }
        if (probe_graph) { if (cudaGraphDestroy(probe_graph) != cudaSuccess) { return false; } probe_graph = nullptr; }
        for (auto & graph : executables) { if (graph) { if (cudaGraphExecDestroy(graph) != cudaSuccess) { return false; } graph = nullptr; } }
        for (auto & graph : graphs) { if (graph) { if (cudaGraphDestroy(graph) != cudaSuccess) { return false; } graph = nullptr; } }
        if (compute && !ggml_cuda_moe_fidelity_release_cublas(*compute)) { return false; }
        for (auto & capture : captures) {
            if (capture.plan) { if (cudaEventDestroy(capture.plan) != cudaSuccess) { return false; } capture.plan = nullptr; }
            if (capture.transfer) { if (cudaEventDestroy(capture.transfer) != cudaSuccess) { return false; } capture.transfer = nullptr; }
        }
        if (io) { if (cudaStreamDestroy(io) != cudaSuccess) { return false; } io = nullptr; }
        if (host) { if (cudaFreeHost(host) != cudaSuccess) { return false; } host = alias = nullptr; }
        if (arena) { ggml_backend_buffer_free(arena); arena = nullptr; }
        compute.reset();
        compiled.allocation.reset();
        resource_leases.clear();
        if (cpu_service) {
            for (auto & l : compiled.layers) { if (l.cpu_region && compiled.query.cpu_api->destroy_region(cpu_service, &l.cpu_region) != 0) { return false; } }
            if (compiled.query.cpu_api->destroy(&cpu_service) != 0) { return false; }
        }
        if (cpu_module) { if (!ggml_backend_moe_module_release_v1(cpu_module)) { return false; } cpu_module = nullptr; }
        if (graph_retained) { if (compiled.query.release_graph(compiled.query.graph_owner) != 0) { return false; } graph_retained = false; }
        return true;
    } catch (...) { return false; }
};

static int32_t make_session(ggml_backend_t backend, const ggml_cuda_moe_fidelity_window_query_v1 * query,
        std::unique_ptr<window_session> & session) try {
    session = std::make_unique<window_session>();
    session->backend = backend;
    if (!backend || !query || query->struct_size != sizeof(*query) || !query->retain_graph || !query->release_graph) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    session->context = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!session->compiled.compile(backend, query)) {
        fprintf(stderr, "moe-fidelity: graph capability rejected\n");
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    if (query->retain_graph(query->graph_owner) != 0) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    session->graph_retained = true;
    session->compiled.state.storage.metadata_bytes += sizeof(window_session) - sizeof(compiled_window) +
        query->n_regions * sizeof(layer_capture) + (query->n_regions + 1) * (sizeof(cudaGraph_t) + sizeof(cudaGraphExec_t));
    for (const auto & l : session->compiled.layers) {
        session->compiled.state.storage.metadata_bytes += session->compiled.reference ? 0 : l.sources.size() * l.rows.transfer_capacity * sizeof(tile_job);
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t measure(ggml_backend_t backend, const ggml_cuda_moe_fidelity_window_query_v1 * query,
        ggml_cuda_moe_fidelity_window_state_v1 * state) try {
    if (!state) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    window_session session;
    session.backend = backend;
    if (!session.compiled.compile(backend, query)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    const int32_t status = session.measure_cpu(0);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
    session.compiled.state.storage.metadata_bytes += sizeof(window_session) - sizeof(compiled_window) +
        query->n_regions * sizeof(layer_capture) + (query->n_regions + 1) * (sizeof(cudaGraph_t) + sizeof(cudaGraphExec_t));
    for (const auto & l : session.compiled.layers) {
        session.compiled.state.storage.metadata_bytes += session.compiled.reference ? 0 : l.sources.size() * l.rows.transfer_capacity * sizeof(tile_job);
    }
    *state = session.compiled.state;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t prepare(ggml_backend_t backend, const ggml_cuda_moe_fidelity_window_query_v1 * query,
        const ggml_cuda_moe_fidelity_window_storage_v1 * limits, uint32_t arm, void ** output) try {
    if (!output) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    *output = nullptr;
    if (arm != GGML_CUDA_MOE_FIDELITY_SEGMENTED && arm != GGML_CUDA_MOE_FIDELITY_POLL) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    if (ggml_moe_fidelity_selection().reference && arm != GGML_CUDA_MOE_FIDELITY_POLL) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
#if !defined(USE_CUDA_GRAPH) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || CUDART_VERSION < 12030
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
#else
    if (arm == GGML_CUDA_MOE_FIDELITY_POLL && backend && ggml_backend_is_cuda(backend) &&
            ggml_cuda_info().devices[static_cast<ggml_backend_cuda_context *>(backend->context)->device].cc < 600) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
#endif
    std::unique_ptr<window_session> session;
    int32_t status = make_session(backend, query, session);
    if (!status && limits) {
        const auto required = session->compiled.state.storage;
        if ((limits->device_bytes && limits->device_bytes < required.device_bytes) ||
                (limits->pinned_bytes && limits->pinned_bytes < required.pinned_bytes) ||
                (limits->metadata_bytes && limits->metadata_bytes < required.metadata_bytes)) { status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    }
    if (!status) { status = session->measure_cpu(limits ? limits->cpu_bytes : 0); }
    if (!status) {
        status = session->prepare_cpu(limits ? limits->cpu_bytes : 0);
        if (status) { fprintf(stderr, "moe-fidelity: CPU service preparation rejected status=%d\n", status); }
    }
    if (!status) {
        session->arm = arm;
        if (!session->allocate()) { status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
    }
    if (status) {
        if (session && !session->release()) { *output = session.release(); }
        return status;
    }
    *output = session.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t replay(void * opaque, const ggml_cuda_moe_fidelity_window_replay_v1 * input,
        ggml_cuda_moe_fidelity_sample_v1 * sample) {
    if (!opaque || !input || !sample) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    return static_cast<window_session *>(opaque)->replay(*input, *sample);
}
static bool state(void * opaque, ggml_cuda_moe_fidelity_window_state_v1 * output) {
    if (!opaque || !output) { return false; }
    auto & session = *static_cast<window_session *>(opaque);
    std::lock_guard<std::mutex> lock(session.mutex);
    *output = session.compiled.state;
    output->active_jobs = session.working || session.caller_active;
    return true;
}
static int32_t set_cpu_test_hook(void * opaque, ggml_backend_moe_cpu_test_hook_v1_t hook, void * data) {
    if (!opaque) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    auto & session = *static_cast<window_session *>(opaque);
    std::lock_guard<std::mutex> lock(session.mutex);
    if (session.closed || session.armed || session.working || session.caller_active || session.external_drains || session.resources_live || !session.cpu_service) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    const auto status = session.compiled.query.cpu_api->set_test_hook(session.cpu_service, hook ? window_session::cpu_test_hook : nullptr, &session);
    if (!status) { session.test_hook = hook; session.test_hook_data = data; }
    return status;
}
static bool test_snapshot(void * opaque, ggml_cuda_moe_fidelity_window_test_snapshot_v1 * output) {
    if (!opaque || !output) { return false; }
    auto & session = *static_cast<window_session *>(opaque);
    std::lock_guard<std::mutex> lock(session.mutex);
    if (!session.test_hook_active || session.test_phase != GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT ||
            session.cpu_layer >= session.compiled.layers.size()) { return false; }
    const auto & l = session.compiled.layers[session.cpu_layer];
    const auto * b = session.bell(session.cpu_layer);
    *output = {};
    output->epoch = b->epoch;
    output->claimed = b->claimed.load();
    output->cpu_terminal = b->done.load();
    output->wake = b->wake.load();
    output->stopped = b->stop.load();
    output->input_hash = fingerprint(1469598103934665603ull, session.host + l.host_input_offset, l.rows.host_input_bytes);
    output->control_hash = fingerprint(1469598103934665603ull, session.rows(session.cpu_layer, true).ticket, l.rows.control_bytes);
    output->control_hash = fingerprint(output->control_hash, session.host_runtime(), sizeof(window_runtime));
    output->output_hash = fingerprint(1469598103934665603ull, session.host + l.host_output_offset, l.rows.private_output_bytes);
    output->host_address = reinterpret_cast<uintptr_t>(session.host);
    output->device_address = reinterpret_cast<uintptr_t>(session.data());
    const auto compute = cudaStreamQuery(session.stream);
    const auto io = cudaStreamQuery(session.io);
    if ((compute != cudaSuccess && compute != cudaErrorNotReady) || (io != cudaSuccess && io != cudaErrorNotReady)) { return false; }
    output->streams_complete = compute == cudaSuccess && io == cudaSuccess;
    if (output->streams_complete) {
        const auto & record = *session.join(session.cpu_layer, true);
        output->join_complete = record.complete;
        output->join_cpu_terminal = record.cpu_terminal;
        output->imported_bytes = record.imported_bytes;
    }
    return true;
}
static int32_t close(void * opaque) {
    return opaque && static_cast<window_session *>(opaque)->close() ? 0 : GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
}
static int32_t drain(void * opaque) {
    return opaque && static_cast<window_session *>(opaque)->drain() ? 0 : GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
}
static int32_t destroy_with_state(void ** opaque, ggml_cuda_moe_fidelity_window_state_v1 * final_state) {
    if (!opaque || !*opaque) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    auto * session = static_cast<window_session *>(*opaque);
    if (!session->release()) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
    if (final_state) { state(session, final_state); }
    delete session;
    *opaque = nullptr;
    return 0;
}
static int32_t destroy(void ** opaque) { return destroy_with_state(opaque, nullptr); }

}

const ggml_cuda_moe_fidelity_window_api_v1 * ggml_cuda_moe_fidelity_window_api() {
    static const ggml_cuda_moe_fidelity_window_api_v1 api{
        sizeof(api), 1, measure, prepare, replay, state, close, drain, destroy, set_cpu_test_hook, test_snapshot};
    return &api;
}

namespace {

struct fidelity_query_copy {
    ggml_backend_moe_cpu_region_query_v1 query = {};
    ggml_cgraph graph = {};
    std::vector<const ggml_tensor *> originals;
    std::vector<ggml_tensor> tensors;
    std::vector<ggml_tensor *> graph_nodes;
    std::vector<ggml_tensor *> graph_leafs;
    std::vector<const ggml_tensor *> body;
    std::vector<const ggml_tensor *> dynamic;
    std::vector<const ggml_tensor *> live_outputs;
    std::vector<ggml_backend_moe_cpu_region_source_v1> sources;

    bool copy(const ggml_backend_moe_cpu_region_query_v1 * input) {
        if (!input || input->struct_size != sizeof(*input) || !input->graph || !input->body_nodes ||
                !input->dynamic_inputs || !input->live_outputs || !input->sources) { return false; }
        const auto add = [&](const ggml_tensor * tensor) {
            if (tensor && std::find(originals.begin(), originals.end(), tensor) == originals.end()) { originals.push_back(tensor); }
        };
        for (int i = 0; i < input->graph->n_nodes; ++i) { add(input->graph->nodes[i]); }
        for (int i = 0; i < input->graph->n_leafs; ++i) { add(input->graph->leafs[i]); }
        for (uint32_t i = 0; i < input->n_body_nodes; ++i) { add(input->body_nodes[i]); }
        for (uint32_t i = 0; i < input->n_dynamic_inputs; ++i) { add(input->dynamic_inputs[i]); }
        for (uint32_t i = 0; i < input->n_live_outputs; ++i) { add(input->live_outputs[i]); }
        for (uint32_t i = 0; i < input->n_sources; ++i) { add(input->sources[i].tensor); }
        for (size_t i = 0; i < originals.size(); ++i) {
            for (const auto * source : originals[i]->src) { add(source); }
            add(originals[i]->view_src);
        }
        tensors.resize(originals.size());
        const auto resolve = [&](const ggml_tensor * tensor) -> ggml_tensor * {
            if (!tensor) { return nullptr; }
            const auto found = std::find(originals.begin(), originals.end(), tensor);
            return found == originals.end() ? nullptr : &tensors[found - originals.begin()];
        };
        for (size_t i = 0; i < originals.size(); ++i) {
            tensors[i] = *originals[i];
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                tensors[i].src[source] = resolve(originals[i]->src[source]);
                if (originals[i]->src[source] && !tensors[i].src[source]) { return false; }
            }
            tensors[i].view_src = resolve(originals[i]->view_src);
            if (originals[i]->view_src && !tensors[i].view_src) { return false; }
            tensors[i].buffer = nullptr;
            tensors[i].extra = nullptr;
        }
        graph_nodes.reserve(input->graph->n_nodes);
        graph_leafs.reserve(input->graph->n_leafs);
        for (int i = 0; i < input->graph->n_nodes; ++i) { graph_nodes.push_back(resolve(input->graph->nodes[i])); }
        for (int i = 0; i < input->graph->n_leafs; ++i) { graph_leafs.push_back(resolve(input->graph->leafs[i])); }
        graph = *input->graph;
        graph.nodes = graph_nodes.data();
        graph.leafs = graph_leafs.data();
        graph.grads = nullptr;
        graph.grad_accs = nullptr;
        graph.use_counts = nullptr;
        graph.visited_hash_set = {};
        body.reserve(input->n_body_nodes);
        dynamic.reserve(input->n_dynamic_inputs);
        live_outputs.reserve(input->n_live_outputs);
        sources.assign(input->sources, input->sources + input->n_sources);
        for (uint32_t i = 0; i < input->n_body_nodes; ++i) { body.push_back(resolve(input->body_nodes[i])); }
        for (uint32_t i = 0; i < input->n_dynamic_inputs; ++i) { dynamic.push_back(resolve(input->dynamic_inputs[i])); }
        for (uint32_t i = 0; i < input->n_live_outputs; ++i) { live_outputs.push_back(resolve(input->live_outputs[i])); }
        for (uint32_t i = 0; i < input->n_sources; ++i) {
            sources[i].tensor = resolve(input->sources[i].tensor);
            if (!sources[i].tensor) { return false; }
        }
        if (std::any_of(body.begin(), body.end(), [](const auto * tensor) { return tensor == nullptr; }) ||
                std::any_of(dynamic.begin(), dynamic.end(), [](const auto * tensor) { return tensor == nullptr; }) ||
                std::any_of(live_outputs.begin(), live_outputs.end(), [](const auto * tensor) { return tensor == nullptr; })) { return false; }
        query = *input;
        query.graph = &graph;
        query.body_nodes = body.data();
        query.dynamic_inputs = dynamic.data();
        query.live_outputs = live_outputs.data();
        query.sources = sources.data();
        query.activation = resolve(input->activation);
        query.ids = resolve(input->ids);
        return query.activation && query.ids;
    }
};

struct fidelity_hybrid_session;

struct fidelity_hybrid_region {
    fidelity_hybrid_session * session = nullptr;
    fidelity_hybrid_region * retired_next = nullptr;
    ggml_backend_moe_hybrid_region_v1 descriptor = {};
    std::vector<std::unique_ptr<fidelity_query_copy>> queries;
    std::vector<const ggml_backend_moe_cpu_region_query_v1 *> query_ptrs;
    std::vector<std::unique_ptr<fidelity_query_copy>> batch_queries;
    std::vector<const ggml_backend_moe_cpu_region_query_v1 *> batch_query_ptrs;
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    const ggml_backend_moe_cpu_fidelity_requirements_api_v1 * cpu_requirements_api = nullptr;
};

struct fidelity_hybrid_session {
    ggml_backend_t backend = nullptr;
    ggml_backend_moe_hybrid_config_v1 config = {};
    ggml_backend_moe_source_owner_v1 source_owner = {};
    void * window = nullptr;
    uint64_t epoch = 0;
    uint64_t graph_uid = 0;
    uint64_t window_uid = 0;
    uint64_t leases = 0;
    bool closed = false;
    fidelity_hybrid_region * retired_regions = nullptr;
    std::mutex lifecycle_mutex;
    std::condition_variable lifecycle_condition;
    uint32_t active_computes = 0;
    uint32_t active_states = 0;
    uint32_t region_count = 0;
    uint32_t window_rows = 0;
    uint32_t window_regions = 0;
    uint32_t window_arm = GGML_CUDA_MOE_FIDELITY_SEGMENTED;
    uint32_t requested_arm = GGML_CUDA_MOE_FIDELITY_SEGMENTED;
    bool quiescing = false;
    bool window_ready = false;
    bool log_summary = getenv("GGML_MOE_HYBRID_LOG") != nullptr;
    bool samples_overflow = false;
    uint64_t samples_valid = 0;
    uint64_t samples_incomplete = 0;
    ggml_cuda_moe_fidelity_sample_v1 samples = {};
    ggml_cuda_moe_fidelity_window_state_v1 last = {};
    std::vector<const ggml_backend_moe_hybrid_region_v1 *> descriptors;
    std::vector<ggml_tensor *> public_outputs;

    bool begin_compute() {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (closed || quiescing || active_computes != 0) { return false; }
        active_computes = 1;
        return true;
    }

    void finish_compute() {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        GGML_ASSERT(active_computes == 1);
        active_computes = 0;
        lifecycle_condition.notify_all();
    }

    bool available() {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        return !closed && !quiescing;
    }

    bool cancel_requested() {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        return closed || quiescing;
    }

    void record_sample(const ggml_cuda_moe_fidelity_sample_v1 & sample) {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (!sample.whole_ns) {
            if (samples_incomplete == UINT64_MAX) { samples_overflow = true; }
            else { ++samples_incomplete; }
            return;
        }
        const uint64_t values[] = {sample.whole_ns, sample.launch_ns, sample.observation_ns, sample.continuation_ns, sample.cpu_service_ns};
        uint64_t * totals[] = {&samples.whole_ns, &samples.launch_ns, &samples.observation_ns, &samples.continuation_ns, &samples.cpu_service_ns};
        if (samples_valid == UINT64_MAX) { samples_overflow = true; return; }
        for (size_t i = 0; i < 5; ++i) {
            if (values[i] > UINT64_MAX - *totals[i]) { samples_overflow = true; return; }
        }
        for (size_t i = 0; i < 5; ++i) { *totals[i] += values[i]; }
        ++samples_valid;
    }

    void report_summary(const ggml_cuda_moe_fidelity_window_state_v1 & state) const {
        if (!log_summary) { return; }
        const auto & storage = state.storage;
        fprintf(stderr, "moe-fidelity-summary: graph_uid=%llu identity=%llu arm=%u rows=%u regions=%u segments=%u cleanup=1"
            " requested_arm=%u mapped_alias=%u protocol_probes=%llu source_banks=%llu mapped_source_banks=%llu"
            " captures=%llu launches=%llu epochs=%llu accepted=%llu rejected=%llu"
            " pipeline=%s source_pool=%u pcie_num=%u executed_selected_bytes=%llu planning_callbacks=%llu transfer_callbacks=%llu cpu_pool_phases_per_call=%u cpu_max_group_entries=%llu cpu_calls=%llu cpu_routes=%llu resident_routes=%llu transfer_routes=%llu transfer_bytes=%llu"
            " producers_expected=%llu producers_completed=%llu admission_reserved=%llu admission_aborted=%llu admission_committed=%llu"
            " fallbacks=%llu retries=%llu finite_drains=%llu address_changes=%llu"
            " device_bytes=%llu pinned_bytes=%llu cpu_bytes=%llu metadata_bytes=%llu"
            " device_runtime_bytes=%llu device_rows_bytes=%llu device_resident_body_bytes=%llu device_transfer_body_bytes=%llu"
            " device_clone_bytes=%llu device_workspace_bytes=%llu device_cublas_bytes=%llu device_staged_input_bytes=%llu pinned_runtime_bytes=%llu pinned_rows_bytes=%llu"
            " whole_ns=%llu launch_ns=%llu observation_ns=%llu continuation_ns=%llu cpu_service_ns=%llu"
            " samples_valid=%llu samples_incomplete=%llu samples_overflow=%u\n",
            (unsigned long long) window_uid, (unsigned long long) state.identity, window_arm, window_rows, window_regions, state.segments,
            requested_arm, state.mapped_alias, (unsigned long long) state.protocol_probes,
            (unsigned long long) state.source_banks, (unsigned long long) state.mapped_source_banks,
            (unsigned long long) state.captures, (unsigned long long) state.launches, (unsigned long long) state.epochs,
            (unsigned long long) state.accepted, (unsigned long long) state.rejected,
            ggml_moe_fidelity_selection().reference ? "reference" : "control", unsigned(ggml_moe_fidelity_selection().source_pool), ggml_moe_fidelity_selection().pcie_num,
            (unsigned long long) state.executed_selected_bytes, (unsigned long long) state.planning_callbacks,
            (unsigned long long) state.transfer_callbacks, (ggml_moe_fidelity_selection().reference || ggml_moe_fidelity_selection().source_pool) ? 2u : 1u,
            (unsigned long long) state.cpu_max_group_entries,
            (unsigned long long) state.cpu_calls, (unsigned long long) state.cpu_routes, (unsigned long long) state.resident_routes,
            (unsigned long long) state.transfer_routes, (unsigned long long) state.transfer_bytes,
            (unsigned long long) state.producers_expected, (unsigned long long) state.producers_completed,
            (unsigned long long) state.admission_reserved, (unsigned long long) state.admission_aborted, (unsigned long long) state.admission_committed,
            (unsigned long long) state.fallbacks, (unsigned long long) state.retries, (unsigned long long) state.finite_drains, (unsigned long long) state.address_changes,
            (unsigned long long) storage.device_bytes, (unsigned long long) storage.pinned_bytes, (unsigned long long) storage.cpu_bytes, (unsigned long long) storage.metadata_bytes,
            (unsigned long long) storage.device_runtime_bytes, (unsigned long long) storage.device_rows_bytes,
            (unsigned long long) storage.device_resident_body_bytes, (unsigned long long) storage.device_transfer_body_bytes,
            (unsigned long long) storage.device_clone_bytes, (unsigned long long) storage.device_workspace_bytes, (unsigned long long) storage.device_cublas_bytes,
            (unsigned long long) storage.device_staged_input_bytes,
            (unsigned long long) storage.pinned_runtime_bytes, (unsigned long long) storage.pinned_rows_bytes,
            (unsigned long long) samples.whole_ns, (unsigned long long) samples.launch_ns, (unsigned long long) samples.observation_ns,
            (unsigned long long) samples.continuation_ns, (unsigned long long) samples.cpu_service_ns,
            (unsigned long long) samples_valid, (unsigned long long) samples_incomplete, unsigned(samples_overflow));
    }

    void release_retired_regions() {
        while (retired_regions) {
            auto * region = retired_regions;
            retired_regions = region->retired_next;
            delete region;
        }
    }

    bool reset_window() {
        if (!window) {
            release_retired_regions();
            return true;
        }
        auto * api = ggml_cuda_moe_fidelity_window_api();
        ggml_cuda_moe_fidelity_window_state_v1 final_state = {};
        const bool ok = api->close(window) == 0 && api->drain(window) == 0 && destroy_with_state(&window, &final_state) == 0;
        if (ok) {
            {
                std::lock_guard<std::mutex> lock(lifecycle_mutex);
                last = final_state;
                report_summary(final_state);
                graph_uid = window_uid = 0;
                window_rows = window_regions = 0;
                samples_valid = samples_incomplete = 0;
                samples_overflow = false;
                samples = {};
            }
            release_retired_regions();
        }
        return ok;
    }

    bool reset_serialized(bool reopen) {
        void * cancel_window = nullptr;
        {
            std::unique_lock<std::mutex> lock(lifecycle_mutex);
            lifecycle_condition.wait(lock, [&] { return !quiescing; });
            quiescing = true;
            if (window_ready && window && active_computes != 0) {
                cancel_window = window;
                ++active_states;
            }
            window_ready = false;
            if (!reopen) { closed = true; }
        }
        if (cancel_window) {
            (void) ggml_cuda_moe_fidelity_window_api()->close(cancel_window);
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            GGML_ASSERT(active_states != 0);
            --active_states;
            lifecycle_condition.notify_all();
        }
        {
            std::unique_lock<std::mutex> lock(lifecycle_mutex);
            lifecycle_condition.wait(lock, [&] { return active_computes == 0 && active_states == 0; });
        }
        const bool ok = reset_window();
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            if (!ok) { closed = true; }
            quiescing = false;
            lifecycle_condition.notify_all();
        }
        return ok;
    }
};

static int32_t fidelity_graph_retain(void * opaque) {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session) { return 1; }
    std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
    if (session->closed || session->leases == UINT64_MAX) { return 1; }
    ++session->leases;
    return 0;
}

static int32_t fidelity_graph_release(void * opaque) {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session) { return 1; }
    std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
    if (!session->leases) { return 1; }
    --session->leases;
    return 0;
}

static int32_t fidelity_hybrid_create(const ggml_backend_moe_hybrid_config_v1 * config, void ** output) try {
    if (!output) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    *output = nullptr;
    if (!config || config->struct_size != sizeof(*config) || !config->backend || !config->source_owner ||
            config->executor != GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY || config->admission_quota != 0 ||
            !config->n_threads || !config->max_regions) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    const auto & selection = ggml_moe_fidelity_selection();
    if (!selection.valid || (selection.reference && getenv("GGML_MOE_FIDELITY_NO_HOST_ALIAS"))) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    const char * requested = getenv("GGML_MOE_FIDELITY_ARM");
    if (selection.reference && requested && strcmp(requested, "poll")) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    auto session = std::make_unique<fidelity_hybrid_session>();
    session->backend = config->backend;
    session->config = *config;
    session->source_owner = *config->source_owner;
    session->config.source_owner = &session->source_owner;
    session->descriptors.reserve(config->max_regions);
    session->public_outputs.reserve(8);
    fprintf(stderr, "moe-fidelity: required pipeline=%s source_pool=%u transport=%s pcie_num=%u gpu_misses=%u threads=%u\n",
        selection.reference ? "reference" : "control", unsigned(selection.source_pool), selection.reference ? "poll" : "segmented", selection.pcie_num,
        config->gpu_miss_quota, config->n_threads);
    *output = session.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t fidelity_hybrid_prepare(void * opaque, const ggml_backend_moe_hybrid_region_v1 * input,
        const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api, ggml_backend_moe_cpu_service_v1_t cpu_service,
        const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions, void ** output) try {
    if (!output) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    *output = nullptr;
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session || !session->available() || !input || !cpu_api || !cpu_service || !cpu_regions ||
            ggml_backend_moe_hybrid_validate_buckets_v1(input) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ||
            input->n_cpu_batch_queries != 1) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    auto region = std::make_unique<fidelity_hybrid_region>();
    region->session = session;
    region->descriptor = *input;
    region->cpu_api = cpu_api;
    const auto cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_device) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    const auto cpu_module = ggml_backend_dev_backend_reg(cpu_device);
    const auto service_proc = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_module, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME));
    const auto requirements_proc = reinterpret_cast<ggml_backend_moe_cpu_fidelity_requirements_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_module, GGML_BACKEND_MOE_CPU_FIDELITY_REQUIREMENTS_V1_PROC_NAME));
    if (!service_proc || service_proc() != cpu_api || !requirements_proc) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    region->cpu_requirements_api = requirements_proc();
    region->queries.reserve(input->n_cpu_queries);
    region->query_ptrs.reserve(input->n_cpu_queries);
    for (uint32_t i = 0; i < input->n_cpu_queries; ++i) {
        if (!cpu_regions[i]) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        auto query = std::make_unique<fidelity_query_copy>();
        if (!query->copy(input->cpu_queries[i])) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
        region->query_ptrs.push_back(&query->query);
        region->queries.push_back(std::move(query));
    }
    for (uint32_t i = 0; i < input->n_cpu_batch_queries; ++i) {
        if (!cpu_regions[input->n_cpu_queries + i]) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        auto query = std::make_unique<fidelity_query_copy>();
        if (!query->copy(input->cpu_batch_queries[i])) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
        region->batch_query_ptrs.push_back(&query->query);
        region->batch_queries.push_back(std::move(query));
    }
    region->descriptor.query = region->query_ptrs[0];
    region->descriptor.cpu_queries = region->query_ptrs.data();
    region->descriptor.cpu_batch_queries = region->batch_query_ptrs.data();
    if (ggml_backend_moe_hybrid_validate_buckets_v1(&region->descriptor) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    *output = region.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static enum ggml_status fidelity_compute_failure(const ggml_cgraph * graph, uint32_t regions,
        const char * phase, const char * reason, int32_t status = GGML_STATUS_FAILED) {
    const ggml_graph_execution_certificate certificate = graph ? graph->execution_certificate : ggml_graph_execution_certificate{};
    fprintf(stderr, "moe-fidelity-failure: graph_uid=%llu source_uid=%llu split_uid=%llu owner_generation=%llu"
        " domain=%u row_semantics=%u rows=%u sequences=%u regions=%u phase=%s reason=%s status=%d\n",
        (unsigned long long) (graph ? graph->uid : 0), (unsigned long long) certificate.source_graph_uid,
        (unsigned long long) certificate.split_graph_uid, (unsigned long long) certificate.owner_generation,
        certificate.domain, certificate.row_semantics, certificate.n_rows, certificate.n_sequences, regions, phase, reason, status);
    return GGML_STATUS_FAILED;
}

static enum ggml_status fidelity_hybrid_compute(void * opaque, ggml_cgraph * graph, void * const * regions, uint32_t count) try {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session || !graph || !regions || !count || count > session->config.max_regions ||
            graph->execution_certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ||
            graph->execution_certificate.domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
            graph->execution_certificate.split_graph_uid != graph->uid) { return fidelity_compute_failure(graph, count, "entry", "invalid_query"); }
    if (!session->begin_compute()) { return fidelity_compute_failure(graph, count, "entry", "session_unavailable"); }
    struct compute_guard {
        fidelity_hybrid_session * session;
        ~compute_guard() { session->finish_compute(); }
    } guard{session};
    {
        std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
        session->region_count = count;
    }
    session->descriptors.clear();
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    const ggml_backend_moe_cpu_fidelity_requirements_api_v1 * cpu_requirements_api = nullptr;
    uint32_t last = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto * region = static_cast<fidelity_hybrid_region *>(regions[i]);
        const char * mismatch = !region ? "missing_region" : region->session != session ? "session_owner" :
            i && region->descriptor.first_node < last ? "region_order" :
            region->descriptor.split_graph_uid != graph->uid ? "split_uid" :
            memcmp(&region->descriptor.certificate, &graph->execution_certificate,
                sizeof(graph->execution_certificate)) != 0 ? "certificate" :
            cpu_api && cpu_api != region->cpu_api ? "cpu_api" :
            cpu_requirements_api && cpu_requirements_api != region->cpu_requirements_api ? "cpu_requirements_api" : nullptr;
        if (mismatch) {
            if (region) {
                const auto & prepared = region->descriptor.certificate;
                const auto & dispatched = graph->execution_certificate;
                fprintf(stderr, "moe-fidelity-region-reject: index=%u reason=%s first=%u previous_last=%u split_uid=%llu expected_split_uid=%llu"
                    " magic=%u/%u abi=%u/%u size=%u/%u flags=%u/%u domain=%u/%u semantics=%u/%u rows=%u/%u sequences=%u/%u"
                    " owner_namespace=%llu/%llu owner_generation=%llu/%llu source_uid=%llu/%llu certificate_split_uid=%llu/%llu\n",
                    i, mismatch, region->descriptor.first_node, last, (unsigned long long) region->descriptor.split_graph_uid, (unsigned long long) graph->uid,
                    prepared.magic, dispatched.magic, prepared.abi_version, dispatched.abi_version,
                    prepared.struct_size, dispatched.struct_size, prepared.flags, dispatched.flags, prepared.domain, dispatched.domain,
                    prepared.row_semantics, dispatched.row_semantics, prepared.n_rows, dispatched.n_rows, prepared.n_sequences, dispatched.n_sequences,
                    (unsigned long long) prepared.owner_namespace, (unsigned long long) dispatched.owner_namespace,
                    (unsigned long long) prepared.owner_generation, (unsigned long long) dispatched.owner_generation,
                    (unsigned long long) prepared.source_graph_uid, (unsigned long long) dispatched.source_graph_uid,
                    (unsigned long long) prepared.split_graph_uid, (unsigned long long) dispatched.split_graph_uid);
            }
            return fidelity_compute_failure(graph, count, "regions", mismatch);
        }
        last = region->descriptor.last_node;
        cpu_api = region->cpu_api;
        cpu_requirements_api = region->cpu_requirements_api;
        session->descriptors.push_back(&region->descriptor);
    }
    session->public_outputs.clear();
    for (int i = 0; i < graph->n_nodes; ++i) {
        if ((graph->nodes[i]->flags & GGML_TENSOR_FLAG_OUTPUT) && graph->nodes[i]->op != GGML_OP_NONE) {
            session->public_outputs.push_back(graph->nodes[i]);
        }
    }
    if (session->public_outputs.empty() || uint32_t(graph->n_nodes) <= last) {
        fprintf(stderr, "moe-fidelity: unsupported split outputs=%zu nodes=%d last_region=%u\n",
            session->public_outputs.size(), graph->n_nodes, last);
        return fidelity_compute_failure(graph, count, "outputs", "unsupported_split");
    }
    if (!session->window || session->graph_uid != graph->uid) {
        {
            std::unique_lock<std::mutex> lock(session->lifecycle_mutex);
            session->window_ready = false;
            session->lifecycle_condition.wait(lock, [&] { return session->active_states == 0; });
        }
        if (!session->reset_window()) { return fidelity_compute_failure(graph, count, "reset", "cleanup_failed"); }
        ggml_cuda_moe_fidelity_window_query_v1 query = {};
        query.struct_size = sizeof(query);
        query.n_regions = count;
        query.graph = graph;
        query.public_output = session->public_outputs[0];
        query.n_public_outputs = session->public_outputs.size();
        query.public_outputs = session->public_outputs.data();
        query.graph_owner = session;
        query.retain_graph = fidelity_graph_retain;
        query.release_graph = fidelity_graph_release;
        query.arena_generation = session->descriptors[0]->allocator_generation;
        query.regions = session->descriptors.data();
        query.source_owner = &session->source_owner;
        query.cpu_api = cpu_api;
        query.cpu_requirements_api = cpu_requirements_api;
        query.certificate = graph->execution_certificate;
        query.n_threads = session->config.n_threads;
        query.gpu_miss_quota = session->config.gpu_miss_quota;
        query.admission_quota = session->config.admission_quota;
        query.no_host_alias = getenv("GGML_MOE_FIDELITY_NO_HOST_ALIAS") != nullptr;
        const auto & selection = ggml_moe_fidelity_selection();
        if (!selection.valid || (selection.reference && query.no_host_alias)) {
            return fidelity_compute_failure(graph, count, "prepare", "invalid_pipeline_configuration");
        }
        const char * requested = getenv("GGML_MOE_FIDELITY_ARM");
        if (requested && strcmp(requested, "segmented") != 0 && strcmp(requested, "poll") != 0) {
            return fidelity_compute_failure(graph, count, "prepare", "unknown_arm");
        }
        if (selection.reference && requested && strcmp(requested, "poll")) { return fidelity_compute_failure(graph, count, "prepare", "reference_requires_poll"); }
        session->requested_arm = selection.reference || (requested && strcmp(requested, "poll") == 0) ? GGML_CUDA_MOE_FIDELITY_POLL : GGML_CUDA_MOE_FIDELITY_SEGMENTED;
        session->window_arm = query.no_host_alias ? GGML_CUDA_MOE_FIDELITY_SEGMENTED : session->requested_arm;
        if (session->log_summary) {
            fprintf(stderr, "moe-fidelity: transport requested=%u selected=%u reason=%s\n", session->requested_arm, session->window_arm,
                query.no_host_alias ? "no_host_alias_segmented" : requested ? "explicit" : "default_segmented");
        }
        ggml_cuda_moe_fidelity_window_storage_v1 limits = {};
        limits.device_bytes = session->config.device_bytes;
        limits.pinned_bytes = session->config.pinned_bytes;
        limits.cpu_bytes = session->config.cpu_bytes;
        void * prepared_window = nullptr;
        const int32_t prepared = ggml_cuda_moe_fidelity_window_api()->prepare(
            session->backend, &query, &limits, session->window_arm, &prepared_window);
        {
            std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
            session->window = prepared_window;
            session->window_ready = prepared == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
            if (prepared_window) {
                session->window_uid = graph->uid;
                session->window_rows = query.certificate.n_rows;
                session->window_regions = count;
            }
            if (session->window_ready) { session->graph_uid = graph->uid; }
        }
        if (prepared != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            fprintf(stderr, "moe-fidelity: window prepare failed status=%d regions=%u rows=%u\n",
                prepared, count, graph->execution_certificate.n_rows);
            return fidelity_compute_failure(graph, count, "prepare", "window_prepare", prepared);
        }
    }
    ggml_cuda_moe_fidelity_window_state_v1 current = {};
    if (!ggml_cuda_moe_fidelity_window_api()->state(session->window, &current)) {
        return fidelity_compute_failure(graph, count, "state_before", "state_unavailable");
    }
    {
        std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
        session->last = current;
    }
    if (session->cancel_requested()) { return fidelity_compute_failure(graph, count, "replay", "cancel_requested"); }
    ggml_cuda_moe_fidelity_window_replay_v1 replay = {};
    replay.epoch = ++session->epoch;
    replay.identity = current.identity;
    replay.active_rows = graph->execution_certificate.n_rows;
    ggml_cuda_moe_fidelity_sample_v1 sample = {};
    const int32_t status = ggml_cuda_moe_fidelity_window_api()->replay(session->window, &replay, &sample);
    session->record_sample(sample);
    if (!ggml_cuda_moe_fidelity_window_api()->state(session->window, &current)) {
        return fidelity_compute_failure(graph, count, "state_after", "state_unavailable", status);
    }
    {
        std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
        session->last = current;
    }
    return status == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ? GGML_STATUS_SUCCESS :
        fidelity_compute_failure(graph, count, "replay", "window_replay", status);
} catch (...) { return fidelity_compute_failure(graph, count, "compute", "exception"); }

static void fidelity_hybrid_destroy_region(void * opaque, void * prepared) {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    auto * region = static_cast<fidelity_hybrid_region *>(prepared);
    if (session && region && region->session == session && !session->reset_serialized(true)) {
        region->retired_next = session->retired_regions;
        session->retired_regions = region;
        return;
    }
    delete region;
}

static void fidelity_hybrid_destroy(void * opaque) {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session) { return; }
    const bool released = session->reset_serialized(false);
    GGML_ASSERT(released && session->leases == 0);
    delete session;
}

static bool fidelity_hybrid_state(void * opaque, ggml_backend_moe_hybrid_state_v1 * output) {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session || !output || output->struct_size != sizeof(*output)) { return false; }
    void * window = nullptr;
    ggml_cuda_moe_fidelity_window_state_v1 state = {};
    uint32_t region_count = 0;
    uint32_t active_computes = 0;
    bool quiescing = false;
    {
        std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
        state = session->last;
        region_count = session->region_count;
        active_computes = session->active_computes;
        quiescing = session->quiescing;
        if (session->window_ready && session->window && (!session->quiescing || session->active_computes != 0)) {
            window = session->window;
            ++session->active_states;
        }
    }
    if (window) {
        const bool ok = ggml_cuda_moe_fidelity_window_api()->state(window, &state);
        {
            std::lock_guard<std::mutex> lock(session->lifecycle_mutex);
            if (ok) { session->last = state; }
            region_count = session->region_count;
            active_computes = session->active_computes;
            quiescing = session->quiescing;
            GGML_ASSERT(session->active_states != 0);
            --session->active_states;
            session->lifecycle_condition.notify_all();
        }
        if (!ok) { return false; }
    }
    *output = {};
    output->struct_size = sizeof(*output);
    output->resident_routes = state.resident_routes;
    output->transfer_routes = state.transfer_routes;
    output->cpu_routes = state.cpu_routes;
    output->h2d_bytes = state.transfer_bytes;
    output->device_bytes = state.storage.device_bytes;
    output->pinned_bytes = state.storage.pinned_bytes;
    output->cpu_jobs = state.cpu_calls;
    output->errors = state.rejected;
    output->cpu_active_jobs = state.active_jobs;
    output->dispatch_active = active_computes;
    output->quiescing = quiescing;
    output->prepared_device_bytes = state.storage.device_bytes;
    output->prepared_cpu_bytes = state.storage.cpu_bytes;
    output->packet_regions = state.epochs * region_count;
    output->producer_events = state.producers_expected;
    output->producer_fences = state.producers_completed;
    output->window_launches = state.launches;
    output->window_captures = state.captures;
    output->window_waits = state.epochs;
    output->window_fallbacks = state.fallbacks;
    output->cpu_execute_calls = state.cpu_calls;
    output->cpu_batch_rows = state.cpu_routes;
    return true;
}

static void fidelity_hybrid_quiesce(void * opaque) {
    auto * session = static_cast<fidelity_hybrid_session *>(opaque);
    if (!session) { return; }
    (void) session->reset_serialized(true);
}

static bool fidelity_hybrid_test_hook(void *, ggml_backend_moe_hybrid_test_hook_v1_t hook, void *) {
    return hook == nullptr;
}

}

const ggml_backend_moe_hybrid_api_v1 * ggml_cuda_moe_fidelity_hybrid_api() {
    static const ggml_backend_moe_hybrid_api_v1 api = {
        sizeof(api), 1, fidelity_hybrid_create, fidelity_hybrid_prepare, fidelity_hybrid_compute,
        fidelity_hybrid_destroy_region, fidelity_hybrid_destroy, fidelity_hybrid_state,
        fidelity_hybrid_quiesce, fidelity_hybrid_test_hook};
    return &api;
}
