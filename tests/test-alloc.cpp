#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size        = 64;
    size_t capacity               = SIZE_MAX; // what the device can hold at one time
    size_t alignment              = 8;
    bool   fail_alloc             = false;
    bool   unique_alloc_addresses = false;
    int    graph_compute_count    = 0;
    enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    const char * registry_name = "dummy";
    bool buffer_is_host = true;
    bool fail_backend_init = false;
    bool fail_event_init = false;
    int transfer_backend_count = 0;
    int event_wait_count = 0;
    int set_tensor_async_count = 0;
    size_t set_tensor_async_bytes = 0;
    size_t alloc_size_pad = 0;

    // what the backend was asked to hold and to move, so a test can check the entries a transport ring lays out and the bytes it delivers into them
    struct tensor_binding {
        const ggml_tensor *   tensor;
        ggml_backend_buffer_t buffer;
        const char *          data;
        size_t                size;
    };
    struct tensor_delivery {
        const ggml_tensor * tensor;
        const char *        src;
        size_t              offset;
        size_t              size;
    };
    struct event_step {
        bool                 is_wait;
        ggml_backend_event_t event;
    };
    std::vector<tensor_binding>  bindings;
    std::vector<tensor_delivery> deliveries;
    std::vector<event_step>      event_steps;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_i backend_interface = {};

    ggml_backend_buffer_i              buffer_interface;
    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<void *>                buffer_bases;
    uintptr_t                          next_base = (uintptr_t) alloc_base;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    if (ctx->fail_alloc || size > ctx->capacity - ctx->allocated_total()) {
        return nullptr;
    }
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    ctx->next_base += std::max<size_t>(size, 4096);
    ctx->buffer_bases.push_back((void *) ctx->next_base);
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static size_t dummy_backend_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    // only a tensor that is no op's output may ask for more than its data [TAG_ALLOC_SIZE_EXPAND]
    return ggml_nbytes(tensor) + (ggml_op_is_empty(tensor->op) ? ctx->alloc_size_pad : 0);
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return ((dummy_backend_context *) buft->context)->buffer_is_host;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffer_bases.erase(ctx->buffer_bases.begin() + (i - ctx->buffers.begin()));
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;
    if (!ctx->unique_alloc_addresses) {
        return alloc_base;
    }
    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    return ctx->buffer_bases[i - ctx->buffers.begin()];
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;
    const size_t size = ggml_backend_buffer_get_alloc_size(buffer, tensor);

    for (auto & b : ctx->bindings) {
        if (b.tensor == tensor) {
            b = { tensor, buffer, (const char *) tensor->data, size };
            return GGML_STATUS_SUCCESS;
        }
    }
    ctx->bindings.push_back({ tensor, buffer, (const char *) tensor->data, size });
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t, uint8_t) {}

// dummy backend for gallocr and scheduler allocation tests

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    std::unique_ptr<ggml_backend_reg>      registry;
    std::unique_ptr<ggml_backend_device>   device;
    std::unique_ptr<ggml_backend>          handle;
    ggml_backend_buffer_type               buffer_type;
};

static const char * dummy_backend_get_name(ggml_backend_t backend) {
    return ((dummy_backend_context *) backend->context)->registry_name;
}

static void dummy_backend_free(ggml_backend_t backend) {
    dummy_backend_context * ctx = (dummy_backend_context *) backend->context;
    ctx->transfer_backend_count--;
    delete backend;
}

static void dummy_backend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    dummy_backend_context * ctx = (dummy_backend_context *) backend->context;
    ctx->set_tensor_async_count++;
    ctx->set_tensor_async_bytes += size;
    ctx->deliveries.push_back({ tensor, (const char *) data, offset, size });
}

static void dummy_backend_synchronize(ggml_backend_t) {}

static void dummy_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ((dummy_backend_context *) backend->context)->event_steps.push_back({ false, event });
}

static void dummy_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    dummy_backend_context * ctx = (dummy_backend_context *) backend->context;
    ctx->event_wait_count++;
    ctx->event_steps.push_back({ true, event });
}

static enum ggml_status dummy_backend_graph_compute(ggml_backend_t backend, ggml_cgraph *) {
    dummy_backend_context * ctx = (dummy_backend_context *) backend->context;
    ctx->graph_compute_count++;
    return GGML_STATUS_SUCCESS;
}

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t dev) {
    return ((dummy_backend_context *) dev->context)->device_type;
}

static void dummy_backend_device_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = SIZE_MAX;
    *total = SIZE_MAX;
}

static ggml_backend_t dummy_backend_device_init(ggml_backend_dev_t dev, const char *) {
    dummy_backend_context * ctx = (dummy_backend_context *) dev->context;
    if (ctx->fail_backend_init) {
        return nullptr;
    }
    ggml_backend_t backend = new ggml_backend{};
    backend->iface = ctx->backend_interface;
    backend->device = dev;
    backend->context = ctx;
    ctx->transfer_backend_count++;
    return backend;
}

static ggml_backend_buffer_type_t dummy_backend_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ((dummy_backend_context *) dev->context)->buffer_type;
}

static ggml_backend_event_t dummy_backend_device_event_new(ggml_backend_dev_t dev) {
    dummy_backend_context * ctx = (dummy_backend_context *) dev->context;
    if (ctx->fail_event_init) {
        return nullptr;
    }
    ggml_backend_event_t event = new ggml_backend_event;
    event->device = dev;
    event->context = nullptr;
    return event;
}

static void dummy_backend_device_event_free(ggml_backend_dev_t, ggml_backend_event_t event) {
    delete event;
}

static void dummy_backend_device_event_synchronize(ggml_backend_dev_t, ggml_backend_event_t) {}

static const char * dummy_backend_registry_get_name(ggml_backend_reg_t reg) {
    return ((dummy_backend_context *) reg->context)->registry_name;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

static dummy_backend dummy_backend_init(
        size_t max_buffer_size,
        size_t alignment = 8,
        bool unique_alloc_addresses = false,
        enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_CPU,
        const char * registry_name = "dummy",
        bool buffer_is_host = true) {
    dummy_backend b{};
    b.context                         = std::make_unique<dummy_backend_context>();
    b.context->alignment              = alignment;
    b.context->max_buffer_size        = max_buffer_size;
    b.context->unique_alloc_addresses = unique_alloc_addresses;
    b.context->device_type            = device_type;
    b.context->registry_name          = registry_name;
    b.context->buffer_is_host         = buffer_is_host;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;

    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.get_alloc_size = dummy_backend_buffer_type_get_alloc_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    b.context->buffer_type            = &b.buffer_type;

    b.registry = std::make_unique<ggml_backend_reg>();
    b.registry->iface.get_name = dummy_backend_registry_get_name;
    b.registry->context = b.context.get();

    b.device = std::make_unique<ggml_backend_device>();
    b.device->iface.get_memory        = dummy_backend_device_get_memory;
    b.device->iface.get_type          = dummy_backend_device_get_type;
    b.device->iface.init_backend      = dummy_backend_device_init;
    b.device->iface.get_buffer_type   = dummy_backend_device_get_buffer_type;
    b.device->iface.supports_op       = dummy_backend_device_supports_op;
    b.device->iface.supports_buft     = dummy_backend_device_supports_buft;
    b.device->iface.event_new         = dummy_backend_device_event_new;
    b.device->iface.event_free        = dummy_backend_device_event_free;
    b.device->iface.event_synchronize = dummy_backend_device_event_synchronize;
    b.device->reg                     = b.registry.get();
    b.device->context                 = b.context.get();

    b.buffer_type.device = b.device.get();

    b.handle = std::make_unique<ggml_backend>();
    b.context->backend_interface.get_name         = dummy_backend_get_name;
    b.context->backend_interface.free             = dummy_backend_free;
    b.context->backend_interface.set_tensor_async = dummy_backend_set_tensor_async;
    b.context->backend_interface.synchronize      = dummy_backend_synchronize;
    b.context->backend_interface.graph_compute    = dummy_backend_graph_compute;
    b.context->backend_interface.event_record     = dummy_backend_event_record;
    b.context->backend_interface.event_wait       = dummy_backend_event_wait;
    b.handle->iface                               = b.context->backend_interface;
    b.handle->device                              = b.device.get();
    b.handle->context                             = b.context.get();
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

struct transport_graph {
    test_context_with_graph ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * source;
    ggml_tensor * output;
};

static transport_graph make_transport_graph(dummy_backend & cpu, size_t size) {
    GGML_ASSERT(size % sizeof(float) == 0);
    auto result = make_context();
    ggml_tensor * source = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, size/sizeof(float));
    ggml_tensor * output = ggml_scale(result.ctx, source, 2.0f);
    source->flags |= GGML_TENSOR_FLAG_TRANSPORT;
    ggml_build_forward_expand(result.graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, size));
    source->buffer = buffer.get();
    source->data = ggml_backend_buffer_get_base(buffer.get());

    return { std::move(result), std::move(buffer), source, output };
}

struct transport_graph_pair {
    test_context_with_graph ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * sources[2];
    ggml_tensor * output;
};

// two transported inputs in one split, so that the ring lays out more than one entry per slot
static transport_graph_pair make_transport_graph_pair(dummy_backend & cpu, size_t size) {
    GGML_ASSERT(size % sizeof(float) == 0);
    auto result = make_context();
    ggml_tensor * s0 = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, size/sizeof(float));
    ggml_tensor * s1 = ggml_new_tensor_1d(result.ctx, GGML_TYPE_F32, size/sizeof(float));
    ggml_tensor * output = ggml_add(result.ctx, s0, s1);
    s0->flags |= GGML_TENSOR_FLAG_TRANSPORT;
    s1->flags |= GGML_TENSOR_FLAG_TRANSPORT;
    ggml_build_forward_expand(result.graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, 2*size));
    char * base = (char *) ggml_backend_buffer_get_base(buffer.get());
    s0->buffer = buffer.get();
    s0->data   = base;
    s1->buffer = buffer.get();
    s1->data   = base + size;

    return { std::move(result), std::move(buffer), { s0, s1 }, output };
}

static void transport_stats(
        ggml_backend_sched_t sched,
        int64_t * deliveries,
        int64_t * early,
        int64_t * late) {
    ggml_backend_sched_get_transport_pipeline_stats(sched, deliveries, early, late);
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto   buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

static auto make_resizable_add_graph(int64_t n_elements) {
    auto result = make_context();
    ggml_tensor * a = make_input_1d(result.ctx, n_elements);
    ggml_tensor * b = make_input_1d(result.ctx, n_elements);
    ggml_tensor * out = ggml_add(result.ctx, a, b);
    ggml_set_output(out);
    ggml_build_forward_expand(result.graph, out);
    return result;
}

static uint64_t gallocr_generation(ggml_gallocr_t alloc) {
    uint64_t generation = 0;
    uint64_t shrink_generation = 0;
    ggml_gallocr_get_resizable_state(alloc, &generation, &shrink_generation);
    return generation;
}

static uint64_t gallocr_shrink_generation(ggml_gallocr_t alloc) {
    uint64_t generation = 0;
    uint64_t shrink_generation = 0;
    ggml_gallocr_get_resizable_state(alloc, &generation, &shrink_generation);
    return shrink_generation;
}

static uint64_t sched_generation(ggml_backend_sched_t sched) {
    uint64_t generation = 0;
    uint64_t shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched, &generation, &shrink_generation);
    return generation;
}

static void test_resizable_buffers_grow_shrink_grow() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4);

    auto small  = make_resizable_add_graph(4);
    auto medium = make_resizable_add_graph(16);
    auto large  = make_resizable_add_graph(24);

    {
        ggml_gallocr_ptr alloc(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(alloc.get(), nullptr));

        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), large.graph));
        const size_t large_size = backend.context->allocated_total();
        const uint64_t large_generation = gallocr_generation(alloc.get());
        GGML_ASSERT(large_size > 0);

        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), small.graph));
        GGML_ASSERT(backend.context->allocated_total() == large_size);
        GGML_ASSERT(gallocr_generation(alloc.get()) == large_generation);

        ggml_gallocr_request_shrink(alloc.get());
        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), small.graph));
        const size_t small_size = backend.context->allocated_total();
        const uint64_t small_generation = gallocr_generation(alloc.get());
        GGML_ASSERT(small_size < large_size);
        GGML_ASSERT(small_generation > large_generation);

        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), medium.graph));
        const size_t medium_size = backend.context->allocated_total();
        const uint64_t medium_generation = gallocr_generation(alloc.get());
        GGML_ASSERT(medium_size > small_size);
        GGML_ASSERT(medium_size < large_size);
        GGML_ASSERT(medium_generation > small_generation);

        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), large.graph));
        GGML_ASSERT(backend.context->allocated_total() == large_size);
        GGML_ASSERT(gallocr_generation(alloc.get()) > medium_generation);
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), large.graph));
        check_all_allocated(large.graph);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_fail_closed_on_allocation_failure() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);

    auto small = make_resizable_add_graph(4);
    auto large = make_resizable_add_graph(24);

    {
        ggml_gallocr_ptr alloc(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(alloc.get(), nullptr));

        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), small.graph));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), small.graph));
        check_all_allocated(small.graph);

        const uint64_t generation = gallocr_generation(alloc.get());
        ggml_backend_buffer_t buffer = small.graph->nodes[0]->buffer;
        GGML_ASSERT(buffer != nullptr);
        void * base = ggml_backend_buffer_get_base(buffer);

        backend.context->fail_alloc = true;
        GGML_ASSERT(!ggml_gallocr_reserve(alloc.get(), large.graph));
        const uint64_t failed_generation = gallocr_generation(alloc.get());
        GGML_ASSERT(backend.context->allocated_total() == 0);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 0) == 0);
        GGML_ASSERT(failed_generation > generation);
        GGML_ASSERT(std::find(backend.context->buffer_bases.begin(), backend.context->buffer_bases.end(), base) == backend.context->buffer_bases.end());

        backend.context->fail_alloc = false;
        GGML_ASSERT(ggml_gallocr_reserve(alloc.get(), large.graph));
        GGML_ASSERT(gallocr_generation(alloc.get()) > failed_generation);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(ggml_backend_buffer_get_base(backend.context->buffers[0]) != base);
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), large.graph));
        check_all_allocated(large.graph);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_multi_entry_allocation_failure() {
    dummy_backend backend_a = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);

    auto small = make_resizable_add_graph(4);
    auto large = make_resizable_add_graph(24);

    GGML_ASSERT(small.graph->n_leafs == 2 && small.graph->n_nodes == 1);
    GGML_ASSERT(large.graph->n_leafs == 2 && large.graph->n_nodes == 1);
    int leaf_buffer_ids[2] = { 0, 1 };
    int node_buffer_ids[1] = { 0 };
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    {
        ggml_gallocr_ptr alloc(ggml_gallocr_new_n(bufts, 2));
        GGML_ASSERT(ggml_gallocr_set_resizable(alloc.get(), nullptr));

        GGML_ASSERT(ggml_gallocr_reserve_n(alloc.get(), small.graph, node_buffer_ids, leaf_buffer_ids));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), small.graph));
        check_all_allocated(small.graph);

        const size_t size_a_small = backend_a.context->allocated_total();
        const size_t size_b_small = backend_b.context->allocated_total();
        const uint64_t generation_small = gallocr_generation(alloc.get());
        ggml_backend_buffer_t buffer_a_small = small.graph->nodes[0]->buffer;
        ggml_backend_buffer_t buffer_b_small = small.graph->leafs[1]->buffer;
        void * data_a_small = small.graph->nodes[0]->data;
        void * data_b_small = small.graph->leafs[1]->data;
        GGML_ASSERT(size_a_small > 0 && size_b_small > 0);
        GGML_ASSERT(buffer_a_small != nullptr && buffer_b_small != nullptr);
        GGML_ASSERT(data_a_small != nullptr && data_b_small != nullptr);
        void * base_a_small = ggml_backend_buffer_get_base(buffer_a_small);
        void * base_b_small = ggml_backend_buffer_get_base(buffer_b_small);

        backend_b.context->fail_alloc = true;
        GGML_ASSERT(!ggml_gallocr_reserve_n(alloc.get(), large.graph, node_buffer_ids, leaf_buffer_ids));
        const size_t size_a_partial = backend_a.context->allocated_total();
        const uint64_t generation_partial = gallocr_generation(alloc.get());
        GGML_ASSERT(backend_a.context->buffers.size() == 1);
        ggml_backend_buffer_t buffer_a_partial = backend_a.context->buffers[0];
        GGML_ASSERT(size_a_partial > size_a_small);
        GGML_ASSERT(backend_b.context->allocated_total() == 0);
        GGML_ASSERT(generation_partial > generation_small);
        GGML_ASSERT(ggml_backend_buffer_get_base(buffer_a_partial) != base_a_small);
        GGML_ASSERT(std::find(backend_b.context->buffer_bases.begin(), backend_b.context->buffer_bases.end(), base_b_small) == backend_b.context->buffer_bases.end());
        GGML_ASSERT(small.graph->nodes[0]->buffer == buffer_a_small);
        GGML_ASSERT(small.graph->nodes[0]->data == data_a_small);

        // Entry A published before entry B failed, so discard the graph with stale entry A addresses.
        small.ctx_ptr.reset();
        small.ctx = nullptr;
        small.graph = nullptr;

        backend_b.context->fail_alloc = false;
        GGML_ASSERT(ggml_gallocr_reserve_n(alloc.get(), large.graph, node_buffer_ids, leaf_buffer_ids));
        GGML_ASSERT(backend_b.context->buffers.size() == 1);
        ggml_backend_buffer_t buffer_b_large = backend_b.context->buffers[0];
        GGML_ASSERT(backend_a.context->allocated_total() == size_a_partial);
        GGML_ASSERT(backend_b.context->allocated_total() > size_b_small);
        GGML_ASSERT(gallocr_generation(alloc.get()) > generation_partial);
        GGML_ASSERT(ggml_backend_buffer_get_base(buffer_b_large) != base_b_small);

        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), large.graph));
        GGML_ASSERT(large.graph->nodes[0]->buffer == buffer_a_partial);
        GGML_ASSERT(large.graph->nodes[0]->data != data_a_small);
        GGML_ASSERT(large.graph->leafs[1]->buffer == buffer_b_large);
        GGML_ASSERT(large.graph->leafs[1]->data != data_b_small);
        check_all_allocated(large.graph);
    }
    GGML_ASSERT(backend_a.context->allocated_total() == 0);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

static void test_resizable_buffers_scheduler_allocation_failure() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);

    auto small       = make_resizable_add_graph(4);
    auto small_retry = make_resizable_add_graph(4);
    auto large       = make_resizable_add_graph(24);
    auto large_retry = make_resizable_add_graph(24);

    ggml_backend_t backends[] = { backend.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &backend.buffer_type };
    {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 1, 128, false, false));
        GGML_ASSERT(ggml_backend_sched_set_resizable(sched.get(), nullptr));

        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), small.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(backend.context->graph_compute_count == 1);
        const uint64_t generation = sched_generation(sched.get());
        ggml_backend_buffer_t buffer = small.graph->nodes[0]->buffer;
        GGML_ASSERT(buffer != nullptr);
        void * base = ggml_backend_buffer_get_base(buffer);

        ggml_backend_sched_reset(sched.get());
        backend.context->fail_alloc = true;
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), large.graph) == GGML_STATUS_ALLOC_FAILED);
        GGML_ASSERT(backend.context->graph_compute_count == 1);
        const uint64_t failed_generation = sched_generation(sched.get());
        GGML_ASSERT(backend.context->allocated_total() == 0);
        GGML_ASSERT(failed_generation > generation);
        GGML_ASSERT(std::find(backend.context->buffer_bases.begin(), backend.context->buffer_bases.end(), base) == backend.context->buffer_bases.end());

        GGML_ASSERT(!ggml_backend_sched_reserve(sched.get(), small_retry.graph));
        GGML_ASSERT(backend.context->graph_compute_count == 1);
        GGML_ASSERT(backend.context->allocated_total() == 0);

        ggml_backend_sched_reset(sched.get());
        backend.context->fail_alloc = false;
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), large_retry.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(backend.context->graph_compute_count == 2);
        GGML_ASSERT(sched_generation(sched.get()) > failed_generation);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_alias_duplicate_buffer_types() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a = make_input_with_size(ctx, 16);
    ggml_tensor * b = make_input_with_size(ctx, 16);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    GGML_ASSERT(graph->n_leafs == 2);
    GGML_ASSERT(graph->n_nodes == 1);
    int leaf_buffer_ids[2] = { 0, 1 };
    int node_buffer_ids[1] = { 1 };
    ggml_backend_buffer_type_t bufts[2] = { &backend.buffer_type, &backend.buffer_type };

    {
        ggml_gallocr_ptr alloc(ggml_gallocr_new_n(bufts, 2));
        GGML_ASSERT(ggml_gallocr_set_resizable(alloc.get(), nullptr));
        GGML_ASSERT(ggml_gallocr_reserve_n(alloc.get(), graph, node_buffer_ids, leaf_buffer_ids));

        const size_t backing_size = ggml_gallocr_get_buffer_size(alloc.get(), 0);
        GGML_ASSERT(backing_size > 0);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(alloc.get(), 1) == 0);
        GGML_ASSERT(backend.context->allocated_total() == backing_size);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), graph));
        check_all_allocated(graph);
        check_no_overlap(graph);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_owner_borrower_maximum() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);

    auto owner_small     = make_resizable_add_graph(4);
    auto owner_medium    = make_resizable_add_graph(16);
    auto owner_large     = make_resizable_add_graph(24);
    auto borrower_medium = make_resizable_add_graph(16);
    auto borrower_large  = make_resizable_add_graph(24);

    {
        ggml_gallocr_ptr owner(ggml_gallocr_new(&backend.buffer_type));
        ggml_gallocr_ptr borrower(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(owner.get(), nullptr));
        GGML_ASSERT(ggml_gallocr_set_resizable(borrower.get(), owner.get()));

        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_medium.graph));
        const size_t owner_medium_size = backend.context->allocated_total();
        GGML_ASSERT(owner_medium_size > 0);

        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_large.graph));
        const size_t shared_large_size = backend.context->allocated_total();
        const uint64_t shared_large_generation = gallocr_generation(owner.get());
        GGML_ASSERT(shared_large_size > owner_medium_size);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(owner.get(), 0) == shared_large_size);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(borrower.get(), 0) == shared_large_size);
        GGML_ASSERT(gallocr_generation(borrower.get()) == shared_large_generation);

        const uint64_t shrink_generation = gallocr_shrink_generation(owner.get());
        ggml_gallocr_request_shrink(owner.get());
        GGML_ASSERT(gallocr_shrink_generation(owner.get()) > shrink_generation);
        GGML_ASSERT(gallocr_shrink_generation(borrower.get()) == gallocr_shrink_generation(owner.get()));

        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_small.graph));
        GGML_ASSERT(backend.context->allocated_total() == shared_large_size);
        GGML_ASSERT(gallocr_generation(owner.get()) == shared_large_generation);

        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_medium.graph));
        const size_t shared_medium_size = backend.context->allocated_total();
        GGML_ASSERT(shared_medium_size < shared_large_size);
        GGML_ASSERT(shared_medium_size > 0);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(gallocr_generation(owner.get()) > shared_large_generation);

        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_large.graph));
        GGML_ASSERT(backend.context->allocated_total() == shared_large_size);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(ggml_gallocr_alloc_graph(owner.get(), owner_large.graph));
        check_all_allocated(owner_large.graph);
        GGML_ASSERT(ggml_gallocr_alloc_graph(borrower.get(), borrower_medium.graph));
        check_all_allocated(borrower_medium.graph);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_owner_borrower_compatible_placement() {
    dummy_backend backend_shared   = dummy_backend_init(SIZE_MAX, /*align*/ 4);
    dummy_backend backend_owner    = dummy_backend_init(SIZE_MAX, /*align*/ 4);
    dummy_backend backend_borrower = dummy_backend_init(SIZE_MAX, /*align*/ 4);

    auto owner_graph    = make_resizable_add_graph(16);
    auto borrower_graph = make_resizable_add_graph(24);
    int leaf_buffer_ids[2] = { 0, 1 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t owner_bufts[2] = {
        &backend_shared.buffer_type,
        &backend_owner.buffer_type,
    };
    ggml_backend_buffer_type_t borrower_bufts[2] = {
        &backend_shared.buffer_type,
        &backend_borrower.buffer_type,
    };

    {
        ggml_gallocr_ptr owner(ggml_gallocr_new_n(owner_bufts, 2));
        ggml_gallocr_ptr borrower(ggml_gallocr_new_n(borrower_bufts, 2));
        GGML_ASSERT(ggml_gallocr_set_resizable(owner.get(), nullptr));
        GGML_ASSERT(ggml_gallocr_set_resizable(borrower.get(), owner.get()));

        GGML_ASSERT(ggml_gallocr_reserve_n(owner.get(), owner_graph.graph, node_buffer_ids, leaf_buffer_ids));
        GGML_ASSERT(ggml_gallocr_reserve_n(borrower.get(), borrower_graph.graph, node_buffer_ids, leaf_buffer_ids));

        const size_t shared_size = backend_shared.context->allocated_total();
        GGML_ASSERT(shared_size > 0);
        GGML_ASSERT(backend_shared.context->buffers.size() == 1);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(owner.get(), 0) == shared_size);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(borrower.get(), 0) == shared_size);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(owner.get(), 1) == backend_owner.context->allocated_total());
        GGML_ASSERT(ggml_gallocr_get_buffer_size(borrower.get(), 1) == backend_borrower.context->allocated_total());

        GGML_ASSERT(ggml_gallocr_alloc_graph(owner.get(), owner_graph.graph));
        GGML_ASSERT(ggml_gallocr_alloc_graph(borrower.get(), borrower_graph.graph));
        check_all_allocated(owner_graph.graph);
        check_all_allocated(borrower_graph.graph);
    }
    GGML_ASSERT(backend_shared.context->allocated_total() == 0);
    GGML_ASSERT(backend_owner.context->allocated_total() == 0);
    GGML_ASSERT(backend_borrower.context->allocated_total() == 0);
}

static void test_resizable_buffers_owner_borrower_allocation_failure() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);

    auto owner_small    = make_resizable_add_graph(4);
    auto borrower_small = make_resizable_add_graph(4);
    auto owner_large    = make_resizable_add_graph(24);

    {
        ggml_gallocr_ptr owner(ggml_gallocr_new(&backend.buffer_type));
        ggml_gallocr_ptr borrower(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(owner.get(), nullptr));
        GGML_ASSERT(ggml_gallocr_set_resizable(borrower.get(), owner.get()));

        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_small.graph));
        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_small.graph));
        const uint64_t generation = gallocr_generation(owner.get());

        backend.context->fail_alloc = true;
        GGML_ASSERT(!ggml_gallocr_reserve(owner.get(), owner_large.graph));
        const uint64_t failed_generation = gallocr_generation(owner.get());
        GGML_ASSERT(failed_generation > generation);
        GGML_ASSERT(gallocr_generation(borrower.get()) == failed_generation);
        GGML_ASSERT(backend.context->allocated_total() == 0);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(owner.get(), 0) == 0);
        GGML_ASSERT(ggml_gallocr_get_buffer_size(borrower.get(), 0) == 0);

        backend.context->fail_alloc = false;
        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_small.graph));
        GGML_ASSERT(backend.context->allocated_total() > 0);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(gallocr_generation(owner.get()) > failed_generation);
        GGML_ASSERT(ggml_gallocr_alloc_graph(owner.get(), owner_large.graph));
        check_all_allocated(owner_large.graph);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_owner_borrower_scheduler_failure() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4, /*unique_alloc_addresses*/ true);

    auto owner_small     = make_resizable_add_graph(4);
    auto borrower_small  = make_resizable_add_graph(4);
    auto owner_large     = make_resizable_add_graph(24);
    auto owner_retry     = make_resizable_add_graph(24);
    auto borrower_retry  = make_resizable_add_graph(4);

    ggml_backend_t backends[] = { backend.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &backend.buffer_type };
    {
        ggml_backend_sched_ptr owner(ggml_backend_sched_new(backends, bufts, 1, 128, false, false));
        ggml_backend_sched_ptr borrower(ggml_backend_sched_new(backends, bufts, 1, 128, false, false));
        GGML_ASSERT(ggml_backend_sched_set_resizable(owner.get(), nullptr));
        GGML_ASSERT(ggml_backend_sched_set_resizable(borrower.get(), owner.get()));

        GGML_ASSERT(ggml_backend_sched_graph_compute(owner.get(), owner_small.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_sched_graph_compute(borrower.get(), borrower_small.graph) == GGML_STATUS_SUCCESS);
        const uint64_t generation = sched_generation(owner.get());

        ggml_backend_sched_reset(owner.get());
        backend.context->fail_alloc = true;
        GGML_ASSERT(ggml_backend_sched_graph_compute(owner.get(), owner_large.graph) == GGML_STATUS_ALLOC_FAILED);
        const uint64_t failed_generation = sched_generation(owner.get());
        GGML_ASSERT(failed_generation > generation);
        GGML_ASSERT(sched_generation(borrower.get()) == failed_generation);
        GGML_ASSERT(backend.context->allocated_total() == 0);

        ggml_backend_sched_reset(owner.get());
        ggml_backend_sched_reset(borrower.get());
        backend.context->fail_alloc = false;
        GGML_ASSERT(ggml_backend_sched_reserve(borrower.get(), borrower_retry.graph));
        GGML_ASSERT(sched_generation(owner.get()) > failed_generation);
        GGML_ASSERT(backend.context->buffers.size() == 1);
        GGML_ASSERT(ggml_backend_sched_graph_compute(borrower.get(), borrower_retry.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_sched_graph_compute(owner.get(), owner_retry.graph) == GGML_STATUS_SUCCESS);
    }
    GGML_ASSERT(backend.context->allocated_total() == 0);
}

static void test_resizable_buffers_owner_borrower_teardown_order() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, /*align*/ 4);

    auto owner_graph       = make_resizable_add_graph(16);
    auto borrower_graph    = make_resizable_add_graph(24);
    auto owner_retry       = make_resizable_add_graph(4);
    auto borrower_retry    = make_resizable_add_graph(4);

    {
        ggml_gallocr_ptr owner(ggml_gallocr_new(&backend.buffer_type));
        ggml_gallocr_ptr borrower(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(owner.get(), nullptr));
        GGML_ASSERT(ggml_gallocr_set_resizable(borrower.get(), owner.get()));
        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_graph.graph));
        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_graph.graph));

        owner.reset();
        GGML_ASSERT(backend.context->allocated_total() > 0);
        ggml_gallocr_request_shrink(borrower.get());
        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_retry.graph));
        GGML_ASSERT(ggml_gallocr_alloc_graph(borrower.get(), borrower_retry.graph));
        borrower.reset();
        GGML_ASSERT(backend.context->allocated_total() == 0);
    }

    {
        ggml_gallocr_ptr owner(ggml_gallocr_new(&backend.buffer_type));
        ggml_gallocr_ptr borrower(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(owner.get(), nullptr));
        GGML_ASSERT(ggml_gallocr_set_resizable(borrower.get(), owner.get()));
        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_graph.graph));
        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_graph.graph));

        borrower.reset();
        GGML_ASSERT(backend.context->allocated_total() > 0);
        borrower.reset(ggml_gallocr_new(&backend.buffer_type));
        GGML_ASSERT(ggml_gallocr_set_resizable(borrower.get(), owner.get()));
        ggml_gallocr_request_shrink(owner.get());
        GGML_ASSERT(ggml_gallocr_reserve(owner.get(), owner_retry.graph));
        GGML_ASSERT(ggml_gallocr_reserve(borrower.get(), borrower_retry.graph));
        GGML_ASSERT(ggml_gallocr_alloc_graph(owner.get(), owner_retry.graph));
        GGML_ASSERT(ggml_gallocr_alloc_graph(borrower.get(), borrower_retry.graph));
        borrower.reset();
        owner.reset();
        GGML_ASSERT(backend.context->allocated_total() == 0);
    }
}

static void test_transport_prefix_and_configuration() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_transport_graph(cpu, 64);

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 1024));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
        ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());

        ggml_set_stable_prefix(graph.source, 32);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

        int64_t deliveries = 0;
        int64_t early = 0;
        int64_t late = 0;
        transport_stats(sched.get(), &deliveries, &early, &late);
        GGML_ASSERT(deliveries == 1 && early == 32 && late == 32);
        GGML_ASSERT(!ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 0));
        GGML_ASSERT(!ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 0));

        ggml_set_stable_prefix(graph.source, 0);
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);
        transport_stats(sched.get(), &deliveries, &early, &late);
        GGML_ASSERT(deliveries == 2 && early == 32 && late == 96);

        ggml_set_stable_prefix(graph.source, 64);
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);
        transport_stats(sched.get(), &deliveries, &early, &late);
        GGML_ASSERT(deliveries == 3 && early == 96 && late == 96);
        GGML_ASSERT(cuda.context->event_wait_count > 0);
    }
    GGML_ASSERT(cuda.context->transfer_backend_count == 0);
}

// The ring must hold what the backend allocates for an entry, deliver every byte of it exactly once, and never wait on an event it has not recorded.
static void test_transport_entry_allocation() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);

    // ask for more room per entry than its data needs, the way a quantized tensor does
    cuda.context->alloc_size_pad = 32;

    const size_t nbytes = 128;
    auto graph = make_transport_graph_pair(cpu, nbytes);

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 4096));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
    ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());

    // one input goes early in part, the other whole
    ggml_set_stable_prefix(graph.sources[0], nbytes/2);
    ggml_set_stable_prefix(graph.sources[1], nbytes);

    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

    int64_t deliveries = 0;
    int64_t early = 0;
    int64_t late = 0;
    transport_stats(sched.get(), &deliveries, &early, &late);
    GGML_ASSERT(deliveries == 1 && early == (int64_t) (nbytes + nbytes/2) && late == (int64_t) (nbytes/2));

    std::vector<const dummy_backend_context::tensor_binding *> entries;
    for (const auto & b : cuda.context->bindings) {
        if (strncmp(b.tensor->name, "CUDA#", 5) == 0) {
            entries.push_back(&b);
        }
    }
    GGML_ASSERT(entries.size() == 2);

    for (const auto * e : entries) {
        // bound through the buffer, with the room the buffer type asks for
        GGML_ASSERT(e->size == ggml_nbytes(e->tensor) + cuda.context->alloc_size_pad);

        const char * base = (const char *) ggml_backend_buffer_get_base(e->buffer);
        GGML_ASSERT(e->data >= base);
        GGML_ASSERT(e->data + e->size <= base + ggml_backend_buffer_get_size(e->buffer));

        // and no entry, padding included, reaches into another one
        for (const auto * other : entries) {
            GGML_ASSERT(other == e || other->data + other->size <= e->data || other->data >= e->data + e->size);
        }

        // every byte of the entry is delivered once, in order, from the matching source offset
        std::vector<dummy_backend_context::tensor_delivery> parts;
        for (const auto & d : cuda.context->deliveries) {
            if (d.tensor == e->tensor) {
                parts.push_back(d);
            }
        }
        GGML_ASSERT(!parts.empty());
        std::sort(parts.begin(), parts.end(),
                [](const dummy_backend_context::tensor_delivery & a, const dummy_backend_context::tensor_delivery & b) {
                    return a.offset < b.offset;
                });
        const char * src = parts.front().src;
        GGML_ASSERT(src == (const char *) graph.sources[0]->data || src == (const char *) graph.sources[1]->data);
        size_t covered = 0;
        for (const auto & d : parts) {
            GGML_ASSERT(d.offset == covered);
            GGML_ASSERT(d.src == src + d.offset);
            covered += d.size;
        }
        GGML_ASSERT(covered == ggml_nbytes(e->tensor));
    }

    // nothing waits on an event before it is recorded
    const auto & steps = cuda.context->event_steps;
    GGML_ASSERT(!steps.empty());
    for (size_t i = 0; i < steps.size(); i++) {
        if (!steps[i].is_wait) {
            continue;
        }
        bool recorded = false;
        for (size_t j = 0; j < i && !recorded; j++) {
            recorded = !steps[j].is_wait && steps[j].event == steps[i].event;
        }
        GGML_ASSERT(recorded);
    }
}

// A window over several streams sits a fixed stride apart in one tensor, with cells between one
// stream's window and the next that the graph never reads. The delivery has to cover each stream's
// window from its own offset and leave those cells alone.
static void test_transport_multi_stream_ranges() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);

    const int64_t n_stream = 4;
    const int64_t n_row    = 8;    // rows of a stream the graph reads
    const int64_t kv_size  = 12;   // rows a stream holds, so 4 rows of every stream stay unread
    const int64_t n_embd   = 4;

    auto ctx = make_context();
    ggml_tensor * store = ggml_new_tensor_3d(ctx.ctx, GGML_TYPE_F32, n_embd, kv_size, n_stream);
    store->flags |= GGML_TENSOR_FLAG_TRANSPORT;

    // the same shape a host-resident KV window has: heads split across dims 0 and 1, rows on 2, streams on 3
    ggml_tensor * window = ggml_view_4d(ctx.ctx, store, n_embd/2, 2, n_row, n_stream,
            (size_t) (n_embd/2)*sizeof(float), store->nb[1], store->nb[2], 0);
    ggml_tensor * output = ggml_cont(ctx.ctx, window);
    ggml_build_forward_expand(ctx.graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, ggml_nbytes(store)));
    store->buffer = buffer.get();
    store->data   = ggml_backend_buffer_get_base(buffer.get());

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 1u << 20));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
    ggml_backend_sched_set_tensor_backend(sched.get(), output, cuda.handle.get());

    // half of every stream's window is stable, so each stream splits into an early and a late range
    const size_t row_bytes  = (size_t) n_embd*sizeof(float);
    const size_t used_bytes = (size_t) n_row*row_bytes;
    ggml_set_stable_prefix(store, used_bytes/2);

    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), ctx.graph) == GGML_STATUS_SUCCESS);

    std::vector<dummy_backend_context::tensor_delivery> parts = cuda.context->deliveries;
    GGML_ASSERT(!parts.empty());
    std::sort(parts.begin(), parts.end(),
            [](const dummy_backend_context::tensor_delivery & a, const dummy_backend_context::tensor_delivery & b) {
                return a.offset < b.offset;
            });

    // every stream's window is covered exactly once, from its own source offset, and the unread cells never move
    const size_t stride = (size_t) window->nb[3];
    size_t total = 0;
    for (const auto & d : parts) {
        GGML_ASSERT(d.offset/stride < (size_t) n_stream);
        GGML_ASSERT(d.offset%stride + d.size <= used_bytes);
        GGML_ASSERT(d.src == (const char *) window->data + d.offset);
        total += d.size;
    }
    GGML_ASSERT(total == (size_t) n_stream*used_bytes);

    for (int64_t st = 0; st < n_stream; st++) {
        size_t covered = 0;
        for (const auto & d : parts) {
            if (d.offset/stride == (size_t) st) {
                GGML_ASSERT(d.offset%stride == covered);
                covered += d.size;
            }
        }
        GGML_ASSERT(covered == used_bytes);
    }

    int64_t early = 0, late = 0;
    transport_stats(sched.get(), NULL, &early, &late);
    GGML_ASSERT(early == (int64_t) ((size_t) n_stream*used_bytes/2));
    GGML_ASSERT(late  == (int64_t) ((size_t) n_stream*used_bytes/2));
}

static void test_transport_empty_graph() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_context();

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.graph) == GGML_STATUS_SUCCESS);
}

static size_t transport_fallback_buffer_size(int depth) {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_context();

    ggml_tensor * weight = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * source = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * output = ggml_mul_mat(graph.ctx, weight, source);
    source->flags |= GGML_TENSOR_FLAG_TRANSPORT;
    ggml_set_stable_prefix(source, ggml_nbytes(source));
    ggml_build_forward_expand(graph.graph, output);

    ggml_backend_buffer_ptr weight_buffer(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, ggml_nbytes(weight)));
    ggml_backend_buffer_ptr source_buffer(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, ggml_nbytes(source)));
    weight->buffer = weight_buffer.get();
    weight->data   = ggml_backend_buffer_get_base(weight_buffer.get());
    source->buffer = source_buffer.get();
    source->data   = ggml_backend_buffer_get_base(source_buffer.get());

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), depth));
    ggml_backend_sched_set_tensor_backend(sched.get(), output, cuda.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.graph));
    return ggml_backend_sched_get_buffer_size(sched.get(), cuda.handle.get());
}

static void test_transport_fallback_keeps_allocator_plan() {
    const size_t ordered   = transport_fallback_buffer_size(0);
    const size_t pipelined = transport_fallback_buffer_size(1);
    GGML_ASSERT(ordered > 0 && pipelined == ordered);
}

static size_t transport_scale_buffer_size(int depth, size_t nbytes, size_t capacity, int64_t * deliveries, int * transfers) {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    cuda.context->capacity = capacity;

    auto graph = make_transport_graph(cpu, nbytes);
    ggml_set_stable_prefix(graph.source, nbytes);

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), depth));
    ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

    if (deliveries) {
        transport_stats(sched.get(), deliveries, nullptr, nullptr);
    }
    if (transfers) {
        *transfers = cuda.context->transfer_backend_count;
    }
    return ggml_backend_sched_get_buffer_size(sched.get(), cuda.handle.get());
}

// the ring is optional, so a device that cannot hold it next to the graph keeps the graph
static void test_transport_releases_ring_for_graph() {
    const size_t nbytes  = 256;
    const size_t ring    = 3*nbytes; // depth 1 plus the margin, one entry per slot
    const size_t ordered = transport_scale_buffer_size(0, nbytes, SIZE_MAX, nullptr, nullptr);
    GGML_ASSERT(ordered > 0);

    int64_t deliveries = -1;
    int transfers = -1;
    const size_t pipelined = transport_scale_buffer_size(1, nbytes, std::max(ordered, ring), &deliveries, &transfers);
    GGML_ASSERT(pipelined == ordered);
    GGML_ASSERT(deliveries == 0);
    GGML_ASSERT(transfers == 0);
}

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s(name, value) == 0);
#else
    GGML_ASSERT(setenv(name, value, 1) == 0);
#endif
}

static void restore_test_env(const char * name, bool had_value, const std::string & value) {
#ifdef _WIN32
    GGML_ASSERT(_putenv_s(name, had_value ? value.c_str() : "") == 0);
#else
    GGML_ASSERT(had_value ? setenv(name, value.c_str(), 1) == 0 : unsetenv(name) == 0);
#endif
}

static void test_transport_environment_is_fallback() {
    const char * depth_env = getenv("GGML_KV_PIPELINE_DEPTH");
    const char * budget_env = getenv("GGML_KV_PIPELINE_BUDGET_MIB");
    const bool had_depth = depth_env != nullptr;
    const bool had_budget = budget_env != nullptr;
    const std::string depth_old = depth_env ? depth_env : "";
    const std::string budget_old = budget_env ? budget_env : "";

    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };

    set_test_env("GGML_KV_PIPELINE_DEPTH", "4");
    set_test_env("GGML_KV_PIPELINE_BUDGET_MIB", "8");
    {
        auto graph = make_transport_graph(cpu, 64);
        ggml_set_stable_prefix(graph.source, 64);
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
        ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);
        int64_t deliveries = 0;
        transport_stats(sched.get(), &deliveries, nullptr, nullptr);
        GGML_ASSERT(deliveries == 1);
    }
    {
        auto graph = make_transport_graph(cpu, 64);
        ggml_set_stable_prefix(graph.source, 64);
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 0));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 0));
        ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);
        int64_t deliveries = -1;
        transport_stats(sched.get(), &deliveries, nullptr, nullptr);
        GGML_ASSERT(deliveries == 0);
    }

    set_test_env("GGML_KV_PIPELINE_DEPTH", "bad");
    set_test_env("GGML_KV_PIPELINE_BUDGET_MIB", "bad");
    {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 0));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 0));
    }

    restore_test_env("GGML_KV_PIPELINE_DEPTH", had_depth, depth_old);
    restore_test_env("GGML_KV_PIPELINE_BUDGET_MIB", had_budget, budget_old);
}

static void test_transport_depth_zero() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_transport_graph(cpu, 64);
    ggml_set_stable_prefix(graph.source, 64);

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 0));
    ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

    int64_t deliveries = -1;
    int64_t early = -1;
    int64_t late = -1;
    transport_stats(sched.get(), &deliveries, &early, &late);
    GGML_ASSERT(deliveries == 0 && early == 0 && late == 0);
    GGML_ASSERT(cuda.context->transfer_backend_count == 0);
}

static void test_transport_budget_recovers() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto large = make_transport_graph(cpu, 256);
    auto small = make_transport_graph(cpu, 64);
    ggml_set_stable_prefix(large.source, 256);
    ggml_set_stable_prefix(small.source, 64);

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_budget(sched.get(), 384));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));

    ggml_backend_sched_set_tensor_backend(sched.get(), large.output, cuda.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), large.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), large.ctx.graph) == GGML_STATUS_SUCCESS);
    int64_t deliveries = 0;
    transport_stats(sched.get(), &deliveries, nullptr, nullptr);
    GGML_ASSERT(deliveries == 0);

    ggml_backend_sched_reset(sched.get());
    ggml_backend_sched_set_tensor_backend(sched.get(), small.output, cuda.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), small.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), small.ctx.graph) == GGML_STATUS_SUCCESS);
    transport_stats(sched.get(), &deliveries, nullptr, nullptr);
    GGML_ASSERT(deliveries == 1);
}

static void test_transport_partial_backend_failure() {
    dummy_backend cuda_fail = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cuda_ok   = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu       = dummy_backend_init(SIZE_MAX, 8, true);
    cuda_fail.context->fail_event_init = true;

    auto graph = make_context();
    ggml_tensor * source_fail = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, 16);
    ggml_tensor * source_ok   = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, 16);
    ggml_tensor * output_fail = ggml_scale(graph.ctx, source_fail, 2.0f);
    ggml_tensor * output_ok   = ggml_scale(graph.ctx, source_ok, 2.0f);
    ggml_tensor * output      = ggml_add(graph.ctx, output_fail, output_ok);
    source_fail->flags |= GGML_TENSOR_FLAG_TRANSPORT;
    source_ok->flags   |= GGML_TENSOR_FLAG_TRANSPORT;
    ggml_set_stable_prefix(source_fail, 64);
    ggml_set_stable_prefix(source_ok, 64);
    ggml_build_forward_expand(graph.graph, output);

    ggml_backend_buffer_ptr buffer_fail(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, 64));
    ggml_backend_buffer_ptr buffer_ok(ggml_backend_buft_alloc_buffer(&cpu.buffer_type, 64));
    source_fail->buffer = buffer_fail.get();
    source_fail->data   = ggml_backend_buffer_get_base(buffer_fail.get());
    source_ok->buffer   = buffer_ok.get();
    source_ok->data     = ggml_backend_buffer_get_base(buffer_ok.get());

    ggml_backend_t backends[] = { cuda_fail.handle.get(), cuda_ok.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda_fail.buffer_type, &cuda_ok.buffer_type, &cpu.buffer_type };
    {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 3, 128, false, false));
        GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
        ggml_backend_sched_set_tensor_backend(sched.get(), output_fail, cuda_fail.handle.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), output_ok, cuda_ok.handle.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), output, cpu.handle.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.graph));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.graph) == GGML_STATUS_SUCCESS);

        int64_t deliveries = 0;
        transport_stats(sched.get(), &deliveries, nullptr, nullptr);
        GGML_ASSERT(deliveries == 1);
        GGML_ASSERT(cuda_fail.context->transfer_backend_count == 0);
        GGML_ASSERT(cuda_ok.context->transfer_backend_count == 1);
    }
    GGML_ASSERT(cuda_ok.context->transfer_backend_count == 0);
}

static void test_transport_excludes_meta() {
    dummy_backend meta = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_META, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_transport_graph(cpu, 64);
    ggml_set_stable_prefix(graph.source, 64);

    ggml_backend_t backends[] = { meta.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &meta.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
    ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, meta.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

    int64_t deliveries = -1;
    transport_stats(sched.get(), &deliveries, nullptr, nullptr);
    GGML_ASSERT(deliveries == 0);
    GGML_ASSERT(meta.context->transfer_backend_count == 0);
}

static void test_transport_requires_annotation() {
    dummy_backend cuda = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_transport_graph(cpu, 64);
    graph.source->flags &= ~GGML_TENSOR_FLAG_TRANSPORT;
    ggml_set_stable_prefix(graph.source, 64);

    ggml_backend_t backends[] = { cuda.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &cuda.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
    ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, cuda.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

    int64_t deliveries = -1;
    transport_stats(sched.get(), &deliveries, nullptr, nullptr);
    GGML_ASSERT(deliveries == 0);
    GGML_ASSERT(cuda.context->transfer_backend_count == 0);
}

static void test_transport_excludes_non_cuda() {
    dummy_backend sycl = dummy_backend_init(SIZE_MAX, 8, true, GGML_BACKEND_DEVICE_TYPE_GPU, "SYCL", false);
    dummy_backend cpu  = dummy_backend_init(SIZE_MAX, 8, true);
    auto graph = make_transport_graph(cpu, 64);
    ggml_set_stable_prefix(graph.source, 64);

    ggml_backend_t backends[] = { sycl.handle.get(), cpu.handle.get() };
    ggml_backend_buffer_type_t bufts[] = { &sycl.buffer_type, &cpu.buffer_type };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, 128, false, false));
    GGML_ASSERT(ggml_backend_sched_set_transport_pipeline_depth(sched.get(), 1));
    ggml_backend_sched_set_tensor_backend(sched.get(), graph.output, sycl.handle.get());
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph.ctx.graph));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph.ctx.graph) == GGML_STATUS_SUCCESS);

    int64_t deliveries = -1;
    transport_stats(sched.get(), &deliveries, nullptr, nullptr);
    GGML_ASSERT(deliveries == 0);
    GGML_ASSERT(sycl.context->transfer_backend_count == 0);
}

static void test_backend_graph_optimize(ggml_backend_t, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(graph->n_nodes == 3);
    params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[2]);
}

static bool graph_reuses_allocation(bool add_alloc_dep) {
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    x[2] = ggml_scale(ctx, x[1], 2.0f);
    x[3] = ggml_scale(ctx, x[2], 2.0f);

    ggml_set_output(x[3]);
    ggml_build_forward_expand(graph, x[3]);

    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    if (add_alloc_dep) {
        backend.handle->iface.graph_optimize = test_backend_graph_optimize;
    }

    ggml_backend_t             backend_ptr = backend.handle.get();
    ggml_backend_buffer_type_t buft        = &backend.buffer_type;
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(&backend_ptr, &buft, 1, 8, false, true));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    return x[1]->data == x[2]->data;
}

static void test_graph_optimize_alloc_dep() {
    GGML_ASSERT(graph_reuses_allocation(false));
    GGML_ASSERT(!graph_reuses_allocation(true));
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reallocation", test_reallocation);
    run("test_resizable_buffers_grow_shrink_grow", test_resizable_buffers_grow_shrink_grow);
    run("test_resizable_buffers_fail_closed_on_allocation_failure", test_resizable_buffers_fail_closed_on_allocation_failure);
    run("test_resizable_buffers_multi_entry_allocation_failure", test_resizable_buffers_multi_entry_allocation_failure);
    run("test_resizable_buffers_scheduler_allocation_failure", test_resizable_buffers_scheduler_allocation_failure);
    run("test_resizable_buffers_alias_duplicate_buffer_types", test_resizable_buffers_alias_duplicate_buffer_types);
    run("test_resizable_buffers_owner_borrower_maximum", test_resizable_buffers_owner_borrower_maximum);
    run("test_resizable_buffers_owner_borrower_compatible_placement", test_resizable_buffers_owner_borrower_compatible_placement);
    run("test_resizable_buffers_owner_borrower_allocation_failure", test_resizable_buffers_owner_borrower_allocation_failure);
    run("test_resizable_buffers_owner_borrower_scheduler_failure", test_resizable_buffers_owner_borrower_scheduler_failure);
    run("test_resizable_buffers_owner_borrower_teardown_order", test_resizable_buffers_owner_borrower_teardown_order);
    run("test_transport_prefix_and_configuration", test_transport_prefix_and_configuration);
    run("test_transport_entry_allocation", test_transport_entry_allocation);
    run("test_transport_multi_stream_ranges", test_transport_multi_stream_ranges);
    run("test_transport_empty_graph", test_transport_empty_graph);
    run("test_transport_fallback_keeps_allocator_plan", test_transport_fallback_keeps_allocator_plan);
    run("test_transport_releases_ring_for_graph", test_transport_releases_ring_for_graph);
    run("test_transport_environment_is_fallback", test_transport_environment_is_fallback);
    run("test_transport_depth_zero", test_transport_depth_zero);
    run("test_transport_budget_recovers", test_transport_budget_recovers);
    run("test_transport_partial_backend_failure", test_transport_partial_backend_failure);
    run("test_transport_excludes_meta", test_transport_excludes_meta);
    run("test_transport_requires_annotation", test_transport_requires_annotation);
    run("test_transport_excludes_non_cuda", test_transport_excludes_non_cuda);
    run("test_graph_optimize_alloc_dep", test_graph_optimize_alloc_dep);
    return 0;
}
