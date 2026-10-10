#include "ggml-backend-moe-certificates.h"
#include "moe-fidelity-config.h"
// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-moe.h"
#include "ggml-moe-source-program.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <crt_externs.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
extern "C" char ** environ;
#endif

// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    bool direct_dependencies[GGML_SCHED_MAX_BACKENDS];
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched_hybrid_region {
    uint32_t split_index;
    uint32_t copy_index;
    uint64_t source_graph_uid;
    uint64_t split_graph_uid;
    uint64_t allocator_generation;
    std::vector<ggml_backend_moe_cpu_prepared_region_v1_t> cpu;
    void * device;
    std::vector<std::pair<const ggml_tensor *, ggml_tensor>> witnesses;

    bool matches_binding(uint64_t source_uid, uint64_t split_uid, uint64_t generation, uint32_t copy) const {
        return source_graph_uid == source_uid && split_graph_uid == split_uid &&
            allocator_generation == generation && copy_index == copy;
    }

    bool matches(uint64_t source_uid, uint64_t split_uid, uint64_t generation, uint32_t copy) const {
        if (!matches_binding(source_uid, split_uid, generation, copy)) { return false; }
        for (const auto & witness : witnesses) {
            if (!ggml_moe_source_tensor_matches(*witness.first, witness.second)) { return false; }
        }
        return true;
    }
};

struct ggml_backend_sched_source_cpu {
    ggml_backend_moe_cpu_service_v1_t service = nullptr;
    const ggml_backend_moe_cpu_region_service_api_v1 * api = nullptr;
    struct resource_entry {
        const void * program;
        const void * session;
        ggml_backend_t backend;
        const void * shared_owner;
        uint64_t device_bytes;
        uint64_t pinned_bytes;
        uint64_t shared_device_bytes;
    };
    std::mutex resource_mutex;
    std::vector<resource_entry> resources;
    uint32_t max_programs = 0;
    uint64_t device_limit = 0;
    uint64_t pinned_limit = 0;

    static bool reserve_resources(void * opaque, const void * program, const void * session, ggml_backend_t backend, const void * shared_owner,
                                  uint64_t device_bytes, uint64_t pinned_bytes, uint64_t shared_device_bytes) try {
        auto * owner = static_cast<ggml_backend_sched_source_cpu *>(opaque);
        if (!owner || !program || !session || !backend || (shared_device_bytes && !shared_owner)) { return false; }
        std::lock_guard<std::mutex> lock(owner->resource_mutex);
        if (!device_bytes && !pinned_bytes && !shared_device_bytes) {
            auto retired = std::find_if(owner->resources.begin(), owner->resources.end(),
                [&](const auto & entry) { return entry.session == session; });
            if (retired != owner->resources.end()) {
                if (retired->program != program || retired->backend != backend) { return false; }
                owner->resources.erase(retired);
            }
            return true;
        }
        auto pending = owner->resources;
        auto found = std::find_if(pending.begin(), pending.end(),
            [&](const auto & entry) { return entry.session == session; });
        if (found != pending.end()) {
            if (found->program != program || found->backend != backend) { return false; }
            pending.erase(found);
        }
        if (device_bytes || pinned_bytes || shared_device_bytes) {
            pending.push_back({program, session, backend, shared_owner, device_bytes, pinned_bytes, shared_device_bytes});
        }
        struct program_usage {
            const void * program;
            uint64_t pinned_bytes = 0;
            std::unordered_map<ggml_backend_dev_t, uint64_t> devices;
        };
        std::vector<program_usage> programs;
        std::unordered_map<ggml_backend_dev_t, uint64_t> device_totals;
        std::unordered_map<const void *, std::pair<ggml_backend_dev_t, uint64_t>> shared_totals;
        uint64_t pinned_total = 0;
        for (const auto & entry : pending) {
            auto usage = std::find_if(programs.begin(), programs.end(),
                [&](const auto & item) { return item.program == entry.program; });
            if (usage == programs.end()) {
                if (programs.size() >= owner->max_programs) { return false; }
                programs.push_back({entry.program, 0, {}});
                usage = programs.end() - 1;
            }
            const auto device = ggml_backend_get_device(entry.backend);
            if (!device) { return false; }
            auto & total = device_totals[device];
            auto & program_device = usage->devices[device];
            if (entry.device_bytes > UINT64_MAX - total || entry.pinned_bytes > UINT64_MAX - pinned_total ||
                    entry.device_bytes > UINT64_MAX - program_device || entry.pinned_bytes > UINT64_MAX - usage->pinned_bytes) { return false; }
            total += entry.device_bytes;
            pinned_total += entry.pinned_bytes;
            program_device += entry.device_bytes;
            usage->pinned_bytes += entry.pinned_bytes;
            if (entry.shared_device_bytes) {
                auto & shared = shared_totals[entry.shared_owner];
                if (shared.first && shared.first != device) { return false; }
                shared.first = device;
                shared.second = std::max(shared.second, entry.shared_device_bytes);
            }
        }
        uint64_t pinned_largest = 0;
        std::unordered_map<ggml_backend_dev_t, uint64_t> device_largest, device_shared;
        for (const auto & usage : programs) {
            pinned_largest = std::max(pinned_largest, usage.pinned_bytes);
            for (const auto & item : usage.devices) { device_largest[item.first] = std::max(device_largest[item.first], item.second); }
        }
        if (!owner->max_programs || (!owner->pinned_limit && pinned_largest > UINT64_MAX / owner->max_programs)) { return false; }
        const auto pinned_limit = owner->pinned_limit ? owner->pinned_limit : pinned_largest * owner->max_programs;
        if (pinned_total > pinned_limit) { return false; }
        for (const auto & item : shared_totals) {
            auto & total = device_totals[item.second.first];
            auto & shared = device_shared[item.second.first];
            if (item.second.second > UINT64_MAX - total || item.second.second > UINT64_MAX - shared) { return false; }
            total += item.second.second;
            shared += item.second.second;
        }
        for (const auto & item : device_totals) {
            uint64_t limit = owner->device_limit;
            if (!limit) {
                const auto largest = device_largest[item.first], shared = device_shared[item.first];
                if (largest > (UINT64_MAX - shared) / owner->max_programs) { return false; }
                limit = largest * owner->max_programs + shared;
            }
            if (item.second > limit) { return false; }
        }
        owner->resources.swap(pending);
        return true;
    } catch (...) { return false; }
};

struct ggml_backend_sched_hybrid {
    ggml_backend_t backend = nullptr;
    ggml_backend_reg_t cpu_module = nullptr;
    ggml_backend_reg_t device_module = nullptr;
    const ggml_backend_moe_cpu_region_service_api_v1 * cpu_api = nullptr;
    const ggml_backend_moe_hybrid_api_v1 * device_api = nullptr;
    const ggml_backend_moe_source_core_api_v1 * source_api = nullptr;
    std::mutex source_mutex;
    ggml_backend_moe_cpu_service_v1_t cpu = nullptr;
    std::shared_ptr<ggml_backend_sched_source_cpu> source_cpu;
    void * device = nullptr;
    ggml_backend_moe_hybrid_config_v1 config = {};
    std::vector<std::vector<int32_t>> profile_ranks;
    std::vector<ggml_backend_moe_static_profile_v1> profiles;
    std::vector<std::vector<uint64_t>> statistics_counts;
    std::vector<std::vector<double>> statistics_score_storage;
    std::vector<const double *> statistics_scores;
    std::vector<ggml_backend_moe_source_statistics_v1> statistics;
    ggml_backend_moe_source_owner_v1 source_owner = {};
    uint32_t max_regions = 0;
    std::vector<ggml_backend_sched_hybrid_region> regions;
    std::vector<void *> dispatch;
    std::vector<std::unique_ptr<ggml_backend_sched_hybrid>> split_sessions;

    template <typename F> void each_session(F visit) {
        visit(this);
        for (auto & session : split_sessions) { visit(session.get()); }
    }

    ggml_backend_sched_hybrid * find_split(uint32_t split_index, uint64_t source_uid, uint64_t split_uid, uint64_t generation, uint32_t copy) {
        ggml_backend_sched_hybrid * found = nullptr;
        each_session([&](ggml_backend_sched_hybrid * session) {
            if (!session->regions.empty() && session->regions.front().split_index == split_index &&
                    session->regions.front().matches_binding(source_uid, split_uid, generation, copy)) { found = session; }
        });
        return found;
    }

    int32_t ensure_source(const std::shared_ptr<ggml_backend_sched_source_cpu> & shared = {}) {
        if (!source_api || (cpu && device)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK; }
        if (!cpu && shared) {
            if (!shared->service || shared->api != cpu_api) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
            source_cpu = shared; cpu = shared->service;
        }
        if (!cpu) {
            source_cpu = std::make_shared<ggml_backend_sched_source_cpu>();
            source_cpu->api = cpu_api;
            source_cpu->max_programs = config.max_source_programs;
            source_cpu->device_limit = config.device_bytes;
            source_cpu->pinned_limit = config.pinned_bytes;
            ggml_backend_moe_cpu_service_config_v1 cpu_config = {};
            cpu_config.struct_size = sizeof(cpu_config); cpu_config.abi_version = 1;
            cpu_config.source_owner = &source_owner; cpu_config.n_threads = config.n_threads;
            cpu_config.n_lanes = 1; cpu_config.max_regions = config.max_prepared_regions;
            cpu_config.flags = config.cpu_flags; cpu_config.prepared_payload_limit = config.cpu_bytes;
            const auto status = cpu_api->create(&cpu_config, &cpu);
            source_cpu->service = cpu;
            if (status) { return status; }
        }
        if (config.max_source_programs) {
            config.resource_context = source_cpu.get();
            config.reserve_resources = ggml_backend_sched_source_cpu::reserve_resources;
        }
        return device ? GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK : source_api->create(&config, &device);
    }

    int32_t release_source_region(ggml_backend_sched_hybrid_region & region) {
        if (region.device) {
            const auto status = source_api->release_region(device, &region.device);
            if (status) { return status; }
        }
        for (auto & bucket : region.cpu) {
            if (bucket && cpu_api->destroy_region(cpu, &bucket)) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
        }
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    }

    int32_t clear_source(bool keep_cpu = false) {
        std::lock_guard<std::mutex> lock(source_mutex);
        int32_t retired = GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
        if (device) {
            const auto closed = source_api->close(device);
            const auto drained = source_api->drain(device);
            retired = closed ? closed : drained;
        }
        for (auto & session : split_sessions) {
            const auto status = session->clear_source(keep_cpu);
            if (!retired) { retired = status; }
        }
        if (retired) { return retired; }
        split_sessions.clear();
        for (auto & region : regions) {
            const auto status = release_source_region(region);
            if (status) { return status; }
        }
        if (device) {
            const auto status = source_api->release(&device);
            if (status) { return status; }
        }
        if (cpu && !keep_cpu) {
            if (!source_cpu || source_cpu.use_count() == 1) {
                if (cpu_api->close(cpu) || cpu_api->drain(cpu) || cpu_api->destroy(&cpu)) {
                    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED;
                }
            }
            cpu = nullptr;
            source_cpu.reset();
        }
        regions.clear(); dispatch.clear();
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    }

    void release(ggml_backend_sched_hybrid_region & region) {
        if (source_api) {
            if (region.device) {
                GGML_ASSERT(source_api->close(device) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                GGML_ASSERT(source_api->drain(device) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            }
            GGML_ASSERT(release_source_region(region) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            return;
        }
        if (region.device != nullptr) {
            device_api->destroy_region(device, region.device);
            region.device = nullptr;
        }
        for (auto & bucket : region.cpu) {
            if (bucket != 0) {
                GGML_ASSERT(cpu_api->destroy_region(cpu, &bucket) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            }
        }
    }

    void clear() {
        if (source_api) {
            GGML_ASSERT(clear_source(true) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
            return;
        }
        if (device != nullptr) {
            device_api->quiesce(device);
        }
        if (backend != nullptr) {
            ggml_backend_synchronize(backend);
        }
        for (auto & region : regions) {
            release(region);
        }
        regions.clear();
        dispatch.clear();
    }

    ~ggml_backend_sched_hybrid() {
        if (source_api) {
            GGML_ASSERT(clear_source() == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        } else {
            clear();
        }
        if (device != nullptr) {
            GGML_ASSERT(!source_api);
            device_api->destroy(device);
        }
        if (cpu != nullptr) {
            GGML_ASSERT(cpu_api->close(cpu) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            GGML_ASSERT(cpu_api->drain(cpu) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
            GGML_ASSERT(cpu_api->destroy(&cpu) == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK);
        }
        if (device_module != nullptr) {
            GGML_ASSERT(config.module_release(device_module));
        }
        if (cpu_module != nullptr) {
            GGML_ASSERT(config.module_release(cpu_module));
        }
    }
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;
    struct ggml_cgraph * source_graph;
    uint64_t source_graph_uid;
    uint64_t source_retirement_epoch;
    uint64_t source_preparation_epoch;
    uint64_t source_preparation_graph_uid;
    bool source_retirement_failed;
    uint64_t source_program_token;
    uint64_t source_program_epoch;
    uint64_t source_program_generation;
    uint64_t source_program_shrink_generation;
    uint64_t source_program_source_uid;
    uint64_t source_program_split_uid;
    int source_program_copy;
    int source_program_backend;
    bool source_program_dependencies[GGML_SCHED_MAX_BACKENDS];
    ggml_graph_execution_certificate source_program_certificate;
    ggml_backend_sched_hybrid * source_program_hybrid;
    void * source_program;
    ggml_backend_sched_hybrid * hybrid;
    ggml_backend_sched_hybrid * hybrids[GGML_SCHED_MAX_BACKENDS];
    int n_hybrids;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static int32_t ggml_backend_sched_moe_hybrid_create(
        const ggml_backend_moe_hybrid_config_v1 * config,
        const std::shared_ptr<ggml_backend_sched_source_cpu> & shared,
        std::unique_ptr<ggml_backend_sched_hybrid> & output) try {
    auto state = std::make_unique<ggml_backend_sched_hybrid>();
    state->backend = config->backend;
    state->config = *config;
    state->profile_ranks.resize(config->n_profiles);
    state->profiles.reserve(config->n_profiles);
    uint64_t profile_count = 0;
    for (uint32_t i = 0; i < config->n_profiles; ++i) {
        const auto & profile = config->profiles[i];
        if (!profile.down || !profile.experts || !profile.n_experts || profile.n_experts > 65536) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        profile_count += profile.n_experts;
        if (profile_count > (1u << 22)) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
        auto & ranks = state->profile_ranks[i];
        ranks.assign(profile.experts, profile.experts + profile.n_experts);
        state->profiles.push_back({profile.down, ranks.data(), profile.n_experts});
    }
    state->config.profiles = state->profiles.empty() ? nullptr : state->profiles.data();
    if (!ggml_moe_source_scores_valid(config->statistics, config->statistics_scores, config->n_statistics)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    state->statistics_score_storage.resize(config->statistics_scores ? config->n_statistics : 0);
    state->statistics_scores.resize(state->statistics_score_storage.size());
    state->statistics_counts.resize(config->n_statistics);
    state->statistics.reserve(config->n_statistics);
    for (uint32_t i = 0; i < config->n_statistics; ++i) {
        const auto & source = config->statistics[i];
        auto & counts = state->statistics_counts[i];
        counts.assign(source.counts, source.counts + source.n_experts);
        auto copy = source;
        copy.counts = counts.data();
        if (config->statistics_scores) {
            auto & scores = state->statistics_score_storage[i];
            scores.assign(config->statistics_scores[i], config->statistics_scores[i] + source.n_experts);
            state->statistics_scores[i] = scores.data();
        }
        state->statistics.push_back(copy);
    }
    state->config.statistics = state->statistics.empty() ? nullptr : state->statistics.data();
    state->config.statistics_scores = state->statistics_scores.empty() ? nullptr : state->statistics_scores.data();
    state->source_owner = *config->source_owner;
    state->config.source_owner = &state->source_owner;
    const auto cpu_module = config->cpu_module_acquire();
    const auto device_module = ggml_backend_dev_backend_reg(ggml_backend_get_device(config->backend));
    if (cpu_module == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    state->cpu_module = cpu_module;
    if (!config->module_retain(device_module)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    state->device_module = device_module;
    const char * cpu_proc_name = config->executor == GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY ?
        GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME : GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME;
    const char * device_proc_name = config->executor == GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY ?
        GGML_BACKEND_MOE_HYBRID_FIDELITY_V1_PROC_NAME : GGML_BACKEND_MOE_HYBRID_V1_PROC_NAME;
    const auto cpu_proc = reinterpret_cast<ggml_backend_moe_cpu_region_service_v1_t>(
        ggml_backend_reg_get_proc_address(cpu_module, cpu_proc_name));
    const auto device_proc = reinterpret_cast<ggml_backend_moe_hybrid_v1_t>(
        ggml_backend_reg_get_proc_address(device_module, device_proc_name));
    const bool source_core = config->executor == GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY &&
        ggml_moe_fidelity_selection().source_pool;
    const auto source_proc = source_core ? reinterpret_cast<ggml_backend_moe_source_core_v1_t>(
        ggml_backend_reg_get_proc_address(device_module, GGML_BACKEND_MOE_SOURCE_CORE_V1_PROC_NAME)) : nullptr;
    if (cpu_proc == nullptr || (source_core ? source_proc == nullptr : device_proc == nullptr)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    state->cpu_api = cpu_proc();
    state->device_api = source_core ? nullptr : device_proc();
    state->source_api = source_core ? source_proc() : nullptr;
    const auto * source = state->source_api;
    const auto * device = state->device_api;
    const bool provider_valid = source_core ?
        source && source->struct_size == sizeof(*source) && source->abi_version == 1 &&
            source->create && source->prepare && source->compute && source->close && source->drain &&
            source->release_region && source->release && source->state && source->preflight :
        device && device->struct_size == sizeof(*device) && device->abi_version == 1 &&
            device->create && device->prepare && device->compute && device->destroy_region &&
            device->destroy && device->state && device->quiesce;
    if (state->cpu_api == nullptr || state->cpu_api->struct_size != sizeof(*state->cpu_api) ||
            state->cpu_api->abi_version != 1 || !provider_valid ||
            state->cpu_api->create == nullptr || state->cpu_api->prepare == nullptr || state->cpu_api->execute == nullptr ||
            state->cpu_api->destroy_region == nullptr || state->cpu_api->close == nullptr ||
            state->cpu_api->drain == nullptr || state->cpu_api->destroy == nullptr ||
            state->cpu_api->cancel == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    ggml_backend_moe_cpu_service_config_v1 cpu_config = {};
    cpu_config.struct_size = sizeof(cpu_config);
    cpu_config.abi_version = 1;
    cpu_config.source_owner = config->source_owner;
    cpu_config.n_threads = config->n_threads;
    cpu_config.n_lanes = 1;
    cpu_config.max_regions = config->max_prepared_regions;
    cpu_config.flags = config->cpu_flags;
    cpu_config.prepared_payload_limit = config->cpu_bytes;
    int32_t status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    if (shared) {
        if (!source_core || !shared->service || shared->api != state->cpu_api) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        state->source_cpu = shared;
        state->cpu = shared->service;
    } else {
        if (source_core) {
            state->source_cpu = std::make_shared<ggml_backend_sched_source_cpu>();
            state->source_cpu->api = state->cpu_api;
            state->source_cpu->max_programs = config->max_source_programs;
            state->source_cpu->device_limit = config->device_bytes;
            state->source_cpu->pinned_limit = config->pinned_bytes;
        }
        status = state->cpu_api->create(&cpu_config, &state->cpu);
        if (state->source_cpu) { state->source_cpu->service = state->cpu; }
    }
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return status;
    }
    if (source_core && config->max_source_programs) {
        state->config.resource_context = state->source_cpu.get();
        state->config.reserve_resources = ggml_backend_sched_source_cpu::reserve_resources;
    }
    status = source_core ? state->source_api->create(&state->config, &state->device) : state->device_api->create(config, &state->device);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return status;
    }
    state->max_regions = config->max_regions;
    state->regions.reserve(config->max_regions);
    state->dispatch.reserve(config->max_regions);
    output = std::move(state);
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t ggml_backend_sched_moe_hybrid_configure_impl_v1(
        ggml_backend_sched_t sched, const ggml_backend_moe_hybrid_config_v1 * config,
        const std::shared_ptr<ggml_backend_sched_source_cpu> & shared_cpu = {}) try {
    if (sched == nullptr || config == nullptr || config->struct_size != sizeof(*config) ||
            config->backend == nullptr || config->source_owner == nullptr || config->n_threads == 0 ||
            config->source_owner->struct_size != sizeof(*config->source_owner) ||
            config->source_owner->abi_version != GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION ||
            config->cpu_module_acquire == nullptr || config->module_retain == nullptr || config->module_release == nullptr ||
            config->max_regions == 0 || config->max_prepared_regions < config->max_regions || config->max_source_programs > 64 ||
            (config->max_source_programs && (config->executor != GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY || !ggml_moe_fidelity_selection().source_pool)) ||
            config->admission_quota > config->gpu_miss_quota ||
            config->demand_admission > 1 || config->resident_batch > 1 ||
            config->executor > GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY ||
            config->profile_adaptation > 2 || (config->profile_adaptation && !config->n_profiles && !config->n_statistics) ||
            (config->n_profiles && config->n_statistics) || !ggml_moe_source_scores_valid(config->statistics, config->statistics_scores, config->n_statistics) ||
            config->n_profiles > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS ||
            (config->n_profiles && !config->profiles) ||
            ((config->n_profiles || config->n_statistics) && (config->executor != GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY ||
                !ggml_moe_fidelity_selection().source_pool)) ||
            sched->n_copies != 1 || sched->callback_eval != nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    int backend_id = -1;
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sched->backends[i] == config->backend) {
            if (backend_id >= 0) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
            backend_id = i;
        }
    }
    if (backend_id < 0) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    auto * existing = sched->hybrids[backend_id];
    auto shared = shared_cpu;
    if (!shared) {
        for (int i = 0; i < sched->n_backends; ++i) {
            if (sched->hybrids[i] && sched->hybrids[i]->source_cpu) { shared = sched->hybrids[i]->source_cpu; break; }
        }
    }
    if (sched->hybrid != nullptr) {
        if (!existing && !sched->hybrid->source_api) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        const auto * common = existing ? existing : sched->hybrid;
        const auto & saved = common->config;
        const auto & owner = common->source_owner;
        bool profiles_match = saved.n_profiles == config->n_profiles;
        for (uint32_t i = 0; profiles_match && i < config->n_profiles; ++i) {
            const auto & a = saved.profiles[i];
            const auto & b = config->profiles[i];
            profiles_match = a.down == b.down && a.n_experts == b.n_experts && b.experts &&
                !memcmp(a.experts, b.experts, size_t(a.n_experts) * sizeof(int32_t));
        }
        bool statistics_match = saved.n_statistics == config->n_statistics && bool(saved.statistics_scores) == bool(config->statistics_scores);
        for (uint32_t i = 0; statistics_match && i < config->n_statistics; ++i) {
            const auto & a = saved.statistics[i];
            const auto & b = config->statistics[i];
            statistics_match = a.tensor == b.tensor && a.domain == b.domain && a.observations == b.observations && a.n_experts == b.n_experts &&
                !memcmp(a.counts, b.counts, size_t(a.n_experts) * sizeof(uint64_t));
            if (statistics_match && saved.statistics_scores) {
                statistics_match = config->statistics_scores[i] && !memcmp(saved.statistics_scores[i], config->statistics_scores[i], size_t(a.n_experts) * sizeof(double));
            }
        }
        const bool matches = (!existing || saved.backend == config->backend) && saved.n_threads == config->n_threads &&
            (common->source_api ? config->max_regions <= saved.max_regions : config->max_regions == saved.max_regions) &&
            saved.cpu_flags == config->cpu_flags &&
            (common->source_api ? config->max_prepared_regions <= saved.max_prepared_regions : config->max_prepared_regions == saved.max_prepared_regions) &&
            saved.gpu_miss_quota == config->gpu_miss_quota && saved.admission_quota == config->admission_quota &&
            saved.demand_admission == config->demand_admission &&
            saved.resident_batch == config->resident_batch && saved.profile_adaptation == config->profile_adaptation &&
            saved.executor == config->executor && saved.max_source_programs == config->max_source_programs &&
            profiles_match && statistics_match &&
            saved.cpu_bytes == config->cpu_bytes && saved.device_bytes == config->device_bytes &&
            saved.pinned_bytes == config->pinned_bytes && owner.owner == config->source_owner->owner &&
            saved.cpu_module_acquire == config->cpu_module_acquire && saved.module_retain == config->module_retain &&
            saved.module_release == config->module_release &&
            owner.generation == config->source_owner->generation && owner.retain == config->source_owner->retain &&
            owner.release == config->source_owner->release && owner.validate_span == config->source_owner->validate_span &&
            owner.flags == config->source_owner->flags && owner.reserved32 == config->source_owner->reserved32 &&
            owner.reserved[0] == config->source_owner->reserved[0] && owner.reserved[1] == config->source_owner->reserved[1];
        if (!matches) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        if (existing) { return existing->ensure_source(shared); }
    }
    std::unique_ptr<ggml_backend_sched_hybrid> state;
    auto scoped_config = *config;
    scoped_config.resource_program = sched;
    const auto status = ggml_backend_sched_moe_hybrid_create(&scoped_config, shared, state);
    if (status) { return status; }
    sched->hybrids[backend_id] = state.release();
    if (!sched->hybrid) { sched->hybrid = sched->hybrids[backend_id]; }
    ++sched->n_hybrids;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) {
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
}

int32_t ggml_backend_sched_moe_hybrid_configure_v1(
        ggml_backend_sched_t sched, const ggml_backend_moe_hybrid_config_v1 * config) {
    if (sched) { sched->source_program_token = 0; }
    return ggml_backend_sched_moe_hybrid_configure_impl_v1(sched, config);
}

int32_t ggml_backend_sched_moe_source_clone_v1(ggml_backend_sched_t sched, ggml_backend_sched_t * output) {
    if (!sched || !output || *output || !sched->hybrid || !sched->hybrid->source_api ||
            !sched->hybrid->source_cpu || sched->n_copies != 1 || sched->callback_eval) {
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID;
    }
    auto clone = ggml_backend_sched_new(sched->backends, sched->bufts, sched->n_backends,
        sched->hash_set.size, false, sched->op_offload);
    if (!ggml_gallocr_share_resizable_plan(clone->galloc, sched->galloc)) {
        ggml_backend_sched_free(clone);
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED;
    }
    ggml_backend_sched_set_copy_callback(clone, sched->callback_copy, sched->callback_copy_user_data);
    for (int i = 0; i < sched->n_backends; ++i) {
        const auto * entry = sched->hybrids[i];
        if (!entry) { continue; }
        const auto status = ggml_backend_sched_moe_hybrid_configure_impl_v1(clone, &entry->config, entry->source_cpu);
        if (status) { ggml_backend_sched_free(clone); return status; }
    }
    *output = clone;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

static bool ggml_backend_moe_hybrid_bucket_matches(
        const ggml_backend_moe_cpu_region_query_v1 & query,
        const ggml_backend_moe_cpu_region_query_v1 & reference, uint32_t routes_per_row, uint32_t bucket_rows,
        bool original_body = false) {
    if (query.n_dynamic_inputs < 2 || query.n_dynamic_inputs != reference.n_dynamic_inputs ||
            uint64_t(query.n_dynamic_inputs) + query.n_sources + query.n_body_nodes > INT_MAX ||
            (!original_body && query.n_dynamic_inputs != 2) || query.dynamic_inputs == nullptr ||
            reference.dynamic_inputs == nullptr || query.n_sources == 0 ||
            query.n_sources != reference.n_sources || query.sources == nullptr || reference.sources == nullptr ||
            query.n_body_nodes == 0 || query.n_body_nodes != reference.n_body_nodes ||
            query.body_nodes == nullptr || reference.body_nodes == nullptr || !query.n_live_outputs ||
            (!original_body && query.n_live_outputs != 1) ||
            query.live_outputs == nullptr || (!original_body && query.live_outputs[0] != query.body_nodes[query.n_body_nodes - 1]) ||
            query.activation != query.dynamic_inputs[0] || query.ids != query.dynamic_inputs[1]) {
        return false;
    }
    const auto role = [](const ggml_backend_moe_cpu_region_query_v1 & graph, const ggml_tensor * tensor) {
        if (tensor == nullptr) { return -1; }
        for (uint32_t i = 0; i < graph.n_dynamic_inputs; ++i) {
            if (tensor == graph.dynamic_inputs[i]) { return int(i); }
        }
        for (uint32_t i = 0; i < graph.n_sources; ++i) {
            if (tensor == graph.sources[i].tensor) { return int(graph.n_dynamic_inputs + i); }
        }
        for (uint32_t i = 0; i < graph.n_body_nodes; ++i) {
            if (tensor == graph.body_nodes[i]) { return int(graph.n_dynamic_inputs + graph.n_sources + i); }
        }
        return -2;
    };
    for (uint32_t i = 0; i < query.n_sources; ++i) {
        const auto & a = query.sources[i];
        const auto & b = reference.sources[i];
        if (a.tensor == nullptr || b.tensor == nullptr || a.witness != b.witness || a.data != b.data ||
                a.bytes != b.bytes || a.expert_stride != b.expert_stride || a.generation != b.generation ||
                a.tensor->type != b.tensor->type || memcmp(a.tensor->ne, b.tensor->ne, sizeof(a.tensor->ne)) != 0 ||
                memcmp(a.tensor->nb, b.tensor->nb, sizeof(a.tensor->nb)) != 0) {
            return false;
        }
    }
    if (original_body && (query.n_live_outputs != reference.n_live_outputs || !reference.live_outputs)) { return false; }
    for (uint32_t i = 0; original_body && i < query.n_live_outputs; ++i) {
        const int index = role(query, query.live_outputs[i]);
        if (index < int(query.n_dynamic_inputs + query.n_sources) || index != role(reference, reference.live_outputs[i])) { return false; }
    }
    for (uint32_t i = 0; i < query.n_dynamic_inputs + query.n_body_nodes; ++i) {
        const bool dynamic = i < query.n_dynamic_inputs;
        const auto * a = dynamic ? query.dynamic_inputs[i] : query.body_nodes[i - query.n_dynamic_inputs];
        const auto * b = dynamic ? reference.dynamic_inputs[i] : reference.body_nodes[i - query.n_dynamic_inputs];
        if (!a || !b || a->type != b->type) { return false; }
        if (original_body) {
            if (memcmp(a->ne, b->ne, sizeof(a->ne)) || memcmp(a->nb, b->nb, sizeof(a->nb))) { return false; }
        } else if (a->ne[0] <= 0 || uint64_t(a->ne[0]) > SIZE_MAX / sizeof(float) ||
                (i < 2 && a->type != (i == 0 ? GGML_TYPE_F32 : GGML_TYPE_I32)) ||
                a->type != b->type || a->ne[0] != (i == 1 ? routes_per_row : b->ne[0]) ||
                a->ne[1] != (i == 0 ? 1 : i == 1 ? bucket_rows : routes_per_row) ||
                a->ne[2] != (i == 1 ? 1 : bucket_rows) || a->ne[3] != 1 || a->nb[0] != b->nb[0] ||
                a->nb[1] != (i < 2 ? sizeof(float) * a->ne[0] : b->nb[1]) ||
                a->nb[1] > SIZE_MAX / uint64_t(a->ne[1]) || a->nb[2] != a->nb[1] * a->ne[1] ||
                a->nb[2] > SIZE_MAX / uint64_t(a->ne[2]) || a->nb[3] != a->nb[2] * a->ne[2]) {
            return false;
        }
        if (dynamic) { continue; }
        if (a->op != b->op || a->flags != b->flags || a->view_offs != b->view_offs ||
                memcmp(a->op_params, b->op_params, sizeof(a->op_params)) != 0 ||
                role(query, a->view_src) < -1 || role(query, a->view_src) != role(reference, b->view_src)) {
            return false;
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            if (role(query, a->src[source]) < -1 || role(query, a->src[source]) != role(reference, b->src[source])) {
                return false;
            }
        }
    }
    return true;
}

int32_t ggml_backend_moe_hybrid_get_geometry_v1(
        const ggml_tensor * activation, const ggml_tensor * ids, const ggml_tensor * output,
        uint32_t expert_count, ggml_backend_moe_hybrid_geometry_v1 * geometry) {
    if (activation == nullptr || ids == nullptr || output == nullptr || geometry == nullptr ||
            expert_count == 0 || expert_count > INT32_MAX || ids->type != GGML_TYPE_I32 ||
            ids->ne[0] <= 0 || uint64_t(ids->ne[0]) > expert_count || ids->ne[1] <= 0 ||
            uint64_t(ids->ne[1]) > UINT32_MAX / uint64_t(ids->ne[0]) || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->nb[0] != sizeof(int32_t) || ids->nb[1] < uint64_t(ids->ne[0]) * sizeof(int32_t) ||
            ids->nb[1] % sizeof(int32_t) != 0 || uint64_t(ids->ne[1]) > SIZE_MAX / ids->nb[1] ||
            activation->type != GGML_TYPE_F32 || activation->ne[0] <= 0 ||
            activation->ne[1] != 1 || activation->ne[2] != ids->ne[1] || activation->ne[3] != 1 ||
            uint64_t(activation->ne[0]) > SIZE_MAX / sizeof(float) / uint64_t(ids->ne[1]) ||
            output->type != GGML_TYPE_F32 || output->ne[0] <= 0 ||
            output->ne[1] != ids->ne[0] || output->ne[2] != ids->ne[1] || output->ne[3] != 1 ||
            uint64_t(output->ne[0]) > SIZE_MAX / sizeof(float) / uint64_t(ids->ne[0]) / uint64_t(ids->ne[1]) ||
            !ggml_is_contiguous(activation) || !ggml_is_contiguous(output)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    const uint32_t route_capacity = uint32_t(ids->ne[0]) * uint32_t(ids->ne[1]);
    *geometry = {uint32_t(ids->ne[1]), uint32_t(ids->ne[0]), route_capacity,
        expert_count, std::min(expert_count, route_capacity)};
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

int32_t ggml_backend_moe_hybrid_validate_buckets_v1(const ggml_backend_moe_hybrid_region_v1 * region) {
    if (region && region->struct_size == sizeof(*region) && region->query &&
            region->query->flags == GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) {
        const auto & q = *region->query;
        if (q.struct_size != sizeof(q) || !q.graph || !q.graph->nodes || q.graph->n_nodes != 1 || q.graph->size < 1 ||
                !q.graph_uid || q.graph->uid != q.graph_uid || !q.graph_generation ||
                q.graph_generation != region->allocator_generation || !q.source_generation ||
                !q.body_nodes || q.n_body_nodes != 1 || !q.body_nodes[0] || q.graph->nodes[0] != q.body_nodes[0] ||
                !q.dynamic_inputs || q.n_dynamic_inputs != 2 || q.dynamic_inputs[0] != q.activation || q.dynamic_inputs[1] != q.ids ||
                !q.live_outputs || q.n_live_outputs != 1 || q.live_outputs[0] != q.body_nodes[0] ||
                !q.sources || q.n_sources != 1 || !q.activation || !q.ids || !q.sources[0].tensor ||
                region->n_cpu_queries != 0 || region->cpu_queries != nullptr || region->n_cpu_batch_queries != 1 ||
                !region->cpu_batch_queries || region->cpu_batch_queries[0] != region->query || region->body_query != region->query ||
                region->first_node != region->last_node || !region->source_graph_uid || !region->split_graph_uid || !region->owner_generation ||
                !region->output || region->output->op != GGML_OP_MUL_MAT_ID || region->output->view_src ||
                region->output->src[0] != region->down || region->output->src[1] != region->activation || region->output->src[2] != region->ids ||
                !region->down || !region->activation || !region->ids || !region->down->data || region->down->ne[2] <= 0 ||
                uint64_t(region->down->ne[2]) > INT32_MAX || region->ids->ne[0] <= 0 || region->ids->ne[1] <= 0 ||
                uint64_t(region->ids->ne[0]) > UINT32_MAX || uint64_t(region->ids->ne[1]) > UINT32_MAX / uint64_t(region->ids->ne[0])) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        const auto * projection = q.body_nodes[0];
        const ggml_tensor * copies[] = {q.sources[0].tensor, q.activation, q.ids, projection};
        const ggml_tensor * originals[] = {region->down, region->activation, region->ids, region->output};
        for (uint32_t i = 0; i < 4; ++i) {
            if (copies[i]->type != originals[i]->type || memcmp(copies[i]->ne, originals[i]->ne, sizeof(copies[i]->ne)) ||
                    memcmp(copies[i]->nb, originals[i]->nb, sizeof(copies[i]->nb))) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        const auto & source = q.sources[0];
        const uint32_t rows = region->ids->ne[1], routes = region->ids->ne[0], experts = region->down->ne[2];
        const ggml_backend_moe_hybrid_geometry_v1 geometry{rows, routes, rows * routes, experts, std::min(experts, rows * routes)};
        if (projection->op != GGML_OP_MUL_MAT_ID || projection->src[0] != copies[0] || projection->src[1] != q.activation ||
                projection->src[2] != q.ids || memcmp(projection->op_params, region->output->op_params, sizeof(projection->op_params)) ||
                source.witness != region->down || source.data != region->down->data || source.bytes != ggml_nbytes(region->down) ||
                source.expert_stride != region->down->nb[2] || source.generation != q.source_generation ||
                memcmp(&geometry, &region->geometry, sizeof(geometry)) || q.bucket_rows != rows || q.routes_per_row != routes ||
                q.source_row_capacity != rows || q.scatter_capacity != geometry.route_capacity || q.n_lanes != 1 || !q.n_threads) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t i = 3; i < GGML_MAX_SRC; ++i) {
            if (projection->src[i] || region->output->src[i]) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        }
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    }
    if (region == nullptr || region->struct_size != sizeof(*region) || region->query == nullptr ||
            region->ids == nullptr || region->cpu_queries == nullptr || region->n_cpu_queries == 0 ||
            region->n_cpu_queries != region->ids->ne[0] || region->cpu_queries[0] != region->query ||
            (region->cpu_batch_queries == nullptr) != (region->n_cpu_batch_queries == 0)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    ggml_backend_moe_hybrid_geometry_v1 geometry = {};
    if (ggml_backend_moe_hybrid_get_geometry_v1(region->activation, region->ids, region->output,
            region->geometry.expert_count, &geometry) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ||
            memcmp(&geometry, &region->geometry, sizeof(geometry)) != 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < region->n_cpu_queries; ++i) {
        const auto * query = region->cpu_queries[i];
        if (query == nullptr || query->struct_size != sizeof(*query) || query->bucket_rows != 1 ||
                query->routes_per_row != i + 1 || query->source_row_capacity != geometry.row_capacity || query->n_lanes != 1 ||
                query->scatter_capacity != geometry.route_capacity ||
                query->graph_generation != region->query->graph_generation ||
                query->source_generation != region->query->source_generation || query->n_threads != region->query->n_threads ||
                query->flags != GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_NONE ||
                !ggml_backend_moe_hybrid_bucket_matches(*query, *region->query, i + 1, 1)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t node = 0; node < query->n_body_nodes; ++node) {
            const auto * tensor = query->body_nodes[node];
            if (tensor->op == GGML_OP_MUL_MAT_ID &&
                    (tensor->src[0] == nullptr || tensor->src[0]->ne[2] != geometry.expert_count)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
    }
    if (region->n_cpu_batch_queries > 1) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < region->n_cpu_batch_queries; ++i) {
        const auto * query = region->cpu_batch_queries[i];
        if (query == nullptr || query->struct_size != sizeof(*query) ||
                query->flags != GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES ||
                query->bucket_rows != geometry.route_capacity || query->routes_per_row != 1 ||
                query->source_row_capacity != geometry.row_capacity || query->n_lanes != 1 ||
                query->scatter_capacity != geometry.route_capacity ||
                query->graph_generation != region->query->graph_generation ||
                query->source_generation != region->query->source_generation ||
                query->n_threads != region->query->n_threads ||
                !ggml_backend_moe_hybrid_bucket_matches(*query, *region->query, 1, geometry.route_capacity)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t node = 0; node < query->n_body_nodes; ++node) {
            const auto * tensor = query->body_nodes[node];
            if (tensor->op == GGML_OP_MUL_MAT_ID &&
                    (tensor->src[0] == nullptr || tensor->src[0]->ne[2] != geometry.expert_count)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

int32_t ggml_backend_moe_hybrid_bind_cpu_row_v1(
        const ggml_backend_moe_hybrid_region_v1 * region, const ggml_backend_moe_hybrid_binding_v1 * routes,
        uint64_t epoch, uint32_t source_row, const uint32_t * selected_routes, uint32_t count, uint32_t capacity,
        int32_t * expert_ids, uint32_t * source_rows, uint32_t * scatter, ggml_backend_moe_cpu_region_binding_v1 * binding,
        uint8_t * marks, size_t marks_bytes) {
    if (ggml_backend_moe_hybrid_validate_buckets_v1(region) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ||
            routes == nullptr || routes->struct_size != sizeof(*routes) || epoch == 0 || routes->epoch != epoch ||
            routes->source_graph_uid != region->source_graph_uid || routes->split_graph_uid != region->split_graph_uid ||
            routes->owner_generation != region->owner_generation || routes->allocator_generation != region->allocator_generation ||
            routes->source_generation != region->query->source_generation ||
            routes->active_rows == 0 || routes->active_rows > region->geometry.row_capacity || source_row >= routes->active_rows ||
            routes->n_routes != uint64_t(routes->active_rows) * region->geometry.routes_per_row ||
            routes->n_weights == 0 || routes->n_weights > region->geometry.weight_capacity ||
            routes->weight_experts == nullptr || routes->routes == nullptr || count == 0 || count > capacity ||
            count > region->n_cpu_queries || selected_routes == nullptr || expert_ids == nullptr ||
            source_rows == nullptr || scatter == nullptr || binding == nullptr || marks == nullptr ||
            uint64_t(region->geometry.expert_count) + region->geometry.route_capacity > marks_bytes) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    }
    const size_t expert_count = region->geometry.expert_count;
    std::fill_n(marks, expert_count + region->geometry.route_capacity, uint8_t(0));
    auto * route_marks = marks + expert_count;
    for (uint32_t i = 0; i < routes->n_weights; ++i) {
        const int32_t expert = routes->weight_experts[i];
        if (expert < 0 || uint32_t(expert) >= expert_count || marks[expert] != 0) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        marks[expert] = 1;
    }
    for (uint32_t i = 0; i < routes->n_routes; ++i) {
        const auto & route = routes->routes[i];
        if (route.source_row >= routes->active_rows || route.source_route >= region->geometry.routes_per_row ||
                route.weight_index >= routes->n_weights ||
                route.scatter_destination != uint64_t(route.source_row) * region->geometry.routes_per_row + route.source_route) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        if (route_marks[route.scatter_destination] != 0) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        route_marks[route.scatter_destination] = 1;
        marks[routes->weight_experts[route.weight_index]] = 2;
    }
    for (uint32_t i = 0; i < routes->n_weights; ++i) {
        if (marks[routes->weight_experts[i]] != 2) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (selected_routes[i] >= routes->n_routes || routes->routes[selected_routes[i]].source_row != source_row) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        auto & mark = route_marks[routes->routes[selected_routes[i]].scatter_destination];
        if (mark == 2) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        mark = 2;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto & route = routes->routes[selected_routes[i]];
        expert_ids[i] = routes->weight_experts[route.weight_index];
        source_rows[i] = route.source_row;
        scatter[i] = route.scatter_destination;
    }
    *binding = {sizeof(*binding), 1, count, expert_ids, source_rows, scatter};
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

int32_t ggml_backend_sched_moe_hybrid_prepare_v1(
        ggml_backend_sched_t sched, const ggml_backend_moe_hybrid_region_v1 * region) try {
    if (sched) { sched->source_program_token = 0; }
    if (sched == nullptr || sched->hybrid == nullptr || sched->source_retirement_failed || !sched->is_alloc || region == nullptr ||
            ggml_backend_moe_hybrid_validate_buckets_v1(region) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ||
            region->source_graph_uid != sched->source_graph_uid || region->split_index >= uint32_t(sched->n_splits) ||
            region->owner_generation == 0 || region->allocator_generation == 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    const auto & split = sched->splits[region->split_index];
    auto * selected = sched->hybrids[split.backend_id];
    if (!selected) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    uint64_t n_regions = 0;
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sched->hybrids[i]) {
            sched->hybrids[i]->each_session([&](ggml_backend_sched_hybrid * entry) { n_regions += entry->regions.size(); });
        }
    }
    if (n_regions >= selected->max_regions) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    auto * root = selected;
    uint64_t allocator_generation = 0, shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched, &allocator_generation, &shrink_generation);
    if (selected->source_api) {
        if (auto * found = selected->find_split(region->split_index, region->source_graph_uid,
                region->split_graph_uid, allocator_generation, sched->cur_copy)) { selected = found; }
    }
    if (region->split_graph_uid != split.graph.uid || sched->backends[split.backend_id] != selected->backend ||
            region->allocator_generation != allocator_generation ||
            region->first_node > region->last_node || region->first_node < uint32_t(split.i_start) ||
            region->last_node >= uint32_t(split.i_end) ||
            split.graph.nodes[region->last_node - split.i_start] != region->output ||
            region->last_node - region->first_node + 1 != region->query->n_body_nodes ||
            selected->regions.size() >= selected->max_regions) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto resolved = *region;
    resolved.first_node -= split.i_start;
    resolved.last_node -= split.i_start;
    auto reference = *region->query;
    std::vector<const ggml_tensor *> body(reference.n_body_nodes);
    const ggml_tensor * dynamic[] = {region->activation, region->ids};
    std::vector<ggml_backend_moe_cpu_region_source_v1> sources(reference.n_sources);
    for (uint32_t i = 0; i < reference.n_body_nodes; ++i) {
        body[i] = split.graph.nodes[resolved.first_node + i];
    }
    for (uint32_t i = 0; i < reference.n_sources; ++i) {
        sources[i] = reference.sources[i];
        const auto * witness = static_cast<const ggml_tensor *>(sources[i].witness);
        bool found = false;
        for (uint32_t node = 0; node < reference.n_body_nodes; ++node) {
            for (const auto * source : body[node]->src) {
                found |= source != nullptr && source == witness;
            }
        }
        if (!found || sources[i].data != witness->data || sources[i].bytes != ggml_nbytes(witness) ||
                sources[i].expert_stride != witness->nb[2]) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        sources[i].tensor = witness;
    }
    reference.body_nodes = body.data();
    reference.dynamic_inputs = dynamic;
    reference.sources = sources.data();
    const bool routed_operation = region->query->flags == GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION;
    const ggml_tensor * projection_output[] = {region->output};
    if (routed_operation) { reference.live_outputs = projection_output; }
    if (!ggml_backend_moe_hybrid_bucket_matches(*region->query, reference, 1, 1, routed_operation)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    if (region->body_query) {
        const auto & original = *region->body_query;
        if (original.struct_size != sizeof(original) || original.graph_generation != allocator_generation ||
                original.source_generation != reference.source_generation || original.n_body_nodes != reference.n_body_nodes ||
                !original.graph || !original.graph_uid || original.graph_uid != original.graph->uid ||
                !original.graph->nodes || original.graph->n_nodes <= 0 || original.graph->n_nodes > original.graph->size ||
                uint32_t(original.graph->n_nodes) != original.n_body_nodes ||
                !original.body_nodes || original.n_dynamic_inputs != reference.n_dynamic_inputs ||
                !original.live_outputs || original.n_live_outputs != reference.n_live_outputs ||
                original.n_live_outputs > GGML_BACKEND_MOE_CPU_REGION_MAX_LIVE_OUTPUTS_V1) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t i = 0; i < original.n_body_nodes; ++i) {
            if (original.graph->nodes[i] != original.body_nodes[i]) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
        }
        auto semantic_reference = original;
        std::vector<const ggml_tensor *> outputs;
        for (uint32_t i = 0; i < original.n_live_outputs; ++i) {
            const auto * end = original.body_nodes + original.n_body_nodes;
            const auto * found = std::find(original.body_nodes, end, original.live_outputs[i]);
            if (found == end) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
            const auto * expected_end = region->query->body_nodes + region->query->n_body_nodes;
            const auto * expected = std::find(region->query->body_nodes, expected_end, region->query->live_outputs[i]);
            if (expected == expected_end || expected - region->query->body_nodes != found - original.body_nodes) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
            outputs.push_back(body[found - original.body_nodes]);
        }
        semantic_reference.body_nodes = body.data();
        semantic_reference.dynamic_inputs = dynamic;
        semantic_reference.sources = sources.data();
        semantic_reference.live_outputs = outputs.data();
        if (!ggml_backend_moe_hybrid_bucket_matches(original, semantic_reference, 0, 0, true)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
    }
    std::unique_ptr<ggml_backend_sched_hybrid> pending;
    if (root->source_api && selected == root && !root->regions.empty() &&
            (root->regions.front().split_index != region->split_index || !root->regions.front().matches_binding(
                region->source_graph_uid, region->split_graph_uid, allocator_generation, sched->cur_copy))) {
        root->split_sessions.reserve(root->split_sessions.size() + 1);
        const auto status = ggml_backend_sched_moe_hybrid_create(&root->config, root->source_cpu, pending);
        if (status) { return status; }
        selected = pending.get();
    }
    auto & state = *selected;
    ggml_backend_sched_hybrid_region prepared = {};
    struct preparation_guard {
        ggml_backend_sched_hybrid & state;
        ggml_backend_sched_hybrid_region & region;
        bool published = false;
        ~preparation_guard() { if (!published) { state.release(region); } }
    } guard{state, prepared};
    prepared.split_index = region->split_index;
    prepared.copy_index = sched->cur_copy;
    prepared.source_graph_uid = region->source_graph_uid;
    prepared.split_graph_uid = region->split_graph_uid;
    prepared.allocator_generation = allocator_generation;
    prepared.cpu.resize(uint64_t(region->n_cpu_queries) + region->n_cpu_batch_queries);
    prepared.witnesses.reserve(region->query->n_body_nodes * (GGML_MAX_SRC + 1));
    const auto record = [&](const ggml_tensor * tensor) {
        if (tensor != nullptr && std::none_of(prepared.witnesses.begin(), prepared.witnesses.end(),
                [&](const auto & witness) { return witness.first == tensor; })) {
            prepared.witnesses.emplace_back(tensor, *tensor);
        }
    };
    for (uint32_t i = resolved.first_node; i <= resolved.last_node; ++i) {
        record(split.graph.nodes[i]);
        for (const auto * source : split.graph.nodes[i]->src) {
            record(source);
        }
    }
    for (uint32_t i = 0; i < region->n_cpu_queries; ++i) {
        const auto * query = region->cpu_queries[i];
        ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
        requirements.struct_size = sizeof(requirements);
        requirements.abi_version = 1;
        const int32_t status = state.cpu_api->prepare(state.cpu, query, &requirements, &prepared.cpu[i]);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            GGML_LOG_ERROR("%s: CPU prepare failed status=%d count=%u graph=%llu\n", __func__, status, i + 1,
                           (unsigned long long) query->graph_uid);
            return status;
        }
    }
    for (uint32_t i = 0; i < region->n_cpu_batch_queries; ++i) {
        const auto * query = region->cpu_batch_queries[i];
        ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
        requirements.struct_size = sizeof(requirements);
        requirements.abi_version = 1;
        const uint32_t prepared_index = region->n_cpu_queries + i;
        const int32_t status = state.cpu_api->prepare(state.cpu, query, &requirements, &prepared.cpu[prepared_index]);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            GGML_LOG_ERROR("%s: CPU batch prepare failed status=%d count=%u graph=%llu\n", __func__, status, i + 1,
                           (unsigned long long) query->graph_uid);
            return status;
        }
    }
    const int32_t status = state.source_api ? state.source_api->prepare(
        state.device, &resolved, state.cpu_api, state.cpu, prepared.cpu.data(), &prepared.device) : state.device_api->prepare(
        state.device, &resolved, state.cpu_api, state.cpu, prepared.cpu.data(), &prepared.device);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return status;
    }
    state.regions.push_back(std::move(prepared));
    if (pending) { root->split_sessions.push_back(std::move(pending)); }
    guard.published = true;
    sched->source_preparation_epoch = sched->source_retirement_epoch;
    sched->source_preparation_graph_uid = region->source_graph_uid;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) {
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
}

bool ggml_backend_sched_moe_hybrid_state_v1(
        ggml_backend_sched_t sched, ggml_backend_moe_hybrid_state_v1 * state) {
    if (!sched || !sched->hybrid || !sched->hybrid->device || !state || state->struct_size != sizeof(*state)) { return false; }
    auto & hybrid = *sched->hybrid;
    const bool single = sched->n_hybrids == 1 && hybrid.split_sessions.empty();
    if (single && !hybrid.source_api) { return hybrid.device_api->state(hybrid.device, state); }
    ggml_backend_moe_hybrid_state_v1 aggregate = {}; aggregate.struct_size = sizeof(aggregate);
    if (single && !hybrid.source_api->state(hybrid.device, &aggregate)) { return false; }
    uint64_t ggml_backend_moe_hybrid_state_v1::* additive[] = {
        &ggml_backend_moe_hybrid_state_v1::resident_routes,
        &ggml_backend_moe_hybrid_state_v1::transfer_routes,
        &ggml_backend_moe_hybrid_state_v1::cpu_routes,
        &ggml_backend_moe_hybrid_state_v1::h2d_bytes,
        &ggml_backend_moe_hybrid_state_v1::device_bytes,
        &ggml_backend_moe_hybrid_state_v1::pinned_bytes,
        &ggml_backend_moe_hybrid_state_v1::distinct_experts,
        &ggml_backend_moe_hybrid_state_v1::resident_experts,
        &ggml_backend_moe_hybrid_state_v1::transfer_experts,
        &ggml_backend_moe_hybrid_state_v1::cpu_experts,
        &ggml_backend_moe_hybrid_state_v1::cpu_jobs,
        &ggml_backend_moe_hybrid_state_v1::cpu_upload_bytes,
        &ggml_backend_moe_hybrid_state_v1::cpu_us,
        &ggml_backend_moe_hybrid_state_v1::gpu_enqueue_us,
        &ggml_backend_moe_hybrid_state_v1::join_us,
        &ggml_backend_moe_hybrid_state_v1::errors,
        &ggml_backend_moe_hybrid_state_v1::capacity_errors,
        &ggml_backend_moe_hybrid_state_v1::cancellations,
        &ggml_backend_moe_hybrid_state_v1::submit_to_start_us,
        &ggml_backend_moe_hybrid_state_v1::gpu_branch_us,
        &ggml_backend_moe_hybrid_state_v1::admission_reserved,
        &ggml_backend_moe_hybrid_state_v1::admission_committed,
        &ggml_backend_moe_hybrid_state_v1::admission_replacements,
        &ggml_backend_moe_hybrid_state_v1::admission_no_slot,
        &ggml_backend_moe_hybrid_state_v1::admission_bytes,
        &ggml_backend_moe_hybrid_state_v1::admission_aborted,
        &ggml_backend_moe_hybrid_state_v1::resident_batches,
        &ggml_backend_moe_hybrid_state_v1::resident_body_submissions,
        &ggml_backend_moe_hybrid_state_v1::gpu_body_submissions,
        &ggml_backend_moe_hybrid_state_v1::readback_us,
        &ggml_backend_moe_hybrid_state_v1::publication_us,
        &ggml_backend_moe_hybrid_state_v1::admission_fence_us,
        &ggml_backend_moe_hybrid_state_v1::prepared_device_bytes,
        &ggml_backend_moe_hybrid_state_v1::packet_regions,
        &ggml_backend_moe_hybrid_state_v1::producer_events,
        &ggml_backend_moe_hybrid_state_v1::producer_fences,
        &ggml_backend_moe_hybrid_state_v1::producer_drain_events,
        &ggml_backend_moe_hybrid_state_v1::window_launches,
        &ggml_backend_moe_hybrid_state_v1::window_captures,
        &ggml_backend_moe_hybrid_state_v1::window_waits,
        &ggml_backend_moe_hybrid_state_v1::window_fallbacks,
        &ggml_backend_moe_hybrid_state_v1::window_fused_nodes,
        &ggml_backend_moe_hybrid_state_v1::window_combined_regions,
        &ggml_backend_moe_hybrid_state_v1::window_direct_regions,
        &ggml_backend_moe_hybrid_state_v1::window_compact_select_regions,
        &ggml_backend_moe_hybrid_state_v1::window_fused_expert_bodies,
        &ggml_backend_moe_hybrid_state_v1::cpu_execute_calls,
        &ggml_backend_moe_hybrid_state_v1::cpu_batch_rows,
    };
    bool valid = true;
    for (int i = 0; !single && i < sched->n_backends; ++i) {
        auto * root = sched->hybrids[i];
        if (!root) { continue; }
        root->each_session([&](ggml_backend_sched_hybrid * entry) {
            if (!valid) { return; }
            ggml_backend_moe_hybrid_state_v1 part = {}; part.struct_size = sizeof(part);
            if (!entry->device || !entry->source_api || !entry->source_api->state(entry->device, &part)) { valid = false; return; }
            for (auto field : additive) {
                if (aggregate.*field > UINT64_MAX - part.*field) { valid = false; return; }
                aggregate.*field += part.*field;
            }
            aggregate.work_peak = std::max(aggregate.work_peak, part.work_peak);
            // Every entry borrows the same worker service. Its active jobs are one shared count.
            aggregate.cpu_active_jobs = std::max(aggregate.cpu_active_jobs, part.cpu_active_jobs);
            aggregate.dispatch_active |= part.dispatch_active;
            aggregate.quiescing |= part.quiescing;
        });
    }
    if (!valid) { return false; }
    ggml_backend_moe_cpu_service_state_v1 cpu = {}; cpu.struct_size = sizeof(cpu);
    if (!hybrid.cpu_api || !hybrid.cpu || hybrid.cpu_api->state(hybrid.cpu, &cpu)) { return false; }
    aggregate.cpu_active_jobs = std::max(aggregate.cpu_active_jobs, cpu.active_jobs);
    aggregate.prepared_cpu_bytes = cpu.prepared_payload_bytes;
    *state = aggregate;
    return true;
}

void ggml_backend_sched_moe_hybrid_quiesce_v1(ggml_backend_sched_t sched) {
    if (!sched) { return; }
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * root = sched->hybrids[i];
        if (!root) { continue; }
        root->each_session([&](ggml_backend_sched_hybrid * hybrid) {
            if (hybrid->source_api) {
                if (hybrid->device) {
                    GGML_ASSERT(hybrid->source_api->close(hybrid->device) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                    GGML_ASSERT(hybrid->source_api->drain(hybrid->device) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
                }
            } else { hybrid->device_api->quiesce(hybrid->device); }
        });
    }
}

bool ggml_backend_sched_moe_hybrid_set_test_hook_v1(
        ggml_backend_sched_t sched, ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data) {
    if (!sched || !sched->hybrid || !sched->hybrid->device) { return false; }
    bool result = true;
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * root = sched->hybrids[i];
        if (!root) { continue; }
        root->each_session([&](ggml_backend_sched_hybrid * hybrid) {
            const bool applied = hybrid->device && (hybrid->source_api ? hybrid->source_api->set_test_hook &&
                hybrid->source_api->set_test_hook(hybrid->device, hook, data) :
                hybrid->device_api->set_test_hook && hybrid->device_api->set_test_hook(hybrid->device, hook, data));
            result = applied && result;
        });
    }
    return result;
}

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    sched->source_program_token = 0;
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();
    sched->source_graph = graph;
    sched->source_graph_uid = graph->uid;

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        memset(split->direct_dependencies, 0, sizeof(split->direct_dependencies));
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                memset(split->direct_dependencies, 0, sizeof(split->direct_dependencies));
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                } else if (src_backend_id != cur_backend_id && src->op != GGML_OP_NONE && !ggml_is_empty(src)) {
                    split->direct_dependencies[src_backend_id] = true;
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched, bool reuse_async) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    const bool can_allocate = !backend_ids_changed && (!reuse_async ||
        ggml_gallocr_reserve_n_if_fits(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids));
    if (!can_allocate || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// How a split input breaks into the ranges a graph reads.
// A window over a cache split into streams is one range per stream, keyed on the last dimension: the ranges sit a fixed stride apart and the bytes between them are never read.
// Anything else is one flat range of ggml_nbytes().
struct ggml_backend_sched_ranges {
    int64_t n;      // ranges to deliver
    size_t  stride; // bytes from one range to the next
    size_t  used;   // bytes of a range this graph reads
};

static void ggml_backend_sched_input_ranges(const struct ggml_tensor * input, struct ggml_backend_sched_ranges * out) {
    out->n      = 1;
    out->stride = 0;
    out->used   = ggml_nbytes(input);

    // a range is one stream's byte span, which is what the tensor covers below dimension 3
    const size_t rows = ggml_nbytes(input) - (size_t) (input->ne[3] - 1)*input->nb[3];
    const size_t offs = input->view_src ? input->view_offs : 0;
    if (input->nb[3] < rows || (offs != 0 && (input->nb[3] == 0 || offs % input->nb[3] != 0))) {
        return;
    }

    if (input->ne[3] > 1) {
        out->n      = input->ne[3];
        out->stride = input->nb[3];
        out->used   = rows;
    }
}

static enum ggml_status ggml_backend_sched_hybrid_dispatch_prepare(
        ggml_backend_sched_t sched, int split_id, uint64_t source_graph_uid,
        const struct ggml_graph_execution_certificate & certificate, ggml_backend_sched_hybrid ** output) {
    *output = nullptr;
    auto * hybrid = sched->hybrid;
    const bool independent = certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    const bool speculative = certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    if (!hybrid || (!hybrid->source_api && certificate.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN && !independent && !speculative)) { return GGML_STATUS_SUCCESS; }
    if (!ggml_backend_sched_hybrid_certificate_supported(certificate, hybrid->source_api != nullptr)) {
        GGML_LOG_ERROR("%s: hybrid certificate rejected: graph_uid=%llu split=%d domain=%u semantics=%u rows=%u sequences=%u flags=%u\n",
            __func__, (unsigned long long) source_graph_uid, split_id, certificate.domain,
            certificate.row_semantics, certificate.n_rows, certificate.n_sequences, certificate.flags);
        return GGML_STATUS_FAILED;
    }
    uint64_t allocator_generation = 0, shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched, &allocator_generation, &shrink_generation);
    auto * selected = sched->hybrids[sched->splits[split_id].backend_id];
    if (selected && selected->source_api) {
        selected = selected->find_split(split_id, source_graph_uid, sched->splits[split_id].graph.uid, allocator_generation, sched->cur_copy);
    }
    if (selected) { selected->dispatch.clear(); }
    bool valid = true;
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * root = sched->hybrids[i];
        if (!root) { continue; }
        root->each_session([&](ggml_backend_sched_hybrid * entry) {
            for (const auto & region : entry->regions) {
                if (region.split_index >= uint32_t(sched->n_splits) || !region.matches(source_graph_uid,
                        sched->splits[region.split_index].graph.uid, allocator_generation, sched->cur_copy)) {
                    GGML_LOG_ERROR("%s: hybrid region storage witness rejected: graph_uid=%llu split=%d region_split=%u generation=%llu expected_generation=%llu copy=%u expected_copy=%d\n",
                        __func__, (unsigned long long) source_graph_uid, split_id, region.split_index,
                        (unsigned long long) region.allocator_generation, (unsigned long long) allocator_generation,
                        region.copy_index, sched->cur_copy);
                    valid = false; return;
                }
                if (region.split_index == uint32_t(split_id)) {
                    if (entry != selected) { valid = false; return; }
                    selected->dispatch.push_back(region.device);
                }
            }
        });
    }
    if (!valid) { return GGML_STATUS_FAILED; }
    if (selected && (!selected->source_api || !selected->dispatch.empty())) { *output = selected; }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_dispatch_split(
        ggml_backend_t backend,
        struct ggml_cgraph * graph,
        uint64_t source_graph_uid,
        const struct ggml_graph_execution_certificate & certificate,
        ggml_backend_sched_hybrid * hybrid,
        const uint8_t * phases, size_t n_phases) {
    graph->execution_certificate = ggml_backend_sched_split_certificate(source_graph_uid, graph->uid, certificate);
    graph->execution_phases = phases;
    graph->n_execution_phases = n_phases;
    struct metadata_scope {
        ggml_cgraph * graph;
        ~metadata_scope() {
            graph->execution_certificate = {};
            graph->execution_phases = nullptr;
            graph->n_execution_phases = 0;
        }
    } scope{graph};

    const enum ggml_status status = hybrid != nullptr ? (hybrid->source_api ?
        hybrid->source_api->compute(hybrid->device, graph, hybrid->dispatch.data(), hybrid->dispatch.size()) :
        hybrid->device_api->compute(hybrid->device, graph, hybrid->dispatch.data(), hybrid->dispatch.size())) :
        ggml_backend_graph_compute_async(backend, graph);
    return status;
}

struct ggml_backend_sched_expert_copy_state {
    const ggml_tensor * ids = nullptr;
    int64_t n_expert = 0;
    std::vector<int32_t> ids_data;
    std::vector<ggml_bitset_t> used;
};

static bool ggml_backend_sched_copy_selected_experts(
        ggml_backend_t input_backend, ggml_backend_t backend, const ggml_tensor * src,
        ggml_tensor * dst, ggml_cgraph * graph, ggml_backend_sched_expert_copy_state & st) {
    if (graph->n_nodes == 0) { return false; }
    const ggml_tensor * node = graph->nodes[0];
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) { return false; }
    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) { return true; }
    const int64_t n_expert = src->ne[2];
    const size_t expert_size = src->nb[2];
    ggml_backend_synchronize(input_backend);
    if (ids != st.ids || n_expert != st.n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);
        st.used.assign(ggml_bitset_size(n_expert), 0);
        for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                ggml_bitset_set(st.used.data(), id);
            }
        }
        st.ids = ids;
        st.n_expert = n_expert;
    }
    for (int64_t first = 0; first < n_expert;) {
        if (!ggml_bitset_get(st.used.data(), first)) { ++first; continue; }
        int64_t last = first;
        while (last + 1 < n_expert && ggml_bitset_get(st.used.data(), last + 1)) { ++last; }
        const size_t offset = first*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size = (last + 1 - first)*expert_size + padding;
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *)src->data + offset, offset, size);
        first = last + 1;
    }
    return true;
}

static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input, bool source_input_batch, bool & source_input_pending, ggml_backend_sched_expert_copy_state & expert_copy) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
    auto * input_cpy_buffer = input_cpy->view_src ? input_cpy->view_src->buffer : input_cpy->buffer;
    const bool source_async_input = source_input_batch && split_backend->iface.set_tensor_async &&
        input->buffer && ggml_backend_buffer_is_host(input->buffer) && input_cpy_buffer &&
        input_cpy_buffer->buft == ggml_backend_get_default_buffer_type(split_backend);
    auto copy_input = [&]() {
        if (source_async_input) {
            GGML_ASSERT(ggml_are_same_layout(input, input_cpy));
            if (input == input_cpy) { return; }
            const size_t bytes = ggml_nbytes(input);
            if (bytes) {
                ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, bytes);
                source_input_pending = true;
            }
        } else {
            if (source_input_pending) {
                ggml_backend_synchronize(split_backend);
                source_input_pending = false;
            }
            ggml_backend_tensor_copy(input, input_cpy);
        }
    };

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else if (!source_input_batch) {
            ggml_backend_synchronize(split_backend);
        }
        copy_input();
        return;
    }

    // wait for the split backend to finish using the input before overwriting it
    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
    } else if (!source_input_batch) {
        ggml_backend_synchronize(split_backend);
    }

    if (sched->callback_copy != NULL && ggml_backend_sched_is_host_weight(input) &&
        sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)) {
        return;
    }

    if (sched->callback_copy == nullptr && ggml_backend_sched_is_host_weight(input) &&
            ggml_backend_sched_copy_selected_experts(input_backend, split_backend, input, input_cpy, &split->graph, expert_copy)) {
        return;
    }

    ggml_backend_buffer_t src_buffer = input->view_src ? input->view_src->buffer : input->buffer;
    ggml_backend_sched_ranges ranges;
    ggml_backend_sched_input_ranges(input, &ranges);
    const bool ranged = ranges.n > 1 && src_buffer && ggml_backend_buffer_is_host(src_buffer);

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    if (ranged || !split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        ggml_backend_synchronize(input_backend);
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else if (ranged || !source_async_input) {
            ggml_backend_synchronize(split_backend);
            source_input_pending = false;
        }
        if (ranged) {
            ggml_backend_tensor_set_2d_async(split_backend, input_cpy, input->data, 0,
                ranges.used, ranges.n, ranges.stride, ranges.stride);
            ggml_backend_synchronize(split_backend);
            source_input_pending = false;
        } else {
            copy_input();
        }
    }
}

static enum ggml_status ggml_backend_sched_compute_failure(ggml_backend_sched_t sched, enum ggml_status status,
        ggml_backend_sched_hybrid * failed, bool previous_split_effects, bool required_grouped) {
    if (sched->hybrid && sched->hybrid->source_api) {
        ggml_backend_moe_hybrid_state_v1 state = {}; state.struct_size = sizeof(state);
        const bool safe_rejection = failed && failed->source_api && failed->device && !previous_split_effects &&
            failed->source_api->state(failed->device, &state) &&
            state.ticket_state == GGML_BACKEND_MOE_SOURCE_CORE_TICKET_V1_REJECTED_BEFORE_EFFECTS &&
            !state.quiescing && !state.dispatch_active && !state.cpu_active_jobs;
        if (!safe_rejection) { (void) ggml_backend_sched_moe_source_close_v1(sched); }
        const int32_t drained = ggml_backend_sched_moe_source_drain_v1(sched);
        if (drained != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { (void) ggml_backend_sched_moe_source_close_v1(sched); }
        if (previous_split_effects && drained == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) {
            // Source retirement also covers earlier ordinary backend work.
            for (int i = 0; i < sched->n_backends; ++i) { ggml_backend_synchronize(sched->backends[i]); }
        }
        GGML_LOG_ERROR("%s: source failure status=%d safe_rejection=%d previous_split_effects=%d drain_status=%d\n",
            __func__, int(status), int(safe_rejection), int(previous_split_effects), drained);
        return status;
    }
    if (required_grouped || sched->hybrid != nullptr) {
        for (int i = 0; i < sched->n_backends; ++i) { ggml_backend_synchronize(sched->backends[i]); }
    }
    return status;
}

static enum ggml_status ggml_backend_sched_compute_splits(
        ggml_backend_sched_t sched,
        uint64_t source_graph_uid,
        struct ggml_graph_execution_certificate certificate,
        const uint8_t * phases, size_t n_phases) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;
    const bool required_grouped =
        (certificate.flags & GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0;
    bool previous_split_effects = false;
    const auto fail = [&](enum ggml_status status, ggml_backend_sched_hybrid * failed = nullptr) {
        return ggml_backend_sched_compute_failure(sched, status, failed, previous_split_effects, required_grouped);
    };

    if (certificate.magic == GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC) {
        if (source_graph_uid == 0) {
            if (required_grouped) {
                GGML_LOG_ERROR("%s: required grouped execution has no source graph UID\n", __func__);
                return GGML_STATUS_FAILED;
            }
            certificate = {};
        }
        for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
            if (splits[split_id].graph.uid == 0) {
                if (required_grouped) {
                    GGML_LOG_ERROR("%s: required grouped execution split %d has no graph UID\n", __func__, split_id);
                    return GGML_STATUS_FAILED;
                }
                certificate = {};
                break;
            }
        }
    }

    // A sole split without input copies is validated by compute before graph effects.
    const bool source_checks_before_effects = sched->n_splits == 1 && splits[0].n_inputs == 0;
    if (!sched->callback_eval && sched->hybrid && sched->hybrid->source_api && !source_checks_before_effects) {
        // Check static source metadata before any split copies or execution.
        for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
            ggml_backend_sched_hybrid * hybrid = nullptr;
            const auto prepared = ggml_backend_sched_hybrid_dispatch_prepare(sched, split_id, source_graph_uid, certificate, &hybrid);
            if (prepared != GGML_STATUS_SUCCESS) { return fail(prepared); }
            if (!hybrid) { continue; }
            const auto & graph = splits[split_id].graph;
            const auto projected = ggml_backend_sched_split_certificate(source_graph_uid, graph.uid, certificate);
            const auto status = hybrid->source_api->preflight(hybrid->device, &graph, &projected,
                hybrid->dispatch.data(), hybrid->dispatch.size());
            if (status != GGML_STATUS_SUCCESS) { return fail(status, hybrid); }
        }
    }

    ggml_backend_sched_expert_copy_state expert_copy;

    int prev_backend_id = -1;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // Directly readable inputs and shared allocations still need producer ordering.
        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            const bool previous = backend_id == prev_backend_id && backend_id != split_backend_id &&
                (split->n_inputs == 0 || sched->bufts[backend_id] == sched->bufts[split_backend_id]);
            if (!previous && !split->direct_dependencies[backend_id]) { continue; }
            if (sched->events[backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[backend_id]);
            }
        }

        const auto * source_entry = sched->hybrids[split_backend_id];
        const bool source_input_batch = split->n_inputs > 0 && source_entry && source_entry->source_api &&
            source_entry->backend == split_backend && !sched->callback_eval &&
            ggml_backend_sched_hybrid_certificate_supported(certificate, true) &&
            (certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT ||
             certificate.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE ||
             certificate.domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN) &&
            sched->events[split_backend_id][sched->cur_copy] == NULL;
        if (source_input_batch) {
            ggml_backend_synchronize(split_backend);
        }
        bool source_input_pending = false;

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (!ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id], source_input_batch, source_input_pending, expert_copy);
            }
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id], source_input_batch, source_input_pending, expert_copy);
            }
        }
        if (source_input_pending) {
            // finish host reads before dispatch, validation failure or scheduler return
            ggml_backend_synchronize(split_backend);
        }

        if (!sched->callback_eval) {
            ggml_backend_sched_hybrid * hybrid = nullptr;
            const auto prepared = ggml_backend_sched_hybrid_dispatch_prepare(sched, split_id, source_graph_uid, certificate, &hybrid);
            if (prepared != GGML_STATUS_SUCCESS) { return fail(prepared); }
            enum ggml_status ec = ggml_backend_sched_dispatch_split(
                split_backend, &split->graph, source_graph_uid, certificate, hybrid, phases, n_phases);
            if (ec != GGML_STATUS_SUCCESS) {
                return fail(ec, hybrid);
            }
            previous_split_effects = true;
        } else {
            split->graph.execution_certificate = {};
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_sched_retire_buffer_bindings(void * user_data);

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    ggml_gallocr_set_buffer_replacement_callback(sched->galloc, ggml_backend_sched_retire_buffer_bindings, sched);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * entry = sched->hybrids[i];
        if (entry && entry->source_api) { GGML_ASSERT(entry->clear_source() == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK); }
        delete entry;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

bool ggml_backend_sched_set_resizable(ggml_backend_sched_t sched, ggml_backend_sched_t owner) {
    GGML_ASSERT(sched != nullptr);
    if (sched->hybrid != nullptr && !sched->hybrid->source_api && owner != nullptr) {
        return false;
    }
    return ggml_gallocr_set_resizable(sched->galloc, owner ? owner->galloc : nullptr);
}

void ggml_backend_sched_get_buffer_state(
        ggml_backend_sched_t sched,
        uint64_t * generation,
        uint64_t * shrink_generation) {
    GGML_ASSERT(sched != nullptr);
    ggml_gallocr_get_resizable_state(sched->galloc, generation, shrink_generation);
}

void ggml_backend_sched_request_buffer_shrink(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched != nullptr);
    ggml_gallocr_request_shrink(sched->galloc);
}

bool ggml_backend_sched_refresh_resizable_plan(ggml_backend_sched_t sched) {
    return sched && ggml_gallocr_refresh_resizable_plan(sched->galloc);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    sched->source_program_token = 0;
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sched->hybrids[i]) { sched->hybrids[i]->clear(); }
    }
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
    sched->source_graph = nullptr;
    sched->source_graph_uid = 0;
}

bool ggml_backend_sched_moe_source_selected_v1(ggml_backend_sched_t sched) {
    return sched && sched->hybrid && sched->hybrid->source_api;
}

static int32_t ggml_backend_sched_moe_source_clear(ggml_backend_sched_t sched, bool keep_cpu) {
    sched->source_program_token = 0;
    int32_t result = GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * entry = sched->hybrids[i];
        if (!entry || !entry->source_api) { continue; }
        const auto status = entry->clear_source(keep_cpu);
        if (!result) { result = status; }
    }
    return result;
}

uint64_t ggml_backend_sched_moe_source_retirement_epoch_v1(ggml_backend_sched_t sched) {
    return sched ? sched->source_retirement_epoch : 0;
}

int32_t ggml_backend_sched_moe_source_retire_v1(ggml_backend_sched_t sched) {
    if (!sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    if (sched->source_retirement_epoch == UINT64_MAX) {
        sched->source_retirement_failed = true;
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED;
    }
    ++sched->source_retirement_epoch;
    const auto status = ggml_backend_sched_moe_source_clear(sched, true);
    sched->source_retirement_failed = status != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    if (status) { return status; }
    for (int i = 0; i < sched->n_backends; ++i) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

static bool ggml_backend_sched_retire_buffer_bindings(void * user_data) {
    auto * sched = static_cast<ggml_backend_sched_t>(user_data);
    if (ggml_backend_sched_moe_source_selected_v1(sched)) {
        return ggml_backend_sched_moe_source_retire_v1(sched) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    }
    ggml_backend_sched_synchronize(sched);
    return true;
}

int32_t ggml_backend_sched_moe_source_reset_v1(ggml_backend_sched_t sched) {
    if (!sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    const auto status = ggml_backend_sched_moe_source_clear(sched, false);
    if (status) { return status; }
    sched->source_retirement_failed = false;
    sched->source_preparation_graph_uid = 0;
    ggml_backend_sched_reset(sched);
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

int32_t ggml_backend_sched_moe_source_reset_graph_v1(ggml_backend_sched_t sched) {
    if (!sched || !sched->hybrid || !sched->hybrid->source_api) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    const auto status = ggml_backend_sched_moe_source_clear(sched, true);
    if (status) { return status; }
    sched->source_retirement_failed = false;
    sched->source_preparation_graph_uid = 0;
    ggml_backend_sched_reset(sched);
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

int32_t ggml_backend_sched_moe_source_drain_v1(ggml_backend_sched_t sched) {
    if (!sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    int32_t result = GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * root = sched->hybrids[i];
        if (!root || !root->source_api) { continue; }
        root->each_session([&](ggml_backend_sched_hybrid * state) {
            std::lock_guard<std::mutex> lock(state->source_mutex);
            const auto status = state->device ? state->source_api->drain(state->device) : GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
            if (!result) { result = status; }
        });
    }
    return result;
}

int32_t ggml_backend_sched_moe_source_close_v1(ggml_backend_sched_t sched) {
    if (!sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    int32_t result = GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    for (int i = 0; i < sched->n_backends; ++i) {
        auto * root = sched->hybrids[i];
        if (!root || !root->source_api) { continue; }
        root->each_session([&](ggml_backend_sched_hybrid * state) {
            std::lock_guard<std::mutex> lock(state->source_mutex);
            const auto status = state->device ? state->source_api->close(state->device) : GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
            if (!result) { result = status; }
        });
    }
    return result;
}

int32_t ggml_backend_sched_moe_source_fallback_v1(ggml_backend_sched_t sched) {
    if (!sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    if (!sched->hybrid || !sched->hybrid->source_api) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    const auto status = ggml_backend_sched_moe_source_clear(sched, false);
    if (status) { return status; }
    for (int i = 0; i < sched->n_backends; ++i) {
        delete sched->hybrids[i]; sched->hybrids[i] = nullptr;
    }
    sched->hybrid = nullptr; sched->n_hybrids = 0;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

int32_t ggml_backend_sched_moe_source_free_v1(ggml_backend_sched_t * sched) {
    if (!sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    if (!*sched) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    const auto status = ggml_backend_sched_moe_source_clear(*sched, false);
    if (status) { return status; }
    int32_t result = GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    for (int i = 0; i < (*sched)->n_backends; ++i) {
        auto * state = (*sched)->hybrids[i];
        if (!state || !state->source_api) { continue; }
        for (auto * module : {&state->device_module, &state->cpu_module}) {
            if (*module) {
                if (state->config.module_release(*module)) { *module = nullptr; }
                else { result = GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_FAILED; }
            }
        }
    }
    if (result) { return result; }
    ggml_backend_sched_free(*sched);
    *sched = nullptr;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

static bool ggml_backend_sched_alloc_graph_impl(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool reuse_async) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched, reuse_async)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_sched_alloc_graph_impl(sched, graph, false);
}

bool ggml_backend_sched_alloc_graph_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_sched_alloc_graph_impl(sched, graph, true);
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_sched_graph_compute_ext(sched, graph, nullptr);
}

enum ggml_status ggml_backend_sched_graph_compute_ext(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        const struct ggml_graph_execution_certificate * certificate) {
    return ggml_backend_sched_graph_compute_with_phases(sched, graph, certificate, nullptr, 0);
}

enum ggml_status ggml_backend_sched_graph_compute_with_phases(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        const struct ggml_graph_execution_certificate * certificate,
        const uint8_t * phases, size_t n_phases) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async_with_phases(sched, graph, certificate, phases, n_phases);
    if (ggml_backend_sched_moe_source_selected_v1(sched)) {
        if (err != GGML_STATUS_SUCCESS) { return err; }
        if (ggml_backend_sched_moe_source_drain_v1(sched) != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { return GGML_STATUS_FAILED; }
    }
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_sched_graph_compute_async_ext(sched, graph, nullptr);
}

enum ggml_status ggml_backend_sched_graph_compute_async_ext(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        const struct ggml_graph_execution_certificate * certificate) {
    return ggml_backend_sched_graph_compute_async_with_phases(sched, graph, certificate, nullptr, 0);
}

enum ggml_status ggml_backend_sched_graph_compute_async_with_phases(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * graph,
        const struct ggml_graph_execution_certificate * certificate,
        const uint8_t * phases, size_t n_phases) {
    GGML_ASSERT(sched);
    struct ggml_graph_execution_certificate certificate_value = {};
    const bool certificate_valid = ggml_backend_sched_execution_certificate_valid(certificate);
    if ((phases == nullptr) != (n_phases == 0) ||
            (phases && (!certificate_valid || n_phases != certificate->n_rows))) { return GGML_STATUS_FAILED; }
    for (size_t i = 0; i < n_phases; ++i) {
        if (phases[i] > GGML_GRAPH_EXECUTION_PHASE_GENERATION) { return GGML_STATUS_FAILED; }
    }
    const bool required_grouped = certificate != nullptr &&
        (certificate->flags & GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0;
    if (sched->hybrid != nullptr && (!certificate_valid || sched->callback_eval != nullptr ||
            !ggml_backend_sched_hybrid_certificate_supported(*certificate, sched->hybrid->source_api != nullptr))) {
        GGML_LOG_ERROR("%s: hybrid execution certificate rejected: graph_uid=%llu valid=%d callback=%d domain=%u semantics=%u rows=%u sequences=%u\n",
            __func__, (unsigned long long) graph->uid, certificate_valid, sched->callback_eval != nullptr,
            certificate ? certificate->domain : 0, certificate ? certificate->row_semantics : 0,
            certificate ? certificate->n_rows : 0, certificate ? certificate->n_sequences : 0);
        return GGML_STATUS_FAILED;
    }
    if (required_grouped && !certificate_valid) {
        GGML_LOG_ERROR("%s: invalid required grouped execution certificate\n", __func__);
        return GGML_STATUS_FAILED;
    }
    if (required_grouped && sched->callback_eval != nullptr) {
        GGML_LOG_ERROR("%s: required grouped execution does not support callback evaluation\n", __func__);
        return GGML_STATUS_FAILED;
    }
    if (certificate_valid) {
        certificate_value = *certificate;
    }

    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    if (graph != sched->source_graph || graph->uid != sched->source_graph_uid) {
        if (required_grouped || sched->hybrid != nullptr || phases) {
            GGML_LOG_ERROR("%s: required grouped execution certificate does not match the source graph\n", __func__);
            return GGML_STATUS_FAILED;
        }
        certificate_value = {};
    }

    if (sched->source_retirement_failed || (sched->hybrid && sched->hybrid->source_api &&
            (required_grouped || sched->source_preparation_graph_uid == sched->source_graph_uid) &&
            sched->source_preparation_epoch != sched->source_retirement_epoch)) {
        GGML_LOG_ERROR("%s: retired source bindings require checked preparation\n", __func__);
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_sched_compute_splits(sched, sched->source_graph_uid, certificate_value, phases, n_phases);
}

int32_t ggml_backend_sched_moe_source_program_bind_v1(ggml_backend_sched_t sched, ggml_cgraph * graph,
        const ggml_graph_execution_certificate * certificate, uint64_t * program) {
    if (!program) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    *program = 0;
    if (!sched || !sched->hybrid || !sched->hybrid->source_api || sched->n_splits != 1 ||
            sched->splits[0].n_inputs || sched->callback_eval) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    if (!sched->is_alloc || sched->source_retirement_failed || !graph || graph != sched->source_graph ||
            graph->uid != sched->source_graph_uid || !ggml_backend_sched_execution_certificate_valid(certificate) ||
            sched->source_preparation_epoch != sched->source_retirement_epoch) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    ggml_backend_sched_hybrid * hybrid = nullptr;
    if (ggml_backend_sched_hybrid_dispatch_prepare(sched, 0, sched->source_graph_uid, *certificate, &hybrid) != GGML_STATUS_SUCCESS) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    if (!hybrid || !hybrid->source_api || !hybrid->source_api->bind_program || !hybrid->source_api->compute_program) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    const auto projected = ggml_backend_sched_split_certificate(sched->source_graph_uid, sched->splits[0].graph.uid, *certificate);
    void * prepared = nullptr;
    const auto status = hybrid->source_api->bind_program(hybrid->device, &sched->splits[0].graph, &projected,
        hybrid->dispatch.data(), hybrid->dispatch.size(), &prepared);
    if (status) { return status; }
    if (!prepared) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    static std::atomic<uint64_t> next_token{1};
    auto token = next_token.load(std::memory_order_relaxed);
    do {
        if (token == UINT64_MAX) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    } while (!next_token.compare_exchange_weak(token, token + 1, std::memory_order_relaxed));
    ggml_backend_sched_get_buffer_state(sched, &sched->source_program_generation, &sched->source_program_shrink_generation);
    sched->source_program_hybrid = hybrid;
    sched->source_program = prepared;
    sched->source_program_epoch = sched->source_retirement_epoch;
    sched->source_program_source_uid = sched->source_graph_uid;
    sched->source_program_split_uid = projected.split_graph_uid;
    sched->source_program_certificate = projected;
    sched->source_program_copy = sched->cur_copy;
    sched->source_program_backend = sched->splits[0].backend_id;
    std::copy(std::begin(sched->splits[0].direct_dependencies), std::end(sched->splits[0].direct_dependencies),
        sched->source_program_dependencies);
    sched->source_program_token = token;
    *program = token;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

enum ggml_status ggml_backend_sched_moe_source_program_compute_v1(ggml_backend_sched_t sched, uint64_t program,
        const ggml_graph_execution_certificate * certificate) {
    return ggml_backend_sched_moe_source_program_compute_with_phases_v1(sched, program, certificate, nullptr, 0);
}

enum ggml_status ggml_backend_sched_moe_source_program_compute_with_phases_v1(ggml_backend_sched_t sched, uint64_t program,
        const ggml_graph_execution_certificate * certificate, const uint8_t * phases, size_t n_phases) {
    if (!sched || !program || program != sched->source_program_token || !sched->is_alloc ||
            sched->source_retirement_failed || sched->callback_eval ||
            sched->source_program_epoch != sched->source_retirement_epoch || sched->source_program_copy != sched->cur_copy ||
            !ggml_backend_sched_execution_certificate_valid(certificate)) { return GGML_STATUS_FAILED; }
    if ((phases == nullptr) != (n_phases == 0) || (phases && n_phases != certificate->n_rows)) { return GGML_STATUS_FAILED; }
    for (size_t i = 0; i < n_phases; ++i) {
        if (phases[i] > GGML_GRAPH_EXECUTION_PHASE_GENERATION) { return GGML_STATUS_FAILED; }
    }
    auto * hybrid = sched->source_program_hybrid;
    if (!hybrid || (phases && !hybrid->source_api->compute_program_with_phases)) { return GGML_STATUS_FAILED; }
    uint64_t generation = 0, shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched, &generation, &shrink_generation);
    const auto projected = ggml_backend_sched_split_certificate(sched->source_program_source_uid, sched->source_program_split_uid, *certificate);
    if (generation != sched->source_program_generation || shrink_generation != sched->source_program_shrink_generation ||
            memcmp(&projected, &sched->source_program_certificate, sizeof(projected))) {
        GGML_LOG_ERROR("%s: prepared storage/certificate rejected generation=%llu expected=%llu shrink=%llu expected_shrink=%llu\n",
            __func__, (unsigned long long) generation, (unsigned long long) sched->source_program_generation,
            (unsigned long long) shrink_generation, (unsigned long long) sched->source_program_shrink_generation);
        return GGML_STATUS_FAILED;
    }
    for (int i = 0; i < sched->n_backends; ++i) {
        if (!sched->source_program_dependencies[i]) { continue; }
        if (sched->events[i][sched->cur_copy]) { ggml_backend_event_synchronize(sched->events[i][sched->cur_copy]); }
        else { ggml_backend_synchronize(sched->backends[i]); }
    }
    const auto status = phases ? hybrid->source_api->compute_program_with_phases(
        hybrid->device, sched->source_program, &projected, phases, n_phases) :
        hybrid->source_api->compute_program(hybrid->device, sched->source_program, &projected);
    if (status != GGML_STATUS_SUCCESS) {
        return ggml_backend_sched_compute_failure(sched, status, hybrid, false,
            (certificate->flags & GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED) != 0);
    }
    const auto event = sched->events[sched->source_program_backend][sched->cur_copy];
    if (event) { ggml_backend_event_record(event, sched->backends[sched->source_program_backend]); }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_moe_source_selected_v1(sched)) {
        GGML_ASSERT(ggml_backend_sched_moe_source_drain_v1(sched) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    }
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->source_program_token = 0;
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->source_program_token = 0;
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int32_t ggml_backend_sched_region_finalize_v1(
        ggml_backend_sched_t                                  sched,
        const ggml_backend_sched_region_query_v1 *            region,
        ggml_backend_sched_region_handoff_v1 *                handoff) {
    if (handoff == nullptr || handoff->struct_size != sizeof(*handoff) ||
            handoff->abi_version != GGML_BACKEND_SCHED_REGION_FINALIZE_V1_VERSION) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    *handoff = {};
    ggml_backend_sched_region_handoff_v1 result = {};
    result.struct_size = sizeof(result);
    result.abi_version = GGML_BACKEND_SCHED_REGION_FINALIZE_V1_VERSION;
    if (sched == nullptr || region == nullptr || region->struct_size != sizeof(*region) ||
            region->abi_version != GGML_BACKEND_SCHED_REGION_FINALIZE_V1_VERSION || region->reserved32 != 0 ||
            region->graph == nullptr || region->body_nodes == nullptr || region->n_body_nodes == 0 ||
            (region->n_dynamic_inputs != 0 && region->dynamic_inputs == nullptr) ||
            (region->n_live_outputs != 0 && region->live_outputs == nullptr)) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    const ggml_cgraph * graph = region->graph;
    if (graph->uid == 0 || graph->n_nodes <= 0 || graph->nodes == nullptr || graph->n_nodes > graph->size ||
            !sched->is_alloc || sched->source_graph != graph || sched->source_graph_uid != graph->uid) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == nullptr) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED;
        }
    }
    if (sched->callback_eval != nullptr) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_CALLBACK;
    }
    if (region->n_dynamic_inputs > GGML_BACKEND_SCHED_REGION_MAX_DYNAMIC_INPUTS_V1) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_CAPACITY;
    }

    int first = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == region->body_nodes[0]) {
            first = i;
            break;
        }
    }
    if (first < 0 || region->n_body_nodes > (uint32_t) (graph->n_nodes - first)) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
    }
    for (uint32_t i = 0; i < region->n_body_nodes; ++i) {
        if (region->body_nodes[i] == nullptr || graph->nodes[first + i] != region->body_nodes[i]) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (region->body_nodes[i] == region->body_nodes[j]) {
                return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
            }
        }
    }
    const int last = first + region->n_body_nodes - 1;
    const int tail = last + 1;
    if ((tail < graph->n_nodes ? graph->nodes[tail] : nullptr) != region->tail_resume) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
    }

    int split_index = -1;
    for (int i = 0; i < sched->n_splits; ++i) {
        const auto & split = sched->splits[i];
        if (first >= split.i_start && last < split.i_end) {
            split_index = i;
            break;
        }
    }
    if (split_index < 0 || sched->splits[split_index].graph.uid == 0) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_CROSS_SPLIT;
    }
    const auto & split = sched->splits[split_index];
    if (split.graph.n_nodes != split.i_end - split.i_start || split.graph.nodes == nullptr ||
            split.graph.nodes != graph->nodes + split.i_start) {
        return GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED;
    }

    const auto body_index = [&](const ggml_tensor * tensor) {
        for (uint32_t i = 0; i < region->n_body_nodes; ++i) {
            if (region->body_nodes[i] == tensor) {
                return (int) i;
            }
        }
        return -1;
    };
    result.n_dynamic_inputs = region->n_dynamic_inputs;
    for (uint32_t i = 0; i < region->n_dynamic_inputs; ++i) {
        const ggml_tensor * input = region->dynamic_inputs[i];
        if (input == nullptr || body_index(input) >= 0) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (region->dynamic_inputs[j] == input) {
                return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        const ggml_tensor * finalized = input;
        bool referenced = false;
        for (uint32_t node = 0; node < region->n_body_nodes && !referenced; ++node) {
            const auto * body = region->body_nodes[node];
            referenced = body->view_src == input;
            for (int src = 0; src < GGML_MAX_SRC && !referenced; ++src) {
                referenced = body->src[src] == input;
            }
        }
        for (int j = 0; !referenced && j < split.n_inputs; ++j) {
            if (split.inputs[j] != input) {
                continue;
            }
            finalized = tensor_copy(const_cast<ggml_tensor *>(input), split.backend_id, sched->cur_copy);
            for (uint32_t node = 0; node < region->n_body_nodes && !referenced; ++node) {
                const auto * body = region->body_nodes[node];
                referenced = body->view_src == finalized;
                for (int src = 0; src < GGML_MAX_SRC && !referenced; ++src) {
                    referenced = body->src[src] == finalized;
                }
            }
        }
        if (!referenced || finalized == nullptr || body_index(finalized) >= 0) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
        }
        result.dynamic_inputs[i] = finalized;
    }

    const auto dynamic_input = [&](const ggml_tensor * tensor) {
        for (uint32_t i = 0; i < result.n_dynamic_inputs; ++i) {
            if (result.dynamic_inputs[i] == tensor) {
                return true;
            }
        }
        return false;
    };
    for (uint32_t i = 0; i < region->n_body_nodes; ++i) {
        const auto * node = region->body_nodes[i];
        const bool metadata_view = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                                   node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
        if (node->view_src != nullptr && !metadata_view) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
        }
        for (int src = 0; src < GGML_MAX_SRC; ++src) {
            const auto * tensor = node->src[src];
            const int dependency = body_index(tensor);
            if (dependency >= (int) i ||
                    (tensor != nullptr && dependency < 0 && !dynamic_input(tensor) && tensor->op != GGML_OP_NONE)) {
                return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
            }
        }
        const int view_dependency = body_index(node->view_src);
        if (view_dependency >= (int) i ||
                (node->view_src != nullptr && view_dependency < 0 && !dynamic_input(node->view_src) &&
                 node->view_src->op != GGML_OP_NONE)) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
        }
    }

    for (uint32_t i = 0; i < region->n_live_outputs; ++i) {
        const auto & live = region->live_outputs[i];
        if (live.reserved != 0 || live.tensor == nullptr || body_index(live.tensor) < 0 ||
                (live.n_consumers == 0 && (live.tensor->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) ||
                (live.n_consumers != 0 && live.consumers == nullptr)) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (region->live_outputs[j].tensor == live.tensor) {
                return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        for (uint32_t j = 0; j < live.n_consumers; ++j) {
            const ggml_tensor * consumer = live.consumers[j];
            for (uint32_t previous = 0; previous < j; ++previous) {
                if (live.consumers[previous] == consumer) {
                    return GGML_BACKEND_SCHED_REGION_STATUS_V1_INVALID_ARGUMENT;
                }
            }
            int consumer_index = -1;
            for (int node = last + 1; node < graph->n_nodes; ++node) {
                if (graph->nodes[node] == consumer) {
                    consumer_index = node;
                    break;
                }
            }
            const bool consumes = ggml_backend_sched_region_consumes_v1(consumer, live.tensor);
            if (consumer_index < 0 || !consumes) {
                return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
            }
        }
    }
    for (uint32_t body = 0; body < region->n_body_nodes; ++body) {
        if ((region->body_nodes[body]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            continue;
        }
        bool declared = false;
        for (uint32_t i = 0; i < region->n_live_outputs; ++i) {
            declared = declared || region->live_outputs[i].tensor == region->body_nodes[body];
        }
        if (!declared) {
            return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
        }
    }
    for (int node = 0; node < graph->n_nodes; ++node) {
        if (node >= first && node <= last) {
            continue;
        }
        const auto * consumer = graph->nodes[node];
        for (uint32_t body = 0; body < region->n_body_nodes; ++body) {
            const auto * tensor = region->body_nodes[body];
            const bool consumes = ggml_backend_sched_region_consumes_v1(consumer, tensor);
            if (!consumes) {
                continue;
            }
            bool declared = false;
            for (uint32_t i = 0; i < region->n_live_outputs && !declared; ++i) {
                const auto & live = region->live_outputs[i];
                if (live.tensor != tensor) {
                    continue;
                }
                for (uint32_t j = 0; j < live.n_consumers; ++j) {
                    declared = live.consumers[j] == consumer;
                    if (declared) {
                        break;
                    }
                }
            }
            if (!declared) {
                return GGML_BACKEND_SCHED_REGION_STATUS_V1_INCOMPLETE_CUT;
            }
        }
    }

    result.source_graph_uid = graph->uid;
    result.split_graph_uid  = split.graph.uid;
    result.split_index      = (uint32_t) split_index;
    result.first_node_index = (uint32_t) first;
    result.last_node_index  = (uint32_t) last;
    result.tail_node_index  = (uint32_t) tail;
    *handoff = result;
    return GGML_BACKEND_SCHED_REGION_STATUS_V1_OK;
}

uint64_t ggml_backend_moe_graph_uid_v1(const ggml_cgraph * graph) {
    return graph != nullptr ? graph->uid : 0;
}

bool ggml_backend_moe_graph_assign_uid_v1(ggml_cgraph * graph) {
    if (graph == nullptr) {
        return false;
    }
    graph->uid = ggml_graph_next_uid();
    return graph->uid != 0;
}

bool ggml_backend_sched_region_snapshot_graph_v1(
        ggml_cgraph *                                  graph,
        const ggml_backend_sched_region_handoff_v1 *   handoff,
        ggml_tensor * const *                          nodes,
        uint32_t                                       n_nodes,
        ggml_tensor * const *                          leafs,
        uint32_t                                       n_leafs) {
    if (graph == nullptr || handoff == nullptr || handoff->struct_size != sizeof(*handoff) ||
            handoff->abi_version != GGML_BACKEND_SCHED_REGION_FINALIZE_V1_VERSION ||
            handoff->source_graph_uid == 0 || handoff->split_graph_uid == 0 || n_nodes == 0 || nodes == nullptr ||
            (n_leafs != 0 && leafs == nullptr) || graph->size <= 0 || graph->nodes == nullptr || graph->leafs == nullptr ||
            n_nodes > (uint32_t) graph->size || n_leafs > (uint32_t) graph->size) {
        return false;
    }
    if ((size_t) n_leafs > SIZE_MAX - (size_t) n_nodes ||
            (size_t) n_nodes > SIZE_MAX / sizeof(*nodes) || (size_t) n_leafs > SIZE_MAX / sizeof(*leafs)) {
        return false;
    }
    const size_t n_tensors = (size_t) n_leafs + n_nodes;
    if (n_tensors > graph->visited_hash_set.size) {
        return false;
    }
    ggml_hash_set_reset(&graph->visited_hash_set);
    memset(graph->use_counts, 0, graph->visited_hash_set.size * sizeof(*graph->use_counts));
    for (size_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * tensor = i < n_leafs ? leafs[i] : nodes[i - n_leafs];
        if (tensor == nullptr || ggml_hash_insert(&graph->visited_hash_set, tensor) == GGML_HASHSET_ALREADY_EXISTS) {
            ggml_hash_set_reset(&graph->visited_hash_set);
            return false;
        }
    }
    for (size_t i = 0; i < n_tensors; ++i) {
        const ggml_tensor * tensor = i < n_leafs ? leafs[i] : nodes[i - n_leafs];
        for (const auto * src : tensor->src) {
            if (src == nullptr) {
                continue;
            }
            const size_t source = ggml_hash_find(&graph->visited_hash_set, src);
            if (source == GGML_HASHSET_FULL || !ggml_bitset_get(graph->visited_hash_set.used, source) ||
                    graph->use_counts[source] == INT32_MAX) {
                ggml_hash_set_reset(&graph->visited_hash_set);
                return false;
            }
            graph->use_counts[source]++;
        }
    }
    graph->n_nodes = n_nodes;
    graph->n_leafs = n_leafs;
    memcpy(graph->nodes, nodes, n_nodes * sizeof(*nodes));
    if (n_leafs != 0) {
        memcpy(graph->leafs, leafs, n_leafs * sizeof(*leafs));
    }
    return true;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    if (tensor_backend_id(node) != backend_index) { sched->source_program_token = 0; }
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}

// This process selection is fixed before a prepared region is measured.
const ggml_moe_fidelity_config & ggml_moe_fidelity_selection() {
    static const ggml_moe_fidelity_config config = [] {
        ggml_moe_fidelity_config c;
        const char * mode = std::getenv("GGML_MOE_FIDELITY_PIPELINE");
        if (mode && std::strcmp(mode, "control") && std::strcmp(mode, "reference") &&
                std::strcmp(mode, "conversion") && std::strcmp(mode, "reference-conversion")) { c.valid = false; }
        c.reference = mode && (!std::strcmp(mode, "reference") || !std::strcmp(mode, "reference-conversion"));
        c.source_pool = mode && (!std::strcmp(mode, "conversion") || !std::strcmp(mode, "reference-conversion"));
        const char * executor = std::getenv("GGML_MOE_HYBRID_EXECUTOR");
        const char * hybrid = std::getenv("GGML_MOE_HYBRID");
        const bool source_executor = executor ? !std::strcmp(executor, "source") :
            hybrid && !std::strcmp(hybrid, "required");
        if (source_executor) {
            if (mode && std::strcmp(mode, "reference-conversion")) { c.valid = false; }
            c.reference = c.source_pool = true;
        }
        const char * value = std::getenv(source_executor ? "GGML_MOE_SOURCE_GPU_MISS_FRACTION" : "GGML_MOE_FIDELITY_PCIE_FRAC");
        const char * tuning = std::getenv("GGML_MOE_SOURCE_MISS_TUNING");
        if (tuning && std::strcmp(tuning, "on") && std::strcmp(tuning, "off")) { c.valid = false; }
        c.tune_misses = source_executor && !value && (!tuning || !std::strcmp(tuning, "on"));
        const char * keep = std::getenv("GGML_MOE_SOURCE_KEEP_RANKS");
        if (keep) {
            char * end = nullptr;
            errno = 0;
            const unsigned long rank = std::strtoul(keep, &end, 10);
            if (errno || end == keep || *end || std::strchr(keep, '-') || rank > UINT32_MAX) { c.valid = false; }
            else { c.keep_ranks = uint32_t(rank); }
        }
        if (source_executor && !value) { value = "0.17"; }
        if (c.reference && !value) { c.valid = false; }
        if (value) {
            char * end = nullptr;
            errno = 0;
            const double fraction = std::strtod(value, &end);
            if (errno || end == value || *end || !std::isfinite(fraction) || fraction < 0 || fraction > 1) { c.valid = false; }
            else { c.pcie_num = unsigned(std::floor(fraction * 256.0 + 0.5)); }
        }
        const char * cache = std::getenv("GGML_MOE_FIDELITY_CACHE_POLICY");
        if (cache && std::strcmp(cache, "static")) { c.valid = false; }
        const auto unsupported_adapt = [](const char * entry) {
            constexpr char prefix[] = "GGML_MOE_FIDELITY_ADAPT_";
            for (size_t i = 0; i < sizeof(prefix) - 1; ++i) {
                char character = entry[i];
#ifdef _WIN32
                if (character >= 'a' && character <= 'z') { character -= 'a' - 'A'; }
#endif
                if (character != prefix[i]) { return false; }
            }
            return true;
        };
#ifdef _WIN32
        char * environment = GetEnvironmentStringsA();
        if (!environment) { c.valid = false; }
        else {
            for (const char * entry = environment; *entry; entry += std::strlen(entry) + 1) {
                if (unsupported_adapt(entry)) { c.valid = false; break; }
            }
            FreeEnvironmentStringsA(environment);
        }
#else
#ifdef __APPLE__
        char ** environment = *_NSGetEnviron();
#else
        char ** environment = environ;
#endif
        for (char ** entry = environment; entry && *entry; ++entry) {
            if (unsupported_adapt(*entry)) { c.valid = false; break; }
        }
#endif
        return c;
    }();
    return config;
}
