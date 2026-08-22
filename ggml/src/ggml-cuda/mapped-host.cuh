#pragma once

#include "ggml-backend.h"

// Return the device-0 alias resolved when a CUDA_MappedHost buffer was
// allocated. Zero-sized no-alloc buffers and other buffer types return null.
void * ggml_cuda_mapped_host_buffer_device_base(ggml_backend_buffer_t buffer);
