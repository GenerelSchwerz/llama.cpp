#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-moe.h"
#include "ggml-cpu.h"
#include "moe-fidelity.h"
#include "moe-source-pool.h"
#include "../moe-fidelity-config.h"
#include "tiled/tiled.h"
#include "ops.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#    include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#    include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types() {
    static std::vector<ggml_backend_buffer_type_t> bufts = []() {
        std::vector<ggml_backend_buffer_type_t> bufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        if (ggml_backend_amx_buffer_type()) {
            bufts.push_back(ggml_backend_amx_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
        if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
        if (ggml_backend_cpu_kleidiai_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
        }
#endif

#ifdef GGML_USE_CPU_REPACK
        if (ggml_backend_cpu_repack_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_repack_buffer_type());
        }
#endif

        return bufts;
    }();

    return bufts;
}

static ggml_backend_buffer_type_t * ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device) {
    static std::vector<ggml_backend_buffer_type_t> extra_bufts = [] {
        std::vector<ggml_backend_buffer_type_t> bufts = ggml_backend_cpu_get_extra_buffer_types();
        bufts.push_back(nullptr);
        return bufts;
    }();

    return extra_bufts.data();

    GGML_UNUSED(device);
}

static bool ggml_backend_cpu_is_extra_buffer_type(ggml_backend_buffer_type_t buft) {
    for (auto * extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra == buft) {
            return true;
        }
    }
    return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool                use_ref;  // use reference implementation
    ggml_backend_get_rows_callback get_rows_callback = nullptr;
    void * get_rows_callback_data = nullptr;
};

static const char * ggml_backend_cpu_get_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete cpu_ctx;
    delete backend;
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
    ggml_backend_t backend;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = new ggml_backend_plan_cpu;

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy
    cpu_plan->backend = backend;

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
        if (cpu_plan->cplan.work_data == NULL) {
            delete cpu_plan;
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.use_ref             = cpu_ctx->use_ref;
    cpu_plan->cplan.get_rows_callback = cpu_ctx->get_rows_callback;
    cpu_plan->cplan.get_rows_callback_data = cpu_ctx->get_rows_callback_data;

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    delete[] cpu_plan->cplan.work_data;
    delete cpu_plan;

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.use_ref             = cpu_ctx->use_ref;
    cplan.get_rows_callback = cpu_ctx->get_rows_callback;
    cplan.get_rows_callback_data = cpu_ctx->get_rows_callback_data;

    return ggml_graph_compute(cgraph, &cplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    // initialize CPU backend now to avoid slowing the first graph computation
    ggml_cpu_init();

    struct ggml_backend_cpu_context * ctx = new ggml_backend_cpu_context;
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->threadpool          = NULL;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->use_ref             = false;

    ggml_backend_t cpu_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cpu_guid(),
        /* .iface   = */ ggml_backend_cpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ctx,
    };

    if (cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }

    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

bool ggml_backend_cpu_graph_plan_set_mmid_route_filter(ggml_backend_t backend_cpu,
        ggml_backend_graph_plan_t plan, ggml_cpu_mmid_route_filter filter, void * user_data) {
    if (!ggml_backend_is_cpu(backend_cpu) || !plan) { return false; }
    auto * cpu_plan = static_cast<ggml_backend_plan_cpu *>(plan);
    if (cpu_plan->backend != backend_cpu) { return false; }
    if (filter) {
        for (int i = 0; i < cpu_plan->cgraph.n_nodes; ++i) {
            const auto * node = cpu_plan->cgraph.nodes[i];
            if (node->op == GGML_OP_MUL_MAT_ID && !ggml_cpu_extra_supports_mmid_route_filter(node)) { return false; }
        }
    }
    cpu_plan->cplan.mmid_route_filter = filter;
    cpu_plan->cplan.mmid_route_filter_data = filter ? user_data : nullptr;
    return true;
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;

    if (ctx->threadpool && ctx->threadpool != threadpool) {
        // already had a different threadpool, pause/suspend it before switching
        ggml_threadpool_pause(ctx->threadpool);
    }
    ctx->threadpool = threadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->use_ref = use_ref;
}

static void ggml_backend_cpu_set_get_rows_callback(ggml_backend_t backend, ggml_backend_get_rows_callback callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend));
    auto * ctx = static_cast<ggml_backend_cpu_context *>(backend->context);
    ctx->get_rows_callback = callback;
    ctx->get_rows_callback_data = user_data;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
    std::string description = "CPU";

    ggml_backend_cpu_device_context() {
#ifdef __APPLE__
        size_t len = 0;
        if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
            description.resize(len);
            sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
        }
#elif defined(__linux__)
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) {
                if (strncmp(buf, "model name", 10) == 0) {
                    char * p = strchr(buf, ':');
                    if (p) {
                        p++;
                        while (std::isspace(*p)) {
                            p++;
                        }
                        while (std::isspace(p[strlen(p) - 1])) {
                            p[strlen(p) - 1] = '\0';
                        }
                        description = p;
                        break;
                    }
                }
            }
            fclose(f);
        }
#elif defined(_WIN32)
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                        0,
                        KEY_READ,
                        &hKey) == ERROR_SUCCESS) {
            DWORD cpu_brand_size = 0;
            if (RegQueryValueExA(hKey,
                                "ProcessorNameString",
                                NULL,
                                NULL,
                                NULL,
                                &cpu_brand_size) == ERROR_SUCCESS) {
                description.resize(cpu_brand_size);
                if (RegQueryValueExA(hKey,
                                    "ProcessorNameString",
                                    NULL,
                                    NULL,
                                    (LPBYTE)&description[0], // NOLINT
                                    &cpu_brand_size) == ERROR_SUCCESS) {
                    if (description.find('\0') != std::string::npos) {
                        description.resize(description.find('\0'));
                    }
                }
            }
            RegCloseKey(hKey);
        }
#endif
    }
};

static const char * ggml_backend_cpu_device_get_name(ggml_backend_dev_t dev) {
    return "CPU";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_cpu_device_get_description(ggml_backend_dev_t dev) {
    struct ggml_backend_cpu_device_context * ctx = (struct ggml_backend_cpu_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_cpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_cpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_cpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cpu_device_get_name(dev);
    props->description = ggml_backend_cpu_device_get_description(dev);
    props->type        = ggml_backend_cpu_device_get_type(dev);
    ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_cpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_cpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->op == GGML_OP_GATED_DELTA_NET && !ggml_gated_delta_net_validate(op)) {
        return false;
    }

    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW || op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }

    // check extra buffer types
    // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
    for (int i = 0; i < 4; i++) {
        if (op->src[i] && op->src[i]->buffer &&
            ggml_backend_cpu_is_extra_buffer_type(op->src[i]->buffer->buft)) {
            auto * buf_extra = (ggml::cpu::extra_buffer_type *) op->src[i]->buffer->buft->context;
            return buf_extra->supports_op(dev, op);
        }
    }

    switch (op->op) {
        case GGML_OP_SET_ROWS:
            if (op->src[0] != nullptr && op->src[0]->type == op->type) {
                return true;
            }
            [[fallthrough]];
        case GGML_OP_CPY:
            return
                op->type != GGML_TYPE_IQ3_XXS &&
                op->type != GGML_TYPE_IQ3_S   &&
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ2_S   &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16;
            }
            // BF16 in src1 is widened into the F32 work buffer
            {
                struct ggml_cpu_mul_mat_kernel kernel;
                return ggml_cpu_get_mul_mat_kernel(op, &kernel) &&
                    (src1->type == GGML_TYPE_F32 || src1->type == kernel.input_type ||
                     (src1->type == GGML_TYPE_BF16 && kernel.input_type == GGML_TYPE_F32));
            }
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_cpu_mul_mat_kernel kernel;
                return ggml_cpu_get_mul_mat_kernel(op, &kernel) &&
                    (src1->type == kernel.input_type || (src1->type == GGML_TYPE_F32 && kernel.from_float != nullptr));
            }
        case GGML_OP_SOFT_MAX_BACK: {
            if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
                return false;
            }
            float max_bias = 0.0f;

            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return max_bias == 0.0f;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            const ggml_tensor * mask = op->src[3];
            if (mask == nullptr || mask->type == GGML_TYPE_F16) {
                return true;
            }

            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));

            return mask->type == GGML_TYPE_I64 && mask->ne[0] == src0->ne[1] &&
                mask->ne[1] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1 && max_bias == 0.0f;
        }
        case GGML_OP_IM2COL_BACK:
            return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        case GGML_OP_GET_ROWS_BACK:
            return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16;
        case GGML_OP_OUT_PROD:
            return (src0->type == GGML_TYPE_F32 ||
                    ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3])) &&
                src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SSM_SCAN:
            return ggml_get_op_params_i32(op, 0) == 1 || op->src[3]->ne[0] == 1;
        default:
            return true;
    }
}

static bool ggml_backend_cpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ ggml_backend_cpu_device_get_name,
    /* .get_description      = */ ggml_backend_cpu_device_get_description,
    /* .get_memory           = */ ggml_backend_cpu_device_get_memory,
    /* .get_type             = */ ggml_backend_cpu_device_get_type,
    /* .get_props            = */ ggml_backend_cpu_device_get_props,
    /* .init_backend         = */ ggml_backend_cpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_cpu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_cpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_cpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_cpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_cpu_device_context ctx;
    static ggml_backend_device ggml_backend_cpu_device = {
        /* .iface   = */ ggml_backend_cpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };

    return &ggml_backend_cpu_device;
}

static bool ggml_backend_moe_cpu_region_add_v1(uint64_t a, uint64_t b, uint64_t * result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    *result = a + b;
    return true;
}

static bool ggml_backend_moe_cpu_region_mul_v1(uint64_t a, uint64_t b, uint64_t * result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

static constexpr uint64_t GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 =
    CACHE_LINE_SIZE > GGML_MEM_ALIGN ? CACHE_LINE_SIZE : GGML_MEM_ALIGN;

static bool ggml_backend_moe_cpu_region_align_v1(uint64_t value, uint64_t * result) {
    uint64_t padded;
    return ggml_backend_moe_cpu_region_add_v1(value, GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 - 1, &padded) &&
           (*result = padded / GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 *
                      GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1, true);
}

static bool ggml_backend_moe_cpu_region_graph_has_v1(const struct ggml_cgraph * graph,
                                                     const struct ggml_tensor * tensor) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == tensor) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_moe_cpu_region_array_has_v1(const struct ggml_tensor * const * tensors,
                                                     uint32_t                           n_tensors,
                                                     const struct ggml_tensor *         tensor) {
    for (uint32_t i = 0; i < n_tensors; ++i) {
        if (tensors[i] == tensor) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_moe_cpu_region_referenced_v1(const struct ggml_cgraph * graph,
                                                      const struct ggml_tensor * tensor) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (graph->nodes[i]->src[j] == tensor) {
                return true;
            }
        }
    }
    return false;
}

static const struct ggml_backend_moe_cpu_region_source_v1 * ggml_backend_moe_cpu_region_find_source_v1(
    const struct ggml_backend_moe_cpu_region_query_v1 * query,
    const struct ggml_tensor *                          tensor) {
    const struct ggml_backend_moe_cpu_region_source_v1 * result = nullptr;
    for (uint32_t i = 0; i < query->n_sources; ++i) {
        if (query->sources[i].tensor == tensor) {
            if (result != nullptr) {
                return nullptr;
            }
            result = &query->sources[i];
        }
    }
    return result;
}

static bool ggml_backend_moe_cpu_region_supported_op_v1(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_GLU:
        case GGML_OP_VIEW:
        case GGML_OP_ADD_ID:
        case GGML_OP_UNARY:
        case GGML_OP_SQR:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_moe_cpu_region_valid_tensor_layout_v1(const struct ggml_tensor * tensor,
                                                               uint64_t *                 span_bytes);

static bool ggml_backend_moe_cpu_region_tensor_elements_v1(const struct ggml_tensor * tensor, uint64_t * elements) {
    uint64_t result = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] <= 0 || !ggml_backend_moe_cpu_region_mul_v1(result, (uint64_t) tensor->ne[dim], &result) ||
            result > INT64_MAX) {
            return false;
        }
    }
    *elements = result;
    return true;
}

static bool ggml_backend_moe_cpu_region_row_bytes_v1(enum ggml_type type, uint64_t elements, uint64_t * bytes) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        return false;
    }
    const int64_t block_size = ggml_blck_size(type);
    const size_t  type_size  = ggml_type_size(type);
    return block_size > 0 && type_size > 0 && elements > 0 && elements % block_size == 0 &&
           ggml_backend_moe_cpu_region_mul_v1(type_size, elements / block_size, bytes) && *bytes > 0;
}

static bool ggml_backend_moe_cpu_region_exact_layout_v1(const struct ggml_tensor * tensor) {
    uint64_t row_bytes;
    if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(tensor, nullptr) ||
        !ggml_backend_moe_cpu_region_row_bytes_v1(tensor->type, tensor->ne[0], &row_bytes) ||
        tensor->nb[1] != row_bytes) {
        return false;
    }
    for (int dim = 2; dim < GGML_MAX_DIMS; ++dim) {
        uint64_t stride;
        if (!ggml_backend_moe_cpu_region_mul_v1(tensor->nb[dim - 1], tensor->ne[dim - 1], &stride) ||
            tensor->nb[dim] != stride) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_moe_cpu_region_valid_glu_v1(const struct ggml_tensor * node, uint32_t n_threads) {
    if (node->src[0] == nullptr || node->src[1] == nullptr || node->type != GGML_TYPE_F32 ||
        node->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32 ||
        !ggml_backend_moe_cpu_region_valid_tensor_layout_v1(node, nullptr) ||
        !ggml_backend_moe_cpu_region_valid_tensor_layout_v1(node->src[0], nullptr) ||
        !ggml_backend_moe_cpu_region_valid_tensor_layout_v1(node->src[1], nullptr) ||
        !ggml_are_same_shape(node, node->src[0]) || !ggml_are_same_shape(node->src[0], node->src[1]) ||
        !ggml_is_contiguous_1(node) || !ggml_is_contiguous_1(node->src[0]) || !ggml_is_contiguous_1(node->src[1]) ||
        node->op_params[1] != 0) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] != nullptr) {
            return false;
        }
    }
    uint64_t rows = 1;
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (!ggml_backend_moe_cpu_region_mul_v1(rows, node->src[0]->ne[dim], &rows)) {
            return false;
        }
    }
    uint64_t rows_with_threads;
    if (node->src[0]->ne[0] > INT_MAX || rows > INT_MAX ||
        !ggml_backend_moe_cpu_region_add_v1(rows, n_threads - 1, &rows_with_threads) || rows_with_threads > INT_MAX) {
        return false;
    }
    const enum ggml_glu_op op = ggml_get_glu_op(node);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    for (size_t i = 2; i < sizeof(node->op_params) / sizeof(node->op_params[0]); ++i) {
        if (node->op_params[i] != 0) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_moe_cpu_region_valid_tensor_layout_v1(const struct ggml_tensor * tensor,
                                                               uint64_t *                 span_bytes) {
    if (tensor == nullptr || tensor->type < 0 || tensor->type >= GGML_TYPE_COUNT) {
        return false;
    }
    const int64_t block_size = ggml_blck_size(tensor->type);
    const size_t  type_size  = ggml_type_size(tensor->type);
    if (block_size <= 0 || type_size == 0 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 ||
        tensor->ne[3] <= 0 || tensor->ne[0] % block_size != 0 || tensor->nb[0] != type_size) {
        return false;
    }
    uint64_t row_bytes;
    if (!ggml_backend_moe_cpu_region_mul_v1(type_size, (uint64_t) tensor->ne[0] / block_size, &row_bytes) ||
        row_bytes == 0 || tensor->nb[1] < row_bytes) {
        return false;
    }
    for (int dim = 2; dim < GGML_MAX_DIMS; ++dim) {
        uint64_t required_stride;
        if (tensor->nb[dim - 1] == 0 || tensor->nb[dim] == 0 ||
            !ggml_backend_moe_cpu_region_mul_v1(tensor->nb[dim - 1], tensor->ne[dim - 1], &required_stride) ||
            tensor->nb[dim] < required_stride) {
            return false;
        }
    }
    uint64_t span = row_bytes;
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        uint64_t extent;
        if (!ggml_backend_moe_cpu_region_mul_v1((uint64_t) tensor->ne[dim] - 1, tensor->nb[dim], &extent) ||
            !ggml_backend_moe_cpu_region_add_v1(span, extent, &span)) {
            return false;
        }
    }
    uint64_t elements;
    if (span > SIZE_MAX || !ggml_backend_moe_cpu_region_tensor_elements_v1(tensor, &elements)) {
        return false;
    }
    if (span_bytes != nullptr) {
        *span_bytes = span;
    }
    return true;
}

static int32_t ggml_backend_moe_cpu_region_check_mmid_v1(const struct ggml_backend_moe_cpu_region_query_v1 * query,
                                                         const struct ggml_tensor *                          node,
                                                         uint32_t * expert_count) {
    const struct ggml_tensor * weight = node->src[0];
    const struct ggml_tensor * input  = node->src[1];
    const struct ggml_tensor * ids    = node->src[2];
    if (weight == nullptr || input == nullptr || ids == nullptr || ids != query->ids || ids->type != GGML_TYPE_I32 ||
        node->type != GGML_TYPE_F32 || weight->extra != nullptr ||
        (weight->buffer != nullptr && ggml_backend_cpu_is_extra_buffer_type(weight->buffer->buft)) ||
        !ggml_backend_moe_cpu_region_exact_layout_v1(weight) ||
        !ggml_backend_moe_cpu_region_exact_layout_v1(input) ||
        ((query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0 ?
            !ggml_backend_moe_cpu_region_valid_tensor_layout_v1(ids, nullptr) : !ggml_backend_moe_cpu_region_exact_layout_v1(ids)) ||
        !ggml_backend_moe_cpu_region_exact_layout_v1(node) || weight->ne[3] != 1 || input->ne[3] != 1 ||
        ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[1] != input->ne[2] || weight->ne[0] != input->ne[0] ||
        input->ne[1] <= 0 || ids->ne[0] <= 0 || ids->ne[0] % input->ne[1] != 0 || weight->ne[2] <= 0 ||
        weight->ne[2] > (INT_MAX - CACHE_LINE_SIZE) / CACHE_LINE_SIZE || weight->ne[0] > INT_MAX ||
        weight->ne[1] > INT_MAX || input->ne[0] > INT_MAX || input->ne[1] > INT_MAX || input->ne[2] > INT_MAX ||
        node->ne[0] > INT_MAX || node->ne[1] > INT_MAX || node->ne[2] > INT_MAX || node->ne[0] != weight->ne[1] ||
        node->ne[1] != ids->ne[0] || node->ne[2] != input->ne[2] || node->ne[3] != 1 || node->nb[0] != sizeof(float)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    uint64_t per_expert_elements;
    uint64_t input_elements;
    uint64_t output_elements;
    uint64_t kernel_chunks;
    if (!ggml_backend_moe_cpu_region_mul_v1(weight->ne[0], weight->ne[1], &per_expert_elements) ||
        !ggml_backend_moe_cpu_region_tensor_elements_v1(input, &input_elements) ||
        !ggml_backend_moe_cpu_region_tensor_elements_v1(node, &output_elements) ||
        !ggml_backend_moe_cpu_region_mul_v1(weight->ne[1], query->bucket_rows * (uint64_t) query->routes_per_row,
                                            &kernel_chunks) ||
        per_expert_elements > INT_MAX || input_elements > INT_MAX || output_elements > INT_MAX ||
        kernel_chunks > INT_MAX) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    for (int i = 3; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] != nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
    }
    if ((query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) == 0) {
        if (node->op_params[0] != GGML_PREC_UNDEFINED || node->op_params[3] != GGML_PREC_UNDEFINED) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_PRECISION;
        }
        for (size_t i = 1; i < sizeof(node->op_params) / sizeof(node->op_params[0]); ++i) {
            if (i != 3 && node->op_params[i] != 0) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
        }
    }
    struct ggml_cpu_mul_mat_kernel kernel;
    if (!ggml_cpu_get_mul_mat_kernel(node, &kernel) ||
        (input->type != GGML_TYPE_F32 && input->type != kernel.input_type) ||
        (input->type == GGML_TYPE_F32 && kernel.convert_input && kernel.from_float == nullptr)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    const auto * source = ggml_backend_moe_cpu_region_find_source_v1(query, weight);
    if (source == nullptr || source->expert_stride != weight->nb[2]) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE;
    }
    if (*expert_count == 0) {
        *expert_count = (uint32_t) weight->ne[2];
    } else if (*expert_count != (uint32_t) weight->ne[2]) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static bool ggml_backend_moe_cpu_region_valid_view_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query, const ggml_tensor * node) {
    const auto * source = node->src[0];
    if (source == nullptr || node->view_src != source || source->view_src != nullptr ||
            !ggml_backend_moe_cpu_region_graph_has_v1(query->graph, source) ||
            node->type != GGML_TYPE_F32 || source->type != GGML_TYPE_F32 ||
            !ggml_backend_moe_cpu_region_valid_tensor_layout_v1(node, nullptr) ||
            !ggml_backend_moe_cpu_region_exact_layout_v1(source)) {
        return false;
    }
    uint64_t row_end;
    if (node->view_offs % sizeof(float) != 0 ||
            !ggml_backend_moe_cpu_region_add_v1(node->view_offs, uint64_t(node->ne[0]) * sizeof(float), &row_end) ||
            row_end > source->nb[1]) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (node->ne[dim] != source->ne[dim] || node->nb[dim] != source->nb[dim]) {
            return false;
        }
    }
    for (int i = 1; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] != nullptr) {
            return false;
        }
    }
    size_t offset;
    memcpy(&offset, node->op_params, sizeof(offset));
    const auto * params = reinterpret_cast<const uint8_t *>(node->op_params);
    return offset == node->view_offs &&
        std::all_of(params + sizeof(offset), params + sizeof(node->op_params), [](uint8_t value) { return value == 0; });
}

static bool ggml_backend_moe_cpu_region_valid_add_id_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query, const ggml_tensor * node, uint32_t expert_count) {
    const auto * input = node->src[0];
    const auto * bias = node->src[1];
    if (input == nullptr || bias == nullptr || node->src[2] != query->ids || expert_count == 0 ||
            node->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            !ggml_backend_moe_cpu_region_graph_has_v1(query->graph, input) ||
            ggml_backend_moe_cpu_region_find_source_v1(query, bias) == nullptr ||
            !ggml_backend_moe_cpu_region_exact_layout_v1(node) ||
            !ggml_backend_moe_cpu_region_valid_tensor_layout_v1(input, nullptr) ||
            !ggml_backend_moe_cpu_region_exact_layout_v1(bias) || !ggml_are_same_shape(node, input) ||
            node->ne[0] > INT_MAX || node->ne[1] != query->routes_per_row || node->ne[2] != query->bucket_rows ||
            node->ne[3] != 1 || bias->ne[0] != node->ne[0] || bias->ne[1] != expert_count ||
            bias->ne[2] != 1 || bias->ne[3] != 1 ||
            uint64_t(query->routes_per_row) * query->bucket_rows > uint64_t(INT_MAX) - query->n_threads + 1) {
        return false;
    }
    for (int i = 3; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] != nullptr) {
            return false;
        }
    }
    return std::all_of(std::begin(node->op_params), std::end(node->op_params), [](int32_t value) { return value == 0; });
}

static bool ggml_backend_moe_cpu_region_checked_add_component_v1(uint64_t * total,
                                                                 uint64_t   count,
                                                                 uint64_t   size,
                                                                 uint64_t   padding) {
    uint64_t bytes;
    return ggml_backend_moe_cpu_region_mul_v1(count, size, &bytes) &&
           ggml_backend_moe_cpu_region_add_v1(bytes, padding, &bytes) &&
           ggml_backend_moe_cpu_region_add_v1(*total, bytes, total);
}

static int32_t ggml_backend_moe_cpu_region_plan_work_v1(const struct ggml_backend_moe_cpu_region_query_v1 * query,
                                                        const struct ggml_tensor *                          node,
                                                        uint64_t * work_bytes) {
    if (node->op != GGML_OP_MUL_MAT_ID) {
        *work_bytes = 0;
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    }

    const struct ggml_tensor *          weight = node->src[0];
    const struct ggml_tensor *          input  = node->src[1];
    const struct ggml_tensor *          ids    = node->src[2];
    struct ggml_cpu_mul_mat_kernel kernel;
    size_t input_work;
    if (!ggml_cpu_get_mul_mat_kernel(node, &kernel) ||
            !ggml_cpu_mul_mat_input_work_size(input, &kernel, &input_work)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    uint64_t work = input_work;
    if (work != 0 && !ggml_backend_moe_cpu_region_add_v1(work, sizeof(int64_t), &work)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    uint64_t routes;
    if (!ggml_backend_moe_cpu_region_mul_v1(ids->ne[0], ids->ne[1], &routes) ||
        !ggml_backend_moe_cpu_region_checked_add_component_v1(&work, weight->ne[2], sizeof(int64_t), sizeof(int64_t)) ||
        !ggml_backend_moe_cpu_region_mul_v1(weight->ne[2], routes, &routes) ||
        !ggml_backend_moe_cpu_region_checked_add_component_v1(&work, routes, 2 * sizeof(int32_t), sizeof(int64_t)) ||
        !ggml_backend_moe_cpu_region_checked_add_component_v1(&work, weight->ne[2], CACHE_LINE_SIZE, CACHE_LINE_SIZE)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    if (!ggml_backend_moe_cpu_region_add_v1(work, 63, &work)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    work = work / 64 * 64;
    const uint64_t tiled_work = ggml_tiled_wdata_size(query->n_threads, const_cast<ggml_tensor *>(node));
    if (!ggml_backend_moe_cpu_region_add_v1(work, tiled_work, &work)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }

    if (work > SIZE_MAX ||
            (query->lane_execution_byte_limit != 0 && work > query->lane_execution_byte_limit)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    size_t extra_work = 0;
    if (ggml_cpu_extra_work_size(query->n_threads, node, &extra_work)) {
        work = extra_work;
    }
    if (work > SIZE_MAX ||
            (query->lane_execution_byte_limit != 0 && work > query->lane_execution_byte_limit)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    *work_bytes = work;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_query_impl_v1(
    const struct ggml_backend_moe_cpu_region_query_v1 *  query,
    struct ggml_backend_moe_cpu_region_requirements_v1 * requirements, bool graph_execution = true) {
    if (query == nullptr || requirements == nullptr || query->struct_size < sizeof(*query) ||
        requirements->struct_size < sizeof(*requirements) || query->graph == nullptr ||
        query->graph->nodes == nullptr || query->graph->n_nodes <= 0 || query->graph->n_nodes > query->graph->size ||
        query->graph_uid == 0 || query->graph_uid != query->graph->uid || query->graph_generation == 0 ||
        query->source_generation == 0 ||
        query->body_nodes == nullptr || query->n_body_nodes != (uint32_t) query->graph->n_nodes ||
        query->activation == nullptr || query->ids == nullptr || query->dynamic_inputs == nullptr ||
        query->n_dynamic_inputs < 2 || query->live_outputs == nullptr || query->n_live_outputs == 0 ||
        query->n_live_outputs > GGML_BACKEND_MOE_CPU_REGION_MAX_LIVE_OUTPUTS_V1 || query->sources == nullptr ||
        query->n_sources == 0 || query->bucket_rows == 0 || query->routes_per_row == 0 ||
        query->source_row_capacity == 0 || query->scatter_capacity == 0 || query->n_threads == 0 ||
        query->n_threads > GGML_MAX_N_THREADS || query->n_threads > INT_MAX || query->n_lanes == 0 ||
        query->ids->type != GGML_TYPE_I32 || query->ids->ne[0] != query->routes_per_row ||
        query->ids->ne[1] != query->bucket_rows || query->ids->ne[2] != 1 || query->ids->ne[3] != 1 ||
        (query->activation->ne[1] != 1 &&
         (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) == 0) ||
        query->activation->ne[2] != query->bucket_rows ||
        !ggml_backend_moe_cpu_region_array_has_v1(query->dynamic_inputs, query->n_dynamic_inputs, query->activation) ||
        !ggml_backend_moe_cpu_region_array_has_v1(query->dynamic_inputs, query->n_dynamic_inputs, query->ids)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    struct ggml_backend_moe_cpu_region_requirements_v1 result = {};
    result.struct_size                                        = sizeof(result);
    if ((query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_HAS_LORA) != 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    const bool routed_operation = (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0;
    if ((query->flags & ~(GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_HAS_LORA |
                         GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES |
                         GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION)) != 0 ||
            ((query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES) != 0 &&
             query->routes_per_row != 1)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    if (routed_operation && (query->flags != GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION ||
            query->n_body_nodes != 1 || query->n_dynamic_inputs != 2 || query->n_live_outputs != 1 ||
            !query->body_nodes[0] || query->body_nodes[0]->op != GGML_OP_MUL_MAT_ID ||
            query->live_outputs[0] != query->body_nodes[0] || query->body_nodes[0]->src[1] != query->activation)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    uint64_t route_capacity;
    if (!ggml_backend_moe_cpu_region_mul_v1(query->bucket_rows, query->routes_per_row, &route_capacity) ||
        route_capacity > INT_MAX || query->scatter_capacity < route_capacity) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }

    for (uint32_t i = 0; i < query->n_body_nodes; ++i) {
        if (query->body_nodes[i] == nullptr || query->body_nodes[i] != query->graph->nodes[i]) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (query->body_nodes[i] == query->body_nodes[j]) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            const struct ggml_tensor * src = query->body_nodes[i]->src[src_index];
            if (src == query->body_nodes[i]) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
            if (src != nullptr && ggml_backend_moe_cpu_region_graph_has_v1(query->graph, src)) {
                bool before = false;
                for (uint32_t j = 0; j < i; ++j) {
                    before = before || query->body_nodes[j] == src;
                }
                if (!before) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
                }
            }
        }
    }
    for (uint32_t i = 0; i < query->n_dynamic_inputs; ++i) {
        if (query->dynamic_inputs[i] == nullptr ||
            ggml_backend_moe_cpu_region_graph_has_v1(query->graph, query->dynamic_inputs[i]) ||
            !ggml_backend_moe_cpu_region_referenced_v1(query->graph, query->dynamic_inputs[i])) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(query->dynamic_inputs[i], nullptr)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (query->dynamic_inputs[i] == query->dynamic_inputs[j]) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
    }
    for (uint32_t i = 0; i < query->n_live_outputs; ++i) {
        if (query->live_outputs[i] == nullptr ||
            !ggml_backend_moe_cpu_region_graph_has_v1(query->graph, query->live_outputs[i])) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(query->live_outputs[i], nullptr)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        if (query->live_outputs[i]->view_src != nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (query->live_outputs[i] == query->live_outputs[j]) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
    }

    uint32_t expert_count        = 0;
    bool     activation_used     = false;
    uint64_t node_data_bytes     = 0;
    uint64_t expected_graph_work = 0;
    for (int i = 0; i < query->graph->n_nodes; ++i) {
        const struct ggml_tensor * node = query->graph->nodes[i];
        if (node == nullptr || (node->op != GGML_OP_VIEW && (node->view_src != nullptr || node->view_offs != 0)) ||
            !ggml_backend_moe_cpu_region_supported_op_v1(node->op)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int32_t status = ggml_backend_moe_cpu_region_check_mmid_v1(query, node, &expert_count);
            if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                return status;
            }
            if (routed_operation && (!ggml_backend_cpu_device_supports_op(nullptr, node) ||
                    !ggml_cpu_extra_supports_mmid_route_filter(node))) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
            activation_used = activation_used || node->src[1] == query->activation;
        } else if (node->op == GGML_OP_GLU) {
            if (!ggml_backend_moe_cpu_region_graph_has_v1(query->graph, node->src[0]) ||
                !ggml_backend_moe_cpu_region_graph_has_v1(query->graph, node->src[1]) ||
                !ggml_backend_moe_cpu_region_valid_glu_v1(node, query->n_threads)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
        } else if (node->op == GGML_OP_VIEW) {
            if (!ggml_backend_moe_cpu_region_valid_view_v1(query, node)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
        } else if (node->op == GGML_OP_ADD_ID) {
            if (!ggml_backend_moe_cpu_region_valid_add_id_v1(query, node, expert_count)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
        } else if (node->op == GGML_OP_UNARY || node->op == GGML_OP_SQR) {
            const auto * input = node->src[0];
            if (!input || !ggml_backend_moe_cpu_region_graph_has_v1(query->graph, input) ||
                    node->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || !ggml_are_same_shape(node, input) ||
                    !ggml_backend_moe_cpu_region_exact_layout_v1(node) || !ggml_backend_moe_cpu_region_exact_layout_v1(input) ||
                    (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) != GGML_UNARY_OP_RELU)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
            for (int s = 1; s < GGML_MAX_SRC; ++s) {
                if (node->src[s]) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
            }
            for (size_t p = node->op == GGML_OP_UNARY ? 1 : 0; p < sizeof(node->op_params) / sizeof(node->op_params[0]); ++p) {
                if (node->op_params[p]) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
            }
        }
        uint64_t span;
        uint64_t bytes;
        if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(node, &span) ||
            !ggml_backend_moe_cpu_region_align_v1(span, &bytes) ||
            (graph_execution && node->op != GGML_OP_VIEW &&
             !ggml_backend_moe_cpu_region_add_v1(node_data_bytes, bytes, &node_data_bytes))) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        if (graph_execution) {
            uint64_t node_work;
            const int32_t work_status = ggml_backend_moe_cpu_region_plan_work_v1(query, node, &node_work);
            if (work_status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                return work_status;
            }
            expected_graph_work = std::max(expected_graph_work, node_work);
        }
    }
    if (expert_count == 0 || !activation_used) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }

    uint32_t immutable_sources     = 0;
    uint64_t borrowed_source_bytes = 0;
    for (int i = 0; i < query->graph->n_nodes; ++i) {
        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            const struct ggml_tensor * tensor = query->graph->nodes[i]->src[src_index];
            if (tensor == nullptr || ggml_backend_moe_cpu_region_graph_has_v1(query->graph, tensor) ||
                ggml_backend_moe_cpu_region_array_has_v1(query->dynamic_inputs, query->n_dynamic_inputs, tensor)) {
                continue;
            }
            bool seen = false;
            for (int previous_node = 0; previous_node <= i && !seen; ++previous_node) {
                const int end = previous_node == i ? src_index : GGML_MAX_SRC;
                for (int previous_src = 0; previous_src < end; ++previous_src) {
                    if (query->graph->nodes[previous_node]->src[previous_src] == tensor) {
                        seen = true;
                        break;
                    }
                }
            }
            if (seen) {
                continue;
            }
            uint64_t source_span;
            if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(tensor, &source_span)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
            }
            const auto * source = ggml_backend_moe_cpu_region_find_source_v1(query, tensor);
            if (source == nullptr || source->witness == nullptr || source->data == nullptr ||
                source->bytes < source_span || source->generation != query->source_generation ||
                !ggml_backend_moe_cpu_region_add_v1(borrowed_source_bytes, source->bytes, &borrowed_source_bytes)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE;
            }
            ++immutable_sources;
        }
    }
    if (immutable_sources != query->n_sources) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE;
    }

    uint64_t dynamic_input_bytes = 0;
    for (uint32_t i = 0; i < query->n_dynamic_inputs; ++i) {
        uint64_t span;
        uint64_t bytes;
        if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(query->dynamic_inputs[i], &span)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        if (!ggml_backend_moe_cpu_region_align_v1(span, &bytes) ||
            (graph_execution && !ggml_backend_moe_cpu_region_add_v1(dynamic_input_bytes, bytes, &dynamic_input_bytes))) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
    }
    uint64_t binding_metadata_bytes;
    if (!ggml_backend_moe_cpu_region_mul_v1(route_capacity,
            2 * sizeof(uint32_t) + (routed_operation ? sizeof(uint8_t) : 0), &binding_metadata_bytes) ||
        !ggml_backend_moe_cpu_region_align_v1(binding_metadata_bytes, &binding_metadata_bytes)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }

    uint64_t output_staging_bytes = 0;
    for (uint32_t i = 0; i < query->n_live_outputs; ++i) {
        uint64_t span;
        uint64_t offset;
        uint64_t bytes;
        if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(query->live_outputs[i], &span)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        if (!ggml_backend_moe_cpu_region_align_v1(output_staging_bytes, &offset) ||
            !ggml_backend_moe_cpu_region_align_v1(span, &bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(offset, bytes, &output_staging_bytes)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        result.live_outputs[i].tensor = query->live_outputs[i];
        result.live_outputs[i].offset = offset;
        result.live_outputs[i].bytes  = span;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            result.live_outputs[i].ne[dim] = query->live_outputs[i]->ne[dim];
            result.live_outputs[i].nb[dim] = query->live_outputs[i]->nb[dim];
        }
    }
    if (query->output_staging_limit != 0 && output_staging_bytes > query->output_staging_limit) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }

    if (expected_graph_work > 0) {
        uint64_t thread_padding;
        if (!ggml_backend_moe_cpu_region_mul_v1(CACHE_LINE_SIZE, query->n_threads, &thread_padding) ||
            !ggml_backend_moe_cpu_region_add_v1(expected_graph_work, thread_padding, &expected_graph_work) ||
            expected_graph_work > SIZE_MAX ||
            (query->lane_execution_byte_limit != 0 && expected_graph_work > query->lane_execution_byte_limit)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
    }

    // All planner inputs and work terms are checked above. ggml_graph_plan scans metadata and does not allocate.
    const struct ggml_cplan plan = graph_execution ? ggml_graph_plan(query->graph, query->n_threads, nullptr) : ggml_cplan{};
    if (plan.work_size != expected_graph_work) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
    }
    uint64_t lane_execution_bytes = 0;
    for (const uint64_t bytes : { (uint64_t) plan.work_size, node_data_bytes, dynamic_input_bytes,
                                  binding_metadata_bytes, output_staging_bytes }) {
        uint64_t aligned;
        if (!ggml_backend_moe_cpu_region_align_v1(bytes, &aligned) ||
            !ggml_backend_moe_cpu_region_add_v1(lane_execution_bytes, aligned, &lane_execution_bytes)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
    }
    uint64_t all_lane_execution_bytes;
    if (!ggml_backend_moe_cpu_region_mul_v1(lane_execution_bytes, query->n_lanes, &all_lane_execution_bytes) ||
            (query->lane_execution_byte_limit != 0 && all_lane_execution_bytes > query->lane_execution_byte_limit)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }

    result.n_threads                = query->n_threads;
    result.n_lanes                  = query->n_lanes;
    result.graph_nodes              = graph_execution ? query->n_body_nodes : 0;
    result.bucket_rows              = query->bucket_rows;
    result.routes_per_row           = query->routes_per_row;
    result.route_capacity           = (uint32_t) route_capacity;
    result.expert_count             = expert_count;
    result.immutable_sources        = immutable_sources;
    result.n_live_outputs           = query->n_live_outputs;
    result.graph_work_bytes         = plan.work_size;
    result.node_data_bytes          = node_data_bytes;
    result.dynamic_input_bytes      = dynamic_input_bytes;
    result.binding_metadata_bytes   = binding_metadata_bytes;
    result.output_staging_bytes     = output_staging_bytes;
    result.lane_execution_bytes     = lane_execution_bytes;
    result.all_lane_execution_bytes = all_lane_execution_bytes;
    result.borrowed_source_bytes    = borrowed_source_bytes;
    *requirements                   = result;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_validate_binding_impl_v1(
    const struct ggml_backend_moe_cpu_region_query_v1 *   query,
    const struct ggml_backend_moe_cpu_region_binding_v1 * binding) {
    if (binding == nullptr || binding->struct_size < sizeof(*binding)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    }
    struct ggml_backend_moe_cpu_region_requirements_v1 requirements = {};
    requirements.struct_size                                        = sizeof(requirements);
    const bool compact = (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES) != 0;
    if (ggml_backend_moe_cpu_region_query_impl_v1(query, &requirements) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ||
        binding->n_rows == 0 || binding->n_routes == 0 ||
        (compact ? binding->n_rows != binding->n_routes || binding->n_routes > requirements.route_capacity :
                   binding->n_rows != requirements.bucket_rows || binding->n_routes != requirements.route_capacity) ||
        binding->expert_ids == nullptr || binding->source_rows == nullptr || binding->scatter_destinations == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    }
    for (uint32_t route = 0; route < binding->n_routes; ++route) {
        if (binding->expert_ids[route] < 0 || (uint32_t) binding->expert_ids[route] >= requirements.expert_count ||
            binding->source_rows[route] >= query->source_row_capacity ||
            binding->scatter_destinations[route] >= query->scatter_capacity) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        const uint32_t row_start = route - route % requirements.routes_per_row;
        if (binding->source_rows[route] != binding->source_rows[row_start]) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        if (route == row_start &&
                (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES) == 0) {
            for (uint32_t previous = 0; previous < row_start; previous += requirements.routes_per_row) {
                if (binding->source_rows[previous] == binding->source_rows[route]) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
                }
            }
        }
        for (uint32_t previous = 0; previous < route; ++previous) {
            if (binding->scatter_destinations[previous] == binding->scatter_destinations[route]) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
            }
        }
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_query_checked_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        struct ggml_backend_moe_cpu_region_requirements_v1 * requirements) {
    try {
        return ggml_backend_moe_cpu_region_query_impl_v1(query, requirements);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static int32_t ggml_backend_moe_cpu_region_validate_binding_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        const struct ggml_backend_moe_cpu_region_binding_v1 * binding) {
    try {
        return ggml_backend_moe_cpu_region_validate_binding_impl_v1(query, binding);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static const struct ggml_backend_moe_cpu_region_query_api_v1 * ggml_backend_moe_cpu_region_query_v1(void) {
    static const struct ggml_backend_moe_cpu_region_query_api_v1 api = {
        /* .struct_size      = */ sizeof(api),
        /* .abi_version      = */ 1,
        /* .query            = */ ggml_backend_moe_cpu_region_query_checked_v1,
        /* .validate_binding = */ ggml_backend_moe_cpu_region_validate_binding_v1,
    };
    return &api;
}

struct ggml_backend_moe_cpu_prepared_lane_v1 {
    ggml_moe_cpu_fidelity * fidelity = nullptr;
    std::unique_ptr<uint8_t[]> metadata;
    std::unique_ptr<uint8_t[]> execution_storage;
    std::unique_ptr<uint64_t[]> execution_offsets;
    uint64_t work_offset = 0;
    uint64_t binding_offset = 0;
    uint64_t staging_offset = 0;
    uint8_t * execution = nullptr;
    uint8_t * binding_metadata = nullptr;
    uint8_t * output_staging = nullptr;
    struct ggml_context * context = nullptr;
    struct ggml_cgraph * graph = nullptr;
    struct ggml_cplan plan = {};
    std::unique_ptr<struct ggml_tensor *[]> dynamic_inputs;
    struct ggml_tensor * live_outputs[GGML_BACKEND_MOE_CPU_REGION_MAX_LIVE_OUTPUTS_V1] = {};

    ~ggml_backend_moe_cpu_prepared_lane_v1() {
        ggml_moe_cpu_fidelity_destroy(fidelity);
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

static bool ggml_backend_moe_cpu_owned_route_v1(const ggml_tensor * op,
        int64_t row, int64_t column, int32_t expert, void * data) {
    GGML_UNUSED(expert);
    return static_cast<const uint8_t *>(data)[row * op->ne[1] + column] != 0;
}

struct ggml_backend_moe_cpu_service_impl_v1;

struct ggml_backend_moe_cpu_prepared_region_impl_v1 {
    ggml_backend_moe_cpu_prepared_region_v1_t id = 0;
    ggml_backend_moe_cpu_prepared_requirements_v1 requirements = {};
    uint64_t charged_payload_bytes = 0;
    std::unique_ptr<ggml_backend_moe_source_span_v1[]> sources;
    std::unique_ptr<std::unique_ptr<ggml_backend_moe_cpu_prepared_lane_v1>[]> lanes;
    uint32_t n_sources = 0;
    uint32_t n_lanes = 0;
    uint64_t graph_uid = 0;
    uint64_t graph_generation = 0;
    uint64_t source_generation = 0;
    uint32_t n_dynamic_inputs = 0;
    uint32_t n_live_outputs = 0;
    uint32_t activation_input = 0;
    uint32_t ids_input = 0;
    uint32_t flags = 0;
    uint32_t source_row_capacity = 0;
    uint32_t scatter_capacity = 0;
    uint32_t active_jobs = 0;
};

static uint64_t moe_cpu_profile_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ggml_backend_moe_cpu_service_impl_v1 {
    ggml_backend_moe_source_owner_v1 owner = {};
    ggml_backend_moe_source_lease_v1 lease = {};
    struct ggml_threadpool * threadpool = nullptr;
    ggml_moe_source_pool * source_pool = nullptr;
    bool cpu_profile = getenv("GGML_MOE_CPU_PROFILE") && !strcmp(getenv("GGML_MOE_CPU_PROFILE"), "1");
    std::atomic<uint64_t> profile_jobs{0}, profile_bind_ns{0}, profile_compute_ns{0}, profile_scatter_ns{0};
    std::mutex mutex;
    std::mutex execute_mutex;
    std::condition_variable drain_condition;
    std::unique_ptr<ggml_backend_moe_cpu_prepared_region_impl_v1 *[]> regions;
    std::unique_ptr<uint8_t[]> routed_storage;
    uint64_t routed_storage_bytes = 0;
    uint32_t routed_regions = 0;
    std::atomic<uint64_t> cancel_through_epoch = 0;
    std::atomic<bool> closing = false;
    uint64_t prepared_payload_limit = 0;
    uint64_t prepared_payload_bytes = 0;
    uint64_t prepared_payload_peak = 0;
    uint64_t service_control_bytes = 0;
    uint32_t n_threads = 0;
    uint32_t n_lanes = 0;
    uint32_t max_regions = 0;
    uint32_t n_regions = 0;
    uint32_t active_jobs = 0;
    uint32_t next_lane = 0;
    bool runtime_allocations_unproven = false;
    bool closed = false;
    ggml_backend_moe_cpu_test_hook_v1_t test_hook = nullptr;
    void * test_hook_data = nullptr;

    ~ggml_backend_moe_cpu_service_impl_v1() {
        if (cpu_profile) {
            fprintf(stderr, "moe-cpu-service-profile: jobs=%llu bind_ns=%llu compute_ns=%llu scatter_ns=%llu\n",
                (unsigned long long) profile_jobs.load(), (unsigned long long) profile_bind_ns.load(),
                (unsigned long long) profile_compute_ns.load(), (unsigned long long) profile_scatter_ns.load());
        }
        ggml_moe_source_pool_free(source_pool);
        if (threadpool != nullptr) {
            ggml_threadpool_free(threadpool);
        }
    }
};

static uint64_t ggml_backend_moe_cpu_region_next_id_v1() {
    static std::atomic<uint64_t> next { 1 };
    uint64_t id = next.load(std::memory_order_relaxed);
    while (id != UINT64_MAX && !next.compare_exchange_weak(id, id + 1, std::memory_order_relaxed)) {
    }
    return id == UINT64_MAX ? 0 : id;
}

static int32_t ggml_backend_moe_cpu_region_service_release_v1(ggml_backend_moe_cpu_service_impl_v1 * service) {
    if (service->lease.owner == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    }
    ggml_backend_moe_source_lease_v1 lease = service->lease;
    int32_t status;
    try {
        status = service->owner.release(&lease);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    if (status != GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK &&
            status != GGML_BACKEND_MOE_SOURCE_STATUS_V1_ALREADY_RELEASED) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    service->lease = lease;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_service_requirements_impl_v1(
        const ggml_backend_moe_cpu_service_config_v1 * config, uint64_t * bytes, bool * runtime_unproven, bool source_pool = false) {
    if (config == nullptr || bytes == nullptr || runtime_unproven == nullptr ||
            config->struct_size != sizeof(*config) || config->abi_version != 1 || config->source_owner == nullptr ||
            config->source_owner->struct_size != sizeof(*config->source_owner) ||
            config->source_owner->abi_version != GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION ||
            config->source_owner->owner == nullptr || config->source_owner->generation == 0 ||
            config->source_owner->flags != GGML_BACKEND_MOE_SOURCE_OWNER_FLAG_V1_NONE ||
            config->source_owner->reserved32 != 0 || config->source_owner->reserved[0] != 0 ||
            config->source_owner->reserved[1] != 0 ||
            config->source_owner->retain == nullptr || config->source_owner->release == nullptr ||
            config->source_owner->validate_span == nullptr || config->n_threads == 0 ||
            config->n_threads > GGML_MAX_N_THREADS || config->n_threads > INT_MAX || config->n_lanes == 0 ||
            config->max_regions == 0 || config->reserved[0] != 0 ||
            config->reserved[1] != 0 ||
            (config->flags & ~(
                GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES |
                GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS)) != 0 ||
            (config->flags & GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES) == 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
#ifdef GGML_USE_OPENMP
    const bool runtime_allocations_unproven = true;
#else
    const bool runtime_allocations_unproven = ggml_is_numa();
#endif
    if (runtime_allocations_unproven &&
            (config->flags & GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS) == 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }

    uint64_t region_table_bytes;
    uint64_t service_control_bytes;
    if (!ggml_backend_moe_cpu_region_mul_v1(config->max_regions, sizeof(void *), &region_table_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(sizeof(ggml_backend_moe_cpu_service_impl_v1), region_table_bytes,
                                                &service_control_bytes) ||
            (source_pool && !ggml_backend_moe_cpu_region_add_v1(service_control_bytes,
                ggml_moe_source_pool_bytes(config->n_threads), &service_control_bytes)) ||
            service_control_bytes > SIZE_MAX ||
            (config->prepared_payload_limit != 0 && service_control_bytes > config->prepared_payload_limit)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    *bytes = service_control_bytes;
    *runtime_unproven = runtime_allocations_unproven;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_create_impl_v1(
        const struct ggml_backend_moe_cpu_service_config_v1 * config,
        ggml_backend_moe_cpu_service_v1_t * service_out, bool source_pool = false) {
    if (service_out == nullptr || *service_out != nullptr) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT; }
    uint64_t service_control_bytes = 0;
    bool runtime_allocations_unproven = false;
    const int32_t measured = ggml_backend_moe_cpu_service_requirements_impl_v1(config, &service_control_bytes, &runtime_allocations_unproven, source_pool);
    if (measured != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return measured; }
    std::unique_ptr<ggml_backend_moe_cpu_service_impl_v1> service(
        new (std::nothrow) ggml_backend_moe_cpu_service_impl_v1());
    if (!service) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    service->owner                  = *config->source_owner;
    service->prepared_payload_limit = config->prepared_payload_limit;
    service->prepared_payload_bytes = service_control_bytes;
    service->prepared_payload_peak = service_control_bytes;
    service->service_control_bytes  = service_control_bytes;
    service->n_threads              = config->n_threads;
    service->n_lanes                = config->n_lanes;
    service->max_regions            = config->max_regions;
    service->runtime_allocations_unproven = runtime_allocations_unproven;
    service->regions.reset(new (std::nothrow) ggml_backend_moe_cpu_prepared_region_impl_v1 *[config->max_regions]());
    if (!service->regions) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    if (source_pool) {
        service->source_pool = ggml_moe_source_pool_create(config->n_threads);
        if (!service->source_pool) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
    } else {
        struct ggml_threadpool_params params = ggml_threadpool_params_default(config->n_threads);
        params.paused = true;
        service->threadpool = ggml_threadpool_new_persistent(&params);
        if (service->threadpool == nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
    }
    service->lease.struct_size = sizeof(service->lease);
    service->lease.abi_version = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
    int32_t status;
    try {
        const int32_t retain_status = service->owner.retain(&service->owner, service->owner.generation, &service->lease);
        status = retain_status == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK ?
            GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK :
            retain_status == GGML_BACKEND_MOE_SOURCE_STATUS_V1_CLOSED ?
                GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CLOSED : GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE;
    } catch (...) {
        status = GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        if (ggml_backend_moe_cpu_region_service_release_v1(service.get()) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            service->closed = true;
            service->closing.store(true, std::memory_order_release);
            *service_out = service.release();
        }
        return status;
    }
    *service_out = service.release();
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_create_v1(
        const struct ggml_backend_moe_cpu_service_config_v1 * config,
        ggml_backend_moe_cpu_service_v1_t * service_out) {
    try {
        return ggml_backend_moe_cpu_region_service_create_impl_v1(config, service_out);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static int32_t ggml_backend_moe_cpu_fidelity_create_v1(
        const ggml_backend_moe_cpu_service_config_v1 * config, ggml_backend_moe_cpu_service_v1_t * service) {
    try {
        const auto & selection = ggml_moe_fidelity_selection();
        if (!selection.valid) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
        return ggml_backend_moe_cpu_region_service_create_impl_v1(config, service, selection.source_pool);
    } catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
}

static bool ggml_backend_moe_cpu_region_metadata_bytes_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        uint64_t * tensor_count,
        uint64_t * metadata_bytes) {
    uint64_t count;
    uint64_t tensors_bytes;
    uint64_t graph_capacity = std::max<uint64_t>(query->n_body_nodes, query->n_dynamic_inputs + query->n_sources);
    return ggml_backend_moe_cpu_region_add_v1(query->n_body_nodes, query->n_dynamic_inputs, &count) &&
           ggml_backend_moe_cpu_region_add_v1(count, query->n_sources, &count) &&
           count <= INT_MAX && graph_capacity <= INT_MAX &&
           ggml_backend_moe_cpu_region_mul_v1(count, ggml_tensor_overhead(), &tensors_bytes) &&
           ggml_backend_moe_cpu_region_add_v1(
               tensors_bytes, ggml_graph_overhead_custom(std::max<uint64_t>(graph_capacity, 1), false), metadata_bytes) &&
           *metadata_bytes <= SIZE_MAX && (*tensor_count = count, true);
}

static struct ggml_tensor * ggml_backend_moe_cpu_region_clone_find_v1(
        const std::vector<const struct ggml_tensor *> & originals,
        const std::vector<struct ggml_tensor *> & clones,
        const struct ggml_tensor * tensor) {
    const auto it = std::find(originals.begin(), originals.end(), tensor);
    return it == originals.end() ? nullptr : clones[it - originals.begin()];
}

static bool ggml_backend_moe_cpu_region_lane_offset_v1(uint64_t bytes, uint64_t * offset, uint64_t * cursor) {
    uint64_t aligned;
    return ggml_backend_moe_cpu_region_align_v1(*cursor, &aligned) &&
           ggml_backend_moe_cpu_region_add_v1(aligned, bytes, cursor) &&
           (*offset = aligned, true);
}

static int32_t ggml_backend_moe_cpu_region_build_lane_v1(
        ggml_backend_moe_cpu_service_impl_v1 * service,
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        const struct ggml_backend_moe_cpu_region_requirements_v1 * execution,
        uint64_t metadata_bytes,
        uint64_t allocation_bytes,
        uint8_t * shared_storage,
        std::unique_ptr<ggml_backend_moe_cpu_prepared_lane_v1> & lane_out) {
    try {
        auto lane = std::make_unique<ggml_backend_moe_cpu_prepared_lane_v1>();
        lane->metadata.reset(new (std::nothrow) uint8_t[metadata_bytes]);
        if (shared_storage == nullptr) {
            lane->execution_storage.reset(new (std::nothrow) uint8_t[allocation_bytes]);
        } else {
            lane->execution_offsets.reset(new (std::nothrow) uint64_t[query->n_dynamic_inputs + query->n_body_nodes]());
        }
        if (!lane->metadata || (shared_storage ? !lane->execution_offsets : !lane->execution_storage)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        lane->dynamic_inputs.reset(new (std::nothrow) struct ggml_tensor *[query->n_dynamic_inputs]());
        if (!lane->dynamic_inputs) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        uint8_t * storage = shared_storage ? shared_storage : lane->execution_storage.get();
        const uintptr_t raw = reinterpret_cast<uintptr_t>(storage);
        const uintptr_t padding = (GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 -
                                   raw % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1) %
                                  GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1;
        if (padding > allocation_bytes || execution->lane_execution_bytes > allocation_bytes - padding) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        lane->execution = storage + padding;
        GGML_ASSERT(reinterpret_cast<uintptr_t>(lane->execution) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 == 0);
        lane->context = ggml_init({(size_t) metadata_bytes, lane->metadata.get(), true});
        if (lane->context == nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }

        std::vector<const struct ggml_tensor *> originals;
        originals.reserve(query->n_dynamic_inputs + query->n_sources + query->n_body_nodes);
        originals.insert(originals.end(), query->dynamic_inputs, query->dynamic_inputs + query->n_dynamic_inputs);
        for (uint32_t i = 0; i < query->n_sources; ++i) {
            originals.push_back(query->sources[i].tensor);
        }
        originals.insert(originals.end(), query->body_nodes, query->body_nodes + query->n_body_nodes);

        std::vector<struct ggml_tensor *> clones;
        clones.reserve(originals.size());
        for (const auto * tensor : originals) {
            auto * clone = ggml_new_tensor_1d(lane->context, GGML_TYPE_F32, 1);
            *clone = *tensor;
            clone->buffer = nullptr;
            clone->data   = nullptr;
            clone->extra  = nullptr;
            std::fill(std::begin(clone->src), std::end(clone->src), nullptr);
            clone->view_src  = nullptr;
            clone->view_offs = 0;
            clones.push_back(clone);
        }

        uint64_t cursor = 0;
        uint64_t work_offset;
        if (!ggml_backend_moe_cpu_region_lane_offset_v1(execution->graph_work_bytes, &work_offset, &cursor)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        for (uint32_t i = 0; i < query->n_dynamic_inputs; ++i) {
            uint64_t span;
            uint64_t offset;
            if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(query->dynamic_inputs[i], &span) ||
                    !ggml_backend_moe_cpu_region_lane_offset_v1(span, &offset, &cursor)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
            }
            clones[i]->data = lane->execution + offset;
            if (lane->execution_offsets) { lane->execution_offsets[i] = offset; }
            lane->dynamic_inputs[i] = clones[i];
            GGML_ASSERT(reinterpret_cast<uintptr_t>(clones[i]->data) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 == 0);
        }
        for (uint32_t i = 0; i < query->n_sources; ++i) {
            clones[query->n_dynamic_inputs + i]->data = const_cast<void *>(query->sources[i].data);
        }
        for (uint32_t i = 0; i < query->n_body_nodes; ++i) {
            const uint32_t clone_index = query->n_dynamic_inputs + query->n_sources + i;
            const auto * original = query->body_nodes[i];
            if (original->op == GGML_OP_VIEW) {
                auto * source = ggml_backend_moe_cpu_region_clone_find_v1(originals, clones, original->view_src);
                if (source == nullptr || source->data == nullptr) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
                }
                clones[clone_index]->view_src = source;
                clones[clone_index]->view_offs = original->view_offs;
                clones[clone_index]->data = static_cast<uint8_t *>(source->data) + original->view_offs;
                continue;
            }
            uint64_t span;
            uint64_t offset;
            if (!ggml_backend_moe_cpu_region_valid_tensor_layout_v1(query->body_nodes[i], &span) ||
                    !ggml_backend_moe_cpu_region_lane_offset_v1(span, &offset, &cursor)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
            }
            clones[clone_index]->data = lane->execution + offset;
            if (lane->execution_offsets) { lane->execution_offsets[query->n_dynamic_inputs + i] = offset; }
            GGML_ASSERT(reinterpret_cast<uintptr_t>(clones[clone_index]->data) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 == 0);
        }
        uint64_t binding_offset;
        uint64_t staging_offset;
        if (!ggml_backend_moe_cpu_region_lane_offset_v1(execution->binding_metadata_bytes, &binding_offset, &cursor) ||
                !ggml_backend_moe_cpu_region_lane_offset_v1(execution->output_staging_bytes, &staging_offset, &cursor) ||
                cursor > execution->lane_execution_bytes) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        lane->binding_metadata = lane->execution + binding_offset;
        lane->output_staging   = lane->execution + staging_offset;
        lane->work_offset = work_offset;
        lane->binding_offset = binding_offset;
        lane->staging_offset = staging_offset;
        GGML_ASSERT(reinterpret_cast<uintptr_t>(lane->binding_metadata) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 == 0);
        GGML_ASSERT(reinterpret_cast<uintptr_t>(lane->output_staging) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 == 0);

        for (uint32_t i = 0; i < query->n_body_nodes; ++i) {
            auto * clone = clones[query->n_dynamic_inputs + query->n_sources + i];
            const auto * original = query->body_nodes[i];
            for (int src = 0; src < GGML_MAX_SRC; ++src) {
                if (original->src[src] != nullptr) {
                    clone->src[src] = ggml_backend_moe_cpu_region_clone_find_v1(originals, clones, original->src[src]);
                    if (clone->src[src] == nullptr) {
                        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
                    }
                }
            }
        }

        const size_t graph_capacity = std::max<size_t>(query->n_body_nodes, query->n_dynamic_inputs + query->n_sources);
        lane->graph = ggml_new_graph_custom(lane->context, std::max<size_t>(graph_capacity, 1), false);
        for (uint32_t i = 0; i < query->n_live_outputs; ++i) {
            auto * output = ggml_backend_moe_cpu_region_clone_find_v1(originals, clones, query->live_outputs[i]);
            if (output == nullptr) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
            lane->live_outputs[i] = output;
            ggml_build_forward_expand(lane->graph, output);
        }
        if ((uint32_t) lane->graph->n_nodes != query->n_body_nodes) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t i = 0; i < query->n_body_nodes; ++i) {
            if (lane->graph->nodes[i] != clones[query->n_dynamic_inputs + query->n_sources + i]) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        lane->graph->uid = ggml_graph_next_uid();
        lane->plan = ggml_graph_plan(lane->graph, service->n_threads, service->threadpool);
        if (lane->plan.work_size != execution->graph_work_bytes) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION;
        }
        lane->plan.work_data = execution->graph_work_bytes == 0 ? nullptr : lane->execution + work_offset;
        if ((query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0) {
            lane->plan.mmid_route_filter = ggml_backend_moe_cpu_owned_route_v1;
            lane->plan.mmid_route_filter_data = lane->binding_metadata +
                size_t(execution->route_capacity) * 2 * sizeof(uint32_t);
        }
        GGML_ASSERT(lane->plan.work_data == nullptr ||
                    reinterpret_cast<uintptr_t>(lane->plan.work_data) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 == 0);
        lane_out = std::move(lane);
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    } catch (const std::bad_alloc &) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static int32_t ggml_backend_moe_cpu_fidelity_build_lane_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        const struct ggml_backend_moe_cpu_prepared_requirements_v1 & requirements,
        uint64_t fidelity_offset, uint64_t fidelity_bytes,
        std::unique_ptr<ggml_backend_moe_cpu_prepared_lane_v1> & lane_out) {
    auto lane = std::make_unique<ggml_backend_moe_cpu_prepared_lane_v1>();
    lane->metadata.reset(new (std::nothrow) uint8_t[requirements.lane_metadata_bytes]);
    lane->execution_storage.reset(new (std::nothrow) uint8_t[requirements.lane_allocation_bytes]);
    lane->dynamic_inputs.reset(new (std::nothrow) ggml_tensor *[query->n_dynamic_inputs]());
    if (!lane->metadata || !lane->execution_storage || !lane->dynamic_inputs) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    const uintptr_t raw = reinterpret_cast<uintptr_t>(lane->execution_storage.get());
    const uintptr_t padding = (GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 -
        raw % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1;
    if (padding > requirements.lane_allocation_bytes ||
            requirements.execution.lane_execution_bytes > requirements.lane_allocation_bytes - padding) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    lane->execution = lane->execution_storage.get() + padding;
    uint64_t cursor = 0, binding_offset, staging_offset;
    if (!ggml_backend_moe_cpu_region_lane_offset_v1(requirements.execution.binding_metadata_bytes, &binding_offset, &cursor) ||
            !ggml_backend_moe_cpu_region_lane_offset_v1(requirements.execution.output_staging_bytes, &staging_offset, &cursor) ||
            cursor > fidelity_offset || fidelity_offset > requirements.execution.lane_execution_bytes ||
            fidelity_bytes > requirements.execution.lane_execution_bytes - fidelity_offset) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    lane->binding_metadata = lane->execution + binding_offset;
    lane->output_staging = lane->execution + staging_offset;
    auto * views = reinterpret_cast<ggml_tensor *>(lane->metadata.get());
    for (uint32_t i = 0; i < requirements.tensor_count; ++i) {
        const auto * original = i < query->n_dynamic_inputs ? query->dynamic_inputs[i] : query->live_outputs[i - query->n_dynamic_inputs];
        auto * view = new (views + i) ggml_tensor{};
        view->type = original->type;
        std::copy(std::begin(original->ne), std::end(original->ne), view->ne);
        std::copy(std::begin(original->nb), std::end(original->nb), view->nb);
        if (i < query->n_dynamic_inputs) {
            lane->dynamic_inputs[i] = view;
        } else {
            const auto output = i - query->n_dynamic_inputs;
            view->data = lane->output_staging + requirements.execution.live_outputs[output].offset;
            lane->live_outputs[output] = view;
        }
    }
    lane->fidelity = ggml_moe_cpu_fidelity_prepare(query, lane->execution + fidelity_offset, fidelity_bytes);
    if (!lane->fidelity) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    lane_out = std::move(lane);
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_prepared_requirements_impl_v1(
        const struct ggml_backend_moe_cpu_region_query_v1 * query, bool fidelity, bool runtime_allocations_unproven,
        ggml_backend_moe_cpu_prepared_requirements_v1 * requirements, uint64_t & fidelity_bytes, uint64_t & fidelity_offset) {
    if (!query || query->struct_size < sizeof(*query) || !requirements ||
            requirements->struct_size != sizeof(*requirements) || requirements->abi_version != 1) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    fidelity = fidelity && (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) == 0;
    ggml_backend_moe_cpu_prepared_requirements_v1 result = {};
    result.struct_size = sizeof(result);
    result.abi_version = 1;
    result.execution.struct_size = sizeof(result.execution);
    const int32_t query_status = ggml_backend_moe_cpu_region_query_impl_v1(query, &result.execution, !fidelity);
    if (query_status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return query_status;
    }
    fidelity_bytes = fidelity_offset = 0;
    if (fidelity) {
        const auto status = ggml_moe_cpu_fidelity_measure(query, &fidelity_bytes);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
        if (!ggml_backend_moe_cpu_region_align_v1(result.execution.lane_execution_bytes, &fidelity_offset) ||
                !ggml_backend_moe_cpu_region_add_v1(fidelity_offset, fidelity_bytes, &result.execution.lane_execution_bytes) ||
                !ggml_backend_moe_cpu_region_mul_v1(result.execution.lane_execution_bytes, query->n_lanes,
                    &result.execution.all_lane_execution_bytes) ||
                (query->lane_execution_byte_limit && result.execution.all_lane_execution_bytes > query->lane_execution_byte_limit)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
    }
    uint64_t tensor_count = 0;
    if (fidelity ?
            !ggml_backend_moe_cpu_region_add_v1(query->n_dynamic_inputs, query->n_live_outputs, &tensor_count) ||
            !ggml_backend_moe_cpu_region_mul_v1(tensor_count, sizeof(ggml_tensor), &result.lane_metadata_bytes) ||
            tensor_count > UINT32_MAX || result.lane_metadata_bytes > SIZE_MAX :
            !ggml_backend_moe_cpu_region_metadata_bytes_v1(query, &tensor_count, &result.lane_metadata_bytes)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    result.tensor_count           = tensor_count;
    result.external_source_count = query->n_sources;
    result.flags                 = GGML_BACKEND_MOE_CPU_PREPARED_FLAG_V1_THREAD_STACK_BYTES_UNKNOWN;
    if (runtime_allocations_unproven) {
        result.flags |= GGML_BACKEND_MOE_CPU_PREPARED_FLAG_V1_RUNTIME_ALLOCATIONS_UNPROVEN;
    }
    result.thread_stack_bytes    = UINT64_MAX;
    result.lane_context_bytes    = fidelity ? 0 : ggml_context_overhead();
    result.lane_alignment       = GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1;
    uint64_t lane_metadata_total;
    uint64_t lane_allocation_total;
    uint64_t lane_control_bytes;
    uint64_t dynamic_input_pointer_bytes;
    uint64_t lane_context_total;
    uint64_t execution_offset_bytes = 0;
    if ((query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0 &&
            (!ggml_backend_moe_cpu_region_mul_v1(tensor_count - query->n_sources, sizeof(uint64_t), &execution_offset_bytes) ||
             !ggml_backend_moe_cpu_region_mul_v1(execution_offset_bytes, query->n_lanes, &execution_offset_bytes))) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    if (!ggml_backend_moe_cpu_region_add_v1(result.execution.lane_execution_bytes,
                                            GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 - 1,
                                            &result.lane_allocation_bytes) ||
            !ggml_backend_moe_cpu_region_mul_v1(result.lane_metadata_bytes, query->n_lanes, &lane_metadata_total) ||
            !ggml_backend_moe_cpu_region_mul_v1(result.lane_allocation_bytes, query->n_lanes,
                                                &lane_allocation_total) ||
            !ggml_backend_moe_cpu_region_mul_v1(
                query->n_lanes,
                sizeof(ggml_backend_moe_cpu_prepared_lane_v1) +
                    sizeof(std::unique_ptr<ggml_backend_moe_cpu_prepared_lane_v1>),
                &lane_control_bytes) ||
            !ggml_backend_moe_cpu_region_mul_v1(query->n_dynamic_inputs, sizeof(struct ggml_tensor *),
                                                &dynamic_input_pointer_bytes) ||
            !ggml_backend_moe_cpu_region_mul_v1(dynamic_input_pointer_bytes, query->n_lanes,
                                                &dynamic_input_pointer_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(lane_control_bytes, dynamic_input_pointer_bytes,
                                                &lane_control_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(lane_control_bytes, execution_offset_bytes, &lane_control_bytes) ||
            !ggml_backend_moe_cpu_region_mul_v1(result.lane_context_bytes, query->n_lanes,
                                                &lane_context_total) ||
            !ggml_backend_moe_cpu_region_add_v1(lane_control_bytes, lane_context_total,
                                                &lane_control_bytes) ||
            !ggml_backend_moe_cpu_region_mul_v1(query->n_sources, sizeof(ggml_backend_moe_source_span_v1),
                                                &result.control_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(result.control_bytes, sizeof(ggml_backend_moe_cpu_prepared_region_impl_v1),
                                                &result.control_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(result.control_bytes, lane_control_bytes, &result.control_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(lane_metadata_total, lane_allocation_total,
                                                &result.prepared_payload_bytes) ||
            !ggml_backend_moe_cpu_region_add_v1(result.prepared_payload_bytes, result.control_bytes,
                                                &result.prepared_payload_bytes) ||
            result.lane_allocation_bytes > SIZE_MAX || result.control_bytes > SIZE_MAX) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }

    *requirements = result;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_validate_sources_v1(
        const ggml_backend_moe_source_owner_v1 * owner, const struct ggml_backend_moe_cpu_region_query_v1 * query) {
    for (uint32_t i = 0; i < query->n_sources; ++i) {
        const auto & source = query->sources[i];
        ggml_backend_moe_source_span_v1 span = {};
        span.struct_size   = sizeof(span);
        span.abi_version   = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
        span.witness       = source.witness;
        span.data          = source.data;
        span.bytes         = source.bytes;
        span.expert_stride = source.expert_stride;
        span.type          = source.tensor->type;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            span.ne[dim] = source.tensor->ne[dim];
            span.nb[dim] = source.tensor->nb[dim];
        }
        int32_t source_status;
        try {
            source_status = owner->validate_span(owner, query->source_generation, &span);
        } catch (...) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE;
        }
        if (source_status != GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_MISSING_SOURCE;
        }
    }

    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_prepare_impl_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        struct ggml_backend_moe_cpu_prepared_requirements_v1 * requirements,
        ggml_backend_moe_cpu_prepared_region_v1_t * region_out, bool fidelity = false) {
    if (service_handle == nullptr || query == nullptr || query->struct_size < sizeof(*query) ||
            requirements == nullptr || region_out == nullptr ||
            *region_out != 0 || requirements->struct_size != sizeof(*requirements) ||
            requirements->abi_version != 1) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    fidelity = fidelity && (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) == 0;
    if (query->n_threads != service->n_threads || query->n_lanes != service->n_lanes ||
            query->source_generation != service->owner.generation) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }

    ggml_backend_moe_cpu_prepared_requirements_v1 result = {};
    result.struct_size = sizeof(result);
    result.abi_version = 1;
    uint64_t fidelity_bytes = 0, fidelity_offset = 0;
    const int32_t measured = ggml_backend_moe_cpu_prepared_requirements_impl_v1(
        query, fidelity, service->runtime_allocations_unproven, &result, fidelity_bytes, fidelity_offset);
    if (measured != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return measured; }

    const int32_t source_status = ggml_backend_moe_cpu_validate_sources_v1(&service->owner, query);
    if (source_status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return source_status; }

    std::lock_guard<std::mutex> execute_lock(service->execute_mutex);
    std::lock_guard<std::mutex> lock(service->mutex);
    uint64_t total;
    const uint64_t graph_pool_bytes = !fidelity && !service->threadpool ?
        ggml_threadpool_host_size(service->n_threads) : 0;
    if (service->closed) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CLOSED;
    }
    const bool shared = (query->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0;
    uint64_t charged_payload = result.prepared_payload_bytes;
    if (shared) {
        uint64_t lane_allocation_total;
        if (!ggml_backend_moe_cpu_region_mul_v1(result.lane_allocation_bytes, query->n_lanes, &lane_allocation_total) ||
                lane_allocation_total > charged_payload) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        charged_payload -= lane_allocation_total;
    }
    const bool grow = shared && result.lane_allocation_bytes > service->routed_storage_bytes;
    const uint64_t growth = grow ? result.lane_allocation_bytes - service->routed_storage_bytes : 0;
    uint64_t peak;
    if (service->n_regions >= service->max_regions ||
            !ggml_backend_moe_cpu_region_add_v1(service->prepared_payload_bytes,
                                                charged_payload, &total) ||
            !ggml_backend_moe_cpu_region_add_v1(total, graph_pool_bytes, &total) ||
            !ggml_backend_moe_cpu_region_add_v1(total, growth, &total) ||
            !ggml_backend_moe_cpu_region_add_v1(total, grow ? service->routed_storage_bytes : 0, &peak) ||
            (service->prepared_payload_limit != 0 && peak > service->prepared_payload_limit)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    std::unique_ptr<uint8_t[]> next_storage;
    if (grow) {
        next_storage.reset(new (std::nothrow) uint8_t[result.lane_allocation_bytes]);
        if (!next_storage) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
        service->prepared_payload_peak = std::max(service->prepared_payload_peak, peak);
    }
    if (graph_pool_bytes) {
        auto params = ggml_threadpool_params_default(service->n_threads);
        params.paused = true;
        params.poll = 0;
        service->threadpool = ggml_threadpool_new_persistent(&params);
        if (!service->threadpool) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
        service->service_control_bytes += graph_pool_bytes;
        service->prepared_payload_bytes += graph_pool_bytes;
        service->prepared_payload_peak = std::max(service->prepared_payload_peak, service->prepared_payload_bytes);
    }
    std::unique_ptr<ggml_backend_moe_cpu_prepared_region_impl_v1> region(
        new (std::nothrow) ggml_backend_moe_cpu_prepared_region_impl_v1());
    if (!region) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    region->requirements      = result;
    region->charged_payload_bytes = charged_payload;
    region->graph_uid         = query->graph_uid;
    region->graph_generation  = query->graph_generation;
    region->source_generation = query->source_generation;
    region->n_dynamic_inputs    = query->n_dynamic_inputs;
    region->n_live_outputs      = query->n_live_outputs;
    region->flags               = query->flags;
    region->source_row_capacity = query->source_row_capacity;
    region->scatter_capacity    = query->scatter_capacity;
    region->n_sources           = query->n_sources;
    region->n_lanes             = query->n_lanes;
    for (uint32_t i = 0; i < query->n_dynamic_inputs; ++i) {
        if (query->dynamic_inputs[i] == query->activation) {
            region->activation_input = i;
        }
        if (query->dynamic_inputs[i] == query->ids) {
            region->ids_input = i;
        }
    }
    region->sources.reset(new (std::nothrow) ggml_backend_moe_source_span_v1[query->n_sources]());
    region->lanes.reset(new (std::nothrow) std::unique_ptr<ggml_backend_moe_cpu_prepared_lane_v1>[query->n_lanes]());
    if (!region->sources || !region->lanes) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    for (uint32_t i = 0; i < query->n_sources; ++i) {
        const auto & source = query->sources[i];
        auto & span = region->sources[i];
        span.struct_size   = sizeof(span);
        span.abi_version   = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
        span.witness       = source.witness;
        span.data          = source.data;
        span.bytes         = source.bytes;
        span.expert_stride = source.expert_stride;
        span.type          = source.tensor->type;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            span.ne[dim] = source.tensor->ne[dim];
            span.nb[dim] = source.tensor->nb[dim];
        }
    }
    for (uint32_t lane_index = 0; lane_index < query->n_lanes; ++lane_index) {
        std::unique_ptr<ggml_backend_moe_cpu_prepared_lane_v1> lane;
        const int32_t status = fidelity ? ggml_backend_moe_cpu_fidelity_build_lane_v1(
            query, result, fidelity_offset, fidelity_bytes, lane) : ggml_backend_moe_cpu_region_build_lane_v1(
                service, query, &result.execution, result.lane_metadata_bytes, result.lane_allocation_bytes,
                shared ? (grow ? next_storage.get() : service->routed_storage.get()) : nullptr, lane);
        if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
            return status;
        }
        region->lanes[lane_index] = std::move(lane);
    }
    region->id = ggml_backend_moe_cpu_region_next_id_v1();
    if (region->id == 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    if (grow) {
        service->routed_storage = std::move(next_storage);
        service->routed_storage_bytes = result.lane_allocation_bytes;
    }
    if (shared) { ++service->routed_regions; }
    *region_out = region->id;
    service->regions[service->n_regions++] = region.release();
    service->prepared_payload_bytes = total;
    service->prepared_payload_peak = std::max(service->prepared_payload_peak, total);
    *requirements = result;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_prepare_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        const struct ggml_backend_moe_cpu_region_query_v1 * query,
        struct ggml_backend_moe_cpu_prepared_requirements_v1 * requirements,
        ggml_backend_moe_cpu_prepared_region_v1_t * region_out) {
    try {
        return ggml_backend_moe_cpu_region_service_prepare_impl_v1(
            service_handle, query, requirements, region_out);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static int32_t ggml_backend_moe_cpu_fidelity_prepare_v1(
        ggml_backend_moe_cpu_service_v1_t service, const struct ggml_backend_moe_cpu_region_query_v1 * query,
        ggml_backend_moe_cpu_prepared_requirements_v1 * requirements, ggml_backend_moe_cpu_prepared_region_v1_t * region) {
    try {
        return ggml_backend_moe_cpu_region_service_prepare_impl_v1(service, query, requirements, region, true);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

struct ggml_backend_moe_cpu_active_job_v1 {
    ggml_backend_moe_cpu_service_impl_v1 * service;
    ggml_backend_moe_cpu_prepared_region_impl_v1 * region;

    ~ggml_backend_moe_cpu_active_job_v1() {
        std::lock_guard<std::mutex> lock(service->mutex);
        GGML_ASSERT(service->active_jobs > 0 && region->active_jobs > 0);
        --service->active_jobs;
        --region->active_jobs;
        service->drain_condition.notify_all();
    }
};

struct ggml_backend_moe_cpu_abort_state_v1 {
    ggml_backend_moe_cpu_service_impl_v1 * service;
    uint64_t epoch;
    const ggml_backend_moe_cpu_execute_control_v1 * control;
};

static bool ggml_backend_moe_cpu_region_abort_v1(void * data) {
    const auto * state = static_cast<const ggml_backend_moe_cpu_abort_state_v1 *>(data);
    if (state->service->closing.load(std::memory_order_acquire)) { return true; }
    if (state->control) { return state->control->abort && state->control->abort(state->control->abort_data); }
    return state->service->cancel_through_epoch.load(std::memory_order_acquire) >= state->epoch;
}

static ggml_backend_moe_cpu_prepared_region_impl_v1 * ggml_backend_moe_cpu_region_find_v1(
        const ggml_backend_moe_cpu_service_impl_v1 * service,
        ggml_backend_moe_cpu_prepared_region_v1_t id) {
    for (uint32_t i = 0; i < service->n_regions; ++i) {
        if (service->regions[i]->id == id) {
            return service->regions[i];
        }
    }
    return nullptr;
}

static bool ggml_backend_moe_cpu_region_span_valid_v1(const void * data, uint64_t bytes) {
    return data != nullptr && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(data);
}

static bool ggml_backend_moe_cpu_region_spans_overlap_v1(
        const void * a, uint64_t a_bytes, const void * b, uint64_t b_bytes) {
    if (!ggml_backend_moe_cpu_region_span_valid_v1(a, a_bytes) ||
            !ggml_backend_moe_cpu_region_span_valid_v1(b, b_bytes)) {
        return true;
    }
    const uintptr_t a_start = reinterpret_cast<uintptr_t>(a);
    const uintptr_t b_start = reinterpret_cast<uintptr_t>(b);
    return a_start < b_start + b_bytes && b_start < a_start + a_bytes;
}

static int32_t ggml_backend_moe_cpu_region_validate_execute_v1(
        const ggml_backend_moe_cpu_prepared_region_impl_v1 * region,
        const ggml_backend_moe_cpu_execute_v1 * execution,
        const ggml_backend_moe_cpu_prepared_lane_v1 * lane,
        const ggml_backend_moe_cpu_execute_result_v1 * result) {
    if (execution == nullptr || execution->struct_size != sizeof(*execution) ||
            (execution->flags & ~uint32_t(GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_PRIVATE_OUTPUTS)) || execution->epoch == 0 ||
            execution->graph_uid != region->graph_uid ||
            execution->graph_generation != region->graph_generation ||
            execution->source_generation != region->source_generation ||
            execution->binding == nullptr || execution->dynamic_inputs == nullptr || execution->outputs == nullptr ||
            execution->n_dynamic_inputs != region->n_dynamic_inputs || execution->n_outputs != region->n_live_outputs ||
            execution->reserved[0] != 0 || execution->reserved[1] != 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    const auto * binding = execution->binding;
    const auto & requirements = region->requirements.execution;
    const bool compact = (region->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES) != 0;
    if (binding->struct_size != sizeof(*binding) || binding->n_rows == 0 || binding->n_routes == 0 ||
            (compact ? binding->n_rows != binding->n_routes || binding->n_routes > requirements.route_capacity :
                       binding->n_rows != requirements.bucket_rows || binding->n_routes != requirements.route_capacity) ||
            binding->expert_ids == nullptr ||
            binding->source_rows == nullptr || binding->scatter_destinations == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    }
    auto * sorted_rows = reinterpret_cast<uint32_t *>(lane->binding_metadata);
    auto * sorted_destinations = sorted_rows + binding->n_routes;
    uint32_t n_source_rows = 0;
    for (uint32_t route = 0; route < binding->n_routes; ++route) {
        if (binding->expert_ids[route] < 0 || (uint32_t) binding->expert_ids[route] >= requirements.expert_count ||
                binding->source_rows[route] >= region->source_row_capacity ||
                binding->scatter_destinations[route] >= region->scatter_capacity) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        const uint32_t row_start = route - route % requirements.routes_per_row;
        if (binding->source_rows[route] != binding->source_rows[row_start]) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        if (route == row_start && !compact) {
            sorted_rows[n_source_rows++] = binding->source_rows[route];
        }
        sorted_destinations[route] = binding->scatter_destinations[route];
    }
    std::sort(sorted_rows, sorted_rows + n_source_rows);
    std::sort(sorted_destinations, sorted_destinations + binding->n_routes);
    if (std::adjacent_find(sorted_rows, sorted_rows + n_source_rows) != sorted_rows + n_source_rows ||
            std::adjacent_find(sorted_destinations, sorted_destinations + binding->n_routes) != sorted_destinations + binding->n_routes) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    }

    for (uint32_t i = 0; i < region->n_dynamic_inputs; ++i) {
        const auto & input = execution->dynamic_inputs[i];
        const auto * tensor = lane->dynamic_inputs[i];
        // Private lane layouts are validated during preparation.
        if (i == region->ids_input) {
            if (input.data != nullptr || input.bytes != 0 || input.row_stride != 0) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
            continue;
        }
        if (!ggml_backend_moe_cpu_region_span_valid_v1(input.data, input.bytes)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        if (i == region->activation_input) {
            const uint64_t row_bytes = tensor->nb[2];
            uint64_t required;
            if (input.row_stride < row_bytes ||
                    !ggml_backend_moe_cpu_region_mul_v1(binding->source_rows[binding->n_routes - 1],
                                                        input.row_stride, &required)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
            for (uint32_t row = 0; row < binding->n_rows; ++row) {
                uint64_t offset;
                if (!ggml_backend_moe_cpu_region_mul_v1(
                        binding->source_rows[row * requirements.routes_per_row], input.row_stride, &offset)) {
                    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
                }
                required = std::max(required, offset);
            }
            if (!ggml_backend_moe_cpu_region_add_v1(required, row_bytes, &required) || required > input.bytes) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        } else if (input.row_stride != 0 || input.bytes < ggml_nbytes(tensor)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
    }

    for (uint32_t output_index = 0; output_index < region->n_live_outputs; ++output_index) {
        const auto & output = execution->outputs[output_index];
        const auto * tensor = lane->live_outputs[output_index];
        const uint64_t route_bytes = tensor->nb[1];
        if (!ggml_backend_moe_cpu_region_span_valid_v1(output.data, output.bytes) || output.route_stride < route_bytes) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        const auto overlaps = [&](const void * data, uint64_t bytes) {
            return ggml_backend_moe_cpu_region_spans_overlap_v1(output.data, output.bytes, data, bytes);
        };
        if (overlaps(execution, sizeof(*execution)) || overlaps(result, sizeof(*result)) ||
                overlaps(binding, sizeof(*binding)) ||
                overlaps(execution->dynamic_inputs, uint64_t(region->n_dynamic_inputs) * sizeof(execution->dynamic_inputs[0])) ||
                overlaps(execution->outputs, uint64_t(region->n_live_outputs) * sizeof(execution->outputs[0])) ||
                overlaps(binding->expert_ids, uint64_t(binding->n_routes) * sizeof(binding->expert_ids[0])) ||
                overlaps(binding->source_rows, uint64_t(binding->n_routes) * sizeof(binding->source_rows[0])) ||
                overlaps(binding->scatter_destinations, uint64_t(binding->n_routes) * sizeof(binding->scatter_destinations[0]))) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        for (uint32_t previous = 0; previous < output_index; ++previous) {
            if (overlaps(execution->outputs[previous].data, execution->outputs[previous].bytes)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        for (uint32_t i = 0; i < region->n_sources; ++i) {
            if (overlaps(region->sources[i].data, region->sources[i].bytes)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
        for (uint32_t route = 0; route < binding->n_routes; ++route) {
            uint64_t end;
            if (!ggml_backend_moe_cpu_region_mul_v1(binding->scatter_destinations[route], output.route_stride, &end) ||
                    !ggml_backend_moe_cpu_region_add_v1(end, route_bytes, &end) || end > output.bytes) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static bool ggml_backend_moe_cpu_region_private_outputs_v1(
        const ggml_backend_moe_cpu_prepared_region_impl_v1 * region,
        const ggml_backend_moe_cpu_prepared_lane_v1 * lane,
        const ggml_backend_moe_cpu_execute_v1 * execution) {
    if (!lane->fidelity || !(execution->flags & GGML_BACKEND_MOE_CPU_EXECUTE_FLAG_V1_PRIVATE_OUTPUTS)) {
        return false;
    }
    for (uint32_t i = 0; i < region->n_live_outputs; ++i) {
        const auto & output = execution->outputs[i];
        const auto * tensor = lane->live_outputs[i];
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float) ||
                tensor->nb[1] != uint64_t(tensor->ne[0]) * sizeof(float) ||
                reinterpret_cast<uintptr_t>(output.data) % alignof(float) || output.route_stride % alignof(float)) {
            return false;
        }
        for (uint32_t input = 0; input < region->n_dynamic_inputs; ++input) {
            const auto & value = execution->dynamic_inputs[input];
            if (value.bytes && ggml_backend_moe_cpu_region_spans_overlap_v1(output.data, output.bytes, value.data, value.bytes)) {
                return false;
            }
        }
    }
    return true;
}

static int32_t ggml_backend_moe_cpu_region_service_execute_impl_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        ggml_backend_moe_cpu_prepared_region_v1_t region_handle,
        const struct ggml_backend_moe_cpu_execute_v1 * execution,
        struct ggml_backend_moe_cpu_execute_result_v1 * result,
        const uint8_t * ownership = nullptr, uint32_t n_ownership = 0,
        const ggml_backend_moe_cpu_execute_control_v1 * control = nullptr) {
    if (service_handle == nullptr || region_handle == 0 || result == nullptr ||
            result->struct_size != sizeof(*result)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    const auto profile_start = service->cpu_profile ? moe_cpu_profile_now() : 0;
    ggml_backend_moe_cpu_prepared_region_impl_v1 * region;
    ggml_backend_moe_cpu_test_hook_v1_t test_hook;
    void * test_hook_data;
    uint32_t lane_index;
    {
        std::lock_guard<std::mutex> lock(service->mutex);
        if (service->closed) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CLOSED;
        }
        region = ggml_backend_moe_cpu_region_find_v1(service, region_handle);
        if (region == nullptr) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        if (service->active_jobs >= service->n_lanes) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
        }
        test_hook = control ? control->hook : service->test_hook;
        test_hook_data = control ? control->hook_data : service->test_hook_data;
        lane_index = service->next_lane++ % region->n_lanes;
        ++service->active_jobs;
        ++region->active_jobs;
    }
    ggml_backend_moe_cpu_active_job_v1 active = { service, region };
    if (test_hook != nullptr) {
        test_hook(test_hook_data, GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_ADMITTED);
    }
    std::lock_guard<std::mutex> execute_lock(service->execute_mutex);
    auto * lane = region->lanes[lane_index].get();
    if (lane->execution_offsets) {
        // Prepared graphs keep offsets; the service slab can grow between executions.
        const uintptr_t raw = reinterpret_cast<uintptr_t>(service->routed_storage.get());
        const uintptr_t padding = (GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1 -
            raw % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1) % GGML_BACKEND_MOE_CPU_REGION_ALIGNMENT_V1;
        GGML_ASSERT(service->routed_storage && padding <= service->routed_storage_bytes &&
            region->requirements.execution.lane_execution_bytes <= service->routed_storage_bytes - padding);
        lane->execution = service->routed_storage.get() + padding;
        for (uint32_t i = 0; i < region->n_dynamic_inputs; ++i) {
            lane->dynamic_inputs[i]->data = lane->execution + lane->execution_offsets[i];
        }
        for (int i = 0; i < lane->graph->n_nodes; ++i) {
            lane->graph->nodes[i]->data = lane->execution + lane->execution_offsets[region->n_dynamic_inputs + i];
        }
        lane->binding_metadata = lane->execution + lane->binding_offset;
        lane->output_staging = lane->execution + lane->staging_offset;
        lane->plan.work_data = lane->plan.work_size ? lane->execution + lane->work_offset : nullptr;
        lane->plan.mmid_route_filter_data = lane->binding_metadata +
            size_t(region->requirements.execution.route_capacity) * 2 * sizeof(uint32_t);
    }
    const int32_t validation = ggml_backend_moe_cpu_region_validate_execute_v1(region, execution, lane, result);
    if (validation != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return validation;
    }
    const bool routed_operation = (region->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0;
    const auto & requirements = region->requirements.execution;
    if (routed_operation != (ownership != nullptr) ||
            (routed_operation ? n_ownership != requirements.route_capacity : n_ownership != 0)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
    }
    if (routed_operation) {
        if (!ggml_backend_moe_cpu_region_span_valid_v1(ownership, n_ownership)) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING;
        }
        for (uint32_t route = 0; route < n_ownership; ++route) {
            if (ownership[route] > 1) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; }
        }
        for (uint32_t output = 0; output < execution->n_outputs; ++output) {
            if (ggml_backend_moe_cpu_region_spans_overlap_v1(ownership, n_ownership,
                    execution->outputs[output].data, execution->outputs[output].bytes)) {
                return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
            }
        }
    }
    ggml_backend_moe_cpu_abort_state_v1 abort_state = { service, execution->epoch, control };
    if (ggml_backend_moe_cpu_region_abort_v1(&abort_state)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
    }

    const auto * binding = execution->binding;
    const bool compact = (region->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_COMPACT_ROUTES) != 0;
    auto * bound_rows    = reinterpret_cast<uint32_t *>(lane->binding_metadata);
    auto * bound_scatter = bound_rows + binding->n_routes;
    memcpy(bound_rows, binding->source_rows, binding->n_routes * sizeof(bound_rows[0]));
    memcpy(bound_scatter, binding->scatter_destinations, binding->n_routes * sizeof(bound_scatter[0]));
    const bool private_outputs = !routed_operation && ggml_backend_moe_cpu_region_private_outputs_v1(region, lane, execution);
    const auto * destinations = private_outputs ? bound_scatter : nullptr;
    const auto * output_layouts = private_outputs ? execution->outputs : nullptr;
    auto * bound_ownership = lane->binding_metadata + size_t(requirements.route_capacity) * 2 * sizeof(uint32_t);
    uint32_t published_routes = binding->n_routes;
    if (routed_operation) {
        memcpy(bound_ownership, ownership, n_ownership);
        published_routes = 0;
        for (uint32_t route = 0; route < n_ownership; ++route) { published_routes += bound_ownership[route]; }
    }
    if (compact) {
        for (uint32_t i = 0; i < region->n_dynamic_inputs; ++i) {
            auto * tensor = lane->dynamic_inputs[i];
            if (i == region->ids_input) {
                tensor->ne[1] = binding->n_routes;
                tensor->nb[2] = tensor->nb[1] * tensor->ne[1];
                tensor->nb[3] = tensor->nb[2];
            } else if (i == region->activation_input) {
                tensor->ne[2] = binding->n_routes;
                tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
            }
        }
        if (lane->fidelity) {
            for (uint32_t output = 0; output < region->n_live_outputs; ++output) {
                auto * tensor = lane->live_outputs[output];
                tensor->ne[2] = binding->n_routes;
                tensor->nb[3] = tensor->nb[2] * tensor->ne[2];
            }
        } else {
            for (int node_index = 0; node_index < lane->graph->n_nodes; ++node_index) {
                auto * node = lane->graph->nodes[node_index];
                node->ne[2] = binding->n_routes;
                node->nb[3] = node->nb[2] * node->ne[2];
            }
        }
    }
    for (uint32_t i = 0; lane->fidelity == nullptr && i < region->n_dynamic_inputs; ++i) {
        auto * tensor = lane->dynamic_inputs[i];
        const auto & input = execution->dynamic_inputs[i];
        if (i == region->ids_input) {
            const size_t row_bytes = size_t(requirements.routes_per_row) * sizeof(binding->expert_ids[0]);
            if (routed_operation && tensor->nb[1] != row_bytes) {
                for (uint32_t row = 0; row < binding->n_rows; ++row) {
                    memcpy(static_cast<uint8_t *>(tensor->data) + row * tensor->nb[1],
                        binding->expert_ids + row * requirements.routes_per_row, row_bytes);
                }
            } else {
                memcpy(tensor->data, binding->expert_ids, binding->n_routes * sizeof(binding->expert_ids[0]));
            }
        } else if (i == region->activation_input) {
            const uint64_t row_bytes = tensor->nb[2];
            for (uint32_t row = 0; row < binding->n_rows; ++row) {
                const uint32_t source_row = bound_rows[row * requirements.routes_per_row];
                memcpy(static_cast<uint8_t *>(tensor->data) + row * tensor->nb[2],
                       static_cast<const uint8_t *>(input.data) + source_row * input.row_stride,
                       row_bytes);
            }
        } else {
            memcpy(tensor->data, input.data, ggml_nbytes(tensor));
        }
    }

    const auto profile_bound = service->cpu_profile ? moe_cpu_profile_now() : 0;
    lane->plan.abort_callback = ggml_backend_moe_cpu_region_abort_v1;
    lane->plan.abort_callback_data = &abort_state;
    const auto compute_started = moe_cpu_profile_now();
    enum ggml_status compute_status;
    if (lane->fidelity && service->source_pool) {
        compute_status = ggml_moe_source_pool_run(service->source_pool, lane->fidelity, binding,
            &execution->dynamic_inputs[region->activation_input], lane->live_outputs,
            ggml_backend_moe_cpu_region_abort_v1, &abort_state, test_hook, test_hook_data, output_layouts, destinations);
    } else if (lane->fidelity) {
        compute_status = ggml_moe_cpu_fidelity_run(lane->fidelity, service->threadpool, binding,
            &execution->dynamic_inputs[region->activation_input], lane->live_outputs,
            ggml_backend_moe_cpu_region_abort_v1, &abort_state, test_hook, test_hook_data, output_layouts, destinations);
    } else {
        ggml_moe_source_pool_suspend(service->source_pool);
        compute_status = ggml_graph_compute(lane->graph, &lane->plan);
        if (service->source_pool) { ggml_threadpool_pause(service->threadpool); }
    }
    const auto compute_finished = moe_cpu_profile_now();
    const auto profile_computed = service->cpu_profile ? compute_finished : 0;
    lane->plan.abort_callback = nullptr;
    lane->plan.abort_callback_data = nullptr;
    if (compute_status == GGML_STATUS_ABORTED || ggml_backend_moe_cpu_region_abort_v1(&abort_state)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
    }
    if (compute_status != GGML_STATUS_SUCCESS) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }

    for (uint32_t output_index = 0; output_index < region->n_live_outputs; ++output_index) {
        const auto & layout = requirements.live_outputs[output_index];
        const size_t bytes = compact ? ggml_nbytes(lane->live_outputs[output_index]) : layout.bytes;
        GGML_ASSERT(bytes <= layout.bytes);
        if (!lane->fidelity) {
            memcpy(lane->output_staging + layout.offset, lane->live_outputs[output_index]->data, bytes);
        }
    }
    if (test_hook != nullptr) {
        test_hook(test_hook_data, GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_BEFORE_COMMIT);
    }
    if (ggml_backend_moe_cpu_region_abort_v1(&abort_state)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CANCELED;
    }
    if (test_hook != nullptr) {
        test_hook(test_hook_data, GGML_BACKEND_MOE_CPU_TEST_PHASE_V1_AFTER_COMMIT);
    }
    for (uint32_t output_index = 0; !private_outputs && output_index < region->n_live_outputs; ++output_index) {
        const auto & output = execution->outputs[output_index];
        const auto & layout = requirements.live_outputs[output_index];
        const auto * tensor = lane->live_outputs[output_index];
        const uint64_t route_bytes = tensor->nb[1];
        const auto * staged = lane->output_staging + layout.offset;
        for (uint32_t route = 0; route < binding->n_routes;) {
            const uint32_t first = route++;
            if (routed_operation && !bound_ownership[first]) { continue; }
            const uint32_t row = first / requirements.routes_per_row;
            const uint32_t selected = first % requirements.routes_per_row;
            const uint64_t offset = row * tensor->nb[2] + selected * tensor->nb[1];
            uint64_t bytes = route_bytes;
            while (route < binding->n_routes && (!routed_operation || bound_ownership[route]) &&
                    output.route_stride == route_bytes &&
                    uint64_t(bound_scatter[route]) == uint64_t(bound_scatter[route - 1]) + 1 &&
                    (route / requirements.routes_per_row) * tensor->nb[2] +
                        (route % requirements.routes_per_row) * tensor->nb[1] == offset + bytes) {
                bytes += route_bytes;
                ++route;
            }
            memcpy(static_cast<uint8_t *>(output.data) + bound_scatter[first] * output.route_stride,
                   staged + offset, bytes);
        }
    }

    ggml_backend_moe_cpu_execute_result_v1 published = {};
    published.struct_size       = sizeof(published);
    published.flags             = GGML_BACKEND_MOE_CPU_EXECUTE_RESULT_FLAG_V1_PUBLISHED;
    published.epoch             = execution->epoch;
    published.graph_uid         = region->graph_uid;
    published.graph_generation  = region->graph_generation;
    published.source_generation = region->source_generation;
    published.compute_ns        = compute_finished - compute_started;
    published.lane_index        = lane_index;
    published.published_routes  = published_routes;
    published.published_outputs = region->n_live_outputs;
    *result = published;
    if (service->cpu_profile) {
        service->profile_bind_ns.fetch_add(profile_bound - profile_start, std::memory_order_relaxed);
        service->profile_compute_ns.fetch_add(profile_computed - profile_bound, std::memory_order_relaxed);
        service->profile_scatter_ns.fetch_add(moe_cpu_profile_now() - profile_computed, std::memory_order_relaxed);
        service->profile_jobs.fetch_add(1, std::memory_order_relaxed);
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_execute_v1(
        ggml_backend_moe_cpu_service_v1_t service,
        ggml_backend_moe_cpu_prepared_region_v1_t region,
        const struct ggml_backend_moe_cpu_execute_v1 * execution,
        struct ggml_backend_moe_cpu_execute_result_v1 * result) {
    try {
        return ggml_backend_moe_cpu_region_service_execute_impl_v1(service, region, execution, result);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }
}

static int32_t ggml_backend_moe_cpu_routed_execute_v1(
        ggml_backend_moe_cpu_service_v1_t service, ggml_backend_moe_cpu_prepared_region_v1_t region,
        const ggml_backend_moe_cpu_execute_v1 * execution, const uint8_t * ownership, uint32_t n_ownership,
        ggml_backend_moe_cpu_execute_result_v1 * result) {
    if (!ownership || !n_ownership) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_BINDING; }
    try {
        return ggml_backend_moe_cpu_region_service_execute_impl_v1(service, region, execution, result, ownership, n_ownership);
    } catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }
}

static int32_t ggml_backend_moe_cpu_controlled_execute_v1(
        ggml_backend_moe_cpu_service_v1_t service, ggml_backend_moe_cpu_prepared_region_v1_t region,
        const ggml_backend_moe_cpu_execute_v1 * execution, const uint8_t * ownership, uint32_t n_ownership,
        const ggml_backend_moe_cpu_execute_control_v1 * control, ggml_backend_moe_cpu_execute_result_v1 * result) {
    if (!control || control->struct_size != sizeof(*control) || control->abi_version != 1 || control->flags ||
            control->reserved32 || control->reserved[0] || control->reserved[1]) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    const auto borrowed = *control;
    try {
        return ggml_backend_moe_cpu_region_service_execute_impl_v1(service, region, execution, result, ownership, n_ownership, &borrowed);
    } catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED; }
}

static int32_t ggml_backend_moe_cpu_region_service_state_impl_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        struct ggml_backend_moe_cpu_service_state_v1 * state) {
    if (service_handle == nullptr || state == nullptr || state->struct_size != sizeof(*state)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    ggml_backend_moe_cpu_service_state_v1 result = {};
    result.struct_size = sizeof(result);
    {
        std::lock_guard<std::mutex> lock(service->mutex);
        result.active_jobs    = service->active_jobs;
        result.active_regions = service->n_regions;
        result.closed         = service->closed;
        result.prepared_payload_bytes = service->prepared_payload_bytes;
        result.prepared_payload_peak = service->prepared_payload_peak;
        result.prepared_payload_limit = service->prepared_payload_limit;
    }
    *state = result;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_state_v1(
        ggml_backend_moe_cpu_service_v1_t service,
        struct ggml_backend_moe_cpu_service_state_v1 * state) {
    try {
        return ggml_backend_moe_cpu_region_service_state_impl_v1(service, state);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }
}

static int32_t ggml_backend_moe_cpu_region_service_set_test_hook_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        ggml_backend_moe_cpu_test_hook_v1_t hook, void * data) {
    if (service_handle == nullptr || (hook == nullptr && data != nullptr)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    try {
        auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
        std::lock_guard<std::mutex> lock(service->mutex);
        if (service->active_jobs != 0) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_JOBS;
        }
        service->test_hook = hook;
        service->test_hook_data = data;
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }
}

static int32_t ggml_backend_moe_cpu_region_service_cancel_impl_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle, uint64_t through_epoch) {
    if (service_handle == nullptr || through_epoch == 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    uint64_t current = service->cancel_through_epoch.load(std::memory_order_relaxed);
    while (current < through_epoch && !service->cancel_through_epoch.compare_exchange_weak(
               current, through_epoch, std::memory_order_release, std::memory_order_relaxed)) {
    }
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_cancel_v1(
        ggml_backend_moe_cpu_service_v1_t service, uint64_t through_epoch) {
    try {
        return ggml_backend_moe_cpu_region_service_cancel_impl_v1(service, through_epoch);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }
}

static int32_t ggml_backend_moe_cpu_region_service_drain_impl_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle) {
    if (service_handle == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    std::unique_lock<std::mutex> lock(service->mutex);
    service->drain_condition.wait(lock, [&]() { return service->active_jobs == 0; });
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_drain_v1(ggml_backend_moe_cpu_service_v1_t service) {
    try {
        return ggml_backend_moe_cpu_region_service_drain_impl_v1(service);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_COMPUTE_FAILED;
    }
}

static int32_t ggml_backend_moe_cpu_region_service_close_impl_v1(ggml_backend_moe_cpu_service_v1_t service_handle) {
    if (service_handle == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    std::unique_lock<std::mutex> lock(service->mutex);
    service->closed = true;
    service->closing.store(true, std::memory_order_release);
    service->cancel_through_epoch.store(UINT64_MAX, std::memory_order_release);
    service->drain_condition.wait(lock, [&]() { return service->active_jobs == 0; });
    ggml_moe_source_pool_stop(service->source_pool);
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_close_v1(ggml_backend_moe_cpu_service_v1_t service_handle) {
    try {
        return ggml_backend_moe_cpu_region_service_close_impl_v1(service_handle);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static int32_t ggml_backend_moe_cpu_region_service_destroy_region_impl_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        ggml_backend_moe_cpu_prepared_region_v1_t * region_handle) {
    if (service_handle == nullptr || region_handle == nullptr || *region_handle == 0) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(service_handle);
    ggml_backend_moe_cpu_prepared_region_impl_v1 * region;
    {
        std::lock_guard<std::mutex> lock(service->mutex);
        uint32_t index = 0;
        while (index < service->n_regions && service->regions[index]->id != *region_handle) {
            ++index;
        }
        if (index == service->n_regions) {
            *region_handle = 0;
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
        }
        region = service->regions[index];
        if (region->active_jobs != 0) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_JOBS;
        }
        for (uint32_t i = index + 1; i < service->n_regions; ++i) {
            service->regions[i - 1] = service->regions[i];
        }
        service->regions[--service->n_regions] = nullptr;
        service->prepared_payload_bytes -= region->charged_payload_bytes;
        if ((region->flags & GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION) != 0) {
            GGML_ASSERT(service->routed_regions > 0);
            if (--service->routed_regions == 0) {
                service->prepared_payload_bytes -= service->routed_storage_bytes;
                service->routed_storage_bytes = 0;
                service->routed_storage.reset();
            }
        }
        delete region;
    }
    *region_handle = 0;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
}

static int32_t ggml_backend_moe_cpu_region_service_destroy_region_v1(
        ggml_backend_moe_cpu_service_v1_t service_handle,
        ggml_backend_moe_cpu_prepared_region_v1_t * region_handle) {
    try {
        return ggml_backend_moe_cpu_region_service_destroy_region_impl_v1(service_handle, region_handle);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static int32_t ggml_backend_moe_cpu_region_service_destroy_impl_v1(ggml_backend_moe_cpu_service_v1_t * service_handle) {
    if (service_handle == nullptr || *service_handle == nullptr) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    auto * service = static_cast<ggml_backend_moe_cpu_service_impl_v1 *>(*service_handle);
    {
        std::lock_guard<std::mutex> lock(service->mutex);
        service->closed = true;
        if (service->n_regions != 0) {
            return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_ACTIVE_REGIONS;
        }
    }
    ggml_moe_source_pool_stop(service->source_pool);
    const int32_t status = ggml_backend_moe_cpu_region_service_release_v1(service);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
        return status;
    }
    delete service;
    *service_handle = nullptr;
    return status;
}

static int32_t ggml_backend_moe_cpu_region_service_destroy_v1(ggml_backend_moe_cpu_service_v1_t * service_handle) {
    try {
        return ggml_backend_moe_cpu_region_service_destroy_impl_v1(service_handle);
    } catch (...) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
}

static const struct ggml_backend_moe_cpu_region_service_api_v1 * ggml_backend_moe_cpu_region_service_v1(void) {
    static const struct ggml_backend_moe_cpu_region_service_api_v1 api = {
        /* .struct_size    = */ sizeof(api),
        /* .abi_version    = */ 1,
        /* .create         = */ ggml_backend_moe_cpu_region_service_create_v1,
        /* .prepare        = */ ggml_backend_moe_cpu_region_service_prepare_v1,
        /* .execute        = */ ggml_backend_moe_cpu_region_service_execute_v1,
        /* .state          = */ ggml_backend_moe_cpu_region_service_state_v1,
        /* .set_test_hook  = */ ggml_backend_moe_cpu_region_service_set_test_hook_v1,
        /* .cancel         = */ ggml_backend_moe_cpu_region_service_cancel_v1,
        /* .drain          = */ ggml_backend_moe_cpu_region_service_drain_v1,
        /* .close          = */ ggml_backend_moe_cpu_region_service_close_v1,
        /* .destroy_region = */ ggml_backend_moe_cpu_region_service_destroy_region_v1,
        /* .destroy        = */ ggml_backend_moe_cpu_region_service_destroy_v1,
        /* .execute_routed = */ ggml_backend_moe_cpu_routed_execute_v1,
    };
    return &api;
}

static const ggml_backend_moe_cpu_region_service_api_v1 * ggml_backend_moe_cpu_fidelity_service_v1() {
    static const auto api = [] {
        auto result = *ggml_backend_moe_cpu_region_service_v1();
        result.create = ggml_backend_moe_cpu_fidelity_create_v1;
        result.prepare = ggml_backend_moe_cpu_fidelity_prepare_v1;
        return result;
    }();
    return &api;
}

static int32_t ggml_backend_moe_cpu_fidelity_service_requirements_v1(
        const ggml_backend_moe_cpu_service_config_v1 * config, uint64_t * bytes) try {
    bool runtime_unproven = false;
    if (!ggml_moe_fidelity_selection().valid) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    return ggml_backend_moe_cpu_service_requirements_impl_v1(config, bytes, &runtime_unproven,
        ggml_moe_fidelity_selection().source_pool);
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static int32_t ggml_backend_moe_cpu_fidelity_region_requirements_v1(
        const ggml_backend_moe_cpu_service_config_v1 * config, const struct ggml_backend_moe_cpu_region_query_v1 * query,
        ggml_backend_moe_cpu_prepared_requirements_v1 * requirements) try {
    uint64_t service_bytes = 0, fidelity_bytes = 0, fidelity_offset = 0;
    bool runtime_unproven = false;
    if (!ggml_moe_fidelity_selection().valid) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION; }
    int32_t status = ggml_backend_moe_cpu_service_requirements_impl_v1(config, &service_bytes, &runtime_unproven,
        ggml_moe_fidelity_selection().source_pool);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
    if (!query || query->struct_size < sizeof(*query) || !requirements || requirements->struct_size != sizeof(*requirements) || requirements->abi_version != 1 ||
            query->n_threads != config->n_threads || query->n_lanes != config->n_lanes ||
            query->source_generation != config->source_owner->generation) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_INVALID_ARGUMENT;
    }
    ggml_backend_moe_cpu_prepared_requirements_v1 result = {};
    result.struct_size = sizeof(result);
    result.abi_version = 1;
    status = ggml_backend_moe_cpu_prepared_requirements_impl_v1(query, true, runtime_unproven, &result, fidelity_bytes, fidelity_offset);
    if (status != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { return status; }
    uint64_t total = 0;
    if (!ggml_backend_moe_cpu_region_add_v1(service_bytes, result.prepared_payload_bytes, &total) ||
            (config->prepared_payload_limit && total > config->prepared_payload_limit)) {
        return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY;
    }
    *requirements = result;
    return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK;
} catch (...) { return GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_CAPACITY; }

static const ggml_backend_moe_cpu_fidelity_requirements_api_v1 * ggml_backend_moe_cpu_fidelity_requirements_v1() {
    static const ggml_backend_moe_cpu_fidelity_requirements_api_v1 api = {
        sizeof(api), 1, ggml_backend_moe_cpu_fidelity_service_requirements_v1, ggml_backend_moe_cpu_fidelity_region_requirements_v1,
    };
    return &api;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the same API
static ggml_backend_feature * ggml_backend_cpu_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        ggml_cpu_init();

        std::vector<ggml_backend_feature> features;
        if (ggml_cpu_has_sse3()) {
            features.push_back({ "SSE3", "1" });
        }
        if (ggml_cpu_has_ssse3()) {
            features.push_back({ "SSSE3", "1" });
        }
        if (ggml_cpu_has_avx()) {
            features.push_back({ "AVX", "1" });
        }
        if (ggml_cpu_has_avx_vnni()) {
            features.push_back({ "AVX_VNNI", "1" });
        }
        if (ggml_cpu_has_avx2()) {
            features.push_back({ "AVX2", "1" });
        }
        if (ggml_cpu_has_f16c()) {
            features.push_back({ "F16C", "1" });
        }
        if (ggml_cpu_has_fma()) {
            features.push_back({ "FMA", "1" });
        }
        if (ggml_cpu_has_bmi2()) {
            features.push_back({ "BMI2", "1" });
        }
        if (ggml_cpu_has_avx512()) {
            features.push_back({ "AVX512", "1" });
        }
        if (ggml_cpu_has_avx512_vbmi()) {
            features.push_back({ "AVX512_VBMI", "1" });
        }
        if (ggml_cpu_has_avx512_vnni()) {
            features.push_back({ "AVX512_VNNI", "1" });
        }
        if (ggml_cpu_has_avx512_bf16()) {
            features.push_back({ "AVX512_BF16", "1" });
        }
        if (ggml_cpu_has_amx_int8()) {
            features.push_back({ "AMX_INT8", "1" });
        }
        if (ggml_cpu_has_neon()) {
            features.push_back({ "NEON", "1" });
        }
        if (ggml_cpu_has_arm_fma()) {
            features.push_back({ "ARM_FMA", "1" });
        }
        if (ggml_cpu_has_fp16_va()) {
            features.push_back({ "FP16_VA", "1" });
        }
        if (ggml_cpu_has_matmul_int8()) {
            features.push_back({ "MATMUL_INT8", "1" });
        }
        if (ggml_cpu_has_sve()) {
            features.push_back({ "SVE", "1" });
        }
        if (ggml_cpu_has_dotprod()) {
            features.push_back({ "DOTPROD", "1" });
        }
        if (ggml_cpu_get_sve_cnt() > 0) {
            static std::string sve_cnt = std::to_string(ggml_cpu_get_sve_cnt());
            features.push_back({ "SVE_CNT", sve_cnt.c_str() });
        }
        if (ggml_cpu_has_sme()) {
            features.push_back({ "SME", "1" });
        }
        if (ggml_cpu_has_sme2()) {
            features.push_back({ "SME2", "1" });
        }
        if (ggml_cpu_has_riscv_v()) {
            features.push_back({ "RISCV_V", "1" });
        }
        if (ggml_cpu_get_rvv_vlen() > 0) {
            static std::string rvv_vlen = std::to_string(ggml_cpu_get_rvv_vlen());
            features.push_back({ "RVV_VLEN", rvv_vlen.c_str() });
        }
        if (ggml_cpu_has_vsx()) {
            features.push_back({ "VSX", "1" });
        }
        if (ggml_cpu_has_vxe()) {
            features.push_back({ "VXE", "1" });
        }
        if (ggml_cpu_has_wasm_simd()) {
            features.push_back({ "WASM_SIMD", "1" });
        }
        if (ggml_cpu_has_llamafile()) {
            features.push_back({ "LLAMAFILE", "1" });
        }
    #ifdef GGML_USE_ACCELERATE
        features.push_back({ "ACCELERATE", "1" });
    #endif
    #ifdef GGML_USE_CPU_HBM
        features.push_back({ "CPU_HBM", "1" });
    #endif
    #ifdef GGML_USE_OPENMP
        features.push_back({ "OPENMP", "1" });
    #endif
    #ifdef GGML_USE_CPU_KLEIDIAI
        features.push_back({ "KLEIDIAI", "1" });
    #endif
    #ifdef GGML_USE_CPU_REPACK
        features.push_back({ "REPACK", "1" });
    #endif

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_cpu_graph_plan_set_mmid_route_filter") == 0) {
        return (void *) ggml_backend_cpu_graph_plan_set_mmid_route_filter;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CPU_ROUTED_EXECUTE_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_moe_cpu_routed_execute_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CPU_CONTROLLED_EXECUTE_V1_PROC_NAME) == 0) {
        return (void *) ggml_backend_moe_cpu_controlled_execute_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CPU_FIDELITY_REQUIREMENTS_V1_PROC_NAME) == 0) {
        try {
            (void) ggml_backend_cpu_get_extra_buffer_types();
        } catch (...) { return nullptr; }
        return (void *) ggml_backend_moe_cpu_fidelity_requirements_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CPU_REGION_QUERY_V1_PROC_NAME) == 0) {
        try {
            (void) ggml_backend_cpu_get_extra_buffer_types();
        } catch (...) {
            return nullptr;
        }
        return (void *) ggml_backend_moe_cpu_region_query_v1;
    }
    if (strcmp(name, GGML_BACKEND_MOE_CPU_REGION_SERVICE_V1_PROC_NAME) == 0 ||
            strcmp(name, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME) == 0) {
        try {
            (void) ggml_backend_cpu_get_extra_buffer_types();
        } catch (...) {
            return nullptr;
        }
        return strcmp(name, GGML_BACKEND_MOE_CPU_FIDELITY_SERVICE_V1_PROC_NAME) == 0 ?
            (void *) ggml_backend_moe_cpu_fidelity_service_v1 : (void *) ggml_backend_moe_cpu_region_service_v1;
    }
    if (strcmp(name, "ggml_backend_set_get_rows_callback") == 0) {
        return (void *) ggml_backend_cpu_set_get_rows_callback;
    }
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        ggml_backend_set_n_threads_t fct = ggml_backend_cpu_set_n_threads;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
        return (void *)fct;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cpu_get_features;
    }
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_cpu_set_abort_callback;
    }
    if (strcmp(name, "ggml_backend_cpu_numa_init") == 0) {
        return (void *)ggml_numa_init;
    }
    if (strcmp(name, "ggml_backend_cpu_is_numa") == 0) {
        return (void *)ggml_is_numa;
    }
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        return (void *)ggml_backend_cpu_set_use_ref;
    }

    // threadpool - TODO:  move to ggml-base
    if (strcmp(name, "ggml_threadpool_new") == 0) {
        return (void *)ggml_threadpool_new;
    }
    if (strcmp(name, "ggml_threadpool_free") == 0) {
        return (void *)ggml_threadpool_free;
    }
    if (strcmp(name, "ggml_backend_cpu_set_threadpool") == 0) {
        return (void *)ggml_backend_cpu_set_threadpool;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ ggml_backend_cpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_cpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    // init CPU feature detection
    ggml_cpu_init();

    static struct ggml_backend_reg ggml_backend_cpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_cpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
