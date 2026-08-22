#include "ggml.h"
#include "ggml-backend.h"
#include "llama-kv-cache-store-stage.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

static std::vector<float> make_rows(int64_t width, int64_t rows, float base) {
    std::vector<float> values(width*rows);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < width; ++col) {
            values[col + width*row] = base + 0.25f*row + 0.01f*float(col % 7 - 3);
        }
    }
    return values;
}

static void test_key_compatibility(ggml_backend_buffer_type_t buft) {
    const llama_kv_store_stage_key k0 {
        llama_kv_store_stage_side::K, buft, GGML_TYPE_Q8_0, 32,
    };
    const llama_kv_store_stage_key k1 = k0;
    const llama_kv_store_stage_key v0 {
        llama_kv_store_stage_side::V, buft, GGML_TYPE_Q8_0, 32,
    };
    const llama_kv_store_stage_key different_type {
        llama_kv_store_stage_side::K, buft, GGML_TYPE_Q4_0, 32,
    };
    const llama_kv_store_stage_key different_width {
        llama_kv_store_stage_side::K, buft, GGML_TYPE_Q8_0, 64,
    };
    const llama_kv_store_stage_key different_buft {
        llama_kv_store_stage_side::K, nullptr, GGML_TYPE_Q8_0, 32,
    };

    if (!(k0 == k1) || k0 == v0 || k0 == different_type ||
            k0 == different_width || k0 == different_buft) {
        fail("store-stage compatibility key accepted an incompatible layout");
    }
}

static void test_reused_stage_graph_lifetime(ggml_backend_t backend) {
    constexpr int64_t width = 32;
    constexpr int64_t stage_rows = 4;
    constexpr int64_t writes = 2;
    constexpr int64_t cache_rows = 7;

    ggml_init_params params = {
        /*.mem_size   =*/ 2*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize store-stage test context");
    }

    ggml_tensor * k_stage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, stage_rows);
    ggml_tensor * v_stage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, stage_rows);
    ggml_tensor * stores[4] = {
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, cache_rows),
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, cache_rows),
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, cache_rows),
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width, cache_rows),
    };
    ggml_tensor * sources[4] = {
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, writes),
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, writes),
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, writes),
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, writes),
    };
    ggml_tensor * indices[4] = {
        ggml_new_tensor_1d(ctx, GGML_TYPE_I64, writes),
        ggml_new_tensor_1d(ctx, GGML_TYPE_I64, writes),
        ggml_new_tensor_1d(ctx, GGML_TYPE_I64, writes),
        ggml_new_tensor_1d(ctx, GGML_TYPE_I64, writes),
    };
    ggml_tensor * read_indices[4] = {
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, writes),
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, writes),
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, writes),
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, writes),
    };

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_tensor * observed[4] = {};
    const auto add_store = [&](int index, ggml_tensor * stage) {
        auto * stage_view = ggml_view_2d(
                ctx, stage, width, writes, stage->nb[1], 0);
        auto * converted = ggml_cpy(ctx, sources[index], stage_view);
        auto * written = ggml_set_rows(ctx, stores[index], converted, indices[index]);
        observed[index] = ggml_get_rows_as(
                ctx, written, read_indices[index], GGML_TYPE_F32);
        ggml_build_forward_expand(graph, observed[index]);
    };

    // Two layer-shaped K/V stores reuse their side's backing. The second K
    // producer must not clobber the first K consumer, and likewise for V.
    add_store(0, k_stage);
    add_store(1, v_stage);
    add_store(2, k_stage);
    add_store(3, v_stage);

    int copy_nodes = 0;
    int store_nodes = 0;
    int unmatched_copy = 0;
    for (int node_index = 0; node_index < ggml_graph_n_nodes(graph); ++node_index) {
        const auto op = ggml_graph_node(graph, node_index)->op;
        if (op == GGML_OP_CPY) {
            ++copy_nodes;
            ++unmatched_copy;
        } else if (op == GGML_OP_SET_ROWS) {
            ++store_nodes;
            if (unmatched_copy != 1) {
                fail("pooled stage producer/consumer order changed");
            }
            unmatched_copy = 0;
        }
    }
    if (copy_nodes != 4 || store_nodes != 4 || unmatched_copy != 0) {
        fail("pooled stage graph is missing a producer or consumer");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("failed to allocate store-stage test tensors");
    }

    const int64_t row_indices[] = { 1, 5 };
    const int32_t read_row_indices[] = { 1, 5 };
    std::vector<std::vector<float>> expected;
    for (int index = 0; index < 4; ++index) {
        expected.push_back(make_rows(width, writes, 0.5f + 2.0f*index));
        ggml_backend_tensor_set(
                sources[index], expected.back().data(), 0, ggml_nbytes(sources[index]));
        ggml_backend_tensor_set(indices[index], row_indices, 0, sizeof(row_indices));
        ggml_backend_tensor_set(
                read_indices[index], read_row_indices, 0, sizeof(read_row_indices));
    }

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fail("pooled store-stage graph compute failed");
    }

    for (int index = 0; index < 4; ++index) {
        std::vector<float> got(ggml_nelements(observed[index]));
        ggml_backend_tensor_get(observed[index], got.data(), 0, ggml_nbytes(observed[index]));
        for (size_t element = 0; element < got.size(); ++element) {
            if (std::fabs(got[element] - expected[index][element]) > 0.06f) {
                std::fprintf(stderr,
                        "store-stage cache mismatch output=%d element=%zu: got %.6f expected %.6f\n",
                        index, element, got[element], expected[index][element]);
                std::exit(1);
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_t backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fail("failed to initialize CPU backend");
    }

    test_key_compatibility(ggml_backend_get_default_buffer_type(backend));
    test_reused_stage_graph_lifetime(backend);

    ggml_backend_free(backend);
    return 0;
}
