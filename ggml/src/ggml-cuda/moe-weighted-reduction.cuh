#pragma once

#include "common.cuh"

struct ggml_backend_graph_optimize_params;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

bool ggml_cuda_match_moe_weighted_reduction(const ggml_cgraph * cgraph, int node_idx,
    ggml_cuda_moe_weighted_reduction_match & match);

void ggml_cuda_moe_weighted_reduction_alloc_deps(const ggml_cuda_moe_weighted_reduction_match & match,
    const ggml_backend_graph_optimize_params * params);

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);
