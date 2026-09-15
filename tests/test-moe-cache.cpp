// MoE cache test runner and focused modes.

#include "test-moe-cache.h"

int main(int argc, char ** argv) {
    test_moe_tensor_split_rejection();
    if (argc == 2 && strcmp(argv[1], "--tensor-policy-only") == 0) {
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--grouped-multigpu-only") == 0) {
        return test_grouped_multigpu();
    }
    if (argc == 2 && strcmp(argv[1], "--grouped-layer-only") == 0) {
        test_grouped_layer_placement();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--pageable-fallback-only") == 0) {
        test_pageable_cache_fallback();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--grouped-staging-bench") == 0) {
        test_grouped_staging_benchmark();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--early-grouped-only") == 0) {
        test_early_grouped_graphs();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--early-graph-only") == 0) {
        ggml_cuda_moe_grouped_context context(nullptr, 0);
        CHECK(ggml_cuda_moe_grouped_context_test_access::early_graph(context));
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--early-select-only") == 0) {
        ggml_cuda_moe_grouped_context context(nullptr, 0);
        CHECK(ggml_cuda_moe_grouped_context_test_access::early_select(context));
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--early-hc-only") == 0) {
        ggml_cuda_moe_grouped_context context(nullptr, 0);
        CHECK(ggml_cuda_moe_grouped_context_test_access::early_hc(context));
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--prepack-only") == 0) {
        ggml_cuda_moe_grouped_context context(nullptr, -1);
        CHECK(ggml_cuda_moe_grouped_context_test_access::prepack(context));
        return 0;
    }
    if (argc == 2 && (strcmp(argv[1], "--early-copy-only") == 0 || strcmp(argv[1], "--early-copy-graph-only") == 0)) {
        fprintf(stderr, "test-moe-cache: old copy transport retired; use --prepack-only\n");
        return 1;
    }
    const bool registry_only = argc == 2 && strcmp(argv[1], "--registry-only") == 0;
    const bool registry_bench = argc == 2 && strcmp(argv[1], "--registry-bench") == 0;
    const bool cached_fusion_only = argc == 2 && strcmp(argv[1], "--cached-fusion-only") == 0;
    const bool grouped_bench = argc == 2 && strcmp(argv[1], "--grouped-bench") == 0;
    const bool grouped_decode_only = argc == 2 && strcmp(argv[1], "--grouped-decode-only") == 0;
    const bool grouped_replay_only = argc == 2 && strcmp(argv[1], "--grouped-replay-only") == 0;
    const bool grouped_multirow_only = argc == 2 && strcmp(argv[1], "--grouped-multirow-only") == 0;
    const bool grouped_nvfp4_only = argc == 2 && strcmp(argv[1], "--grouped-nvfp4-only") == 0;
    const bool legacy_phase_telemetry_only = argc == 2 && strcmp(argv[1], "--legacy-phase-telemetry-only") == 0;
    const bool staging_pipeline_only = argc == 2 && strcmp(argv[1], "--staging-pipeline-only") == 0;
    const bool gemma_q4_parity_only = argc == 2 && strcmp(argv[1], "--gemma-q4-parity-only") == 0;
    const bool prefill_resident_only = argc == 2 && strcmp(argv[1], "--prefill-resident-only") == 0;
    test_moe_cache_proc_api();
    if (prefill_resident_only) {
        test_prefill_resident_biases();
        return 0;
    }
    if (gemma_q4_parity_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_gemma_q4_cached_cuda_parity(dev);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--grouped-lookup-only") == 0) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_active_grouped_lookup_routes(dev);
        return 0;
    }
    if (grouped_multirow_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_speculative_grouped_intent_splits();
        test_graph_execution_certificate_policy();
        test_speculative_required_grouped_backend_capability(dev);
        test_active_grouped_multirow_graph_modes(dev);
        return 0;
    }
    if (grouped_decode_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_grouped_decode(dev);
        return 0;
    }
    if (grouped_replay_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_grouped_graph_replay_lifecycle(dev);
        test_grouped_graph_replay_lifecycle(dev, 262144, 2);
        test_grouped_graph_replay_lifecycle(dev, 393216, 1);
        test_grouped_graph_replay_lifecycle(dev, 4 * 1024 * 1024, 2, 1024);
        return 0;
    }
    if (grouped_nvfp4_only) {
        test_active_grouped_nvfp4_scales();
        return 0;
    }
    if (legacy_phase_telemetry_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_active_grouped_legacy_phase_telemetry(dev);
        return 0;
    }
    if (staging_pipeline_only) {
        int dev = 0;
        CUDA_OK(cudaGetDevice(&dev));
        test_pageable_staging_pipeline(dev);
        return 0;
    }
    test_speculative_grouped_intent_splits();
    test_candidate_graph_coverage_ledger();
    test_candidate_graph_inventory_reuse();
    test_mmid_capabilities();
    test_scheduler_execution_certificate();
    test_graph_execution_certificate_policy();
    test_candidate_generic_physical_truth();
    test_candidate_producer();
    test_candidate_registry(registry_bench);
    test_legacy_owner_leases();
    test_grouped_context_resources();
    test_grouped_graph_preflight(registry_bench);
    test_grouped_graph_mixed_phase();
    if (registry_only || registry_bench) {
        return 0;
    }

    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    test_speculative_required_grouped_backend_capability(dev);
    if (grouped_bench) {
        test_grouped_decode(dev);
        test_grouped_decode_benchmark(dev);
        return 0;
    }

    test_moe_route_publication_lifetime();
    test_cached_mmid_fusion_decline();
    test_cached_mmid_prefill_and_overflow();
    test_pageable_staging_pipeline(dev);
    test_owner_legacy_cache(dev);
    if (cached_fusion_only) {
        test_cached_mmid_routed_separate_chain();
        return 0;
    }
    test_grouped_decode(dev);
    test_grouped_graph_replay_lifecycle(dev);
    test_grouped_graph_replay_lifecycle(dev, 262144, 2);
    test_grouped_graph_replay_lifecycle(dev, 393216, 1);
    test_grouped_graph_replay_lifecycle(dev, 4 * 1024 * 1024, 2, 1024);
    test_strided_copy_graph_update(dev, false);
    test_strided_copy_graph_update(dev, true);
    test_active_grouped_multirow_graph_modes(dev);
    test_active_grouped_dispatch();

    test_cache_slots_and_staging(dev);
    return 0;
}
