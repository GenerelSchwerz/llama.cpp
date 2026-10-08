# Prefill wave scratch correction

The expanded private CPU/GPU prefill test fails at Q5_K,2048 rows before CPU execution. The fixed CUDA pool rejects a request requiring9391104bytes against4803584bytes capacity; used returns to0 after unwinding. No CUDA fatal or memory-limit event accompanies this fixture failure. This does not prove the original Flash Xid/global OOM root.

## Source cause

`capture_prefill_resources` measures one expert containing every route. `execute_bounded_prefill` later divides GPU work into actual expert waves. `ggml_cuda_mmq_mmid_launch_range` selects tiles from each wave's maximum row count. `mmq_stream_k_requirements` can skip fixup for efficient full-row tiling but require it for a smaller wave. A large row count therefore does not bound scratch storage monotonically.

`launch_mul_mat_q` allocates the fixup from the existing fixed pool. A deficient reserve throws before CPU join. `source_compute` drains and returns failure without publishing. The allocation rejection preserved ownership; increasing a global RAM limit would not correct it.

## Correction

When `ggml_cuda_mmq_routed_requirements` is asked for a bound, it now includes the maximum `nsm * I * J` fixup across eligible existing stream-k tile configurations. This follows the existing kernel's block calculation: fixup either uses `nsm` blocks or is absent when all tiles become blocks. Unsupported or shared-memory-ineligible configurations are excluded. Size products are checked before use. Exact, unbounded resource queries retain their previous calculation.

Source prefill also reserves the original-ID source map alongside the routed consumer's scratch, using the existing aligned checked pool-reserve helper. Both allocations remain in the existing program pool. The correction adds no model/type allowlist, kernel fusion, dynamic replay allocation or second residency authority.

For the failing Q5_K shape, preparation now reserves9391104bytes without recapture. Temporary rejection/exception diagnostics were removed before the qualified build.

## Qualification

Frozen `candidate-feature12-06` passes30 private-program checks: Q5_K, IQ4_NL, Q4_K, Q8_0 and F32;9/65/2048 rows; two steps; two layers and six original projections. Actual CPU workers, resident GPU experts and transferred GPU misses execute. Independent original routes and the composed CPU/GPU reference pass the unchanged2e-5 relative-MSE bound. Maximum observed MSE is1.4196435e-5 in the F32,65-row case. The separate all-GPU diagnostic comparison remains intact.

The same30 checks pass CUDA memcheck with zero errors. Ordinary execution takes9.09s and memcheck13.10s; both hold ordered shared locks, have finite lifetimes and leave empty process trees. These are correctness checks, not serving-speed comparisons. Evidence and exact manifests are in `/home/gencoolpc/moe-cache-tests/results/generic-strata-prefill-20261007/feature12-qualification-20261008/RESULTS.json`.

Main sequential serving admission, aggregate canonical staging accounting, live MTP/context-group/async profile persistence, physical platforms, equal-byte held-out quality and matched performance remain separate acceptance gates. The normal Flash profile test does not substitute for CPU-assisted prefill model qualification.
