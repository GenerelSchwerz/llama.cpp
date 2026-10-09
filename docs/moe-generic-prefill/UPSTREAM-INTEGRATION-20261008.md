# Generic upstream integration, 2026-10-08

The owner authorized a generic checkpoint, upstream merge, regression investigation and push to GenerelSchwerz/llama.cpp. Standalone Strata and the specialized research port are excluded. This integration does not promote the research branch to the default executor or complete the four feature gates.

## Source lineage

| Input | Identity | Composition |
| --- | --- | --- |
| Generic prefill and learned-state checkpoint | 04caf6fdefd7e8a5b2f45982ff6e40032afb0db1 | Preserves frozen candidate17 implementation and qualification; adds generic maintenance policy |
| Profile-aware capacity | 23bb969f1812cdf06635ad3645b90d4f1a6921e0 | Composed as 4d2fc5aa3ec4d90c84381d433f1865256936944d; preserves learned-state and CPU-prefill contracts |
| Upstream llama.cpp | c35b66744f13cb0dcc476af063e112122eee9355 | Actual Git merge with upstream ancestry, including models/backends/server behavior |
| Previously tested conflict resolutions and CUDA composition | Frozen donor based on baf0661e351f3902f71b5dff828e98600f70d78c | Three-way composition against saved base and current generic HEAD, not wholesale replacement |

Evidence is in /home/gencoolpc/moe-cache-tests/results/generic-strata-catchup-20261008/generic-upstream-integration-01. PREFILL-PREMERGE.json, CHECKPOINT-VALIDATION.json, CAPACITY-RESOLUTIONS.json, DONOR-SOURCE.json and COMPOSITION-RESOLUTIONS.json preserve the source inputs, stable donor hashes, explicit resolutions and private-engine exclusions. The donor checkout remains owned by the optimization session and is not edited.

## Reconciled contracts

- Backend buffer interfaces preserve new upstream allocation hooks while retaining canonical grouped residency and original source lifetimes.
- Graph/model construction inherits upstream semantics and new models; fork admission uses graph/tensor contracts rather than a support allowlist.
- Context and sampler integration preserves MTP, backend sampling and target/auxiliary ownership. Upstream --moe-cache-mib and fork --moe-expert-cache-size cannot run competing expert caches in one context.
- Mandatory hybrid prefill retains source-core admission, including zero CPU share; ordinary prefill fallback is not permitted.
- Profile-aware capacity preserves full-model learned statistics and runtime-independent profile coverage. Selected runtime capacity remains a separate policy.
- CUDA conflict resolutions include the previously tested generic dispatch kernels and shared completion/wait optimizations. Result publication retains system release/acquire and final stream ownership fences; the health-probe path remains bounded and cancellation-aware. Combined source behavior and performance still require current-tree validation.

## Validation and publication gates

1. Review conflict semantics, private-engine tree exclusions and whitespace. Preserve explicit failure contracts and unchanged numerical limits.
2. Build the actual combined CUDA/backend/llama/server and affected test targets using the owned build, plain 18 compiler jobs, finite timeout and build-then-GPU locks.
3. Run registry, profile/capacity, CPU routed service, canonical owner and required source-prefill checks. Qualify Tiny Mixtral Q5_K learned reload/continuation with CPU sharing on/off and different capacities.
4. Run affected upstream allocation, argument, architecture and operator checks, then the saved Flash Next 8194/2048 MTP3/.5 serving recipe with profiling/adaptation. Compare frozen candidate17; distinguish changed MTP/output work from execution changes. Add focused no-MTP control when needed to resolve attribution.
5. Investigate failures/regressions before promotion. Record residual platform, concurrency, multi-device, feature3/4 and held-out quality limits; do not imply an upstream-wide CI matrix passed.
6. Commit the verified merge with Assisted-by disclosure, push only the owner fork research branch, and verify remote identity, upstream ancestry and private-engine exclusion. No PR or upstream push is authorized.

Build and serving jobs retain finite deadlines and whole owned-process-tree teardown. No global settings or unrelated processes are changed. Historical limits and incident evidence remain in their original records; the owner removed the recovery build guard and requested plain 18-job builds. Serving teardown on fatal CUDA errors remains mandatory.

## Integration findings

The first compile caught the resolved donor's untracked ggml-backend-moe-certificates.h dependency. Its existing certificate functions are now included without changing their checks; CERTIFICATE-HEADER.json records donor and normalized final identity.

The broader target build also exposed retained fork server-ubatch fixtures using the old batch_builder.make and allocation initializer. The upstream builder now owns llama_batch_ext. These fixtures bind the same mock memory and eight-sequence limit in the builder, then use the existing extended initializer and reference-returning helper. Row, position, sequence and output expectations remain unchanged; BATCH-API-RESOLUTION.json records the correction. This is a test API reconciliation, not an executor workaround.


## Combined-tree validation

Frozen candidate-prefill-integration-18c builds the CUDA/backend/llama/server and affected allocation, argument, batch and recurrent test targets with plain 18 jobs. All 1547 retained source hashes and 25 binary hashes match the tested checkpoint. Focused checks pass: profile codec and allocation, tensor policy and capability registry, 69632 independent prefill policy cases, 96 canonical owner cases, 1620 exact CPU routed replays plus duplicate/range rejection, 40 source-prefill numerical replays and source-core/overlap checks. Numerical limits are unchanged.

Tiny Mixtral Q5_K passes four 9/128-row CPU-sharing on/off runs with mandatory hybrid prefill, exact learned-file cold reload at two capacities and both continued adaptation windows. Affected upstream allocator, argument, registry, batch, statistics, cache-selector and routed-bank checks pass. CUDA operator coverage passes 792 ordinary/shared-input and 102 HC/norm cases. These focused gates do not substitute for the full platform/server CI matrix or prove the compiled recurrent test passed a real model.

Build and test jobs are terminal and their owned process trees are empty. The previously identified staged cached-MMID fixture and mmap-wrapper teardown checks are outside this qualification; they are not represented as fixed. PREPUBLICATION-VALIDATION.json retains the exact checks and unchanged source identities.

## Flash serving observations

All arms use the saved 8194-token prompt and 2048 output tokens,16K F16 context,16 workers,2048 batch/ubatch,load mode none,profile,0.17 miss transfer,decode overlap and boundary overlap,PLE and backend sampling. CPU-assisted prefill is explicitly off; mandatory hybrid admission remains active. Occurrence adaptation is the actual asynchronous mode2, not occurrence-sync. New uniform allocation is selected explicitly for the matched controls because auto packing is now available.

| Mode | Frozen generic17 prefill tok/s | Merged generic18c prefill tok/s | Frozen generic17 decode tok/s | Merged generic18c decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| MTP3/.5, uniform | 875.93 | 1047.59 | 83.50 | 78.98 |
| No MTP, uniform | 1052.02 | 1101.66 | 57.02 | 53.43 |

The initial 3072 resident expert IDs and 5364121600 resident bytes match exactly in both uniform pairs. MTP accepted/offered is 1303/1570 before versus 1300/1544 after. Generated IDs first differ at 58 with MTP and 16 without MTP; routing, copied bytes and execution work therefore differ. These are single forward observations, not exact-work speed claims. The observed 5.4% and 6.3% decode declines remain an investigation and default-promotion gate; MTP acceptance alone and CPU-assisted prefill do not explain them.

A separate merged auto-packing MTP arm reports 1056.56 prefill and 79.01 decode tok/s, with 1250/1519 MTP acceptance. Its capacity/routing differ and it is not an allocation-only speed proof. SERVING-WORK-COMPARISON.json retains actual maps, generated IDs, byte counts and timings. In the no-MTP pair the aggregate map-acquire timer rises 1.668s to 2.664s, while CPU service and copied bytes also rise. That timer includes publication/policy waiting and is not an isolated cause.

CPU sharing remains opt-in via GGML_MOE_SOURCE_CPU_PREFILL=1. Unset or0 retains a zero-CPU partition inside the hybrid source owner; neither requests ordinary prefill. Prior matched CPU-on observations 529.45 prefill/78.62 decode versus CPU-off 1026.87/84.59 do not justify enabling CPU sharing by default. A generic measured benefit policy remains open.

## Fixed-token diagnostic limits

An existing 512-row forced-token replay without profiling/adaptation reports 20.940 versus 21.109 tok/s. It starts from empty maps and a short prompt, so it does not reproduce the serving residency or prove the merged profiled path is faster. Candidate logits agree on 506/512 top IDs with whole-logit NMSE 0.0023047; no numerical limits were weakened and this does not establish unchanged arithmetic.

The first external diagnostic for the saved 8194 prompt failed during initial prefill preparation. Its raw context parameters left n_outputs_max=0, which expands to 2048 output rows, unlike the server's single output limit. It reserved 3983.45MiB compute before a 59MiB grouped allocation was refused by the GPU budget. Fatal-log containment terminated the entire owned tree; swap/OOM counters stayed zero. PROFILED-FIXED-FAILURE.json preserves the failure and source-backed configuration correction. This harness mismatch does not invalidate the successful server runs or explain the historical OOM root. A subsequent helper identity check also caught a stale manifest before model launch. Original failures are retained separately from the corrected run.


The corrected single-output diagnostic completes both arms with the same 8194 prompt IDs and 512 forced continuation IDs, uniform capacity, shipped profile and asynchronous occurrence adaptation. It reports 56.817 versus 53.821 tok/s, a 5.3% decline with generated continuation held fixed. CPU route counts are 92800 versus 92730, resident routes 146457 versus 146517 and copied payload 11.287 versus 11.305GB. Aggregate source replay time is 8.210 versus 8.338s, while total measured loop time is 9.011 versus 9.513s. Map acquisition and CPU service are close in this diagnostic. The larger loop-time gap extends beyond the source replay timer; these overlapping timers do not isolate its cause. Both owned trees are empty, with sampled GPU peaks 12687/12683MiB. PROFILED-FIXED-COMPARISON.json preserves commands, IDs and counters. This API loop synchronizes every row and differs from server overlap; it narrows the investigation but does not prove a steady-state server regression, unchanged routes or full feature parity.
