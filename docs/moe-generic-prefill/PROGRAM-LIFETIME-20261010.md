# Program lifetime investigation

## Observed contract mismatch

Fresh paired traces212/213 show398.37ms without recorded CUDA activity in generic's repeated first-token window, versus4.38ms specialized. This is a diagnostic window, not proof that all398.37ms is host preparation or recoverable.228 confirms2048-row, normal-input1-row and sampled-input1-row graph rebuilds on each uncached request. Generic source preparation logs and CUDA graph/allocation APIs account for portions, but exact fresh function attribution remains incomplete.

The existing source variant cache in src/llama-context.cpp:3253-3360 bounds admission by min(n_ubatch,max(n_outputs_max,sched_decode_outputs)). Server n_outputs_max follows parallel count times enabled draft width, capped by batch capacity (tools/server/server-context.cpp:73-86, common/speculative.cpp:3172-3181). With one request and no draft this excludes2048-row prefill. Sampled-input mode changes also reject/clear variants. Cached snapshots already store sampled mode, but lookup currently compares only outputs and execution certificate. Removing the mode rejection without changing the key is unsafe.

Phase-aware prepare_sched_reserve:2115-2150 requests shrink as token/KV reservations fall and invalidates captured addresses on backing-generation changes. Disabling phase awareness is not a valid workaround:227 explicitly rejects required hybrid because allocator generation is unavailable. Do not bypass retirement or fabricate allocator generations.

## Existing mechanisms to reuse

The execution certificate describes graph domain, rows, sequence semantics and grouped requirement; sequential prefill has a nonzero certificate. llm_graph_result::can_reuse checks the upstream graph parameters and every original input. These checks remain mandatory after restoring any snapshot.

ggml_backend_sched_moe_source_clone_v1 (ggml-backend.cpp:1428-1447) already shares the reference-counted resizable allocation plan through ggml_gallocr_share_resizable_plan (ggml-alloc.c:1026-1046). This is the existing scheduler/owner pattern, not permission to copy an executor or create an independent residency owner. Shared generation and retirement changes must invalidate affected snapshots; drains/synchronization protect readers and captured addresses.

## Bounded next step and decision

Investigate extending the existing variant cache to cover actual prefill plus normal/sampled decode shapes. Key sampled mode explicitly, retain original can_reuse checks, and bound aggregate private resources across all snapshots/devices. Measured normal and sampled single-row pool capacities are5997056bytes each; main prefill capacity is384494592bytes. Shared canonical backing is not duplicated by the existing clone contract, but other private and pinned allocations still need aggregate accounting. Do not infer a complete budget from these pool capacities alone.

First attribute backing-generation/shrink/retirement changes at transitions. A small temporary diagnostic may estimate the value of retaining peak backing while preserving generation checks; it cannot establish production acceptability or equal-memory speed. No implementation of this cache extension or shrink suppression is included here. If a safe change requires a new scheduling/storage pattern rather than the existing bounded cache/shared allocator contracts, present the concrete design under AGENTS.md's large-pattern decision before code. The owner already authorized the generalized hybrid pipeline; avoid unrelated executor redesign.

Source/reference work remains independent of GPU locks. Any later GPU experiment must use the established ordered locks, frozen provenance, finite timeout, fatal-CUDA teardown, exact output/work checks and actual resource accounting.228 is terminal/passed/tree-empty;227 is a contained admission rejection. No new workload is queued.

## Independent lifetime diagnostic

Temporary229 interposes only the original exported allocator state/shrink calls. A CPU mock verifies exact forwarding versus explicit no-shrink behavior; existing test-alloc passes53 CPU/dummy-backend gates, including shared shape plans, replacement callbacks and teardown order.230's original-shrink control confirms backing generations change at prefill/decode transitions.

Source sched_reserve:2195-2218 unconditionally resets source state and prior graphs after the reservation changes, independently of whether physical shrinking occurs. Therefore retaining allocations alone cannot establish program reuse.231 tests physical shrink suppression only; a separate temporary232 probe wraps the existing sched_reserve member using the real context header and actual original plan. It may retain an already sufficient reservation, preserving explicit sched_need_reserve refresh, source-core gating, shared-workspace acquisition and original allocation generation checks.233 is its forwarding control;234 retains the reservation. These probes are external artifacts, not production edits. Runtime comparisons and whole-tree teardown remain required; any added live workspace must be reported rather than called equal-budget optimization.

## Diagnostic results and revised boundary

230/231 are terminal/passed/tree-empty with four exact64-token sequences and12363MiB sampled peaks. Suppressing physical shrink stabilizes allocator generation4, but retains three graph rebuilds per request. Repeated prefill1446.37 ->1452.60 tok/s (+0.43%, one process each) does not qualify a useful gain. ALLOCATOR-SHRINK-DIAGNOSTIC-REPORT.json preserves the full events.

233/234 are also terminal/passed/tree-empty with four exact64-token sequences. The temporary phase-retention wrapper skips126 of131 reservation calls, but repeated prefill1441.62 ->1432.03 tok/s (-0.67%, one process each). Sampled peaks12363 ->12383MiB. Cold1005.34 ->740.85 varies with substantially greater preparation/capture cost; it is not repeatable regression proof. The experiment is rejected as a performance fix. PHASE-RETENTION-DIAGNOSTIC-REPORT.json preserves it; all production source and binaries still match222.

The remaining forced refresh is concrete: src/llama-context.cpp:4768 sets sched_need_reserve when token placement changes or sampled asynchronous input first becomes active. Keeping memory without preserving this transition does not remove the reset. Source graph variants additionally reject mode changes (3271-3274). Merely removing this rejection is unsafe because variant lookup omits the stored sampled mode, and configure max_prepared_regions is derived from the old variant bound (3116-3118).

Cross-request reuse also depends on sampler lifetime: server-context.cpp:476 detaches a request sampler; set_sampler:4028-4081 forces reservation for sampler changes; llm_graph_params::allow_reuse (llama-graph.h:915-935) and llm_graph_input_sampling::can_reuse (llama-graph.cpp:1395-1408) check original sampler identities. Cached sampling graphs cannot outlive or silently substitute those owners. No plan to bypass these checks is approved by this evidence. A correct reuse change must handle reservation state, modes, CPU prepared-region capacity, sampler ownership and aggregate private resources together, while retaining upstream can_reuse and generation retirement.

The prior proposed bounded cache extension remains unimplemented. First investigate grouping existing expert products within each ready owner wave: the mapped kernel geometry differs despite matching basic tile/register settings, and the temporary lifetime shortcuts did not establish a model gain. This does not rule out proper program reuse; it rejects treating allocation retention as that implementation.
