# Profile publication age invariant

2026-10-07. A source-proven invariant mismatch exists in frozen release081cf1d792596a2cac5aa06090751d0850aadaab. This is a concrete defect, but no preserved device assertion/status proves it was the exact first failure in PID2753309. The original CUDA fault and full global-OOM causal chain remain unproven.

## Evidence

- moe_grouped_plan_decode rejects any occupied slot with last_used=0 through moe_grouped_plan_fail/__trap. Reciprocal maps alone do not satisfy its ready-slot invariant.
- Frozen apply_source_profile publishes changed occupants and explicitly zeros their last_used. Frozen complete_source_adaptation does the same after asynchronous payload completion. Both paths can leave coherent ready maps that the subsequent ordinary prefill planner rejects.
- prepare_source_group reconciles an earlier ordinary plan and returns it to BUILDING. Source rows use a separate program plan. Later source dispatch does not guarantee a canonical ordinary plan will repair every profile-written age before the next prefill.
- Existing initialize_placement already seeds changed ready ages to1 and restores frequency/epoch/step/clock statistics. Placement adds no usage credit. This supplies the established policy for the repair.

## Isolated candidate

Both source-profile completion writers now call the existing moe_grouped_set_clock device setter with value1 and check launch status. They keep the existing stream, map invalidation, payload copy, publication, completion dependency, canonical owner and usage counters. Zero remains the vacant-slot age. A newly placed ready expert remains oldest for LRU comparison, without an invented observation or clock advance. No model/format gate or executor tier is added.

This candidate is isolated in /home/gencoolpc/llama-moe-generic-prefill-20261007. All frozen controls, prior candidate binaries, parent/specialized trees and other workers' sources remain untouched. The guarded serial CUDA rebuild now passes; no GPU retry, performance/default promotion or release qualification occurred.

## CPU proof and remaining checks

Artifact PROFILE-AGE-INVARIANT.json records a bounded CPU probe that extracts the actual frozen resident-slot predicate and existing scalar setter. With2/64/127 slots, a zero-age vacancy is accepted, a reciprocal resident published with age zero is rejected, and age one satisfies that invariant. Wrong maps and out-of-range occupants still reject; other slot ages remain unchanged. The candidate source check finds both seed-one writers and no remaining source-profile zero writer. This is a source-derived invariant check, not execution of CUDA streams or a whole planner simulation. Each small compile held the two ordered locks under256MiB, zero swap and30s runtime; teardown is verified.

When GPU investigation can safely resume, execute the prepared retained-resource transition fixture before any matched model comparison. Qualify unchanged logits/state/output work and MTP/concurrency. Preserve the hard limits and fatal whole-tree teardown. No blanket stability or no-regression claim follows from the CPU proof.

The concrete fixture is test_fidelity_real_window in tests/test-moe-cache-multigpu.cpp:5608. Existing source-core-profile/source-core-adapt/source-core-async and routed statistics modes cover static seeds, synchronous learning, held asynchronous copies, clone/retirement and exact resident payloads. Their original phase changes remain source execution and miss ordinary prefill.

The fixture now retires the source scheduler, retains the backend owner and candidate generation, and captures every gate/up/down bank allocation. It builds two128-row sequential prefill requests using the original fixture weights and the same backend. Rotating routes cover the full16-expert validation sample; two-expert staging lanes force multiple waves. It checks that retirement/allocation preserves the initial reciprocal map, compares the actual graph result against ordinary CUDA at the existing2e-5 bounded-prefill relative-MSE limit, requires two grouped prefill completions/all matrix-bank consumers/multiple waves with zero fallback, rollback or preparation/finish errors, and verifies every bank allocation and candidate key after both requests. It neither replaces the manifest nor allocates a new residency owner. No device-age accessor is added; the real ordinary planner tests the retained state.

The extended translation unit passes a CPU-only syntax check using its actual build compile definitions/includes. containment-profile-transition-syntax-02 verifies768MiB hard memory, zero swap,45s runtime, ordered locks and an empty tree; peak accounted memory472469504 bytes, elapsed1.43s. No object, CUDA build or device execution occurred. The fixture is prepared, not passed on CUDA. Candidate-wave/control binaries still contain their original tests and code.

The subsequent containment-profile-transition-build-02 rebuild compiles moe-cache.cu and the extended test translation unit, then links test-moe-cache and its libraries. Both ordered locks cover the full single-job build, with verified4096MiB memory, zero swap and600s runtime. It passes in38.12s with1516716032 bytes peak accounted memory, no memory-limit/OOM events, unchanged source and a removed process tree. The earlier build-01 attempt exited75 immediately under lock contention without starting a compiler. The compiled test/library snapshot is frozen separately in candidate-profile-age/bin; its manifest has no model server. Existing frozen controls are untouched. Device execution and the exact incident assertion remain unqualified.

## Writer and transition audit

PROFILE-AGE-WRITER-AUDIT.json pins current/frozen source hashes and records seven direct timestamp writers. This is a textual source audit, not proof of every indirect alias write or of the original device assertion.

| Writer | Timestamp policy | Readiness |
| --- | --- | --- |
| moe_grouped_commit_admission |Checked positive clock/rank-derived age |Completed ordinary/auxiliary admission adds actual usage |
| moe_grouped_select_hybrid |Checked clock+rank+1 |Actual resident-hit usage |
| moe_grouped_begin_hybrid_admission |0 |Owner/map claim removed before replacement |
| moe_grouped_begin_hybrid_admissions |0 |Owner/map claim removed before replacement |
| initialize_placement |1 for changed ready occupants |Restores original frequency/epoch/step/clock; no usage credit |
| apply_source_profile |1 in isolated correction; frozen release wrote0 |Payload completion precedes ready-map publication |
| complete_source_adaptation |1 in isolated correction; frozen release wrote0 |Async payload completion precedes ready maps/event |

Device-resource initialization sets owners/maps to vacant and ages to0 together. The older per-tensor cache->last_used/access_counter belongs to a different cache; its zero-as-vacant convention is not a precedent for publishing ready grouped occupants.

core_session::acquire_owner calls prepare_source_group, which can reconcile a completed ordinary plan before reading maps. It does not call select_hybrid_group. Source-core itself has no last_used reference, and the rows-plan API borrows the canonical maps while writing a separate plan. Consequently source decode observations do not guarantee repair of every profile-written canonical age before ordinary prefill. The prepared source-retirement regression preserves this boundary rather than replacing the owner. Its before/after device result remains unmeasured.

## Scoped synthetic device investigation

The corrected test-only candidate passes the guarded CPU command/manifest preflight:3072MiB hard accounted memory, zero swap,120s finite runtime and verified whole-tree removal. test_profile_transition.py permits only test-moe-cache --hybrid-metadata-only with source-core-profile/adapt/async; it validates frozen candidate hashes, acquires both ordered locks nonblocking, installs the fatal-log group-kill watcher before child launch and requires all eight retained-owner prefill checks plus four source fixtures. No server or faulty frozen GPU control is launched. GPU-RUNS-BLOCKED.json remains present and continues to block run_models.py. Its explicit tiny-fixture scope follows the now-verified containment prerequisites; it does not release the original Flash workload or establish its root cause. The previous block file is preserved. Another session currently holds both locks with a live runner; no device test has started.

## Completed small device transition

PROFILE-TRANSITION-QUALIFICATION.json qualifies the corrected candidate on static placement, synchronous adaptation and held asynchronous copies:24 retained-owner ordinary-prefill requests, separate/fused bank layouts and pinned/pageable transport, forced waves and exact ordinary-CUDA output agreement. All owner keys and bank allocations remain retained; error counters are zero. Each job uses3072MiB hard memory, zero swap,120s runtime and both ordered locks; RAM peaks423-432MiB, all test PIDs/trees are gone and the interval's kernel log contains no Xid/OOM. This closes only the small age-publication/transition qualification. No model replay, faulty frozen GPU control, default promotion or matched serving-performance claim follows. The exact original incident assertion/root cause remains unproven.

Async command setup initially used0.5, violating the pending all-CPU-miss assertion, then0, violating the seed-transfer assertion. Both failed jobs terminated cleanly and remain preserved. Source gates require floor(8*n/256)>0 and floor(3*n/256)=0; numerator64/fraction0.25 satisfies both. The test assertions and product source were unchanged. Static and synchronous cases use0.5. Follow the actual fixture recipe rather than treating a shared environment as interchangeable across these tests.
