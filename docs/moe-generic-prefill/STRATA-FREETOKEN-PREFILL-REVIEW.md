# Strata and FreeToken prefill review

Source review on 2026-10-08, before the approved CPU workspace reuse change. This extends features 1-4; it does not authorize kernel fusion, a new cache owner or model-specific execution.

## Exact references

| Reference | Revision and verification |
| --- | --- |
| Strata | `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, local `/home/gencoolpc/Strata`; `git ls-remote origin HEAD` matches. |
| FreeToken | Local `FreeToken-current` points to `af71ba43206e124f5ff6419b47ee36c6e9981078`, 48 commits behind the checked upstream HEAD. Review first refreshed the relevant five files at upstream `078132480ea4a0424f677a27eb22085624d331e6`. The owner then authorized updating FreeToken-current; its clean detached checkout was updated to that exact revision. No dependencies, build, engine installation or model run changed. |
| Generic candidate | Published checkpoint `f9fe22c22c2a309da74af92570a2a2e414c8bea5` plus pending frontend admission, capture-order, profile/token reacquisition and empty-wave fixes. CPU scratch remains per prepared lane at this review boundary. |

Frozen FreeToken source and hashes: `/home/gencoolpc/moe-cache-tests/results/generic-strata-prefill-20261007/reference-review-20261008/FreeToken-HEAD.json`. Read the pinned source rather than assuming the local symlink is current. Primary paper: [FreeToken, sections 3.1 and 3.2](https://arxiv.org/html/2608.16157).

## Actual mechanisms

| Mechanism | Strata | FreeToken | Generic consequence |
| --- | --- | --- | --- |
| CPU prompt work | `Prefill::run`, `src/prefill/prefill.cpp`: count routes, exclude resident experts, require stable CPU backing, sort candidates by token count then ID. Only experts with at most `MAXT=8` routed tokens qualify. Launch CPU work beside GPU work and join before combination. | Current `OffloadMoELayer::_prefill_routed` computes routed prompt experts on the GPU. Its CPU/GPU miss partition is a decode policy. | Do not treat FreeToken's decode bandwidth fraction as evidence for a fixed prefill split. Our fixed-fraction prompt policy still needs descriptor/backend-derived cost control and matched evidence. Do not import Strata's specialized eight-token capacity as a universal limit. |
| CPU split tuning | Fixed override or measured share. EWMA per-expert costs use alpha 0.25; CPU share is `gpu_cost / (cpu_cost + gpu_cost)`, clamped to 0.05-0.9. Alternating eligible shared/unshared layers feed a median wall-time gate. Default enablement is limited to measured CUDA/single-GPU/no-batch-slot paths and chunks below 1024 tokens. | `load_hybrid_fetch_fraction` prefers concurrent PCIe and CPU measurements: GPU-fetch share is `pcie_overlap / (pcie_overlap + cpu_overlap)`. Standalone fallback is `pcie / cpu`. | Hardware calibration stays separate from full-model learned counts and online residency. Adjacent-layer timing arms are a heuristic, not a matched-model no-regression proof. |
| CPU scratch lifetime | `Prefill::Impl` keeps reusable `cpu_x`, `cpu_rows`, `cpu_actq` and `cpu_nact`; quantizes only needed input rows. `ExpertPool` owns per-worker and bounded multi-expert scratch and batches work in `run_split_multi_native`. | Native `CpuMoeExecutor` owns persistent scratch across layer tasks. `submit` can grow the intermediate scratch when a larger token batch arrives; its comment requires an idle pool and warmup before capture. | Our existing serialized CPU service can own one largest routed execution slab. Keep per-region graph metadata and source witnesses private, and allocate/grow during preparation, never execution. |
| Cross-layer movement | Long chunks stream complete nonresident expert blobs in a fixed order through a bounded ring; short chunks use known routes. Copy/consumption events protect reuse. | `_wait_prefill_overlap` starts current and next full-layer copies. Two full-layer bank views alias the first `2 * num_experts` canonical cache slots. Ready/release events and an initial fence protect prior decode readers. | These are causal prefetch schedules, not future-route prediction. Extend existing bounded source scheduling for feature 3; account actual heterogeneous bank bytes. |
| Cache lending and retained hits | `Prefill::init`/`relayout` can carve prompt buffers from lent upper expert slots. `generate.cpp` records displaced occupants, marks them nonresident and refills after the prompt. | Borrowed slots are invalidated in the same cache maps. Optional `prefill_hit_d2d` gathers retained experts outside the borrowed range on-device and batches only miss transfers; small banks still use full-layer copies. This requires additional copy/kernel facilities. | Feature 4 must use canonical ownership and reader/publication fences. Borrowing storage alone does not satisfy the requested restore-current-learned-occupants contract. Do not import optional fused hit compaction in this non-fusion slice. |

Neither reference's specialized format dispatch, fixed model geometry or frontend restrictions are generic support contracts. Reuse scheduling and lifetime principles through existing GGML descriptors and arithmetic. CPU/GPU numerical differences remain expected and require the existing numerical/state acceptance policy.

## Approved CPU workspace decision

The owner approved replacing duplicated prepared CPU execution slabs with a largest reusable service slab on 2026-10-08. `ggml_backend_moe_cpu_region_service_execute_impl_v1` already holds `execute_mutex` across binding, compute and scatter; independent services retain independent storage and cancellation control. Keep that serialization and the existing workers.

Use the existing routed-operation contract for shared scratch; preserve ordinary/fidelity lane ownership. Keep immutable source bindings and per-lane graph metadata private. Record checked local offsets and rebind every execution after taking the execution mutex. Preparation and growth take the execution mutex before the service mutex; no executing job may see a freed slab. Grow transactionally, charge metadata plus actual shared storage once, account old-plus-new transient storage against an explicit payload limit, and free the slab when its last prepared routed region is destroyed.

Qualification must prove smaller and larger region reuse, execution after growth, failed-growth rollback, exact service ledger, independent services, queued jobs, cancellation and teardown before another Flash attempt. Sparse routes must also cover GPU ranges with zero rows. This does not change CPU arithmetic, profile scores or residency authority.

The pending implementation now uses the existing execution lock to protect transactional service-slab growth and per-execution offset rebinding. Ordinary/fidelity non-routed lanes keep private execution storage. Checked offset tables are charged as metadata; explicit payload limits include old-plus-new slabs during growth. Last routed-region destruction releases shared storage.

The full CUDA/base/test build passes. CPU service tests pass 810 exact ordinary-reference replays through each of the two service entry points (1620 total), nine types and three input periods, with same-size sharing, growth followed by old-region execution, rejected-growth rollback, independent-service output, cancellation and ledger/release checks. Peak accounted memory is 194142208 bytes, with a verified 1 GiB RAM/zero-swap/120-second guard and empty process tree. `CPU-WORKSPACE-REUSE-QUALIFICATION.json` preserves the exact candidate and records open larger-region arithmetic and queued-growth/concurrency gates.

The expanded 96-case CPU/GPU owner fixture passed its CPU-only harness preflight but deferred on the shared locks without starting CUDA. Its arithmetic, fresh full-context/async profile reload and Flash/model/performance checks are still pending. No new model attempt followed the stopped Flash run.

## Current evidence and open gates

Actual CPU-prefill plus exact learned-profile save/fresh reload has passed small LLAMA/Qwen fixtures, Tiny Mixtral Q5_K and Nemotron Q4_K on their recorded candidates. The larger Flash attempt was stopped by the three-second GPU-monitor timeout before a completed prefill, and its process tree was removed. It reached 50.38 GiB accounted resident memory and 21.62 GiB cgroup swap. Per-region slab duplication is source-proven; it is not proof of the original Xid/OOM root cause or the monitor timeout cause.

No speedup, Flash completion, MTP/concurrency/platform qualification, default promotion or held-out Strata profile-quality claim follows from this source review. Features 1 and 2 remain incomplete under GOAL-1-4.md; 3 and 4 remain pending.

## Later qualification, 2026-10-08

The new server-linked slab candidate passes96 actual CPU/GPU owner cases and Tiny Mixtral fresh context/profile reload with both adaptation continuations. Flash now completes both2049-token prefill controls without a new kernel Xid/OOM marker: generic GPU-prefill1096.54 tok/s, experimental CPU-prefill87.64 tok/s, updated actual specialized port1699.57 tok/s. All three return the same4 IDs. The implementation works on this bounded request; the CPU policy fails its own-control performance gate and stays opt-in.

The updated source distinguishes the policies: Strata sorts nonresident cohorts by token demand, limits each CPU cohort to its multi-row capacity, measures CPU/GPU costs and applies a benefit gate only in its qualified chunk range. Generic prefill currently reuses the distinct-miss decode fraction and lacks that benefit gate. Separate prefill hardware policy is the next source-backed prerequisite; the reference's fixed1024-token default cannot become a model-specific generic eligibility rule. This evidence does not resolve the October7 root incident or close platform/concurrency/held-out quality gates. Full commands, raw traces and qualifications: `/home/gencoolpc/moe-cache-tests/results/strata-specialized-update-20261008/REPORT.md`.
