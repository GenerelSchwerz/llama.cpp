# Priority 2: a measured, owner-local MoE placement frontier

Status: Steps 1-3 are implemented. Step 4 is qualified locally and on the remote physical two-GPU no-P2P host as described below. Step 5 has not begun, so this branch makes no new placement-performance claim. Prepared on 2026-09-20 for `moe-cache-multi-gpu`. Initial inspection used `edb542798b6577451722281fb0288f944f47a1e1`; during the task HEAD advanced to `d50dcda11a0db15635f825172b4591a6408e4a70`. The complete intervening diff was reviewed: six lines in `common/common.cpp` log the positive host-pin budget at info level, with no placement/accounting change. The recorded Qwen measurements remain specifically `edb542798`.

Review revision: the implementation contracts, first experiment manifest, and callback protocol below refine this guide without replacing its source-pinned analysis or acceptance criteria. They permit local, reversible engineering choices and bounded exploration. This revision performs no implementation or model tests.

### Local and unpublished evidence

Repository links are relative; code line anchors refer to the reviewed revision. The following shared artifacts are **local/unpublished, not portable GitHub evidence links**. Paths are relative to the external shared root, normally `../moe-cache-tests/` from the repository root (`/home/gencoolpc/moe-cache-tests/` on the reviewed workstation). A repository clone alone does not contain them. Before public publication, either retain this disclosure or publish appropriately reviewed artifacts and replace these references with stable URLs; do not imply that inaccessible evidence is bundled or independently reproducible.

| Reference | Path within the local shared root | Provenance/use |
| --- | --- | --- |
| S1 | `notes/moe-cache-architecture-conversation-review-20260920.md` | Original architecture review; retired-arena lessons at line 341, priorities at line 360, coverage contract at line 431. |
| S2 | `ornith-residency-profile-20260920/RESIDENCY-PROFILE-REPORT.md` | Historical remote residency profile, starting at line 3; not a current-HEAD benchmark. |
| S3 | `results/qwen36-gist-recheck-edb542798-20260920/` | Existing commands, server responses/logs, sampled GPU memory and summaries for the capacity boundary in E1. |
| S4 | `bench.zsh` | Reusable local runner; capture its actual version/hash with future runs. The remote runner must also be located/versioned before use. |

## A. Redundancy verdict and decision

**The existing report is partially sufficient. A concrete design supplement is warranted.** The [preserved architecture report (local/unpublished S1)](#local-and-unpublished-evidence), at line 360, explicitly calls itself an architectural direction, not an implementation plan. Its Priority 2, reference comparison, retired-arena lessons, and coverage contract remain authoritative. Do not replace them with a new five-priority roadmap.

The missing decisions, resolved below, are:

| Missing decision | This design's decision |
| --- | --- |
| What does respecting `-ncmoe` mean? | Preserve its explicit ordinary CPU placement in residual-selector mode; do not redefine it as GPU refill or introduce CPU-miss execution. Preserve legacy precedence without a selector. |
| What is persistent? | Complete expert groups placed in ordinary device buffers at load time, selected at whole-layer granularity initially. Not cache slots with a permanent pin bit. |
| Where is placement selected? | Before real weight allocation, using metadata/no-allocation probes and existing placement parameters. Never mutate buffers in cache finalization or during decode. |
| What is uneven allocation? | Different cache byte budgets and resulting slot counts across physical owners, plus persistent layers. Keep one slot count per model/owner and independent context/group pools. |
| How is a configuration chosen? | First compare explicit configurations using existing controls. Then, only if worthwhile, add a bounded, offline, profile-assisted neighbor advisor to the existing fit utility. No default server autotuner. |
| What is optimized? | Complete request latency and accepted output throughput under per-device peak-memory and host-memory limits, not hit rate, equal GPU utilization, or total VRAM alone. |
| How are drafting and fit handled? | Measure the actual target and every live drafting context/model jointly; count shared storage once but independent execution resources separately. A failed required-draft estimate invalidates the candidate. |
| When is this good enough? | Correctness and no-regression gates at low, half, and full residency; a repeatable placement win must survive matched-memory, unprofiled, multi-prompt comparisons. |

The smallest sound implementation is **placement policy around the existing engine**, not a new engine. Priority 1 already expresses most candidate configurations. Reproduce the suspected selector edge before deciding whether it needs a fix, make joint fit/reporting trustworthy, and exercise explicit frontiers before adding an advisor. If explicit recipes solve the useful cases, stop there; the advisor is a gated convenience, not a prerequisite for claiming a measured placement improvement.

### What was actually rechecked

- Current Priority 1 is committed, not the old uncommitted staging-lane proposal: `6673d4043` added telemetry, `c8780dd01ee5ae4ae36460aedac0c5b02682cd9c` added owner-aware controls/accounting, and `edb542798` added coverage. The worktree was clean when inspected. The subsequent `d50dcda11` merge incorporates logging commit `9174043f044ec191775f0f2ef0eeb10f1438e590`; it does not alter these mechanisms or the cited core implementation line references.
- [The feature guide](fork-features.md#L11) documents layer selectors, independent draft controls, backend-owned host/device sizing, and independent owner pools. These supersede stale intermediate proposals, including an MTP lane multiplier.
- The [preserved report (local/unpublished S1)](#local-and-unpublished-evidence) was read in full; its archived-arena discussion begins at line 341. The archived shared-arena experiment remains negative evidence, not a starting branch.
- The [discussion](https://github.com/ggml-org/llama.cpp/discussions/24528) was reread, including its root, 39 top-level comments and 54 replies returned by the completed API snapshot. Reference heads and relevant primary code/docs were checked separately; forum comparisons are not treated as matched benchmarks.

## B. Implementation-ready design

### B1. Preserve the baseline and separate three decisions

Keep these separate:

1. **Placement:** which device owns a layer, and which complete expert groups are ordinary resident, cached, or ordinary CPU-placed.
2. **Capacity:** how much memory each owner/model/context is allowed and how many slots that affords.
3. **Execution:** today's grouped GPU engine, ordinary resident operations, and existing explicitly eligible non-grouped paths.

Changing placement must not implicitly change eviction, expert routing, arithmetic kernels, graph certification, prediction, or sampling. Priority 2 does not implement CPU misses, expert parallelism, or a shared allocator.

This priority optimizes **owner-local, layer-split placement**. Completing it does not complete the long-term multi-GPU MoE objective: deeper parallel scheduling, expert parallelism and other justified execution changes remain later priorities, with separate evidence and design approval.

The remote machine's unavailable P2P is a supported baseline condition. Keep its normal contiguous layer split and host-mediated boundary transport. Do not assume a P2P enablement, equal link bandwidth, identical GPUs, or a two-device-only data structure.

### B2. Ownership and placement precedence

The unit of ownership is the complete typed source group from `llama_model::build_moe_sources()`, including the banks and eligible auxiliaries its graph consumes. A stable diagnostic key is:

`(model_instance_id, context_role, semantic_group_index, group_domain, physical_device_id)`

The lifetime key also includes the existing registry/source generation and backend context. Layer number alone is not sufficient: normal and chunk expert groups, separate models, target verification, and nextn execution must not collide. Use the existing typed manifest rather than infer architectures from model names or three hard-coded tensor names.

The initial optimization move operates on a **whole layer's routed groups**. If a layer contains multiple expert groups, promote/demote them together. Their dense/shared-expert tensors keep ordinary placement and are included in the fit ledger. No splitting gate/up/down, fused banks, or auxiliaries across independently selected cache owners.

Precedence is explicit:

| Mode | Effective order | Meaning |
| --- | --- | --- |
| Cache disabled | Existing ordinary loader/override rules | Unchanged baseline. |
| Cache enabled, no layer selector | Legacy cache override, then user overrides | Preserve existing scripts and the warning that matching CPU overrides do not win. Do not silently change this default. |
| Explicit cache layer selector | User overrides in their existing first-match order, then cache for selected remaining groups, then ordinary placement | This is the persistent-region/residual-cache contract. |
| Offline advisor | Same rules, applied to a new emitted configuration | It cannot alter locked controls or turn explicit CPU intent into cache placement. |

The implementation is already visible in [src/llama.cpp](../src/llama.cpp#L384). Do not invent a new priority order among `-ot`, `--cpu-moe`, and `-ncmoe`; they currently contribute ordered overrides. Expose the winning override in diagnostics.

**Investigate one possible prerequisite, test first:** [src/llama-model.cpp](../src/llama-model.cpp#L2521) builds `unmatched_selected_layers`, but erases a layer only after finding cached banks at line 2558. Static inspection suggests that a selected, real MoE layer wholly claimed by an explicit CPU/GPU override could reach the error at line 2705 as though it had no routed group. This would conflict with the intended precedence; it is not yet a reproduced failure.

Step 1 first adds a focused reproducer to existing registry/model fixture infrastructure, with parser coverage in [test-arg-parser.cpp](../tests/test-arg-parser.cpp#L268). Use a tiny two-MoE-layer model/metadata fixture (for example eight experts, top-k two, four slots), exercising the real override-resolution and model-finalization path without generation or a large model load. Select layers `0,1`; compare no override with a complete CPU override of layer 0, the equivalent `-ncmoe 1`, and a complete ordinary-GPU override. Layer 1 must remain cached. Also select only overridden layer 0 (zero active cache groups), override just one base bank (partial-group rejection), and retain nonexistent/dense-only selector and selector-free legacy controls. A hand-built backend snapshot or parser-only pass is insufficient to settle loader precedence. Fixture construction and exact shapes may follow existing helpers; a missing CUDA backend is an explicit skip, not a reproduction result.

Record the unmodified baseline's resolved buffers, load result/error and relevant call path. **Make a production change only if the fixture reproduces the mismatch.** Otherwise retain useful regression coverage, explain which earlier validation/transformation prevents the static reading from manifesting, and call back with that evidence; do not patch the suspicious line merely to match this plan.

If confirmed, the proposed correction separates **selector existence validation** from **effective cache participation**. Mark a selected layer as matched when its complete routed source group exists; then validate each effective group. All base banks outside the cache: honor their ordinary overrides and exclude the group from cache; do not impose a new restriction on ordinary mixed-backend placement. Some base banks cached and others overridden: retain the partial-group error. A nonexistent/dense-only selected layer: retain the selector error. If every selected group is overridden, report `active_cached_groups=0`, allocate no cache pools, and retain ordinary execution; do not manufacture a one-slot cache.

#### What respecting `-ncmoe` does, and does not, promise

`-ncmoe N` keeps its ordinary meaning: the first N layers' expert weights receive CPU placement overrides. In selector mode those overrides win. It does **not** mean that CPU arithmetic is competitive, that all phases must execute on CPU regardless of ordinary scheduler behavior, or that cold experts should be executed on CPU inside the grouped cache.

To request GPU refill for a prefix and persistent GPU experts elsewhere, use `-ngl all` with `--moe-expert-cache-layers <prefix>` and a cache budget, **without a conflicting CPU override**. The selector's complement follows normal GPU placement; it is not automatically guaranteed resident if the remaining ordinary settings place it on CPU. The resolved-placement report must show the actual outcome.

Volunteer-1's [specific placement request](https://github.com/ggml-org/llama.cpp/discussions/24528#discussioncomment-18456498) supports preserving deliberate resident regions and uneven MiB allocation. It does not establish that our refill path should become a CPU/GPU hybrid. Their dedicated-attention/expert-device suggestion is a different scheduling design, deferred here.

### B3. Persistent regions and the residual cache

Represent persistent groups with ordinary device weight buffers and ordinary graph operations. They must not become active cache candidates, allocate slot maps/staging, increment cache misses, or wait on cache planning. They may remain in the complete typed manifest with non-cached flags: preserve semantic indexing rather than deleting ordinary-group records. The current snapshot builder includes all source groups ([src/llama-context.cpp](../src/llama-context.cpp#L597)); backend eligibility checks their cached flags separately. Residual groups retain today's grouped engine and safeguards.

This is mostly an existing loader boundary, not a new fast kernel. Audit mixed ordinary/cached graphs in `llama_context::place_moe_regions()` and `refresh_moe_layer_owners()` ([src/llama-context.cpp](../src/llama-context.cpp#L1704)). An inactive cache candidate for an explicitly ordinary group is correct; an inactive candidate for an expected cached group is not.

Do not automatically convert `slots >= experts` into ordinary storage during runtime. Full capacity can still be unpopulated, may consume padded/auxiliary resources, and belongs to different graph lifetimes. Offer a separate, pre-load all-resident configuration when it fits. Compare it with full-capacity cache explicitly.

**Load boundary:** real weight buffers are allocated at [src/llama-model.cpp](../src/llama-model.cpp#L1874); `build_moe_sources()` and `finalize_moe_expert_cache()` run afterward at line 1933. Reassigning buffer types in finalization is too late. Derive any new configuration in a metadata/no-allocation probe, destroy the probe, then perform a fresh real load with frozen parameters. Do not add pointer rebinding, resource migration, or graph recapture to implement static placement.

Initial selection candidates are complete persistent layers, not individually pinned hot experts. Keep the frequency/LRU policy unchanged on residual groups. Hot-expert seeding and admission-policy changes belong to Priority 3 and require domain-shift/cold-start evidence.

### B4. The memory ledger and budget formulas

All arithmetic uses checked byte counts. CLI MiB is `2^20` bytes. Device identity is physical, not merely a CUDA ordinal or position in one model's device list. Preserve selected-device order for public budget arrays and record its physical mapping.

For model `m`, owner `d`, and a fixed set of cached groups:

```text
F[m,d] = sum(group fixed bytes for each live execution-resource instance)
         + sum(context fixed bytes once per live backend context)
P[m,d] = sum(group per-slot bytes for those resource instances)
Emin[m,d] = minimum expert count of participating groups

S[m,d] = min(Emin[m,d], floor((Bcache[m,d] - F[m,d]) / P[m,d]))
C[m,d] = F[m,d] + S[m,d] * P[m,d]
```

An empty group set contributes zero, not division by zero. A positive active group set with insufficient budget is infeasible. This retains [finalize_moe_expert_cache()](../src/llama-model.cpp#L2711), including one owner-local slot count and the conservative global minimum used for graph admission. V2 snapshots carry one `n_slots` per backend owner ([src/llama-context.cpp](../src/llama-context.cpp#L1667)); do not pretend there is an existing per-group capacity API.

Use the backend's current allocation-free `ggml_backend_moe_device_size_query_v1` / result, and staging query, in [ggml-backend-moe.h](../ggml/src/ggml-backend-moe.h#L95). They already cover bank stride/type/row padding, slot auxiliaries, original shadows, prefill copies, metadata, debug state and early-router geometry. Reuse [the CUDA sizing implementation](../ggml/src/ggml-cuda/moe-cache.cu#L2810). Do not reintroduce approximate copies of CUDA structs or allocation formulas in a placement planner.

The physical-device feasibility constraint is:

```text
ordinary unique model allocations[d]        # includes persistent experts
+ all live context KV/recurrent allocations[d]
+ all retained cache allocations C[m,d]
+ measured/estimated peak compute workspaces[d]
+ other retained backend allocations[d]
+ explicit safety margin[d]
<= available device envelope[d]
```

Count an allocation in exactly one category. In particular, do not add `C` twice when it already appears in the context memory breakdown. Use ordinary buffer allocation sizes, not just tensor payload, for persistent banks. Include capture/workspace high-water behavior and retained old resources until their work/leases complete. `no_alloc` is an estimate, not evidence that a first real request or long-context verification fits.

For a proposed persistent group, ask the ordinary owner buffer type for its existing allocation-size/alignment requirements and validate the complete packed candidate in the normal no-allocation loader. A per-group byte attribution is a diagnostic; the candidate's final whole-buffer ledger is authoritative when packing, views or shared tensors prevent an additive decomposition. No second CUDA allocation formula is needed.

Grouped, legacy and grouped-host-staged **device slot storage can alias** through `init_with_pool()` / `owns_slot_pool` ([moe-cache.cu](../ggml/src/ggml-cuda/moe-cache.cu#L13003)). Do not triple-count a borrowed device pool. Their **retained host staging families are independent** and must be summed for eligible pageable groups and live contexts, as current [host fit accounting](../src/llama-model.cpp#L2436) does. Preserve backend ownership, source-registration admission and resource leases; no shared arena and no `load_mtp * 2` heuristic.

For a positive host-pin budget, mandatory staging/auxiliary commitments precede source registration. Charge unique registered source extents once within the actual budget owner and all independently retained staging allocations; shared pointers alone do not prove shared budget ownership. Keep optional growth bounded and distinguish mandatory reserve, actual allocation, and optional headroom. Zero host-pin budget keeps its existing full-pinning/automatic fallback semantics; it does **not** mean pinning disabled or zero host use. Joint host-RAM feasibility is separate from device fit: stock fit explicitly assumes system RAM is unlimited ([common/fit.h](../common/fit.h#L24)).

For an advisor candidate with ordinary resident footprint `R[d]`, derive its available cache envelope from the **remaining memory on that device**, capped by explicit cache limits. Bytes cannot be transferred from one physical GPU to another simply because the sum fits. Promoting one group changes both `F`/`P` and `R`, so recompute all remaining groups' owner slot capacity and staging, not just the promoted group's bytes. Moving a layer also moves its dense/shared weights, context state, compute, and boundary traffic.

The planner must additionally check the workload's grouped route capacity. One slot, currently sufficient for byte-budget admission, is not sufficient for every graph. For existing paths the limit includes routed rows (`top_k * rows`, subject to the actual graph certificate), not merely the unique expert IDs; see [prepare admission](../ggml/src/ggml-cuda/moe-cache.cu#L9299). Use the existing capability/graph checks as authority. Reject an infeasible candidate or retain its existing explicitly supported execution choice; do not silently shorten drafting, change batch size, or enable legacy execution to make a benchmark run.

### B5. Cost model: prioritize exposed traffic, not utilization

Use measured, phase-specific costs for each physical owner and source path. The historical [remote residency profile (local/unpublished S2)](#local-and-unpublished-evidence) measured roughly 3 GiB/s on the slow owner versus roughly 21-23 GiB/s on the other. It also found mapped and asynchronous-copy controls near the slow link's ceiling. These are evidence for what to investigate, not constants valid for current HEAD or a promised speedup.

For a fixed workload/profile, rank a candidate by a conservative estimate of:

```text
request time = prefill + target verification/decode + actual draft work
               + boundary/coordination overhead

group cost = exposed refill bytes / measured effective path bandwidth
             + exposed planning/dispatch time + expert compute time
```

Only sum non-overlapping intervals. For serial layer-split decode, the objective is not `max(GPU0 time, GPU1 time)` as if both GPUs executed the entire request concurrently. CPU wait and its underlying GPU copy are not two independent savings. Use Nsight for exposed critical-path timing and existing telemetry for ownership/byte attribution; do not put blocking CUDA timing in normal inference.

Rank capacity moves by the **net reduction in exposed request time per additional byte on the affected owner**, including lost capacity/higher misses in other residual groups. A slow-link owner can justify more resident experts, but adding its compute-heavy layers can lose more than the saved transfer. Effective host bandwidth also depends on contention, source registration/pageability, quantization, transfer size, and prefill versus decode.

The advisor may use measured miss-byte curves at observed slot counts. Do not extrapolate a frequency histogram into an exact LRU/frequency-cache miss curve. Unknown points remain unknown/exploratory; offline oracle hit rates do not count as runtime results. Profiles are workload hints, not router instructions. Actual routing and all expert contributions remain unchanged.

### B6. Minimum implementation surfaces and optional advisor

#### Always useful: resolved inventory and strict measurement

Add a small read-only placement snapshot using the existing WIP C++ extension seam in [src/llama-ext.h](../src/llama-ext.h#L95). Keep CUDA internals behind the existing private backend size procedures. No new installed public `llama_model_params` fields or backend ABI are necessary for Priority 2.

Suggested internal records, with names illustrative but responsibilities fixed:

| Record | Required fields |
| --- | --- |
| `moe_placement_group` | Model identity, layer, semantic group/domain, context-use mask, physical owner, resolved mode (`ordinary_device`, `residual_cache`, `ordinary_cpu`), winning placement reason, expert count/top-k, bank shape/quant/stride fingerprint, ordinary allocation contribution, backend fixed/per-slot cache contribution. |
| `moe_placement_owner` | Selected-device index and stable identity, active contexts/groups, slot count, cache cap, fixed/per-slot totals, persistent/model/context/compute estimates, mandatory host staging, explicit margin, estimated total and capacity slack. |
| `moe_placement_measurement` | Resolved configuration fingerprint, complete target/draft measurement, validity/errors, estimate-versus-runtime provenance. No owning device buffers or graph pointers. |

Return owned C++ values for metadata callers; do not retain pointers into temporary `no_alloc` models or borrow tensor addresses into a saved profile. Runtime resource IDs/generations stay transient and are never deserialized. Derive this view from current source manifests and memory ledgers, not a second model parser.

Factor a **non-mutating joint configuration measurement** out of [common/fit.cpp](../common/fit.cpp#L30) / [common/fit.h](../common/fit.h#L58). It validates a fully specified candidate and does not run stock fit's placement-changing search. Stock fit currently declines to rearrange user-set `ngl`, tensor splits and overrides ([common/fit.cpp](../common/fit.cpp#L463)); do not bypass those protections or silently clear the user's overrides.

#### Step 2 implementation contract: a view, not a snapshot subsystem

Start with one model-side collector, illustratively `describe_moe_placement(model) -> owned_report`, and a common/tool-side formatter that joins it with effective parameters and context measurements. Its responsibility is to describe resolved ownership and expose the existing ledger, not choose placement, measure GPU performance, own allocations, or reconstruct CUDA sizing. Names and factoring may follow the code; these responsibility boundaries must remain intact.

- **Scope/lifetime:** internal C++ in the existing model/extension seam; no installed public header, stable ABI or new backend proc. Synchronous calls use a live real or `no_alloc` model; return bounded owned strings/records/byte counts usable after probe destruction. No borrowed tensors, graph leases, callbacks or global registry in the result. Model-instance addresses/generations may appear only as transient diagnostics, never persistent identity.
- **Human surface:** one concise owner/role ledger in an explicitly requested fit report and, for real loads, the existing experimental-log channel. Detailed group rows remain opt-in. Preserve the fit utility's existing default fitted-arguments stdout contract and normal inference hot path.
- **Machine surface:** an explicit, tool-only report mode emits one newline-terminated JSON object on stdout, with diagnostics on stderr. A name such as `--fit-moe-report-json` is illustrative, not an existing flag. Initial fields are `schema_version`, `configuration_id`, identity/provenance, groups, owners, measurement status and diagnostics. No server endpoint, streaming event feed, automatic sidecar file or serialized CUDA state. The benchmark runner may save that stdout alongside existing artifacts.
- **Canonical identity v1:** hash an explicitly ordered canonical record of resolved effective settings, not raw CLI text. Include schema/build/backend identity, model/layout identity, ordered physical device mapping, role/context settings, resolved per-layer placement, slots/byte caps, and relevant effective memory/execution flags/environment. Materialize defaults; express sizes as integer bytes and selectors as sorted/coalesced ranges after validation. Preserve first-match override order and device order. Normalize equivalent split spellings through their resolved owner map. Use sorted JSON keys, compact UTF-8 serialization and finite locale-independent numeric values; then `SHA-256` through the [existing hash helper](../vendor/hash/hash.h). Keep the canonical record so an ID can be audited. Exclude pointers, timestamps, free-memory observations, run counters, absolute file locations, ports and secrets. A new hashed-field meaning requires a schema bump.
- **Model identity without hidden I/O:** reuse an available artifact content digest; do not read all GGUF payloads just to generate a report. If only metadata/layout, file sizes and modification stamps are available, label the identity `local_unverified`, include that identity kind in the canonical record, and do not advertise cross-machine/content equivalence. Effective configuration IDs remain distinct from a workload/run ID and from proof of model contents.
- **Provenance:** each byte category records its source and whether it is an exact sizing result, conservative estimate, current observed allocation, or sampled peak (a lower bound on instantaneous peak). Unavailable observations are null/unknown, never zero. Initial JSON can be estimate-only; runtime observations come from existing telemetry and are joined explicitly by configuration/role/device and run window. Producing a snapshot must not allocate cache pools, synchronize GPU work, reset counters or change population.

Split collection/canonicalization from formatting integration (Steps 2a/2b below). Return for architectural review if accurate reporting appears to require a new global allocation tracker, graph/resource ownership, public ABI, per-token synchronization, new reset semantics or a generic snapshot service. A small extra read-only field with clear provenance or a formatter adjustment remains local discretion. Exact field names may evolve through fixtures; preserve the narrow contract and version any saved format change.

#### Gated addition: an offline neighbor advisor, not a serving policy

If explicit frontier experiments demonstrate useful choices that are tedious to derive manually, extend [llama-fit-params](../tools/fit-params/fit-params.cpp#L18), rather than create another runtime or generic tuning service. Its current main path passes no extra draft measurement at line 36; that must be fixed for the new joint mode before advertising MTP-safe recommendations.

A proposed **tool-only** option, `--fit-moe-frontier <manifest.json>`, accepts a versioned local manifest and emits candidate configurations. This option does not exist at the reviewed HEAD. It must neither load tensor payloads nor execute inference, and it must not auto-apply a recommendation to a running server.

Manifest v1 contains:

- `schema_version`, baseline argument vector, exact model/tensor-layout fingerprints and selected device identities;
- explicit per-device usable peak envelopes, safety margins, host/pin limits, and all target/draft context/batch/KV settings;
- `mutable_controls`, drawn from `cache_mib`, `resident_layers`, and `layer_split`; all other controls are hard constraints. Existing explicit controls remain locked unless named here by the caller;
- optional measured profile entries keyed by owner, source path, graph/quant layout, role/phase, slot count, prompt/decode workload and build; bytes, critical-path costs and confidence/provenance are separate;
- `max_candidates`, default 16, bounding fully measured suggestions. No hidden recursive search or automatic benchmarking.

Algorithm v1 is a deterministic local-neighbor advisor, explicitly **not a global optimizer**:

1. Resolve and jointly measure the untouched baseline; always return its result.
2. Enumerate permitted one-move neighbors: promote/demote one complete eligible expert layer; adjust an owner cache envelope at a slot-capacity boundary; or move one layer across one adjacent contiguous owner boundary. Keep owner order and do not intersperse layers. Also retain an all-ordinary-resident boundary candidate if feasible.
3. Recompute the entire joint ledger for each changed placement. Keep per-owner slots uniform within each model. Saturate only within that owner's physical envelope, user cache cap, and route/host constraints.
4. Rank with verified profile costs where available; otherwise label candidates exploratory and use deterministic ordering, not an invented speedup. Filter locked/invalid moves before metadata probes; cap fully probed candidates by `max_candidates` and disclose truncation.
5. Emit resolved selectors, existing target/draft MiB flags, split parameters, predicted memory/slack, expected execution paths, and the reason each candidate exists. Also emit model/device/config fingerprints for reproducibility. Re-resolve the emitted arguments and require the intended ownership to match: tensor-split ratios can round to different layer boundaries.
6. Recommend a speed winner only after external matched validation has supplied evidence. Iterating around a measured winner is a new explicit invocation; the tool does not explore during model load.

Use integer bytes internally. When emitting integer MiB, round the *required* budget upward only if it remains under all caps, otherwise select the lower representable capacity and recompute its slots. Do not emit a command that exceeds a cap through rounding. An all-resident candidate removes cache controls for that model; it must not emit an empty layer selector that restores legacy all-layer caching.

The normal server load consumes only existing parameters. Applying a candidate is a user-approved restart/reload. A stale device/model/profile mapping is rejected or stripped of performance ranking, never silently mapped onto a different GPU.

### B7. Fit, target/draft/MTP, and lifetime semantics

**Separate the weight owner from the execution context.** For normal architectures, DEFAULT and separate DRAFT contexts execute their model's ordinary target groups; MTP executes that model's active nextn groups. Use the existing context-use mask and actual manifest, not a rule that every architecture's MTP is always a disjoint numeric layer range. The current `router_layer`/MTP-only cases in [finalization](../src/llama-model.cpp#L2519) matter.

For a shared target model with active MTP, owner slot derivation must aggregate all concurrently retained target and MTP resources, while each context breakdown reports only its own execution share. Sum the independent context workspaces/cache pools even if only one executes at a time. Shared model storage is charged once. This is the current Priority 1 direction at [lines 2636-2644](../src/llama-model.cpp#L2636), not a new staging-lane multiplier.

A separately loaded draft has independent cache selectors/MiB/slot controls and defaults to cache disabled. Its physical memory competes with the target even if its local device list or context role differs. A separate MTP file can share target weights; inspect real shared ownership and additional head tensors, not just the presence of `-md`. `common_speculative_init_result` assigns MTP versus DRAFT context and `model_shared` in [common/speculative.cpp](../common/speculative.cpp#L2975). Reuse that model/context conversion logic for fit; do not create a different approximation of drafting setup.

Required strict-fit corrections are narrow:

- A required extra model/context that fails measurement invalidates the candidate. Current [add_extra_memory](../common/fit.cpp#L225) warns and fits the target alone after such a failure; the new joint mode must not do that. Preserve any deliberately optional old caller behavior with an explicit requirement parameter/result, not an unannounced global policy change.
- Reuse extra-model measurements only while their full placement/context identity is unchanged. The current cache key is essentially `n_ctx` ([line 210](../common/fit.cpp#L210)); candidate split, devices, selectors, cache budgets, batch/KV settings, flags, or shared-model layout changes require a fresh measurement. Simplest initial implementation: no reuse across candidates.
- `shares_model` means the model bytes really were counted already ([common/fit.h](../common/fit.h#L14)). Do not set it merely because a draft borrows some weights; uniquely allocated head tensors still count. Match devices by backend physical identity and reject unmatched device mappings.
- `fit off` keeps explicit settings; it does not remove safety validation or permit hidden fallback. `fit on` can suggest a changed configuration only for controls the caller permitted to change. Never silently reduce context, parallel slots, draft length, KV precision or safety margin in an otherwise matched comparison.

Keep graph/source generations, immutable source publication, registered-source leases, capture/replay ownership and teardown rules unchanged. Placement changes occur at fresh model load, not by recycling stale group resources. Future drafting workloads must fit the same explicit context/resource ledger without hard-coded "two lanes" assumptions.

#### Step 3 fit-semantics contract

The proposed `measure_joint_configuration(request) -> result` is an internal synchronous measurement, **not another fit search**; names are illustrative. The request supplies immutable target and active extra-model/context descriptions, roles, sharing relationships, per-device envelopes/margins, host limits and a required/optional designation for each extra. Initially support the existing target plus active draft/MTP arrangement; do not build a speculative general resource scheduler. Deep-copy borrowed parameter arrays/strings before adapting probe parameters, hold shared probe objects for the needed call lifetime, and return owned results. No caller settings, live model, cache state or published generation may change; restore temporary logging hooks and retain existing fit thread-safety restrictions.

The minimal result separates **measurement completeness** (`complete`, `incomplete`, `error`) from **capacity assessment** (`within_limits`, `exceeds_limits`, `unknown`). It carries the configuration ID, per-device/role byte breakdown and slack, sharing deductions, excluded/failed extras, bound/provenance tags and diagnostic reasons. Use existing status/record types where they express this faithfully. A total without a requested component is not a successful joint total. A temporary no-allocation backend/probe object is permitted; inference and tensor-payload loading are not.

| Case | Required behavior |
| --- | --- |
| P2 report/candidate with drafting enabled | All enabled target/draft/MTP contexts are required by default. An unmeasurable required extra returns incomplete/error and disqualifies the joint recommendation; never retry as target-only success. |
| Explicitly optional extra | Preserve the target estimate and identify the missing component. A permissive caller may continue with a warning, but the original joint request remains incomplete. A deliberate target-only alternative needs its own resolved configuration/ID, not a relabeled result. |
| Partially shared MTP | Count the union of provably shared weight allocations once, unique target/head weights once each, and independent context/cache/staging/workspace resources separately. Construct probes with actual sharing where supported. A boolean `shares_model` cannot erase a partially shared head's entire model term. If alias ownership cannot be established, a labeled no-dedup upper bound can screen capacity, but precise joint/matched-memory qualification remains unresolved and triggers review; do not invent a sharing registry. |
| `-fit off` | Do not mutate parameters or turn a conservative estimate alone into a new runtime load rejection. The explicit report/advisor may warn, mark a candidate over-envelope/unqualified, and refuse to recommend it. Existing hard errors remain: invalid placement/overflow, a provably insufficient mandatory budget, unsupported required execution and actual allocation failure. An advisory free-VRAM snapshot or conservative workspace overestimate is not an additional hard loader contract. |
| Legacy callers | Keep `common_fit_params`' existing advisory extra-failure behavior for its unopted-in callers, including the current `common_init_result` path in [common/common.cpp](../common/common.cpp#L1303). Preserve the old target-only fit-tool/`common_fit_print` behavior and stdout unless the new joint-report mode is selected. They must not be described as strict joint certification. Migrating a server/CLI caller to strict rejection is an explicit later integration/review decision, not a silent default change in this refactor. |

Steps 3a/3b separate the pure measurement contract from tool/caller integration. Fixtures must establish shared, separate and partially shared cases plus failure behavior; do not memoize across candidate configurations initially. If `no_alloc` cannot describe a needed shared head or actual context geometry, return an honest incomplete result and review the narrow missing seam. The callback at the Step 3 gate is required before relying on the new result as experiment admission evidence; it is not a claim that real peak memory has already been qualified.

### B8. Telemetry and compatibility deltas only

Priority 1 already reports per-owner/per-group execution, domain counters, capture/replay, fill/reset reasons, resources, and prefetch. Reuse [owner telemetry](../ggml/src/ggml-cuda/moe-cache.cu#L5377) and [aggregate summaries](../ggml/src/ggml-cuda/moe-cache.cu#L14975); do not add another parallel statistics system.

Add only what a placement comparison lacks:

- One load-time resolved-placement/ledger report, including ordinary-resident groups and override reasons, selected-device mapping, target/draft attribution, estimated ordinary/cache/staging bytes and capacity slack.
- A configuration ID shared by fit output, runtime placement and benchmark artifacts. A saved profile uses stable semantic IDs, not `owner=%p` addresses.
- If current owner-level traffic cannot rank layers, an **opt-in** bounded per-semantic-group/per-phase byte accumulator or profile export using existing counters. Prove its attribution and cost before adding device counters; do not log or synchronize per token in production. An owner-only first experiment does not require this extension.
- Separate estimated/reserved/allocated/populated values and retained high-water resources. A source-path byte counter represents its defined useful movement, not necessarily PCIe transaction bytes. Verify the counter's update site before deriving bandwidth; inventory bytes are not traffic.

Snapshots/resets must not evict experts, alter prediction, invalidate graphs, or double-count capture plus replay. Keep retired generations visible until safely folded. Request-boundary resets are currently process-wide under concurrency ([feature guide](fork-features.md#L41)); use isolated runs or labeled window totals, not invented exact per-request ownership.

Migration is opt-in. Existing commands without a selector keep their behavior. If Step 1 confirms a defect, existing selector commands gain the stated override correction, not a changed cache policy; otherwise preserve their behavior and record the verified contract. Current CPU, CUDA, cache-off, dense, multimodal, LoRA, and speculative behavior must not silently change. Unsupported cache row/tensor sharding and RPC are not made supported by this plan; preserve precise existing rejection/eligibility boundaries. The cache-off ordinary paths remain available.

All existing speed options remain in scope where the underlying model/path supports them: flash attention, quantized/offloaded KV, backend sampling, batch/microbatch, phase/live-context workspaces, decode/boundary overlap, PLE/prefetch/early router, load/mmap/DIO/lazy modes, and host pin budgets. A concrete current counterexample is required-grouped pageable materialization with `GGML_CUDA_NO_PINNED`, which forbids needed staging ([feature guide](fork-features.md#L27)); preserve that explicit failure, not a blanket claim that all pageable modes are unsupported. Any newly discovered incompatibility needs a reproducer, exact scope, visible error and tracked remediation before excluding it.

## C. Reference alternatives: mechanisms, evidence, and decisions

"Verified" below means inspected primary code or identified documentation at the pinned revision. It does not mean independently benchmarked here. Forum performance statements remain reports; an author's README benchmark is not a matched result on our machines.

| Design/reference | Verified mechanism or evidence boundary | Adopt / test / defer / reject |
| --- | --- | --- |
| Current GenerelSchwerz, `d50dcda11a0db15635f825172b4591a6408e4a70` (P1 core unchanged from `edb542798`) | Grouped GPU execution with host refill, independent owner pools, typed source manifests/certificates, frequency/LRU policy, residual selectors, backend-derived owner byte capacity. [Current contract](fork-features.md). | **Retain** as execution baseline. **Test** uneven bytes, persistent layers, and contiguous owner-boundary moves. Do not call cache-off ordinary placement a pure kernel A/B when tensor placement changes. |
| FreeToken, `cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825` | Distinct fused-resident, GPU-refill/offload, CPU and hybrid strategies; hybrid partitions routes, starts CPU work, refills a GPU subset, and merges contributions. Engine budgets KV/state/weights and chooses hybrid fetch limits using calibrated bandwidth inputs. TP plumbing exists, with substantial format restrictions. See the detailed qualification below. | **Adopt** cost-aware placement reasoning, explicit execution choices, joint capacity discipline and whole-resident boundary comparisons. **Test later** whether a hybrid miss engine wins on our slow owner. **Defer** importing its CPU executor/streams/TP machinery; these are not Priority 2 changes. |
| Lelouch/leloch, `e3096b046bb809f7f80bc47801f6579aed1cbc60` | [Documented CUDA design](https://github.com/leloch/llama.cpp/blob/e3096b046bb809f7f80bc47801f6579aed1cbc60/docs/backend/CUDA-MOE-CACHE.md): GPU hits, CPU misses, asynchronous admission, placement authority and coordinated physical-device target/draft budgets. [Implementation](https://github.com/leloch/llama.cpp/blob/e3096b046bb809f7f80bc47801f6579aed1cbc60/ggml/src/ggml-cuda/moe-cache.cu#L1812) contains the hybrid path/worker coordination. | **Adopt** explicit placement authority and shared-device accounting as principles, not its allocator. **Test** its ordinary/full-resident boundary as a control. **Defer** CPU overlap, miss workers and fusion eligibility heuristics to a separately justified Priority 4. |
| TheTom/turboquant, `4deec5587b2963af00bdf80884f3337e02eb7d64` | [Docs](https://github.com/TheTom/llama-cpp-turboquant/blob/4deec5587b2963af00bdf80884f3337e02eb7d64/docs/backend/MOE-CACHE.md) describe GPU-hit/CPU-miss cache, explicit placement, whole-fit dormancy and target/draft budgets. [Fit code](https://github.com/TheTom/llama-cpp-turboquant/blob/4deec5587b2963af00bdf80884f3337e02eb7d64/common/fit.cpp#L1207) uses spare memory first and searches layer eviction/cache allocations; [cache code](https://github.com/TheTom/llama-cpp-turboquant/blob/4deec5587b2963af00bdf80884f3337e02eb7d64/ggml/src/ggml-cuda/moe-cache.cu#L2417) includes heat-aware eviction. | **Adopt/test** stock-resident-first and persistent-plus-residual alternatives at equal memory. **Reject as universal** its hardware-specific reserve/eligibility thresholds or CPU-miss preference. **Defer** replacement-policy changes and TurboQuant KV as unrelated mechanisms in a placement A/B. |
| buun, `08826ad6e64bf1d08b5f2954a62fdc4c5679e81c` | [Fused expert-parallel code](https://github.com/spiritbuun/buun-llama-cpp/blob/08826ad6e64bf1d08b5f2954a62fdc4c5679e81c/ggml/src/ggml-cuda/moe-cache.cu#L3382) discovers devices, dispatches expert rows to bounded workers, hashes layer/expert ownership, and orders multi-device locks. This is actual expert dispatch across devices, not just a multi-GPU label. | **Adopt** ownership/lock-order/lifetime lessons in future design reviews. **Defer** dispatch/merge execution to Priority 5. Its CPU-miss cache is also a separate engine. Do not infer a win without P2P from the presence of the code. |
| FoMoE, `ca31caffb0bacccb70800e72fc6b697ecbd0ca10` | [README](https://github.com/pmerolla/fomoe/blob/ca31caffb0bacccb70800e72fc6b697ecbd0ca10/README.md#L164) and [HIP code](https://github.com/pmerolla/fomoe/blob/ca31caffb0bacccb70800e72fc6b697ecbd0ca10/src/gpu_kernels.hip#L4206) use alternating layers on two GPUs with host-staged hidden-state transfer. The cited CAR throughput mode substitutes experts and reports a perplexity cost; disabling that mode is a different baseline. | **Adopt** tier-cost and bounded-capacity lessons. **Do not describe this cited path as true expert parallelism. Reject** expert substitution for transparent GGUF inference. **Defer** its NVMe/storage design and do not copy alternating placement onto our no-P2P machine. |
| Blobpager, `f80da77bc6ff77ce9f22a4cd375bd30bccdbe327` | [Prototype code](https://github.com/bdrazn/blobpager/blob/f80da77bc6ff77ce9f22a4cd375bd30bccdbe327/llama.cpp/src/llama-blobpager.cpp#L27) has fixed expert pins, slot maps, separate gate/up/down pools and host sources. Its [paper](https://github.com/bdrazn/blobpager/blob/f80da77bc6ff77ce9f22a4cd375bd30bccdbe327/blobpager/article/paper.md) distinguishes measured harness evidence from projected performance and incomplete clean end-to-end evidence. Narrow shapes and incomplete numerical validation limit transferability. | **Adopt** transparent prototype controls, static-placement comparisons and explicit numerical reporting. **Defer** individual expert pins to Priority 3. **Reject** treating projected throughput, narrow decode/MTP observations or callback surgery as a production-general solution. |
| DSpark plus static-resident-layer reports | DSpark is a drafting method extending DFlash, not a cache-placement engine ([our speculative docs](speculative.md#L81)). The [reported comparison](https://github.com/ggml-org/llama.cpp/discussions/24528#discussioncomment-18253201) contrasts leading resident layers, stock trailing fit and adaptive caching, and reports an extra-layer OOM with drafting. | **Test** leading/trailing/mixed persistent layers while jointly reserving draft resources. **Do not attribute** static placement to a new DSpark allocator or assume resident-layer order is universally best. DSpark implementation/qualification is not required to start this MTP-first work. |
| Fiddler | [Primary paper](https://arxiv.org/abs/2402.07033v3): CPU/GPU orchestration exploits asymmetric transfer and computation costs for MoE inference. Paper-level evidence, not an audited drop-in llama.cpp implementation here. | **Adopt** cost comparisons rather than hit-rate objectives. **Defer** the hybrid engine to Priority 4. Its reported speedups do not predict ours. |
| MoE-Infinity | [Primary paper](https://arxiv.org/abs/2401.14361v3): activation-aware tracing/prefetch/cache management for offloaded MoE serving. | **Adopt** workload-specific profiling and held-out validation lessons. **Defer** profile seeding and scheduling/prefetch changes to Priority 3; no claim that a model-global profile predicts every prompt. |
| Pre-gated MoE | [Primary paper](https://arxiv.org/abs/2308.12066v3): pre-gating co-design enables advance expert selection/loading. | **Defer** algorithm/model changes. **Reject** making altered gates or weights a requirement for general GGUF support. This is not evidence that speculative early routes can replace the real router. |
| LLM in a Flash | [Primary paper](https://arxiv.org/abs/2312.11514v3): flash/DRAM-aware loading, windowing and data-layout techniques. | **Adopt** the discipline of measuring the actual storage/memory tier and amortizing transfers. **Defer** layout/model-specific mechanisms; flash-to-DRAM gains cannot be transferred numerically to host-to-GPU traffic. |

### FreeToken: what is and is not transferable

At the pinned revision, [strategy documentation](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/docs/models.md#L37) and [MoE execution code](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/layers/moe.py#L261) support a concrete comparison:

- Offload and hybrid are distinct. In `_decode_hybrid`, CPU submission precedes refill/GPU work, the route partition prevents double contribution, and the results are merged. This requires an execution pipeline, not just changing our slot budget.
- [Engine sizing and placement](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/engine/engine.py#L589) include KV/state/weights and reserves; later code calibrates hybrid fetch count against CPU and PCIe bandwidth. CPU-locked layers can constrain prefill overlap. The important lesson is joint cost/capacity, not a universally correct CPU fraction.
- TP has actual reduction and shard plumbing: `MoELayer._maybe_all_reduce`, [TP-local intermediate sizing and common eligibility checks](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/layers/quantization/moe/base.py#L58), and [BF16 bank shapes](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/layers/quantization/moe/unquantized.py#L20). However, implementations in [NVFP4](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/layers/quantization/moe/nvfp4.py#L44), [MXFP4](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/layers/quantization/moe/mxfp4.py#L25), and [block FP8](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/layers/quantization/moe/fp8_block.py#L24) explicitly set `tp_ok=False`. Do not summarize this as universal multi-GPU support for every offload format.
- [Cross-rank capacity synchronization](https://github.com/FlashML-org/FreeToken/blob/cc1f5c2c91855f2cc7787ad6b909f7e46a5d5825/python/freetoken/engine/engine.py#L811) checks rank memory consistency and rejects sufficiently large differences. That is not our desired uneven per-owner layer-split policy.

This inspection does not prove FreeToken's complete MTP/quant/model matrix, arbitrary GGUF coverage, or performance on our asymmetric no-P2P pair. Its CPU executor and offload bank schemas have explicit format support; our implementation cannot silently narrow its own compatibility to match them. An open [WSL2 host-pinning report](https://github.com/FlashML-org/FreeToken/issues/443) is an issue report, not proof that either project's pinning is generally correct or broken.

### Discussion claims that should not become design axioms

- [Volunteer-1's newest comparisons](https://github.com/ggml-org/llama.cpp/discussions/24528#discussioncomment-18531512) are useful workload/placement reports, but include separately tuned or patched forks. They do not isolate a current-HEAD kernel regression.
- [Placement and transfer cost](https://github.com/ggml-org/llama.cpp/discussions/24528#discussioncomment-18258303) are directly relevant. Assertions that peer access makes transfer latency effectively zero are not; the remote has no usable P2P, and peer access would not erase bandwidth/synchronization costs anyway.
- Dedicated attention/expert GPUs may be worth a later experiment, but change graph ownership and activation transport. They are not an alternative spelling of `--tensor-split` in today's owner-local cache.
- New UMA/Vulkan/HIP streaming proposals in the discussion target a different memory system. They are adjacent references, not evidence that discrete-GPU bounded host staging should be replaced.

## D. Small commits and stop/go gates

No implementation commits are made by this design task. The proposed sequence deliberately allows stopping after explicit placement recipes demonstrate the useful frontier.

### D1. Proposed commit boundaries

| Step / proposed commit | Scope and dependencies | Stop/go evidence |
| --- | --- | --- |
| 1. Reproduce selector precedence, then fix only if confirmed | First add/run the focused B2 reproducer/controls as an isolated test change and record the unmodified result. A narrow production fix is conditional on reproduction, followed by the same controls. The verified test/fix may land together; preserve the pre-fix evidence rather than require a failing-test commit. No CUDA policy change. | Wholly overridden groups follow explicit placement; partial/invalid selections still fail; legacy behavior remains. If not reproduced, explain the preventing path and omit the fix. **Callback:** send the finding and fixture evidence before treating the placement contract as settled. |
| 2a. Collect resolved placement and canonical identity | Internal owned view of existing manifests/ledgers plus canonicalization fixtures; follow B6. Depends on Step 1's clarified contract. No new formatter or live observation collector yet. | Stable IDs for equivalent effective settings, distinct IDs for meaningful changes, explicit weak model identity, no borrowed lifetime or per-token work. Sums preserve existing accounting. |
| 2b. Add report surfaces | Human opt-in ledger and tool-only JSON serialization; depends on 2a. Reuse experimental logs and existing tool output infrastructure. | JSON/human views agree, defaults/stdout stay compatible, missing observations remain unknown, and no logging-induced synchronization/reset/dispatch change occurs. |
| 3a. Factor immutable joint measurement | `common/fit.*` and minimal reuse of drafting setup, with required/optional, complete/incomplete and sharing fixtures from B7. Depends on 2a; no caller default changes or cross-candidate memoization. | Frozen input remains unchanged; repeated metadata probes agree; separate/shared/partially shared ledgers and failed extras produce truthful results. Unresolved alias ownership cannot become a precise fit guarantee. |
| 3b. Integrate strict tool/candidate admission | Wire the new report mode and later candidate consumer to 3a; depends on 2b/3a. Preserve old advisory callers and `-fit off` runtime semantics. | Required extra failure disqualifies joint recommendations, optional omissions stay visible, output round-trips. **Callback at the Step 3 gate:** review fixture/accounting evidence before experiment admission relies on it. Real first-request peak qualification remains a Step 5/live gate, not a claim of metadata tests. |
| 4. Qualify persistent/residual boundaries | Extend existing MoE graph/ownership/replay fixtures and placement documentation. Use ordinary buffers and existing `place_moe_regions()`; make only narrowly demonstrated integration fixes, if any. Depends on 1-3. | Mixed ordinary/cached graphs, entirely ordinary boundary, nextn/target coexistence and no-P2P owners execute with expected paths and output. No new resident kernel or cache-full aliasing trick is acceptable in this commit. |
| 5. Collect a small explicit placement frontier | Reuse scripts and preserve recipes/results. Depends on 1-4; begin with D2's bounded low/half set, then apply relevant full-residency/local/held-out gates in E. No new execution mechanism. | **Callback after the first frontier**, including negative/infeasible points. Do not promote noise, borrowed-memory or hidden-fallback results; expand only to resolve a stated question. A useful manual recipe may complete P2 on that machine once the applicable acceptance gates pass, not the whole multi-GPU objective. |
| 6. Optional: bounded profile-assisted fit advisor | Tool-only manifest/parser and pure neighbor ranking, then a separate integration commit if needed. Depends on a demonstrated need at 5; follow B6 exactly. Reuse `llama-fit-params`, not a serving autotuner. | Synthetic determinism/constraints pass; emitted commands re-resolve identically; known measurements rank sensibly; missing costs remain labeled unknown. The advisor earns no performance claim until its recommendations survive live tests. Omit this entire step if recipes are sufficient. |
| 7. Publish qualified recommendations and limits | Document exact configurations, evidence, memory/correctness boundaries, and unresolved hardware/model coverage. No default flip. Depends on completed applicable gates. | No headline based only on cache-full, a short answer, profiler timing, or a startup health check. A platform-specific win stays platform-specific. |

Each code commit gets focused ordinary review before its dependent experiment; Astra review is concentrated at the callbacks below, not required for every naming/factoring change. Use existing implementation/test files rather than bundling cleanup. The 2a/2b and 3a/3b splits keep collection, presentation, measurement and caller policy independently reviewable; small adjacent changes may be combined if those boundaries remain clear.

### D1a. Step 4 local qualification (2026-09-21)

The existing placement, scheduler, owner-local cache, and grouped execution mechanisms handle persistent ordinary groups next to residual cached groups. No new allocator, kernel, routing rule, arithmetic path, or placement default was required.

The expanded fixtures cover ordinary/cache, cache/ordinary, entirely ordinary, entirely cached, and full-slot boundaries. They also execute target and MTP contexts with both mixed directions, all-cache, and all-ordinary placement. The real model loader and production graph builder produce finite outputs that exactly match the corresponding ordinary-device control in the synthetic GGUF fixtures. The grouped layer fixture compares every layer against an independent all-ordinary GPU oracle and checks that ordinary groups allocate no cache resources or traffic.

This work exposed a legacy route-publication defect rather than a grouped-refill defect. Fused top-k previously published route IDs for routers whose next MMID consumer used ordinary weights. An ordinary consumer left that publication unconsumed, and a later cached producer could reuse the same scratch address and observe stale IDs. The fix publishes only for the next active cached MMID using the same route tensor. Publication identity now includes the producer tensor, storage address, and stream; per-dispatch validation and an unconsumed-publication guard reject stale ownership. Retained mapped allocations remain bounded at 512 and stable until synchronized backend teardown. Rebuilt graph UIDs reuse the producer-owned slot instead of consuming the bound. If ownership is uncertain or the table is full, execution uses the existing ordered ID-copy path.

On the local RTX 5070 Ti, the complete focused MoE CTest set passed. The physical multi-GPU test returned its registered skip because only one CUDA device was present. Graph-enabled, graph-disabled, fusion-disabled, registry-lifetime, grouped-decode, and cached-fusion focused runs passed. Named production-like mixed and all-cache fixtures recorded zero route-ID synchronizations with fusion enabled; fusion-disabled controls used the expected copy path. The target/MTP fixtures proved placement and numerical equivalence but did not add a separate target/MTP synchronization counter assertion. Fault injection does not yet create an actual in-flight producer whose consumer aborts; the guarded recovery remains covered by code review rather than a dedicated fault fixture.

The physical fixture also passed on the remote RTX 4070 plus RTX 3060 host with peer access unavailable in both directions. It exercised mixed cached/ordinary placement in both directions, exact host-fallback boundary copies, eight grouped dispatches on the eligible owner, and zero legacy/fallback/error activity. The retained log is `build-p3/step4-evidence/multigpu-01.log` in the remote worktree.

A remote Ornith 1.5 35B MTP1 run then exercised a cached target layer beside 40 ordinary target layers and an ordinary MTP context. All 42 model layers were GPU-offloaded across both physical owners. The request produced a coherent 128-token completion; MTP accepted 42 of 84 drafts. The cached target layer completed 85 grouped decodes with zero legacy dispatches, fallbacks, prepare/finish errors, or route-ID synchronizations. The runner reported clean process, port, and GPU teardown. Artifacts are retained under `build-p3/step4-evidence/real-target-cache-mtp-ordinary` in the remote worktree. The observed throughput is not a performance claim because a standing audio workload shared the machine.

These results qualify the persistent/residual integration boundary on the tested CUDA systems. They do not substitute for Step 5's matched placement frontier.

### D2. First experiment manifest: ten adaptive configurations

Start on the remote asymmetric, no-P2P layer-split machine with the existing Ornith model and saved short-prompt + 1,024-output fixture. Freeze a compact experiment record before loading: exact build/model/draft/runner identities, prompt/tokenization/sampling, selected physical device order, resolved split, context/KV/batch/speed flags, source/pin mode, per-device usable byte envelopes/margins, host limits, and predeclared memory/noise tolerances. This is a runner/evidence manifest, **not** a requirement to implement the optional advisor or a new manifest framework first.

Define `L` as the nearest feasible low-residency point to the prior cache30 workload, and `H` as roughly half expert capacity (128 where the model has 256 experts). Resolve both using actual bytes/group geometry and route-capacity checks, not those labels alone. Use one predeclared, previously qualified MTP length/settings throughout the MTP subset (MTP1 is a reasonable initial choice if no better validated paired control exists). Construct separate joint-feasible `L`/`H` MTP baselines inside the same declared envelopes; never silently lower drafting for an individual candidate.

| ID | Band / drafting | Candidate and comparison |
| --- | --- | --- |
| L0 | Low / no MTP | Unchanged uniform-slot/placement baseline. |
| L1 | Low / no MTP | One unequal per-owner MiB tuple, chosen from exposed refill cost and capacity slack; compare with L0. |
| L2 | Low / no MTP | One small complete persistent layer/region on the costly owner plus residual cache; compare with L0. |
| H0 | Half / no MTP | Unchanged baseline at the half-capacity point. |
| H1 | Half / no MTP | A complete persistent/residual choice, using the best supported layer-cost hypothesis rather than assuming a prefix or suffix wins; compare with H0. |
| H2 | Half / no MTP | One layer moved across one contiguous owner boundary, direction chosen from transfer saving versus added compute/KV/boundary cost; compare with H0. |
| LM0 | Low / MTP | Joint-feasible baseline, with actual target and MTP execution. |
| LM1 | Low / MTP | Replay the more promising valid L1/L2 placement choice with joint accounting and matched LM0 limits; derive capacities afresh. |
| HM0 | Half / MTP | Joint-feasible half baseline. |
| HM1 | Half / MTP | Replay the more promising valid H1/H2 choice with joint accounting and matched HM0 limits. |

Selection rules:

- Choose a small promotion (initially one eligible layer); cache byte caps and residual slots must be recomputed for *all* affected groups. Prefer the owner/layer with credible exposed miss cost, not a high raw hit count. If layer attribution is unavailable, a leading/trailing eligible-layer pair is a labeled exploratory substitute, not an assumed optimum.
- Keep every physical device within its own envelope; do not exchange GPU0 headroom for GPU1 bytes. Quantize candidate MiB to actual slot boundaries and leave hard placement overrides locked. If a changed budget uses extra headroom and cannot match baseline peak memory within the predeclared tolerance, report a distinct capacity-frontier point, not a matched-memory win. No padding or post-hoc tolerance increase.
- For LM1/HM1, use the valid candidate with the best complete-request evidence after output/path checks, or the least-uncertain hypothesis if initial differences are noise and mark it exploratory. Selection on these prompts is not held-out validation. Record the chosen layers, budgets and comparison IDs before the MTP run.
- Owner traffic, per-group misses, source path, exposed timing and memory slack may narrow candidates: replace L2/H1 when the intended layer has little exposed refill, or reverse/drop H2 if added compute/boundary traffic defeats its premise. Deduplicate configurations; infeasible points are results. A second justified layer/budget probe may replace a weak candidate within roughly 6-10 configurations; do not force ten runs or expand to every combination. Any extra beyond that initial set needs a recorded question, cap and stop condition under D3.
- If low MTP cannot satisfy certified route capacity, record the boundary and choose the closest explicitly disclosed low feasible point for *both* LM arms, or return for review if it no longer tests low residency. Do not substitute no-MTP and call drafting covered.

Use pilots only for pruning; claimed gains still require E4's matched, repeated, unprofiled complete requests. Callback after this first frontier before broadening it. The next evidence gates replay the unchanged baseline and useful mechanism locally, plus full-cache/ordinary-resident and long-context/held-out controls where feasible. Do not multiply these ten choices across every hardware/model/flag axis; E is representative coverage, not a mandatory Cartesian product.

### D3. Exploration and callback protocol

This protocol guides later authorized implementation/testing; it authorizes no runs during this documentation task. Preserve forward progress within the agreed architecture:

| Category | Proceed locally | Boundary/reporting |
| --- | --- | --- |
| Implementation discretion | Names, file-local factoring, existing-fixture mechanics, serializers and narrow read-only diagnostics; small fields with established provenance. | Keep changes local/reversible, test their contract, and summarize them at the next gate. No Astra round trip for each edit. |
| Bounded exploration | A small metadata/allocation probe, one or two extra placement candidates, or temporary profiling instrumentation isolated from the production patch. | State hypothesis, resource/time or candidate cap and stop condition first; preserve artifacts and compare instrumentation on/off where relevant. No routing/arithmetic/lifetime change disguised as a probe. Safe failed hypotheses are useful results, not a reason to grow the subsystem. |
| Architectural callback before continuing that change | New execution engine; changed expert routing/arithmetic; shared/cross-owner allocator; per-group capacity ABI; runtime migration/autotuning; CPU-miss hybrid execution; TP, expert-parallel or RPC expansion; compatibility exclusions; default changes; or evidence contradicting a core assumption. | Return to this Astra design reviewer with the evidence and smallest alternatives. Do not implement the expansion pending review. Independent safe fixture/reporting work can continue. |

Required callbacks occur after **Step 1**, **Step 3**, and **the first Step 5 frontier**, or sooner when evidence invalidates an assumption. Send a compact packet: exact revision/configuration IDs, what changed, fixture/run artifacts and provenance, output/path correctness, byte/peak/request metrics where applicable, failed hypotheses, and the proposed next bounded step. Missing runtime evidence must be stated, not inferred from a metadata pass.

Callbacks are guidance gates, not invitations to delegate every task or rewrite the plan. While awaiting review, continue independent local/reversible work; do not promote a disputed fit result, broaden the benchmark campaign, change defaults or cross an architectural boundary on presumed approval. If evidence points to Priority 3/4/5, bring back that finding instead of silently absorbing those priorities into this series.

## E. Validation matrix and acceptance

### E1. Preserve the just-observed capacity boundary

Existing artifacts under [qwen36-gist-recheck-edb542798-20260920 (local/unpublished S3)](#local-and-unpublished-evidence) were inspected, not rerun. They use local Qwen3.6-35B-A3B Q4_K_M, a Q4_0 MTP file, a 64,000-token prompt, context 65,536, and **512 generated tokens**, not the separate 64K+1,024 acceptance fixture. The command records `b8192/ub8192`, F16 KV, `fit off`, `load-mode none`, `lazy-mode on`, host-pin budget 0 and existing overlap/workspace/sampling options.

| Current `edb542798` arm | Observed outcome | Additional boundary evidence |
| --- | --- | --- |
| Cache132, MTP2 | OOM on the actual request after startup; no completed output | Sampled peak 15,855 MiB; successful health/startup was insufficient. |
| Cache128, MTP2 | **164.22 tok/s** server-reported generation; **302/416** draft tokens accepted | Sampled peak 15,483 MiB; client TTFT 16.130 s, request wall 19.236 s. |
| Cache128, no MTP | **118.41 tok/s** server-reported generation | Sampled peak 14,319 MiB; client TTFT 15.059 s, request wall 19.378 s. |

The client summaries use a slightly different decode timing boundary (164.73 and 118.42 tok/s); do not substitute those silently for the requested server values. The two successful arms have different output hashes. Their per-arm `all_output_sha256_match` values do not certify cross-arm equality, and MTP/no-MTP token identity was not established by this artifact inspection.

**Interpretation:** this is a concrete capacity-boundary datum for concurrent target/MTP resources, and an example of why decode throughput alone is insufficient when long prefill dominates. It is neither a Priority 2 improvement nor proof that cache128 is universally safe, optimal, or regression-free. Do not deliberately repeat the cache132 OOM before a bounded/synthetic capacity check and adequate memory guard; lowering a test size must be disclosed rather than called the original workload.

### E2. Synthetic coverage before live loading

Extend the existing `tests/test-moe-cache-{registry,plans,graphs,staging,multigpu,fixtures}.cpp`, parser tests and no-allocation fixtures. Add pure planner tests only if the advisor is implemented. Dense tests are unaffected controls, never substitutes for routed MoE.

| Area | Required cases and assertion |
| --- | --- |
| Placement precedence | Legacy mode; selector with CPU override, explicit GPU override, `-ncmoe`, all-selected-overridden, gaps and duplicate/out-of-range/dense-only layers; partial banks rejected; whole-layer multiple/chunk groups stay complete. |
| Capacity arithmetic | Heterogeneous bank sizes/quants/strides/padding, auxiliaries and shadows, zero/empty owners, overflow, caps at exact slot boundaries and MiB rounding. Unequal physical budgets, broadcast/list ordering, reversed visible devices and logical three-owner cases. |
| Resource ownership | Same layer numbers across target/draft; shared source with distinct context pools; DEFAULT/DRAFT versus MTP groups; disjoint nextn and overlapping/MTP-only cases; shared model plus unique head storage. Independent device pools stay independent and aliased bank storage is counted once. |
| Source/staging lifetime | Positive budget below/at/above mandatory staging; source registration succeeds, partially succeeds or fails; pageability changes, mmap/allocated sources; all three retained host families; no late mandatory allocation starved by source admission. |
| Graph execution | All-resident, mixed resident/cache and all-cache; separate/fused/ungated banks, shared/chunk experts, scales/biases, supported activations and quant families; duplicate routes, route-order changes, empty/overflow and varying verification rows. All intended contributions preserved. |
| Replay/reset/teardown | Direct/capture/replay, counter snapshots and resets, graph-shape/source-generation changes, pointer/backing replacement, prefix reuse, rollback and teardown with in-flight work; repeat past 1,050 iterations. No stale ownership or resource disappearance after logging. |
| Strict fit/advisor | Required extra failure is fatal to the candidate; changed split/selector/KV/batch invalidates memoized measurements; profile mismatch/unknown cost/locked controls stay explicit; emitted settings round-trip; candidate cap and deterministic ties work. No tensor payload loads during planning. |
| Existing unsupported boundaries | Preserve explicit tensor-split/cache rejection and required-grouped failure for genuinely unsupported shapes/materialization. Ordinary resident groups do not count as hidden fallback; expected cached groups must not disappear. |

### E3. Live matrix: representative, not a dangerous Cartesian product

Use local single GPU and the remote physical two-GPU asymmetric/no-P2P system. Record the actual available GPUs, PCI identifiers, link state and competing processes at run time; old hardware reports are not a reservation. Coordinate exclusive test access, reuse the existing bench scripts and a persistent remote session, and avoid unnecessary rebuilds. This document authorizes no runs itself. D2 is the initial experiment set; this matrix supplies targeted follow-up coverage, not a Cartesian-product expansion or a demand to run unsafe models.

| Axis | Required representative runs |
| --- | --- |
| Hardware/placement | Local single GPU; remote normal layer split; remote unequal byte budgets; one permitted contiguous boundary move; cached regions on both owners and an owner with no residual group. No physical three-GPU claim from logical fixtures. |
| Residency | Low (roughly 1/8, or the existing cache30 point), half (e.g. 128/256 where appropriate), full cache capacity, and ordinary resident when safely feasible. Report actual bytes, expert counts and population; slot labels alone do not define comparable residency. |
| Drafting | No MTP; actual MTP1 and MTP2 execution/verification at representative low/half/full points; separate draft cache off and on; accept/reject/rollback and varying verification lengths. Inspect MAIN/DRAFT/MTP path evidence, not just presence of a draft argument. |
| Models/graphs | Ornith for continuity on the remote, local Qwen3.6 and Flash Next, plus an available conventional routed MoE and a distinct fused/auxiliary-heavy graph family. Inspect their actual manifests rather than assuming model names prove coverage. Synthetic tests cover families too large/unavailable for safe live use, with that gap stated. |
| Quantization | Live representatives from Q4_K/Q5 or equivalent standard GGUF family and the Flash Next IQ3_XXS workload; add a supported IQ/NVFP4 or other materially different layout where hardware/models permit. Synthetic separate/fused and auxiliary-heavy formats remain mandatory. Do not equate one quant's result with all GGUFs. |
| Workload | Short prompt + 1,024 output; established 64K + 1,024 output where capacity permits; multi-prompt/domain and longer decode; cold first request and warm reuse; a bounded parallel-serving case. The existing 64K+512 result is retained separately. |
| Memory/source modes | Zero and positive pin budgets; deliberately partial/pageable source coverage; mmap/load-none+lazy-on and supported direct/allocated loading. Match host budget and RSS/page faults/storage behavior, not only VRAM. |
| Speed flags | Preserve the fastest known valid configuration in headline A/Bs. Pairwise targeted tests cover FA/KV F16 and Q8_0/Q5_1, KV placement, backend sampling, batch/microbatch, phase/live workspaces, decode/boundary overlap, PLE and early router. Toggle a flag for diagnosis only with both arms matched and the change disclosed. |

Low-capacity MTP points must first pass graph/route-capacity feasibility. A configuration that cannot hold its required verification routes is not a valid grouped comparison. Do not quietly reduce its draft length or substitute no-MTP; explain the constraint and use the closest explicitly qualified point, retaining the unsupported boundary as a test.

For generality, support the existing architecture/flag contracts and require targeted tests for every changed graph family. This is not a promise that every untested model, speculative method, Vulkan/RPC configuration, or quantized tensor-parallel mode is already qualified.

### E4. Controls, metrics, and release gates

Use three distinct comparisons and name them correctly:

1. **Code regression control:** reviewed HEAD versus candidate code, identical placement and workload. Before enabling new policy, the unchanged grouped baseline must not slow down or change established deterministic output.
2. **Placement comparison:** current uniform slots/budgets versus uneven bytes/persistent-plus-residual/split candidates at matched physical memory envelopes. The ordinary-resident boundary is a separate path, not a hidden fallback.
3. **External control:** best ordinary same-fork placement at the same budget; pinned pristine upstream when making an upstream superiority claim and model/flags are comparable. Document kernel/model/speculative differences. Do not use a handicapped all-CPU baseline as the sole comparator.

Every run preserves exact revisions/build options, GGUF and draft hashes/quant mix, prompt/tokenizer/chat-template input, sampling/RNG settings, flags/env, per-owner resolved placement, slot capacities, sources/pin budget, cache state and artifacts. Reuse [bench.zsh (local/unpublished S4)](#local-and-unpublished-evidence) and the established remote runner after locating its current version; do not keep creating/deleting ad hoc runners.

Record complete request wall time, TTFT/prefill rate, target decode/accepted output rate, draft proposed/accepted counts, stalls/p50/p95 where relevant, CPU/RSS/pageable/pinned memory, **per-device peak** VRAM, expected versus actual dispatch, useful source traffic/prefetch and reset reasons. Include load/first-request costs separately. A counter for accepted draft tokens is not itself a full serving throughput metric.

Match each device's usable peak envelope and host limits, not just summed VRAM or equal nominal cache slots. Choose the closest realizable configurations without synthetic padding; disclose irreducible whole-layer/slot allocation granularity and provide a throughput-versus-memory frontier if tight matching is impossible. Predeclare the allowed peak-memory tolerance based on that granularity; a candidate above it is a different capacity point, not a matched win. Sample peaks through load, prefill, verification, decode and cleanup; a coarse sampled peak is a lower bound on the true instantaneous peak.

Correctness gates:

- Preserve token equality for existing deterministic same-path regressions and compare fixed-input logits/intermediates where a divergence begins. Require coherent outputs across multiple prompts and longer runs; one question is insufficient.
- For ordinary-versus-grouped or MTP-versus-no-MTP arithmetic/sampling differences, establish a numerical contract with ordinary controls *before* examining candidate failures. Do not call differing hashes exact, and do not loosen tolerance to excuse an unexplained dispatch error. Router IDs/weights/contribution completeness remain mandatory.
- Check grouped completion on every expected cached owner/role and all required-grouped errors. Cache-full must not silently hand off to another engine. Intentionally ordinary persistent groups should show ordinary execution and zero cache resource participation. Decode fallback/path-change counters must be explained, not simply summed into success.

Performance gates:

- Use at least three paired unprofiled repetitions in an alternating order, increasing repeats if variance masks the decision. Profile representative runs separately; instrumented timing is for attribution, not headline throughput.
- Require a gain larger than the measured paired noise; as an initial preregistered threshold, use the greater of 3% or twice the paired relative spread, and show the raw samples. A near-threshold result needs more repeats, not a confident claim.
- No reproducible regression beyond the noise boundary in unchanged baseline behavior, low/half/full residency, required drafting, or complete request latency for a recommendation advertised as generally better. A workload-specific tradeoff may ship only as an opt-in documented recipe, not a universal default.
- A full-resident-only win does not complete Priority 2. Need a defensible low/half-residency frontier improvement on at least the constrained topology, plus no unjustified regression on the local machine. If placement cannot deliver that, publish the negative result and move to the appropriate next priority rather than rewriting the engine inside this one.

## F. Non-goals and remaining empirical questions

Non-goals: shared arenas; cross-owner eviction; per-group variable slots/V3 manifest redesign; new CUDA compute kernels; CPU-miss hybrid execution; expert-parallel or dedicated-expert-device scheduling; new TP/RPC support; dynamic layer migration; hot-expert pins/profile seeding; changed routes/weights/quantization/model formats; driver/P2P modifications; silent draft/KV/context reductions; automatic changes to existing server defaults.

The following are intentionally unresolved by architecture and require evidence, not another generic plan:

- Does the selector-precedence suspicion reproduce through the real loader, or does an earlier transformation prevent it? Step 1 must settle that before a production fix.
- Can the existing no-allocation probe represent partially shared MTP head allocations and context geometry precisely enough for strict joint reporting, without a new ownership system? Steps 3a/3b must settle this seam or return an explicitly incomplete result for review.
- Which resident layers, owner budgets, and contiguous split best serve each real workload? The old slow-link profile identifies a likely cost, not the optimum.
- How much useful memory is genuinely stranded by uniform slots *within* an owner after persistent-region selection? Only a large measured loss justifies a future per-group-capacity proposal; it does not automatically justify an arena.
- Does bounded pinning change the winning placement through staging/host contention, and does long-context prefill erase a decode-only advantage?
- Can current fit estimates bound first-request and retained-generation peaks tightly enough near the observed MTP capacity edge? Until measured, keep explicit headroom and distinguish estimates from guarantees.
- Do short-profile rankings generalize to long context and held-out prompts? If not, retain explicit recipes rather than overfit an advisor.
- What remains unqualified on Windows, additional physical GPUs, unavailable GGUF families, or future drafting modes? Preserve compatibility intent and record gaps; do not manufacture hardware evidence.

**Start with Step 1's reproducer, the small 2a-3b increments, and explicit bounded placement comparisons. Retain the grouped GPU engine and use D3's evidence callbacks. Promote a convenience advisor only after placement evidence warrants it.** The design is ready for this incremental implementation, not a guarantee of a winning placement or completion of the long-term multi-GPU MoE objective. Adjust local mechanics and candidate choices through evidence; return for review before changing its architectural or acceptance boundaries.
