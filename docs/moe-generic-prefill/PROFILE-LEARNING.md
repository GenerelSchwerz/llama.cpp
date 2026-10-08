# Learned profiles: version3 metadata and owner integration

Scope: feature1 of [GOAL-1-4.md](GOAL-1-4.md). The complete feature is not qualified yet. The current source checkpoint is a69f285ddca71c9446354b687a6d8c44b0c5c9cd with release28d73c87cb90a9b78d4357ff1b456f456bfa1164 merged, plus uncommitted research changes. PROFILE-LEARNING-CODEC-QUALIFICATION.json freezes the exact source/library/test evidence.

## Metadata contract

The existing GGUF source-statistics format now accepts version3 learned snapshots alongside versions1/2 calibration and legacy STRP. The production serializer lives beside the existing parser in llama-context.cpp; the corpus/fixture helper calls it rather than owning a second writer. It stores complete source names/domains/types/shapes and full expert arrays, independently of current slots/selected layers. No whole-model hashing or specialized geometry is introduced.

| Field | Meaning |
| --- | --- |
| expert_counts / observations | Raw integer route occurrences, with checked sum; not decayed heat or confidence |
| learned_heat | Accumulated pre-decay usage at accepted adaptation maintenance boundaries |
| decayed_usage | Current usage, including the unfinished cadence tail, for resumed adaptation |
| prior_indices | Complete original per-expert tie order, distinct from the exported learned rank |
| learning_windows | Actual maintenance window count; preserve the next cadence boundary on reload |
| ranking_scores | Exact normalized ordinal scores derived from heat/prior rank for initial placement; these are not frequency estimates |
| learning_policy | Version1 identifies the existing occurrence cadence4/decay0.7 policy; reject unknown policies |
| provenance | Explicit observation/snapshot scope; no content-identity, confidence or quality claim inferred |

Version3 requires all learned arrays and valid types/lengths/permutations, finite nonnegative heat/usage, raw observation sums, usage bounded by raw per-expert counts, valid window totals, consistent aliases and ranking scores. Zero-observation sources retain complete prior information with zero raw counts/heat/usage and nonzero ordinal ranking scores. Explicit ranking policy is independent of observations; calibration versions1/2 still reject nonzero scores for unobserved sources. Calibration formats cannot silently carry learned-state fields. Learned file capacity is256MiB; calibration/STRP retain64MiB. All profiles retain the existing aggregate4M-expert/source-count bounds. Learning helpers now follow this generic statistics bound instead of the unrelated16-bit STRP geometry limit.

The shared restore helper validates all input before replacing private state. Snapshot can return the original prior as well as current full ranks, and rejects aliased prior/rank outputs. Restoring raw counts/heat/prior and applying the same next observation/heat accumulation reproduces the uninterrupted component exactly. Raw counts and learned heat remain separate policies.

## Qualified evidence

Serial ggml-base build passes under4096MiB hard RAM, zero swap,600s manager/570s subprocess, one compiler job and both ordered locks. Peak accounted RAM403.46MiB and5.10s elapsed; frozen base library in candidate-profile-learning-codec. No CUDA target or model run.

The bounded CPU codec job compiles the actual shared source-program.cpp, actual extracted production codec functions and actual repository profile tests, using real GGUF and existing common JSON test utility. No descriptor/backend mocks or replacement test framework. UBSan passes; the entire llama-context.cpp also passes a syntax check. The probe links only the frozen ggml-base and system CPU libraries, without CUDA libraries. Hard1024MiB/zero-swap/120s bounds,447.77MiB peak,17.23s elapsed and whole-tree teardown are verified.

Evidence includes204 valid windows,816 learning rejection/state checks, expert geometries1/3/17/257/65536/65537, exact metadata reserialization and resumed helper state, unobserved sources,25 malformed learned cases and114 total malformed/coverage cases. Existing weighted-corpus/statistics/alias tests pass. Preserve the first harness link failure; it omitted the existing JSON fixture implementation, and terminated cleanly. No speed, held-out quality, CUDA-owner, controller cadence, native Windows or multi-model serving claim follows.

## Required runtime integration

The private backend protocol now exposes learning snapshot/restore endpoints through the existing registry. They retain the canonical resource lifecycle and use tensor/domain identities. Export settles prior adaptation, rejects active transactions/registry replacement, and borrows complete count/heat/prior/usage/window views only through callback return. Known unprepared sources are explicitly omitted so context composition must supply their original full-model priors. Unknown, duplicate or ambiguous identities fail rather than being assigned to an arbitrary bank group.

Restore validates complete bank groups and matching histories before publication. It creates metadata in existing grouped resources with acquire_group_resources_impl; make_grouped_resource only builds descriptors and no device arena. All allocations/checks precede the first learned-state replacement. Failed validation may leave an unused metadata resource but changes no learned history or residency. Restore rejects a resource that has device preparation or prior learning, so repeated preparations cannot overwrite newer observations. The seed used to detect configuration changes remains separate from the original tie order and projected intent. The lifecycle lock is acquired without waiting; active boundaries defer, and expired deadlines reject before callback/state publication. Caller identities must retain their model lifetime throughout the call; context composition must hold its existing source-owner lease.

The context now invokes cold restore after scheduler reservation and inherited configuration, before execution. Per-device records follow existing cached-buffer ownership. Auxiliary-context inheritance preserves heat, usage, original prior and window fields rather than dropping them. This path has full-context syntax and shared CPU checks; it has not been linked/executed with the current CUDA owner or real models. Do not call this complete reload or claim file compatibility satisfies the goal.

containment-profile-learning-owner-01 passes actual CUDA object builds for moe-cache.cu and ggml-cuda.cu, actual production codec/shared helpers/tests under UBSan, full-context syntax and C11 header compatibility. Peak962.43MiB,40.44s,4096MiB hard RAM, zero swap,600s manager timeout, serial compilers, both ordered locks and verified whole-tree teardown. The later context hook passes containment-profile-learning-context-01 CPU checks and full-context syntax:446.65MiB,17.34s,1024MiB hard RAM/zero swap/120s, both locks and whole-tree teardown. There are15 added ABI/view rejections alongside the previous204 windows/816 state checks/25 learned malformed cases/114 codec malformed cases. No CUDA execution occurred. The first CUDA compilation predates the context hook; source hashes delimit these scopes.

## Full-model composition and publication

The shared full-model baseline builds complete ranks from descriptor-bound learned history, calibration ranking or a checked legacy seed, and includes unseen experts without invented occurrences. Existing version3 history is retained exactly; offline calibration counts are not relabelled online observations. Alias/source geometry and complete coverage are checked by the production codec. Runtime capacity is not an input. The adaptation-off projection now uses explicit ranking scores even when observations are zero; CPU tests preserve the original prior at all eight capacities of the3/5-expert fixtures.

Context snapshot uses the existing publication/caller mutexes and model source-owner lease. Active calls, pending refresh, poisoned contexts or busy owner boundaries defer. Per-device snapshots follow cached-buffer ownership and copy borrowed state before callback return. Unprepared sources retain their complete baseline prior/history. A shared bounded deadline covers snapshot preparation and owner requests; it does not include later file I/O. No live adaptation or residency change occurs during export. The staging callback in llama-ext.h exposes the validated bytes after context/owner locks are released.

common_moe_profile_save bridges that callback to common_moe_profile_write. Publication reuses fs_write_atomic after acquiring a nonblocking per-destination OS lock. The .lock sidecar remains present, so writers do not split across unlinked lock inodes; OS ownership is released on process death. Successful publication gives a complete replacement and failure preserves the previous file. This is atomic visibility, not fsync or power-loss durability. Native Windows locking/replacement remains unqualified. Opt-in production application hooks are implemented and the complete CPU-only libraries/tools plus linked argument/metadata tests pass; no HTTP file-writing endpoint is introduced.

PROFILE-LEARNING-SNAPSHOT-QUALIFICATION.json records three bounded jobs. Actual production codec/baseline/shared helpers and repository tests pass under UBSan, including eight invalid baseline cases, aliases, preserved learned history and a65537-expert Q5_K descriptor. That descriptor has no weight payload and is not model execution. The Linux publisher probe checks exact GGUF replacement, four bad inputs, externally held locks, release after killing only an owned lock-holder child, temporary-write failure preserving the old file, and12 two-writer rounds:22 complete publications and2 safe deferrals. Whole context/common/test sources pass syntax; C11 backend header and actual moe-cache.cu/ggml-cuda.cu objects compile. The final owner build includes the latest header comment; the earlier publication job predates only that comment. Source hashes preserve these scopes.

| Job | Hard RAM | Peak accounted RAM | Elapsed | Scope |
| --- | --- | --- | --- | --- |
| containment-profile-learning-snapshot-01 |1024MiB |446.48MiB |17.79s | Production composition/projection and codec tests, context syntax |
| containment-profile-learning-publication-01 |4096MiB |490.29MiB |20.12s | Linux file publication, codec tests, context/common/test syntax |
| containment-profile-learning-snapshot-owner-01 |4096MiB |923.31MiB |40.23s | Actual CUDA object compilation, C11 header and CPU tests/context syntax |

All jobs verify zero swap, finite timeouts, both ordered locks and whole-process-tree teardown. No CUDA execution or model run occurred. The repository now has a source-learning model exercise that saves, cold-reloads and checks an exact immediate snapshot; it passes syntax but has not been linked or run. Current whole-library linkage, live owner/context reload, next-window controller cadence, multiple runtime capacities, auxiliary/MTP contexts, native platforms/concurrency, matched model performance and independent held-out Strata quality remain required. CPU helper resumption and filesystem correctness do not close those gates.

## Application save boundaries

The new --moe-profile-save FILE option is limited to server/completion and disabled by default. It requires --moe-hybrid on and occurrence/occurrence-sync with an explicit input profile. --moe-profile-save-interval SECONDS selects a minimum time between boundary attempts in[0,86400], default60;0 requests normal shutdown only. It is not a background timer. A busy/deadline/file-lock failure defers until another eligible boundary and preserves the old file. The input calibration file remains distinct from online learned history unless the user deliberately chooses the same destination.

The server attempts periodic saving only when every slot is idle and forces an attempt before releasing speculative contexts/model storage during unload or normal shutdown. Completion attempts between generation iterations and before backend shutdown. Periodic snapshots have a100ms snapshot deadline; forced snapshots use5000ms. File publication starts after context/owner locks release and is outside that deadline. No file work runs in a signal handler; hard termination is not a normal shutdown or a saving guarantee.

MTP shares the target model/source catalog, but context construction creates distinct backend instances and learning procedures address that backend's grouped owner. The server now passes a same-model auxiliary context into a generic context-group snapshot before auxiliary teardown; it does not merge a separately loaded draft model into the target file. Live MTP composition/cold-reload still needs qualification. A separate draft model must not inherit the target save destination: common_base_params_to_speculative clears it. This does not qualify MTP snapshot correctness or automatically save an independent draft; explicit model-bound snapshot APIs remain available. Live MTP and multiple-context snapshot/lifecycle tests remain required.

Example research configuration, not a currently qualified model run:

```sh
llama-server -m MODEL.gguf --moe-hybrid on --moe-expert-cache-size SLOTS \
  --moe-expert-profile calibration.gguf --moe-profile-adapt occurrence \
  --moe-profile-save learned.gguf --moe-profile-save-interval 60
```

Restart with --moe-expert-profile learned.gguf to load the learned arrays/cadence. Do not claim continued runtime equivalence until the actual owner/cold-context and next-window checks pass. The save hook introduces no kernel fusion, graph support rule or extra residency authority.

## Context-group snapshot boundary

The source inventory exposed the target-only export gap: constructor backend initialization creates separate instances, owner snapshot procedures resolve backend->context->moe_grouped_context, and auxiliary inheritance copies configuration rather than the parent's evolving online history. Source evidence is recorded in PROFILE-LEARNING-APPLICATION-QUALIFICATION.json. A full model catalog alone cannot prove live MTP records were captured.

llama_moe_profile_snapshot_contexts accepts up to64 caller-retained contexts sharing the same model object. It sorts context pointers for stable nonblocking acquisition of the existing publication/caller locks and retains one existing model source-owner lease. All listed contexts must be quiescent and healthy. The primary context supplies the complete descriptor-bound prior/history baseline; each actual backend supplies prepared owner records. Cold metadata resources without a device allocation are omitted, preserving their baseline rather than letting a stale copy overwrite an active owner's observations. No context-type/model-name switch selects contributors.

Disjoint tensor/domain records overlay the baseline. Identical repeated records are accepted once, without summing history twice. Conflicting count/heat/usage/prior/window histories for the same source reject the entire unpublished snapshot; they are not averaged into an invented cadence. This is an explicit remaining design/runtime gate if independent live controllers update the same source. A future complete treatment must retain enough per-controller history or establish shared authority using the existing ownership contract. It must not silently choose a larger counter or last writer. Callers must list every context whose history they intend to include; this is not a global scan of unrelated contexts.

The existing single-context API delegates to this boundary. The common save bridge catches allocation failures and the server supplies its same-model auxiliary context at idle/shutdown. File I/O remains outside all context/owner locks. Unlisted contexts and independent draft models are not automatically aggregated.

CPU/UBSan checks qualify the actual overlay helper with disjoint heterogeneous sources/domains, exact codec output, identical-owner deduplication and seven conflict/identity/coverage cases preserving the output. Full-context syntax passes. The complete linked CPU tests exercise11 staging C ABI null/count/deadline rejections with zero callbacks and verify no CUDA libraries in the actual probe's maps. This is not live MTP, shared-context race, multi-GPU, native Windows, performance or held-out quality evidence. Actual owner save/restore/next-window tests remain open.

## Current CUDA linkage qualification

The current complete CUDA backend, llama library and test-moe-cache link pass in containment-profile-learning-owner-cuda-build-01. Compilation is serial, source hashes stay unchanged and the resulting test/library set is frozen as candidate-profile-learning-owner-cuda. Verified limits are4096MiB RAM, zero swap and600s with both ordered locks; peak accounted RAM is1237.90MiB and elapsed time83.52s. The process tree is gone. PROFILE-LEARNING-CUDA-LINK-QUALIFICATION.json preserves the exact source/binary hashes and scope.

containment-profile-learning-owner-transition-preflight-01 passes candidate integrity and containment checks without loading the device. Its verify-only path returns before the scoped GPU command and lock checks, so it does not qualify a device launch. No GPU/model execution or live learning restore occurred. Actual owner continuation and model/context-group gates remain open; the original model-server block and unproven incident root remain unchanged.

## Live synchronous owner and continuation

PROFILE-LEARNING-LIVE-OWNER-QUALIFICATION.json records the actual registered backend snapshot/restore endpoints on the existing tiny source fixture. Full-domain counts, heat, usage, original prior and windows match independent strictly ordered router inputs after windows1..4. Cold restore accepts complete banks at3/10 slots with no device arena, rejects incomplete/repeated install and rejects snapshots during a held transaction. Unprepared cold records are omitted from live export. Actual graph/source preparation, transport copies and controller updates then preserve3->4 decay and4->5 usage-tail continuation using the restored learned seed. Sixteen restore/resume cases pass across separate/fused layouts and direct/staged transport on one CUDA device.

All production library hashes match the qualified complete CUDA-link candidate; only test-moe-cache changes. Eight subsequent128-row prefill transitions retain exact ordinary-reference output and the unchanged2e-5 bound. The final contained job peaks465.90MiB accounted RAM and284MiB sampled GPU use, takes12.17s, holds both ordered locks and removes its entire tree. No interval kernel Xid/OOM marker appears. Failed oracle trials and their source diagnoses remain preserved and excluded. Actual snapshot/state bounds were retained; reading a route intermediate after its allocated lifetime was replaced with an independent expected-ID oracle from the original test inputs.

This is actual owner-controller evidence, not a full llama_context file reload or model run. Async pending adaptation/deadlines, live MTP/context-group history, multiple independent overlapping controllers, physical multi-GPU, native Windows, matched model performance and independent held-out Strata quality remain open. The original model-server block and unproven crash root remain unchanged.
