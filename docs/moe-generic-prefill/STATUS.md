# Status

## Shared scheduler copy identity: single-copy device qualification passes

Prepared-region identity/session lookup now includes original source/split graph UIDs, allocator generation and current scheduler copy. Identity rejects before tensor witness reads; the existing canonical owner, arithmetic, public ABI and single-copy/callback admission guards remain. SCHEDULER-COPY-IDENTITY-QUALIFICATION.json records384 extracted production binding cases,360 stale-before-witness checks,496 session lookups and4 preserved single-copy lookups under an undefined-behavior sanitizer. The actual scheduler passes28 copied-input graph submissions inside each35-test frozen baseline/candidate allocator suite, including two copy rotations, preserved inactive bytes, stable allocation generation, new graph UIDs, synchronization returning to copy0 and teardown. Dummy backends move real bytes but do not qualify matrix arithmetic.

Serial CPU library build and all CPU probes finish within verified hard budgets/zero swap/finite lifetimes with empty process trees. candidate-scheduler-copy-binding retains the CPU-only library. The later candidate-scheduler-copy-device separately freezes the complete tiny test/library set and passes12 single-copy source cases/24 production preparations with the same source patch. All12 use source-core, with minimum CPU120/resident24/selected96 routes, no source/drain failure, peak352.13MiB RAM and22.99s elapsed under3072MiB/zero-swap/120s limits; exact unit/PID teardown and no interval kernel Xid/OOM are verified. SCHEDULER-COPY-IDENTITY-DEVICE-QUALIFICATION.json records that narrower device scope.

Shared asynchronous reader/submission/CPU-job contracts, physical platform/serving/performance and held-out profile gates remain open. Full model-server/original Flash replay is still blocked; this change is not a root-cause or crash-resolution claim. The new source audit/probe in SHARED-ASYNC-CPU-CONTRACT-BOUNDARIES.json confirms that the shared CPU cancel watermark affects older epochs, lane admission returns capacity and the execution mutex still serializes arithmetic.240 extracted cancellation/abort cases pass; cancelling a later hypothetical independent job also cancels the older job in120 cases. This is a counterexample to lifting existing serialization guards, not an observed bug in the guarded runtime. Next extend the same service's request-scoped control/diagnostic and bounded reader contracts.

## Matched component comparison: do not promote pairing

PAIRED-PREFILL-COMPONENT-ABBA.json records four completed off/on/on/off arms,12 shapes each, against the same frozen binary/libraries. All numerical/state gates pass at unchanged2e-5 tolerance; max reference MSE1.27e-14, repeat variation4.08e-15. All12 final output hashes match across arms. Actual resident/staging/compute bytes, transferred bytes and9 waves match per shape. Bounded operations fall3 ->2, but this alone is not a saving.

Observed latency increases: pinned Q4_0 +16.13%, pinned Q5_K +19.44%, pinned IQ3_XXS +22.70%, pageable Q5_K +34.79% and pageable IQ3_XXS +25.63%. Pageable Q4_0 is effectively flat (+0.07%). Pairing stays disabled by default; these regressions are not accepted. This is component evidence, not serving or untouched-release default-path qualification. Each arm takes27.1-27.3s, peaks371-377million bytes under3072MiB/zero-swap/120s limits, holds both locks and removes its entire tree; no interval kernel Xid/OOM appears. No own workload remains live.

candidate-paired-benchmark-numerical contains test-only additions; all15 library files/links match the92-case memchecked candidate. The first attempted benchmark stopped on an overly strict bitwise-repeat assertion in IQ3_XXS after an independent reference check passed. It and the initial logging-contaminated arms remain preserved and excluded. Corrected timing restores inputs, validates every warmup/timed result outside its timing interval, and suppresses debug output. PAIRED-PREFILL-COMPONENT-ABBA.md records all six group means and qualifications.

Next: source-backed paired/single launch-geometry and event-cost diagnosis before any new optimization or promotion. Original Flash replay/root and broader profile/platform/concurrency/no-regression gates remain open.

## Qualified scheduler reuse component

PAIRED-PREFILL-SCHEDULER-QUALIFICATION.json records92 passing contained checks:32 arithmetic,12 canonical owner and48 graph cases, including24 actual scheduler-allocation cases. The scheduler calls backend graph_optimize and exercises allocation dependencies with CUDA first and CPU last. All outputs match independent ordinary CUDA exactly. Twelve eligible scheduler cases use2 bounded operations; twelve exposed-intermediate cases decline the optional pair and use3. Registered weight buffers/data and original routes remain intact. Compute allocation is1720320bytes versus2244608bytes with separate tensor allocations, confirming actual reuse.

candidate-paired-scheduler separately pins source patch6d2da2909645002218842703c6ef40adec1f19a9feeb9264f58cd28424433a03. Its serial build passed. The device fixture took19.49s with343715840bytes peak,3072MiB hard RAM, zero swap,120s limit, both ordered locks and complete tree/PID teardown. No memory-limit/OOM event or interval kernel Xid/OOM appeared. Sentinel checks remain confined to separate allocations because reused intermediates can hold later results. CUDA memcheck subsequently passes all92 cases with zero reported errors (PAIRED-PREFILL-MEMCHECK-QUALIFICATION.json):20.68s,527593472bytes peak, the same3072MiB/zero-swap/120s limits and complete teardown. Matched performance remains pending. No model server ran; pairing remains opt-in and the original incident root remains unproven.

## Qualified optional graph component

The current source adds opt-in cached paired graph selection through the existing graph matcher and allocation dependencies, with canonical physical source/transaction/range checks. Original-ID preparation and final-down retirement are retained. PAIRED-PREFILL-GRAPH-QUALIFICATION.json records68 passing contained device checks:32 paired arithmetic,12 owner and24 graph selection/decline cases. Eligible graphs elide gate/up outputs and use2 bounded operations; exposed intermediates retain original computation and use3. All final outputs match ordinary CUDA exactly. New output/input alias and substituted bank-pointer checks reject before paired arithmetic.

candidate-paired-graph pins the successful source/build; earlier candidates remain separate. Peak343212032bytes (327.31MiB),9.31s,3072MiB hard RAM/zero swap/120s, no limit/OOM events or interval kernel Xid/OOM, tree/PID gone. A prior compile caught and corrected constness in a test saved pointer; CUDA objects compiled but that whole target failed and no device run occurred. Default pairing stays off pending matched performance.

These earlier68 graph checks use separately allocated context tensors; the newer92-case qualification above additionally exercises actual scheduler reuse. No model server or matched speed claim follows; original Flash replay and full platform/concurrency/profile-quality/no-regression gates remain open.

## Qualified small owner extension

The existing bounded prefill method now stages both paired banks within its existing two-lane arena and validates both canonical bindings. Backend-derived alignment avoids a second-bank address error from quantized tail padding. Its extracted byte-capacity calculation passes6921 CPU/sanitizer cases. Serial guarded CUDA builds pass; candidate-paired-owner-certified separately freezes the latest source. The first256MiB syntax attempt hit the hard limit and timed out with complete teardown; the1024MiB attempt exposed and then corrected two test fixture field names.

CANONICAL-PAIRED-PREFILL-QUALIFICATION.json records32 rechecked arithmetic cases plus12 actual canonical-owner cases: Q4_0/Q5_K/IQ3_XXS, pinned/pageable sources and two routing distributions. NaN output sentinels prevent old warmup output from passing. Stale generation and aliased bank slots reject; both resident pointers and resource generation persist, the transaction remains active after GLU, and original down retires it. All outputs match independent ordinary CUDA exactly; owner cases use9 combined pair/down waves. Peak342212608bytes, elapsed8.16s,3072MiB hard RAM/zero swap/120s, whole tree removed, no memory-limit/OOM events or interval kernel Xid/OOM. A held copy verifies pending stream/ownership at entry, not all readers queued before release.

The first owner attempt passed32 arithmetic cases then failed before ownership qualification: its uncertified warmup used PREFILL_STAGED and did not create a bounded arena. The corrected fixture certifies coverage before warmup as the existing bounded test does. The failed candidate, scripts, logs, source diagnosis and complete teardown remain preserved; no assertion or product arithmetic was weakened. Serving graph selection, full expert/down waves, cross-layer scheduling and matched no-regression performance remain open. The original Flash replay remains blocked and its root cause unproven.

## Current recovery and qualification

The small retained-owner transition now passes on the corrected frozen test-only candidate. PROFILE-TRANSITION-QUALIFICATION.json records24 ordinary-prefill requests: eight after static placement, eight after synchronous adaptation and eight after held asynchronous copies. Each retains the same owner/generation/group keys/all bank allocations, exercises multiple bounded staging waves and matches ordinary CUDA outputs exactly. Preparation/completion errors are zero. The async fixture also verifies pending copies remain misses, later windows execute and close waits for publication.

| Mode | Miss transfer fraction | Requests | Accounted RAM peak | Elapsed |
| --- | --- | --- | --- | --- |
| Static profile |0.5 |8 |452980736bytes |10.41s |
| Synchronous adaptation |0.5 |8 |453197824bytes |10.51s |
| Asynchronous adaptation |0.25 |8 |443551744bytes |10.71s |

Every job verified3072MiB hard memory, zero swap,120s runtime, both ordered locks and whole-tree removal. No limit/OOM event or kernel Xid appeared in the interval. Two async setup errors are preserved:0.5 violates the fixture's pending CPU-route check;0 violates its seed-transfer check. The valid range is derived from both source assertions, and0.25 satisfies them. No executor/assertion change was needed. A preceding lock-contention attempt launched no CUDA work.

The source-proven ready-age mismatch is corrected at both publication writers and now device-qualified for this tiny transition. The before counterexample is a CPU proof of the frozen predicate; no faulty frozen GPU replay was run. This is not a model stability, serving-speed, performance-regression or platform qualification. The exact first CUDA fault in frozen Flash PID2753309 and the full global-OOM chain remain unproven. Original evidence is preserved. GPU-RUNS-BLOCKED.json still blocks model servers/original workload replay and permits only scoped corrected tiny transition and paired-arithmetic fixtures under the verified limits. No own workload remains live.

Harness containment passes actual CONSTRAINT_MEMCG OOM/group kill, fatal-log handling, timeout, detached-descendant and parent-death tests. Fatal evidence write failures still attempt exact-unit teardown. GGML_NO_BACKTRACE=1 and zero core limits avoid the source-supported fork/backtrace memory amplifier; the higher-level original fork caller remains unproven. CRASH-RECOVERY.md records the kernel/libc evidence and the external crash-helper limitation.

The longer held-out profile check remains a quality counterexample: at5364121600 resident bytes, shipped Strata hits25.20%, legacy generated24.58% and pooled22.12%; pooled misses4.31% more bytes. Equal-byte16/32/96-slot checks also favor Strata. The earlier favorable short trace does not qualify the tiny training corpus for promotion. See PROFILE-QUALITY.md. Profile-quality/no-regression/platform/concurrency acceptance gates and the specialized prefill gap remain open. The full model-independent goal is active, with no default/handoff/release promotion.

2026-10-07. Base origin/moe-cache: 081cf1d792596a2cac5aa06090751d0850aadaab. Own clean-base worktree; both older dirty research trees remain unchanged. No commit, default installation, push or PR.

## Current checkpoint after crash recovery

Full model-server runs and the original frozen Flash replay remain blocked. Hard RAM/zero-swap/runtime/group-teardown controls pass, including an actual contained cgroup OOM. The ready-age correction passes24 retained-owner transitions; it does not establish the original Xid/global OOM root.

The optional paired optimization passes92 component checks and CUDA memcheck, but the matched component ABBA rejects performance promotion. It remains disabled by default.

Shared pipeline qualification now includes12 original projection-discovery cases,12 routed-scheduler cases and24 sigmoid/bias/scaled-mixture partitions. Another12 discovery cases execute original SCALE/TANH stages outside the optional combined-body matcher. All12 report provider=source-core, with nonzero CPU/resident/selected-transfer routes and zero source/drain failures. No model identity or product admission rule was added. SHARED-PIPELINE-QUALIFICATION.json and SHARED-NEW-OP-QUALIFICATION.json retain the exact frozen sources, counters and containment.

The initial added-stage fixture failed because it mutated dependencies after graph expansion. Its ordered-root graph rebuild and complete placement metadata fix are test-only; the failed candidate and SOURCE-DIAGNOSIS.json are preserved. The combined-hint decline -> routed-projection gate also passes12 cases, with24 optional-body rejections, unchanged canonical residency at rejection and actual source-core execution. SHARED-NEW-OP-FALLBACK-QUALIFICATION.json records this narrower contract qualification; the full context orchestrator remains untested here. Windows, physical multi-GPU, representative held-out profile quality and matched serving no-regression remain open.

## Production preparation refactor

Combined-body decline and routed projection preparation now share the actual graph-region method between context and fixture. The first refactored candidate passes12 tiny cases/24 production preparations. A CPU differential against frozen control flow passes124416 scenarios after correcting temporary metadata lifetime order; the final corrected device rerun passes12 cases/24 production preparations, with nonzero source-core CPU/resident/selected-transfer work, no source/drain failures, peak352.93MiB RAM, clean teardown and no interval Xid/OOM. SHARED-REGION-LIFETIME-DEVICE-QUALIFICATION.json records the exact source and frozen build. No GGML kernel/arithmetic or context-admission guard changed. Follow SHARED-CONTEXT-BOUNDARIES.json for the remaining multi-copy/callback/row-semantic restrictions. Full frontend/serving/platform/no-regression/profile gates remain open.

## Implemented and qualified through build 09

The canonical two-lane bounded prefill owner now also prepares generic matrix inputs/routes once and uses existing CUDA matrix arithmetic on resident/staged views. Existing prepared MMQ remains first choice. Scalar/FP4/IQ1_M formats, non-vector-aligned expert slabs, biases and per-expert scales retain ordinary graph semantics. Generic capability admission is restricted to ordinary sequential prefill.

- 52 bounded cases pass at the unchanged tolerance across 13 geometries/source combinations, two routing variants, pinned and pageable sources.
- CUDA memcheck: zero errors.
- Grouped multirow/decode/speculation/stream-coherence regression passes.
- Seven real models complete all four matched arms, each with two 1024-input/16-output requests and identical generated text: LFM, DeepSeek Coder, Qwen, Nemotron, Ornith, Flash Next and GPT-OSS.
- GPT-OSS prefill: 3053 ->3439 tok/s, +12.6% observed. Other overall means range from -3.3% to +2.2%; these small differences need cold/warm interpretation and do not prove no regressions.

## Flash and the specialized performance requirement

The owner correctly challenged a smaller-ubatch comparison. Generic build 09 at batch/ubatch 2048, 8194 inputs: 1610 cold /1664 warm tok/s. This does not establish parity with another engine's different prompt.

With the same saved 2049 input IDs, batch/ubatch 2048: generic 1011 cold /1331 warm; the current specialized port 1716 with fused prefill disabled and 2122 with fusion enabled. These are bounded observations with different initial residency/profiling and decode recipes. Specialized on/off changes only the fused flag, but placement/output work must still be checked. The real gap remains; the current generic extension is not a complete fast Strata prefill port.

Build 10 is testing a source-backed scheduling correction: use actual routed row counts per expert wave for MMQ launch grid/tile sizing. Strata's MMQ Product uses max_rows for both parameters; the old bounded caller used full prompt length. Counts come from the existing host-ID validation loop. No model name, fixed geometry or new arithmetic kernel is introduced.

## Remaining

Native fused gate/up + activation/requantization + down arithmetic and cross-layer stream-ahead are not implemented in this slice. Their fixed reference geometry cannot be silently treated as universal. Match and measure these benefits through shared backend contracts before declaring the owner request complete. Backend operation gates, real MTP/concurrent/staggered requests and DeepSeek V4 qualification remain open. Windows and physical multi-GPU validation are unavailable here.

Load mode none is retained. Flash uses normal automatic lazy PLE rows; its large PLE file is not forced into RAM. DeepSeek V4 has the owner's explicit mmap exception. All build/model jobs hold both documented flocks in order for their complete lifetime. Extended drains remain skipped.

Artifacts: /home/gencoolpc/moe-cache-tests/results/generic-strata-prefill-20261007. Failed initial setup/fixture artifacts remain preserved beside corrected trials.

## Build 10 qualification and comparison correction

Routed-wave sizing passes all 52 cases and 96/96 focused CUDA-vs-CPU MUL_MAT_ID backend cases. First Flash warm result: 1331 ->1393 tok/s with identical output IDs; observational single pair. An Ornith MTP3 run completes two128-output requests, with59/72 draft acceptance each. GPT-OSS completes two staggered concurrent64-output slots in each of two waves; actual slot IDs differ and client intervals overlap. These are compatibility checks, not matched serving-speed gains.

GPU timeline (diagnostic only) exposed the default server checkpoint policy: 2044-token prefill followed by4-token tail (the API's prompt total includes the additional leading token). Source: tools/server/server-context.cpp:3992-4010; checkpoint default32 in common/common.h. Specialized --prompt-cache0 was unlike this default generic recipe. With generic --ctx-checkpoints0, same2049 saved IDs and batch/ubatch2048, build10 reaches1266 cold /1603 warm tok/s. This setting change is not a code optimization. A build09 no-checkpoint control is next. Specialized same-input observations remain1716 unfused /2122 fused. All examined16-output sequences match across engines/candidates, but cold/warm initialization, profiling and speculative work differ.

The trace contains94 warm prefill-related2D D2H calls, of which47 copy81760 bytes and47 copy160 bytes, confirming2044+4 routed rows at40 bytes/row. CPU API durations include waiting for preceding GPU work; they are not independently additive or wholly removable. Kernel/copy timings overlap. Retain raw trace and approximate request-boundary attribution; exclude profiler-perturbed speeds from normal comparisons.

Completed before the incident: build09 no-checkpoint control1209 cold /1537 warm; build10 counterpart1266 /1603. Native repeat-off1683 /2217; native repeat-on2117 /2483. Same saved2049 IDs, two16-output requests; all outputs examined match. Native profile/adaptation/MTP and generic policy/head differences still limit the speed ratio. Generic's attempted aligned profile/adaptation runs fail on repeated prefill and are not valid speed comparisons. Do not declare parity from the earlier single cold1716 unfused observation.

DeepSeek V4 none-load smoke passes64 input /4 output with required source execution:71.04 prompt tok/s,2.93 decode tok/s. Peak7765MiB VRAM; minimum4419.91MiB host available; clean exit. The earlier mmap/8192MiB recipe rejected read-only registration and lacked room for full-source fallback before prefill; retain it as setup failure. A no-regression control and a substantial prompt remain required.

## Scoped synthetic device investigation

The corrected test-only candidate passes the guarded CPU command/manifest preflight:3072MiB hard accounted memory, zero swap,120s finite runtime and verified whole-tree removal. test_profile_transition.py permits only test-moe-cache --hybrid-metadata-only with source-core-profile/adapt/async; it validates frozen candidate hashes, acquires both ordered locks nonblocking, installs the fatal-log group-kill watcher before child launch and requires all eight retained-owner prefill checks plus four source fixtures. No server or faulty frozen GPU control is launched. GPU-RUNS-BLOCKED.json remains present and continues to block run_models.py. Its explicit tiny-fixture scope follows the now-verified containment prerequisites; it does not release the original Flash workload or establish its root cause. The previous block file is preserved. That preparation stage was followed by the qualified device runs above.

## Fast-prefill source composition audit

A CPU probe of the actual three MMQ source selectors proves why direct paired gate/up cannot simply be combined with the current staged maps/images. Existing direct use is correct; the proposed composition would use wrong second-bank addresses. The next optional fusion must extend the shared source/reader contract rather than relax a buffer guard. The audit also identifies the local compact-body operation switch as unsuitable for universal model admission. Source hashes, exact ranges and counterexamples are in PREFILL-FUSION-SOURCE-AUDIT.json; DESIGN.md records the concrete implementation boundary. No product code, CUDA/model run or performance claim was added.

## Prepared paired-source arithmetic qualification

PAIRED-PREFILL-QUALIFICATION.json now records32 successful tiny device cases against an independently allocated ordinary CUDA graph. Marking reference gate/up tensors as outputs preserves their separate original matrix execution; the candidate retains original routing checks, NaN sentinels and unchanged numerical tolerance. Q4_0/Q5_K, four existing backend GLUs, two routing distributions, reverse-mapped full residency and mixed residency with two staging slots all match exactly (relative MSE0). Mixed residency takes6 waves, full residency1. Invalid residency/staging capacities reject.

The job verified3072MiB hard RAM, zero swap,120s runtime, both ordered locks and complete tree/PID removal. Peak308125696bytes, elapsed6.54s; no limit/OOM event or interval kernel Xid/OOM appeared. Model-server replay remains blocked. Two earlier attempts stopped before arithmetic because of test setup: a two-graph initializer given one graph, then a runner asserting intermediate materialization despite ordinary fusion. Both are preserved with source diagnoses and verified teardown; original helper assertions and product arithmetic were unchanged.

That initial same-stream arithmetic result preceded canonical integration. The later PAIRED-PREFILL-SCHEDULER-QUALIFICATION.json records92 arithmetic/owner/graph checks, including24 actual scheduler cases, and PAIRED-PREFILL-MEMCHECK-QUALIFICATION.json reports zero errors. The canonical implementation remains opt-in and defaults off: the equal-work component ABBA measured regressions in five of six format/source groups. Complete expert/down waves, cross-layer scheduling, serving performance and the original Flash incident remain unqualified. No default or release promotion follows.

## Research checkpoint authorized on 2026-10-08

The owner authorized committing and pushing the current isolated research branch. Before staging, the tracked source patch matches e2d1f430ff2f10523b6709f3018a6c5c91225f1c977b8f8ccef5fe9cdef642ad from the last qualified build. The six frozen executable/library file hashes and their symlinks were rechecked, together with the35-case allocator result and12 tiny device cases/24 production preparations, clean teardown and no interval kernel alerts. CHECKPOINT-PREPUSH-20261008.json retains the local verification. No new build or model execution is part of this checkpoint.

This is an incomplete research checkpoint, not a default installation or release qualification. Optional paired fusion stays disabled after measured regressions. Original Flash root-cause, full serving/no-regression, held-out profile quality and physical platform gates remain open. Request-scoped CPU cancellation and the newly requested non-fusion migration have not been implemented in this checkpoint. PLAN.md and TODO.md record the owner's next task: non-fusion feature parity followed by matched decode comparisons.
