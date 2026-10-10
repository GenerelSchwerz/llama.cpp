# Current prefill implementation comparison

This supersedes the projection-only description in PREFILL-GAP-20261009.md for the current dirty complete-body candidate. It does not promote that candidate or close the performance goal.

## Latest accepted checkpoint

Frozen318 now retains compatible main prefill across sampled decode and request sampler changes. The final318/290/290/318 own-control comparison passes all twelve exact outputs: repeated1547.62 ->1719.69 tok/s, cold1156.86 ->1110.84 and repeated decode58.47 ->57.72. This is bounded single-GPU/no-MTP evidence, not a fresh specialized comparison or full parity claim. Detailed sources and raw reference results below are historical; see STATUS.md for current qualification. No reference engine implementation is included in the generic publication.

## Current source and tested implementations

Frozen290 is the last built sampler-lifetime foundation over279. Unqualified current source adds the owner-approved persistent phase cache and aggregate managed-resource admission. Build/tests298 are running; current source does not match290 and has no serving qualification. Its cache reuses upstream shared allocation plans, keyed by full execution certificate, output count and three input-placement modes, with at most eight prepared programs. Original sampler/input reuse and buffer-retirement checks remain authoritative. No commit, push, promotion or1700 claim.

The table below describes retained279/290 execution and persistent remaining differences. Later sections retain historical measurements with their original qualification.

| Stage | Strata MMQ prefill | Owned generic source | Qualification |
| --- | --- | --- | --- |
| Startup profile | Fill selected expert slots before READY | Same lifecycle through the canonical owner | Implemented and tested in retained100 |
| Root activations | Quantize once in expert order | Share input quantization and routing between compatible roots | Implemented and tested in retained100 |
| Gate/up layout | Combined product stays in expert order | Separate original projections, followed by graph-derived gathers into compact expert order |180/190 reduced gathers/storage without a demonstrated model gain; reverted from owned runtime |
| Complete expert body | Gate/up, activation, hidden quantization and down per group | Complete graph-derived body inside each ready owner wave | Implemented for eligible all-GPU bodies; preserves original operators |
| Down route tables | Relative bounds prepared once per layer | Borrow identity rows and checked relative bounds per compact chunk; hidden quantization remains per chunk | Isolated214 and exact-bound222 pass numerical and matched model gates; speed benefit remains unqualified |
| Transfer issuance | Packed whole-expert blobs, bounded ring and separate issuer spanning layers | Original tensor banks, two64MiB staging lanes, demand from the current body | Cross-layer issuance and packed transport remain different |
| Group sizing |16 experts per MMQ group in this path | Wave width derives from bank bytes and staging budget; body/root subgroup isolation tested and reverted | Body16 slows repeated throughput5.85%; root-only16 slows8.00%, so copying this width alone is rejected |
| Host synchronization | Event-controlled ring reuse; issuer can run ahead | Wait for body completion before proceeding to the next layer | Localized wait deferral passed correctness but slowed repeated throughput0.87%; reverted |
| Workspace | Borrow expert-cache slots during prefill and refill afterward | Separate bounded staging, compact workspace and pool | Lending is absent; matching initial profile bytes does not match live prefill storage |
| Dense backbone | Dedicated dense projection, attention and recurrent-state scheduling | Upstream graph operators and backend dispatch | Fresh traces separate mapped/unmapped MMQ, but do not form a matched dense component benchmark |
| MMQ row bound | Exact maximum for the current product group | Exact compact down bound; roots use wave maxima; rejected subgroup experiment is reverted | Original numerical gates pass; copying subgroup geometry did not establish a speed fix |

Source anchors in the inspected local Strata checkout: src/prefill/prefill.cpp:2236-2359 enumerates nonresident experts across layers and issues the bounded ring;3233-3255 quantizes the root input and prepares relative bounds;3596-3676 gathers native expert blocks and runs grouped products;992-1243 lays out borrowed storage. src/prefill/moe_mmq.cu:192-200 uses maximum-row bounds. Generic ggml/src/ggml-cuda/moe-source-core.cu:506-548 builds original-layout gathers/scatters;2783-2927 prepares shared roots, reuses down bounds and waits for body completion. ggml/src/ggml-cuda/moe-cache.cu:18123-18255 owns wave staging and event-controlled lane reuse. ggml/src/ggml-cuda/mmq.cu:843-977 retains existing preparation/quantization with a checked sorted-route extension.

Strata stream-all lookahead enumerates nonresident experts before future router results are known. It can transfer unused experts; it is not future-route prediction. Native fused int8 prefill is opt-in in the actual frozen specialized source and was not enabled by the matched controls. Its speed cannot be assigned to that optional fusion. Dense native projections dequantize into GEMM scratch; the source does not justify claiming all dense quantized weights are permanently converted.

## Current comparison interpretation

Source reinspection confirms the shared algorithm: group routed rows by expert, share compatible root quantization, execute a complete expert body when its weights are ready, and use relative down-product bounds. Generic now expresses this through graph-derived original operators and the canonical residency owner. It has not reproduced Strata's packed group layout, cross-layer producer or borrowed-cache workspace lifecycle. Strata Prefill::init also checks artifact-specific dimensions and routing geometry at992-999; those assumptions cannot define generic support.

Latest generic frozen272 repeated prefill is1522.39 tok/s versus its own frozen222 control1442.89 (+5.51% bounded observation); cold1080.66 is variable. The earlier matched specialized comparison203-208 measured1780.12 cold/2219.12 repeated. These are different run sets and engine prompt boundaries, so do not present1522 versus2219 as a fresh matched A/B or exact remaining percentage. Earlier paired client traces still establish a real gap independently of engine timers:1493.65 versus944.40ms repeated first-token latency, with398.37 versus4.38ms without recorded CUDA activity. They precede272 and must not be treated as its current idle-time measurement.

Host route planning is now addressed locally in279: reuse existing expert-sized row-count storage during planning to preserve first-seen ordering and all slot/classification checks.277 passes11 gates/264 numerical replays/1456 actual-route checks; sanitized278 compares5040 cases against the exact frozen linear loop. ABBA280-283 observes repeated1517.18 ->1532.32 tok/s (+1.00%), planning39.14 ->19.07ms, eight exact outputs and equal fixed workspace/12363MiB peak. Final control has small adaptive work variation; cold is variable and decode -1.20% observational. This remains a small bounded improvement. Larger cross-layer issuance or cache lending requires a concrete lifetime/resource design, not copying the specialized executor. Earlier packing/grouping/wait experiments show that adopting isolated Strata details without their surrounding pipeline can regress performance.

## Fresh trace and isolated results

Paired diagnostic212/213 wraps two uncached2049-input/64-output requests from retained100 and the frozen specialized port. All four output sequences match exactly. These are first-token windows from the clients, not interchangeable engine prompt timers. Overlapping component times are not additive recoverable savings.

| Repeated first-token trace | Generic100 | Frozen specialized |
| --- | ---: | ---: |
| Client window ms |1493.65 |944.40 |
| Union of kernel activity ms |979.40 |833.07 |
| H2D bytes, decimal GB |26.07 |41.44 |
| H2D activity without kernels ms |112.85 |106.49 |
| No recorded CUDA activity ms |398.37 |4.38 |
| Summed expert mapped-MMQ / private-MMQ ms |399.63 |249.14 |

Specialized moves more host data while keeping the repeated pipeline nearly continuously active. Transfer volume alone does not explain the gap. Generic still has preparation/phase-transition work and slower mapped expert products. The trace does not attribute every idle interval to a function, nor prove all idle time removable. Basic matching Blackwell MMQ type/tile configurations have the same register/shared-memory geometry; a generic occupancy defect is not established. Native fused int8 prefill was disabled in these controls. External PAIRED-FIRST-TOKEN-TRACE-REPORT.json retains the full intervals, runtime API summaries and CUDA unions.

Isolated down-only ABBA217-220: repeated reported prefill1425.51 ->1441.82 tok/s (+1.14% observational); cold1117.68 ->1109.46. Exact compact bound ABBA223-226 against down-only214: repeated1435.65 ->1443.12 (+0.52%); cold1116.01 ->1135.95. Each comparison has only two process samples per arm, and all eight64-token output sequences match within each comparison. Neither qualifies a repeatable speed/no-regression gain. Reports ISOLATED-DOWN-ABBA-REPORT.json and EXACT-CHUNK-BOUND-ABBA-REPORT.json remain separate from the earlier three-arm comparison below.

Diagnostic228 confirms three graph rebuilds per request:2048-row prefill, ordinary-input1-row tail, and sampled-input1-row decode. Both requests remain uncached and emit the exact64 IDs. Source llama-context.cpp:3270-3274 bounds graph variants by decode output capacity and clears variants on sampled-input mode changes. Server output capacity derives from parallel requests and enabled draft width (server-context.cpp:73-86; speculative.cpp:3172-3181), not prompt width. The existing cache therefore does not retain this full2048-row prompt for the single-slot/no-MTP configuration. Phase-aware allocation can additionally invalidate generations; the source checks must remain intact. See PROGRAM-LIFETIME-20261010.md for the bounded next investigation.

Diagnostic227 with phase-aware workspace disabled fails explicit required-hybrid admission: allocator generation unavailable. It is a rejected configuration, not a speed result or CUDA crash. Both227 and228 are terminal with empty process trees. No new model run is queued. External component250 isolates existing mapped/unmapped MMQ interfaces; it is not an end-to-end comparison.

## Product-geometry experiments238-248

Full-body groups ABBA238-241, same frozen237 binary: repeated1351.51 versus1435.41 tok/s (-5.85%). Main compact chunks grow to about1700 versus676. Root-only ABBA245-248, same frozen244 binary and unchanged body grouping: repeated1322.70 versus1437.75 tok/s (-8.00%); cold1041.33 versus1141.46. Root launch groups3374 versus848 over both requests, while compact chunks remain686/673 versus678/676. All eight64-token output sequences match exactly in each comparison; all source/path/guard gates pass and all trees are empty. Internal selected routes and paid bytes differ slightly despite equal outputs, so neither is a matched-route component test. Workspace37124736bytes, pool capacity384494592bytes and peak100288512bytes remain unchanged. Reject both smaller-group settings as speed fixes. Reports PRODUCT-GROUPS-ABBA-REPORT.json and ROOT-GROUPS-ABBA-REPORT.json preserve actual accounting.

The next external component uses existing original mapped/unmapped MMQ interfaces with identical bounded raw model weight slices, fixed activations/routes, original quantization and identity/reversed physical layouts. No source packing, new kernel, copied executor or owner change is implemented. Equal routing/quantization overhead is included on both arms; the result must not be called expert-kernel-only timing or a model speedup.249 failed host-only CUDA-header compilation before GPU execution and has verified empty teardown;250 corrects the compiler in a new result directory.

## Matched64 comparison203-208

All six processes are terminal with passing source/harness gates and empty process trees. Each performs two uncached2049-token prompts and emits64 outputs, with no MTP. All twelve output-ID sequences are exactly equal. Order is candidate198, specialized, retained100, retained100, specialized, candidate198. Means below use two process samples per arm.

| Implementation | Cold reported prefill tok/s | Repeated reported prefill tok/s | Cold first token s | Repeated first token s | Sampled peak GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained generic100 |1030.07 |1424.30 |1.9986 |1.4417 |12363 |
| Combined generic198 |1021.38 |1455.06 |2.0166 |1.4128 |12355 |
| Frozen specialized |1780.12 |2219.12 |1.1991 |0.9391 |10013 |

Candidate198 is2.16% higher on repeated reported throughput and0.84% lower cold. Repeated first-token latency falls28.91ms. These observations do not establish a repeatable speed/no-regression qualification or isolate the sorted-down change. Specialized's reported prompt interval excludes the final prompt-token verify window; generic includes final-token work and can include the following decode submission. Client first-token measurements still show a real gap, with different transports/server work and live storage. See TIMING-BOUNDARIES-20261010.md. Same configured3072 slots/5364121600 profile payload bytes does not mean equal live prefill residency or workspace. Full evidence is external MATCHED-64-ABC-CBA-REPORT.json under generic-prefill-gap-20261009.

Numerical gate197 passes all ten commands,224 main-source replays at unchanged original-oracle maximum relative MSE1.60269249e-7,2736 logged sorted preparations, and50856 independent CPU bounds cases/169081 chunks. Fixed-route component202 passes four processes with matched routes, selected bytes and CPU calls. Summed replay time falls192.707 ->168.832ms, but the F32 fixture slows10.984 ->15.911ms even though it does not execute the optimized body. This component result neither isolates sorted-down reuse nor qualifies a model gain; investigate the unexplained fixture regression before promotion. Failed200 remains preserved: clean execution naturally stopped at88 rather than required128, failing its comparison-length gate. No commit, push or promotion.

## Historical implementation comparisons

The sections below describe their named frozen candidates, not the latest owned source.

## Current root-view candidate190 versus Strata

Current source includes experimental180 expert-order roots and190 root views on top of retained100. The restoration176 comparison below describes retained100, not these prototypes. Strata source remains fb58e0dbc8399662c0e47c76578c6e878b14f6cf, checked locally. No network version claim is made.

| Stage | Strata MMQ path | Current generic candidate190 |
| --- | --- | --- |
| Root preparation | One expert-ordered quantization; precomputed group bounds | Compatible roots share quantization/routes; original bank types/layouts remain |
| Root output layout | Combined gate/up stays expert ordered | Eligible private roots now write expert order and expose compact views; external readers retain original layout and gathers |
| Expert body | Grouped gate/up, activation, hidden quantization, down | Complete graph-derived body per ready all-bank wave, original upstream arithmetic; separate projections remain separate |
| Down route setup | Relative group bounds prepared once per layer | Mapped-MMQ preparation repeats for each compact chunk |
| Transport | Packed expert blobs and bounded ring issuance across layers | Separate original bank planes, two64MiB staging lanes within current body; no cross-layer issuance |
| Storage | Can borrow resident cache slots and refill afterward | Separate bounded compact workspace/staging; no cache-slot lending |
| Lifecycle | Dedicated prompt layout, profile placed before READY | Profile placed before READY, but source graph/resource preparation remains per program |
| MMQ tile bound | Group maximum for both ncols_max and ncols_opt | Wave/chunk maximum for both; this is not a demonstrated mismatch |

Source anchors: Strata prefill.cpp:2238-2273 enumerates nonresident experts across layers without future route knowledge;2330-2359 issues the ring;3233-3255 prepares quantization/bounds;3596-3676 groups expert products. moe_mmq.cu:192-200 uses maximum-row bounds. Generic moe-source-core.cu:527-546 guards private ordering and removes eligible GET_ROWS;2848-2907 shares roots, executes waves, and prepares down per chunk. moe-cache.cu:18040-18106 retains canonical two-lane staging and failure draining. No additional policy owner or kernel fusion was introduced.

Gate189 passes224 main numerical replays, including56 direct-reader,56 nonzero-offset contiguous-view-reader and56 skewed-tail replays;16 ordered-root comparisons are exact. Original noncontiguous public-output rejection remains unresolved. Candidate190 is unpromoted. First model observation191 passes path/source/teardown gates:2049 full uncached prompt tokens each,128 outputs,no MTP;1010.687 tok/s cold,1430.377 repeated,12353MiB sampled peak. Main compact workspace falls37110784 ->26625024 bytes (10MiB);47 bodies,431 waves,689 chunks,94 ordered roots and27043993600 selected bytes remain in the cold main prefill. Repeated residency differs. This verifies the new path and storage reduction, not a speedup; the six-arm candidate190/ordered180/retained100 comparison191-196 is incomplete.

Archived specialized52 reports1766.532 tok/s on2049 input tokens, but requests only4 outputs and borrows1093 cache slots/1.74GiB. The latest generic128-output trial is not a new matched specialized comparison. Same initial profile payload does not mean equal live prefill memory. Cold preparation, down-route reuse and cross-layer transport remain concrete differences; source comparison alone cannot assign the throughput gap among them.

## Direct source comparison after restoration176

Canonical Strata remains fb58e0dbc8399662c0e47c76578c6e878b14f6cf. Generic source and all ten build artifacts were restored to retained100 in176. No new arithmetic or serving experiment is included in this comparison.

| Contract | Strata MMQ prefill | Current generic complete body |
| --- | --- | --- |
| Root input | Quantize activations once in expert order | Checked compatible roots share quantized input and route arrays |
| Root outputs | Combined gate/up stays in expert order, using identity output rows | Separate original projections scatter into original route order; GET_ROWS gathers them back into expert order |
| Body | Group gate/up, SwiGLU, hidden quantization and down | Original graph-derived operators and down run inside a ready all-bank wave |
| Down routes | Precomputed relative bounds and identity rows | Original mapped-MMQ preparation repeats for each compact chunk |
| Transfers | Packed expert blobs, ring issuance across layers | Original quantized banks and two staging lanes within the current body |
| Temporary storage | Can borrow resident cache slots, then refill | Separate bounded staging and compact workspace; no equivalent lending |
| Tile row bound | Group maximum for both ncols_max and ncols_opt | Wave/chunk maximum for both; max versus average is not a demonstrated difference |
| Startup | Profile placement before READY | Canonical source placement before READY is now implemented |

Source anchors: Strata src/prefill/prefill.cpp:2236-2359 for cross-layer issuance,3233-3255 for input quantization and group bounds,3596-3680 for grouped expert products; src/prefill/moe_mmq.cu:192-200 for maximum-row MMQ arguments. Generic ggml/src/ggml-cuda/moe-source-core.cu:505-550 builds input gathers and live-output scatters;2815-2882 shares prepared roots and executes compact chunks. ggml/src/ggml-cuda/mmq.cu:985-1015 retains original output mapping and selects wave bounds.

Archived trace attribution177 narrows trace53 to the main2048-row interval. H2D activity occupies506.22ms, overlapping kernels for387.84ms; GPU idle is106.49ms. Summed GET_ROWS time is54.52ms and route-helper time38.06ms, including unrelated uses. These are overlapping component durations from an older source snapshot, not current timings or additive recoverable savings. Host copy APIs total26.18ms; this does not establish that a new copy-issuer thread would remove most of the gap.

The current unqualified prototype is private expert-ordered root output: retain the existing MMQ arithmetic and gather operators, but use identity output rows for roots with no external live reader or alias. Externally visible roots retain the original layout. Reuse existing gather-index storage with backend-derived tile guard capacity; do not add another owner or change original tensors. 187 passes exact root reordering and direct-reader numerical cases.188 still must qualify supported contiguous view readers. Feature-sliced noncontiguous public output hits an existing HEAD publication restriction and remains open. Measure contiguous-gather performance before attempting to remove gathers or reuse down-route preparation.

Recent retained100 control172/173 averages1145.74 prompt tok/s cold and1427.47 on a second full uncached prompt. Archived specialized52 reports1766.53 on the same2049 input tokens, no MTP and the same initial profile payload. These are different run sets and live prefill resource budgets, not a new matched performance qualification. The first-token guard candidate remains rejected and absent. The1700 goal stays open.

## Setup and rejected first-token followup

Frozen100 probes163/165/167 measure main allocation3.09ms, outer region finalization/preparation9.28ms and CPU embedding3.85ms; these are minor costs. Warm phase reservation remains46.95ms main/27.88ms tail. Server queues the next hybrid decode before first-token publication; its94.00ms blocking call accounts for most of a95.24ms prompt remainder beyond main/tail logical decode. Experimental169 suppresses this first ahead queue, but longer ABBA171-174 does not qualify: repeated TTFT improves, cold first-token and full request worsen,174 output diverges after zero-based index97, and repeated residency differs. Source is restored to100; 175 caught a stale server object; forced rebuild176 and both CPU checks pass, with exact source/server/all10 binary restoration. No compute gain or1700 parity. See SETUP-ATTRIBUTION-20261010.md and FIRST-TOKEN-RELEASE-ABBA-REPORT.json.

## Resource preflight followup rejected

A checked-query omission prototype153 passes13 existing gates and skips523 ordinary resource-measurement kernels after upstream fusion selection. It preserves fused groups, opaque capture and actual replay. It is rejected after matched ABBA155-158 plus full cache_prompt=false diagnostics159/160: cold capture225.01 ->186.36ms is accompanied by replay1128.57 ->1206.08ms, and warm prefill1405.91 versus1418.33 is nearly unchanged. Aggregate ABBA ratios are inflated by a slow control and do not establish a gain. Source is restored to frozen100; rebuild161 passes all seven original checks and restores the mutable build. No specialized rerun, commit/push or promotion. See RESOURCE-PREFLIGHT-20261010.md and external reports. The next setup investigation must measure scheduler/frontend work outside core counters rather than shorten capture in isolation.

## Graph reader indexing candidate100

Current source additionally indexes graph readers during frontend discovery, including consumer view_src and src view_src relationships from the existing scheduler contract. Source body preparation indexes original direct/view readers once and uses them only after its existing full source-witness check. Public closed_cut keeps its live original-reader scan, with checked constant-time clone membership. No arithmetic, canonical residency, transport or GPU budget changes.

Focused98 passes CPU projection discovery with the original consumes oracle, body-program checks, complete-body mixed-layout numerical cases, source-overlap held-reader checks and owner cases. Initial96 passed but CPU probe97 exposed a regression from full witness validation at every public cut query; this was refined before serving. Probe99's synthetic64-layer graph (1728 nodes,8 repetitions) measures discovery7.80 ->1.07ms and body preparation138.42 ->5.50ms. These are metadata component timings, not model speed.

Frozen100 versus frozen95 ABBA102-105 measures candidate1103.45/1062.01 and control958.78/1011.82 prefill tok/s, means1082.73 versus985.30 (+9.89% observational). All process2049 prompt tokens and emit the same four IDs/hash, with3072 startup slots/5364121600 payload, zero first-main profile copies,47 bodies431waves689chunks,47 shared inputs94 plans, zero main CPU/failures/fallback and12349MiB peak. Main replay1126.94 versus1131.84ms and planning44.51 versus44.57ms are nearly unchanged; main preparation367.12 versus339.68ms is slower on average for the candidate. The aggregate difference lies outside these counters: main/tail preparation and replay leave237.03 versus454.78ms of unattributed prompt remainder. This needs direct frontend attribution before claiming an isolated9.89% gain. Diagnostic101 reports16.50ms private storage,64.42ms cumulative layer preparation and248.53ms resource capture. All jobs96-105 are terminal with empty trees; no new specialized comparison, commit, push or promotion.

Direct RTLD_NEXT timing probes107/108 preserve the unchanged exported function calls and pass output/path/teardown checks. Main plus tail discovery takes52.01 ->8.93ms;47 body constructions take14.13 ->4.52ms, nested within source preparation. This verifies approximately52.68ms of setup savings in two diagnostic observations, not the full ABBA remainder difference or an isolated9.89% gain. CPU preload validation passes before serving; the temporary probe lives only in external evidence106. All owned jobs through108 are terminal and locks released. Current source/build match frozen100.

A separate source-backed placement difference remains to measure: upstream llama-model.cpp keeps the input layer on CPU even at full layer offload, while Strata prefill.cpp can perform a batched native embedding gather on GPU. Saved trace53 samples CPU GET_ROWS/dequantize_row_iq3_s before main source allocation. This identifies possible frontend work outside core counters; it does not quantify an embedding penalty or authorize mandatory conversion/model-specific placement. Existing CUDA get_rows supports IQ3_S. Ordinary resource capture remains the larger measured setup cost (~248.53ms in101); inspect existing backend scratch/library contracts before changing admission.

## Canonical startup placement candidate88

The constructor now initializes source residency after learned-state restoration and before READY. It extends the existing owner placement endpoint with a checked source flag; it does not run a fake prompt, create another owner or change arithmetic. Full ranks remain independent of capacity. Existing learning must match the installed seed and retains counts, heat, usage, prior and windows. Physical owners become source placement intent, avoiding an initial recopy merely to reorder slots. Static profile application also accepts an already resident exact selected set after reciprocal-map and transport validation.

Frozen88 passes diagnostic89 and unprofiled ABBA90-93:

| Run | Arm | Prefill tok/s | Main prepare ms | Main replay ms | Model-ready log s | Peak GPU MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 90 | Startup placement | 1125.96 | 281.51 | 1125.79 | 19.615 | 12349 |
| 91 | Frozen65 control | 524.93 | 1079.06 | 1131.43 | 15.616 | 12211 |
| 92 | Frozen65 control | 846.97 | 702.90 | 1126.83 | 17.828 | 12211 |
| 93 | Startup placement | 1001.52 | 402.85 | 1127.27 | 14.013 | 12349 |

All arms process 2049 tokens, emit 846,198,7734,264 with the same hash, and retain the same configured 3072 slots / 5364121600 payload bytes. Main prefill has zero CPU calls, failures and ordinary-provider fallback. Candidate startup copies all 5364121600 bytes in 185.52 and 195.66 ms before READY; first main and tail profile copies are zero. Controls copy 5224857600 bytes in main and 139264000 in tail. Required source routing and complete teardown pass in every arm.

This verifies the lifecycle change and lower request preparation, not a 55% repeatable speedup: the mean ratio is inflated by retained slow control91, and main replay remains approximately 1.13 seconds in every arm. Initialization varies by several seconds. The sum of model-ready log time and prompt time averages 18.747 versus 19.883 seconds; it is an approximate logger-derived lifecycle comparison, not an independent end-to-end measurement. Peak memory increases 138 MiB. Full startup placement also retains layer47's 139264000-byte payload during main prefill, whereas control first fills it in the tail; that is consistent with most of the increase, not an allocation-trace proof of the entire delta. Do not claim equal live prefill residency or equal peak resources.

The earlier specialized 1766.53 tok/s observation remains substantially faster; no fresh specialized ABBA was run here. Candidate88 does not meet the 1700 target. External OWNER-STARTUP-SEED-ABBA-REPORT.json retains all raw runs and caveats.

Numerical gate85 passes the existing 216 source and 96 owner cases plus static, occurrence and statistics cases before its preserved async fixture expectation failure. Focused87 passes explicit GPU-profile all-bank/busy/repeat checks, static first-replay map preservation, statistics and occurrence cases, learned restore/preload/resume, and held-copy asynchronous publication at both 17% and 50% miss fractions. The old pending-route assertion assumed zero GPU misses; its corrected exact count uses the declared split. No numerical tolerance or held-reader/publication assertion was relaxed. Physical multi-GPU, native Windows and mixed CPU/GPU real-model startup remain open. Followup94 passes coherent linkage and focused owner/static/statistics/codec gates for the source-only uncached static-bank filter. Frozen95 matches current source/build; frozen88 remains the exact serving artifact.

## Cold versus repeated full prompt

Frozen65 probe81 sends the same full prompt twice with cache_prompt=false. Both responses report prompt_n=2049 and cache_n=0, and have identical output IDs. Cold prefill is 902.99 tok/s versus repeated 1343.36. Main preparation falls 663.65 -> 72.26 ms, while replay is 1126.12 -> 1112.30 ms. This confirms substantial cold setup without confusing it with KV/prompt reuse, but still leaves a repeated-request gap versus the earlier reference. Main/tail preparation plus replay leaves 315.68 ms cold and 273.57 ms repeated outside those counters; these are timing remainders, not attributed CPU components. Probe79 failed before its log existed, and80 rejected the text-shaped payload before server launch. Both failed setups are preserved with empty trees; corrected81 passed.

## Latest bounded-resource preflight experiment

Build72 passes216 source numerical replays and96 owner cases. Frozen candidate73 omits original-projection MMQ measurement only for complete bodies whose root and compact scratch bounds have already succeeded. Original ordinary-operation capture and other projection measurement remain unchanged. This is resource preflight work, not removal of runtime binding, ownership or reader validation.

Runs74-78 are terminal and passed with complete teardown. Diagnostic74 confirms141 bounded body projections and zero measured projections, but remaining main capture still costs210.25ms. The proposed omission does not remove the expensive ordinary-operation capture. Unprofiled ABBA75-78 measures control833.24 versus candidate845.49 prefill tok/s (+1.47% observational), with main preparation762.95 versus746.48ms and main replay1130.54 versus1166.32ms (+3.16% slower). This does not establish an end-to-end gain; candidate73 was reverted from source before startup placement. Its patch, binaries and results remain preserved.

All four runs process the same2049 prompt tokens and emit846,198,7734,264 with the same output hash. Each retains3072 initial slots/5364121600 payload bytes,47 bodies,431 waves,689 chunks,47 shared inputs,94 shared plans and12211MiB sampled peak GPU memory. CPU calls and failures are zero in the2048-row main prefill. Tail and generation programs are separate and may use CPU work. External BOUNDED-BODY-PREFLIGHT-ABBA-REPORT.json records the final per-program summaries, stage diagnostics and raw artifact paths. Specialized run52 remains the earlier frozen1766.53 tok/s observation, not a newly repeated ABBA arm.

## Matched observation

External evidence root: /home/gencoolpc/moe-cache-tests/results/generic-prefill-gap-20261009.

| Measurement | Generic candidate44, run50 | Frozen specialized port, run52 |
| --- | --- | --- |
| Input tokens | 2049 | 2049 |
| Output token IDs | 846,198,7734,264 | 846,198,7734,264 |
| Prefill tok/s | 903.59 | 1766.53 |
| Initial expert slots | 3072, 64 per layer | 3072, 64 per layer |
| Initial resident payload bytes | 5364121600 | 5364121600 |
| Sampled peak GPU MiB | 12211 | 10009 |
| MTP | No draft work | No draft work |

The prompt text round-trips to the original saved token IDs exactly; proof is matched-2049-inputs-49/TOKENIZER-PROOF.json. Both runs use fresh contexts, ubatch2048, the same balanced-code.bin profile and occurrence adaptation. Four output tokens check this observation's continuation; they do not establish general numerical equivalence or steady decode throughput. These are one observation per arm, not a repeated performance qualification. Equal initial payload does not imply equal live prefill residency, staging or total resources.

Generic throughput is approximately48.85% lower in this observation. The longer8194-token CPU-off storage ABBA averaged1307.34 tok/s, with a6.65% observed gain over its own control. That workload cannot replace this matched2049 comparison. Archived2049 generic was1096.54 tok/s: the current short-prompt regression needs investigation before promotion.

Run51 failed argument validation before any model launch because bench required --served-model; run52 corrects that setup issue in a new directory. Runs50/52/53 are terminal and their FINAL.json records successful teardown. Run53 is an attribution trace, not another unprofiled speed sample.

## Source-backed differences

| Contract | Canonical Strata / frozen specialized prefill | Current generic candidate |
| --- | --- | --- |
| Profile seed timing | Fills selected expert slots before READY; run52 reports0.5s startup fill. | Candidate88 now fills all selected slots before READY through the canonical owner, after restoring learning. First main/tail profile copies are zero in89-93. Earlier run50 performed that work in-request. |
| Complete expert work | Groups experts and executes combined gate/up products, activation, ready-hidden quantization and down before the next group. | Now derives legal complete bodies from the original graph and executes gate/up, original intermediate operators and down inside each owner wave. This portion is implemented; it is no longer projection-only for eligible all-GPU bodies. |
| Input and route preparation | Shared input quantization and grouped route tables for the expert body. | Candidate58 shares prepared quantized input and route arrays between compatible original roots; incompatible roots retain independent upstream preparation. Compact internal MMQ still prepares route tables and quantized hidden input again for each chunk. |
| Chunk granularity | MMQ groups contain16 experts, with ready rows spanning the group. | Expert wave width comes from actual bank bytes and owner budget. Compact ready-row capacity is bounded by original physical rows, so one wave can split into multiple chunks. Run50 has47 bodies,431 waves and689 chunks. |
| Transfer lookahead | Long-prompt stream-all enumerates known nonresident blobs across layers; a separate issuer can copy ahead of current routing. This may copy unused experts and is not a prediction of future router results. | Copies planned from current routed IDs using two canonical staging lanes; overlap stays inside the current body. No equivalent cross-layer stream-all issuer. |
| Weight layout | Stages a complete packed expert blob and gathers group gate/up/down banks. | Preserves original GGUF tensors and separate original banks, staging them under one canonical owner without a mandatory repack. Coalesces eligible consecutive registered-source copies. |
| Storage reuse | Borrows expert-cache storage for temporary prefill buffers, invalidates those residents and refills them afterward. Run52 reports1093 borrowed slots,1.74GiB. This reduces live resident experts during prefill; it is not extra expert capacity. | Keeps resident banks and separate128MiB staging plus compact/workspace storage. Current compact workspace is37109760 bytes; pool capacity384494592 bytes, actual peak100288512. No equivalent loan/refill lifecycle. |
| Synchronization | Ordered copy-ready and release events can cover a gathered group; issuer works independently. | Owner wave events overlap copy/compute, but body completion still waits on the host. First-request graph/resource preparation is also charged to prompt timing. |
| Arithmetic scope | Specialized products and model-specific surrounding operations, with optional fused-expert kernels separately enabled. | Original upstream GGML operations and mapped-MMQ consumer identity. Optional paired fusion remains disabled after earlier regressions. |

Source locations: canonical /home/gencoolpc/Strata/src/prefill/prefill.cpp:241-291,2322-2387,3063,3556-3674 and src/program/generate.cpp:515,4700,6471. The frozen specialized serve command enters main.cpp:71, which dispatches to moe_native_current_main in vendor/strata-kernels/src/program/generate.cpp:1562. CMakeLists.txt:182,200 links this current serving implementation with the port prefill.cpp, rather than using the older target_runtime::serve diagnostic implementation. main.cpp, CMakeLists.txt, generation.cpp, prefill.cpp and vendor generate.cpp hashes were rechecked against candidate-port-0141-01/MANIFEST.json and match. Canonical and specialized sources remain read-only; their adapted source files are not asserted byte-identical. Generic locations are moe-source-core.cu:1366,1547,1872,2738,2822,3006 and mmq.cu:839-931.

## What the evidence establishes

Run50 main prefill preparation is612.11ms. It includes profile application, allocation, graph/resource preparation and validation; it is not a measured612ms profile-only penalty. Main replay is1172.18ms. Additional one-row/decode preparation exists, so these two values are not a complete prompt-time decomposition. Strata's startup profile fill is outside its reported prompt time. Correct that lifecycle mismatch before attributing the entire throughput gap to GPU arithmetic; also retain cold-request and startup costs explicitly.

Time-filtered CPU sampling in run53 places CPU pool execution in trailing one-row/decode work, not the all-GPU2048-row prefill. Disabling that pool has no evidence as the primary prefill remedy. Current source and saved MMQ traces identify repeated preparation, compact chunk overhead and missing copy-ahead as remaining architectural differences. Their individual end-to-end contribution is unproven.

## Next work

1. Startup seeding is implemented through the canonical owner and separately measured in89-93. Finish remaining platform/mixed-placement qualification; investigate ordinary-operation capture and the request-time remainder without hiding cold costs.
2. Compatible original MMQ roots now borrow one prepared quantized input and route image, and compatible projections share the body routing plan. Preserve the checked format, identity, geometry and last-reader contracts; this does not eliminate compact hidden-input preparation.
3. Measure compact chunk fragmentation against bounded scratch and kernel work before changing its budget. Larger temporary memory is a resource change and must be reported.
4. Design cross-layer copy-ahead through the same owner's source leases and release events, with cancellation and per-device budgets. Do not copy fixed model widths, expert counts, thresholds or a second executor from Strata.

Cache lending is a distinct later ownership/lifetime change. Kernel fusion is not required for the archived1700 MMQ reference and remains deferred. Current numerical fixtures pass216 replays, but current short-prompt performance, optimized-body multi-device/concurrency/platform coverage and optional-body budget admission remain open.

## Checked root input sharing candidate

The ordinary CUDA graph planner already shares routed input by route identity and mmq_get_q8_1_ds_layout (ggml-cuda.cu:15168-15225). Its graph-wide reuse plan does not directly describe a canonical-owner wave callback. The existing prepared mapped-MMQ interface owns its quantized input and route arrays, so the local extension lets another prepared consumer borrow those immutable arrays instead of creating a second reuse planner or arithmetic implementation.

Admission requires identical input and IDs tensor identities, device, stream, expert geometry, routed row count and quantized scale/sum layout, plus sufficient tile guard. An incompatible root uses its original independent preparation. The body's original allocation dependencies retain both dynamic inputs and root outputs through the body. Prepared roots remain alive through the final stream completion and are destroyed in reverse preparation order; compact down inputs retain their original independent preparation because their hidden rows become ready later. Gate and up matrix launches remain separate.

Existing owner fixtures add positive borrowing plus rejection of changed input identity, IDs identity, expert geometry and incompatible scale/sum layout. Required source numerical fixtures cover separate/fused bodies, retained outputs, short tails and mixed bank formats. Build57 passes216 source numerical replays and96 owner numerical cases, including positive sharing and structural rejection cases. Attempts54/55 failed host-header compilation;56 exposed an incorrect negative fixture because Q4_0 and Q5_K share the same input layout, corrected to Q2_K. All failed attempts and reviews remain preserved. Frozen candidate58 passes matched serving59/62; controls60/61 use frozen44. All four have the same output IDs,3072 initial slots,5364121600 resident payload bytes,128MiB staging,431 waves and689 chunks. Candidate shares47 root inputs. Actual scratch peak decreases123411456->100288512bytes, but reserved pool capacity and sampled peak VRAM remain unchanged. Candidate mean826.75 versus control795.60 tok/s is observational: large preparation variance and almost identical main replay around1.17s do not establish an isolated compute gain.

Earlier profile initialization needs separate design: source-mode llama_context::initialize_moe_profile is intentionally a no-op after stored configuration, and the legacy initialize_placement endpoint uses ordinary-domain down-group lookup. Calling that endpoint unconditionally at startup would not establish model-independent source placement. Do not trade generic support for moving benchmark costs outside the request timer.

## Separate profile timing and remaining gap

Diagnostic63 passes with complete teardown. It is excluded from the unprofiled ABBA because it enables verbose preparation diagnostics. Main preparation takes709.20ms, of which306.34ms is prepare_profiles plus apply_profiles; compact planning adds85.54ms during replay, not within the preparation accounting. The remaining preparation is not yet fully attributed. The one-row tail preparation is126.03ms with4.64ms profile work. A subsequent generation program reports1.69ms profile preparation/application. Main profile copies5224857600 bytes and tail139264000 bytes, preserving the initial total5364121600.

Main replay is1167.43ms. The frozen reference reports1159.9ms prompt time, excluding its0.5s startup profile fill. These similar times do not prove execution parity: generic main replay excludes tail and other request preparation, and the reference lends cache storage so live prefill resources differ. The next lifecycle change must preserve a matched cold-start total alongside startup and replay reporting, and must use the canonical source owner instead of the ordinary-domain placement initializer. See external shared-root-input-diagnostic-63/ANALYSIS.json and SHARED-ROOT-INPUT-ABBA-REPORT.json. No current jobs remain active. The1700 end-to-end goal remains open.

## Once-per-body routing plan, qualified component evidence

Candidate65 reuses the first plan inside an eligible all-GPU body. It requires the same original IDs tensor, expert/row/route geometry, IDs stride, canonical owner, residency token and map. It copies first-seen order, route weights, counts and classification into existing per-projection storage; per-projection counters, source-access hooks and deduplicated online observations still run through the existing path. It neither replaces route ordering with the previously rejected dense route-index candidate nor adds a residency authority. Other paths keep independent planning.

Build64 passes216 source and96 owner numerical cases. Retained outputs, short-tail, mixed/fused layouts and CPU-enabled projection controls pass. Existing body fixtures confirm plan reuse, and matched frozen serving confirms94 shared plans across47 bodies. The unprofiled ABBA67-70 preserves initial payload,128MiB staging,431 waves,689 chunks, scratch peak100288512bytes, sampled peak12211MiB and the same four output IDs/hash. Planning averages86.80->44.86ms; main replay1170.05->1127.81ms. Aggregate prefill847.11->849.37 tok/s is unchanged within cold preparation variance; candidate preparation746.24ms versus control681.63ms. This qualifies removal of repeated replay work, not the1700 goal.

Diagnostic66 separates cumulative allocation phases: private graph storage23.24ms; layer/operation preparation reaches93.16ms; owner/storage preparation reaches124.69ms; profile preparation/application314.99ms, reaching440.12ms including handoff; capture/resource preflight231.19ms, reaching671.31ms. Compact planning is replay time, not part of allocation time. Reference seeding happens before READY, while current source seeding remains inside allocation.

In preserved trace53, the main capture interval spans652.875-907.488ms. It records1771 cudaLaunchKernel,1620 cudaLaunchKernelExC and20 cuLibraryLoadData calls. Recorded runtime/driver API intervals overlap and do not explain all CPU time; no exact causal decomposition follows. Source capture_prefill_resources still measures all original projections, including down at full route width, whereas complete-body replay uses compact down inputs. Resource admission must follow actual consumers before removing any measurement: keep capture for opaque upstream operators, checked mapped-MMQ bounds, library initialization requests, retained outputs, pool growth rejection and canonical leases.

Module-loading diagnostic71 changes only LAZY->EAGER on frozen65. It passes numerical continuation and teardown but is rejected as a setting change: capture falls231.19->115.88ms while cumulative owner-ready time grows124.69->909.71ms; sampled peak12211->12899MiB and diagnostic throughput661.04 tok/s. One diagnostic per setting is attribution evidence, not a performance qualification. Lazy loading remains. All jobs64-71 are terminal. No commit, push or default promotion.

## Fixed-route mapped addressing component252

252 passes all9 compile/component commands under ordered locks, with immutable frozen244 binaries/source headers and an empty final process tree. Real bounded slices preserve Flash Next's original IQ3_S gate shape2560x640 and Q2_0 down shape640x2560. Synthetic fixed uniform/skewed routes test16/64 experts, identity/reversed physical layouts and two source planes. Input quantization and routing are prepared once through original upstream helpers, outside timing. Both existing explicit MMQ instantiations use identical row maxima, tile arguments, weights, activations and output scatter. No new arithmetic kernel, production source edit or copied private structure is involved.

Mapped outputs equal contiguous outputs exactly in all16 cases. Contiguous results versus the original ordinary-MMQ oracle retain maximum relative MSE1.85434455769e-14. Across tested cases, mapped GPU time is15.55%-96.45% longer. Examples at64 experts/uniform routes: IQ3_S identity contiguous0.1734ms versus mapped0.2581ms; Q2_0 identity0.1275ms versus0.2504ms. This is measured addressing-path overhead at controlled component geometry. It is not proof that repacking will recover that saving in serving; copying, grouping, guards, allocation and live residency remain costs. Full evidence: mapped-contiguous-component-252/ANALYSIS.json.

253's temporary copy+contiguous feasibility passes exact packet bytes/outputs but is slower than mapped in every case. Its copy used one block per expert, unlike the existing canonical/Strata word-parallel copy helpers.254 corrects that decomposition and includes backend allocation-guard clearing in measured packing. Preserve253 as an unsuccessful implementation, rather than presenting contiguous-only times as achievable packet savings. No packet path is implemented in production.

Read-only idle-gap attribution251 reuses212's original trace: the12 listed gaps sum338.40ms of398.37ms without recorded CUDA activity. Captured CPU API union inside those gaps is92.25ms, including26.14ms cudaMallocHost in the68.83ms initial gap and roughly24ms graph instantiation across two late gaps. These observations support lifetime investigation but do not attribute the remaining time to an exact function or prove it removable. No raw trace or primary paired report was overwritten.

## Packing and pointer-mode feasibility254-256

Corrected254 passes all9 commands, exact raw packet bytes and exact output values in all16 cases, immutable sources and empty teardown. Word-parallel copy fixes253's underfilled grid and clears the backend's allocation tail when present. Copy+contiguous execution is9.66%-15.86% faster than mapped for the tested16-expert Q2_0 cases, but1.81%-47.14% slower in the other tested cases. This rejects a blanket packet path on this evidence. The temporary copy has aligned-only feasibility scope and is not a production generic implementation. No actual payload packing was added to owned production source.

255 reuses the existing expert-pointer mode in the original mapped kernel, with pinned pointer-table H2D included per measured iteration. It passes exact outputs and all9 commands/source/teardown gates. Pointer mode is0.06%-7.45% slower than the existing slot map in these cases. No pointer-table fast path is adopted. The addressing path is a measured difference, but pointer substitution or simple packet copying does not deliver a general speed fix.

CPU resource inventory256 reads the immutable244 CUDA library through cuobjdump. Matched type21/42, J32/J128, nonfallback/Q8/single-bank variants show LOCAL:0. Registers at J32 are196/196 for mapped/direct IQ3_S and204/200 for Q2_0; J128 both255. Q2_0 stack bytes differ48/16 at J128. These are compiled resource facts, not transaction counts or proof that register spill or occupancy explains serving. Do not revive an unproven occupancy claim.

Rejected body/root grouping prototypes are restored out of owned core source from the exact pre235/frozen222 control. Qualification257 passes original10 commands,224 original-oracle numerical replays and16 exact cases. All14 source files and ten mutable binaries match222; tree empty. No model run, packet implementation, commit, push or promotion is queued. The current throughput goal remains open.

## Completion-frontier experiment258-265

Existing-stream deferral passes448 original-oracle numerical replays (unchanged maximum relative MSE1.60269249e-7),288 exact no-hook repeats including view readers/tails, and the original10 lifecycle/path gates. Same-binary ABBA261-264 nevertheless gives repeated1442.488 ->1429.921 tok/s (-0.871%), with all eight64-ID outputs exact, equal reported main routed/paid-byte work, equal body/pool resources and12363MiB sampled peak.47 body waits per main prefill were actually deferred. Cold1014.098 ->1107.868 is an observation with large control variability, not a qualified general gain. Reject the scheduling option as a speed fix.265 restores exact source14/binary10 baseline222; frozen260 and raw logs preserve the experiment. This rules out this localized body barrier as the large missing improvement; it does not rule out cross-layer issuance or graph preparation/lifetime improvements.

The saved212 SQLite has no sampling/backtrace/callchain/OSRT tables, so it cannot provide the remaining host function attribution. CPU-only266 validates local user-clock DWARF recording and bounded own-child attachment with successful offline stack extraction. Debuginfod is disabled for local analysis; the earlier network-enabled10s report timeout and asynchronous-stop probe are preserved.267 is a bounded diagnostic of unchanged frozen222, attaching only to the owned model server after READY and recording monotonic request boundaries. Profiled rates are not unprofiled speed evidence.

## Indexed reader closure retained272

CPU diagnostic267 locates approximately100.50ms cold/80.40ms repeated active user CPU in public closed_cut, which rescans every original graph reader despite an existing equivalent index.272 delegates to existing impl::closed_cut after unchanged argument validation; no kernel, arithmetic, owner or graph-cache change.268 passes original10 gates/224 numerical replays at unchanged maximum relative MSE1.60269249e-7, and271 passes1700 independent original-edge differential cases. All CUDA/server arithmetic binaries are identical to222.

ABBA273-276 observes repeated1442.887 ->1522.394 tok/s (+5.510%), first-token1.423291 ->1.348958s and all eight exact64-ID outputs. First three arms match source/body work and12363MiB peak; last candidate has small adaptive variation (942 selected routes/15744000 paid bytes, waves+3/chunks-5, peak+2MiB). Fixed workspaces are equal. Cold +2.46% is variable. Main repeated preparation64.94 ->41.59ms; main replay1086.81 ->1087.39ms is unchanged. Retain the small indexed-reader simplification with these bounded observations, not a universal speed/no-regression or1700 parity claim. Current source/build matches frozen272; all jobs are terminal/tree-empty.
