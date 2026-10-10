# Product groups inside owner waves

## Source evidence and smallest extension

Strata prefill.cpp:3639-3676 runs gate/up, activation, hidden quantization and down for each16-expert product group. The owned core currently launches roots for an entire ready owner wave before running compact down chunks (moe-source-core.cu:2877-2927). Wave size follows original bank bytes and staging budget, so it is not the same product geometry. Fresh212/213 traces show generic mapped expert-MMQ399.63ms versus specialized private MMQ249.14ms in repeated first-token windows. Their basic Blackwell tile/register settings match; differing product geometry remains a concrete hypothesis, not a proven cause.

The original launch_range interface (mmq.cu:1002-1036) already supports checked expert subranges, source-map offsets and row maxima. Extend only the existing ready-wave consumer: partition its expert interval, compute the subgroup maximum, launch original roots for that interval, and execute compact chunks for its rows. Reuse original prepared root inputs, output scatter/gathers, sorted down metadata, bank pointers and upstream kernels. Do not change canonical staging, transfer bytes, lane release events, resident maps or adaptation. Finish every subgroup before returning the wave to its owner.

Use a temporary opt-in GGML_MOE_SOURCE_PREFILL_PRODUCT_EXPERTS bound to measure geometry. Zero or unset retains the complete wave; positive values are clamped to the graph's expert count. This tunes product grouping only and must not define model support. Reject malformed values explicitly. No production default or fixed specialized group width is selected by this experiment.

## Resource and lifetime proof

Groups partition the original expert interval. Empty groups issue no products. Every routed row belongs to exactly one group and compact chunk; roots retain original output indexing, so external readers remain unchanged. The owner releases a lane only after all subgroups have been submitted on the same original stream. No additional host/device allocation or synchronization event is introduced.

The existing append-only down-bounds capacity remains valid: unique expert segments contribute at most E entries; splitting an expert at a compact boundary and each terminal bound contribute at most2 per chunk. Chunk count is at most ceil(R/C)+G-1, with nonempty group count G<=E. Thus3E+2ceil(R/C) remains an upper bound, including sparse expert ranges. Keep original runtime capacity checks and cancellation checks. Exact group maxima cannot exceed the original wave maximum.

## Design decision and qualification

This is a localized use of the existing source wave callback and original MMQ subrange interface. It adds no executor, residency owner, copied kernel or scheduling subsystem, so it stays within the owner's authorized generalized hybrid pipeline and existing-pattern changes. AGENTS.md's large-pattern decision remains applicable if later work requires a different mechanism. The code-review scope is general, security and ggml/backend: inspect range arithmetic, table capacity, zero-row groups, source-map indexing, public readers and lane lifetime.

First run an independent CPU partition/coverage oracle. Then build with plainj18 under the ordered locks and run the original numerical, independent exact/rejection, direct/view-reader, skewed-tail, overlap and owner gates with a small bound that actually subdivides their graphs. Freeze qualified artifacts before matched model comparison against222, unchanged all-GPU settings, source bytes,64 outputs and actual resources. A fixture gain or changed output does not qualify the full prefill goal. No commit, push or promotion.

## Initial qualification

Independent CPU oracle235 passes149544 arbitrary-partition cases,366840 nonempty groups and526901 compact chunks. Build236 passes all11 commands,280 main-source numerical replays with unchanged original-oracle maximum relative MSE1.60269249e-7,16 independent exact sorted-route/rejection cases, original readers, tails, overlap and owner gates. Product diagnostics exercise bounds1,2 and16 (the default full graph expert count), including real subdivision. Source hashes remain unchanged and guard tree is empty. Frozen237 contains the coherent ten binaries and source manifest. Performance is not qualified.238-241 is a prepared same-binary ABBA: group16,whole-wave0,whole-wave0,group16; two uncached2049/64 requests per process, no MTP, unchanged all-GPU settings. Do not launch another arm until the prior authoritative final record proves terminal/tree-empty.

## Root-only geometry isolation

First pair238/239 is terminal/passed/tree-empty with exact64-token outputs. Group16 repeated1357.80 versus whole-wave1441.64 tok/s; cold929.94 versus1133.46 is additionally affected by preparation variability. Group16 main compact chunks1709 versus678 control, while private workspace37124736bytes and pool capacity384494592bytes remain unchanged. Actual paid bytes/routes differ slightly, so this pair does not establish matched-route component performance. Reverse240/241 is complete: full ABBA repeated1351.51 versus1435.41 tok/s (-5.85%). Reject complete-body group16 as a speed fix. Exact8 outputs/source/path/teardown pass; actual internal paid work differs slightly.

A bounded independent root group limit is the next isolation: subdivide only original root MMQ launches within each body group, then run existing compact body chunks unchanged. Keep root quantization shared once per layer. An opt-in GGML_MOE_SOURCE_PREFILL_ROOT_EXPERTS limit lets product limit0/root limit16 retain original down chunking. Zero leaves the body group unchanged. This uses the same checked launch_range contract and introduces no kernel, owner, copy, event or allocation. Body group maxima and sorted down bounds remain original; root subgroups compute their own exact maxima. Existing CPU coverage proof applies to root subranges, and the unchanged body grouping retains its bounds capacity. This is an experimental isolation, not a second production tuning policy or fixed model geometry.

Parse both temporary numeric limits through the existing bounded-decimal pattern once per session. Validate original numerical/readers/tails/default gates plus root-only subdivision and a combined small-body/smaller-root case. Freeze a new source/build closure before testing. Do not attribute any result to fewer source bytes; compare actual source accounting and preserve mismatches explicitly. Existing full-body group237 and baseline222 remain immutable controls.

## Completed root-only isolation

Build243 passes12 commands,336 main numerical replays at unchanged maximum relative MSE1.60269249e-7 and16 independent exact/rejection cases, including root-only subdivision and combined small groups. Frozen244 is coherent. Same-binary ABBA245-248 is terminal/passed/tree-empty: root16 repeated1322.70 versus whole-wave1437.75 tok/s (-8.00%); cold1041.33 versus1141.46. Root groups3374 versus848 across two requests; body chunks686/673 versus678/676. All eight64-token output sequences match, but internal selected routes/bytes differ, so this is not fixed-route evidence. Private workspace37124736bytes, pool capacity384494592bytes and peak100288512bytes stay unchanged. Reject root16 as a speed fix, keep defaults0, and do not publish either temporary tuning knob. External ROOT-GROUPS-ABBA-REPORT.json records complete accounting. Next250 uses existing mapped/unmapped interfaces to isolate fixed-route memory addressing without production edits.

## Reversion

The rejected grouping code is removed by restoring only moe-source-core.cu from the exact pre235/frozen222 source backup; all other owned edits are preserved. Frozen237/244 and all negative experiments remain intact outside the repository.257 passes all original10 build/numerical/owner/reader commands,224 main replays at unchanged maximum relative MSE1.60269249e-7 and16 exact sorted cases. All14 source and ten mutable binary hashes match222; terminal tree empty. Temporary grouping controls are absent from restored production source.
