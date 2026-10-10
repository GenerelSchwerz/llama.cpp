# Generic all-GPU hybrid prefill gap

Owner goal: diagnose and fix approximately1700 specialized versus1000-1200 generic prompt tokens/s at matched settings. This replaces the worker-wait research goal. Work in the existing owned generic feature checkout; preserve the optimization owner checkout and clean specialized control.

## Verified baseline

At the initial inventory, owned generic feature HEAD cbf59b5c6e3ab13fa06f7f9c8c4a160eaa57885a was clean, based on upstream-integrated98a3f3194. Its build has since been regenerated under the ordered locks. Current retained source change is the qualified direct-copy coalescing candidate; the rejected route-index addition has been removed. Other efficiency worktree has ongoing uncommitted MTP scheduling work and is read-only. Specialized control is separate and unchanged.

Saved 2049-input/4-output comparison:1699.57 versus1096.54 prompt tokens/s, four matching output IDs, GPU-prefill generic, CPU sharing off. Reference: /home/gencoolpc/moe-cache-tests/results/strata-specialized-update-20261008/REPORT.md. Historical profiling uses the earlier default prefill; later mandatory source candidate17 measured1026.865 on8194 input with zeroCPU prefill work. Do not conflate these paths or claim old traces identify a current caller.

## Source differences and next proof

Specialized walks bounded expert groups through gate/up, activation and down before moving on; whole expert data is staged once and used in that group. Generic source replay treats each routed projection as a separate prepared layer: GPU prelude, ID readback plus host plan, bounded per-bank waves, host completion wait. This repeats preparation/route handling and has broader materialized intermediates. Kernel differences also remain: archived quantized matmul477 versus248ms, recurrence117 versus34ms; overlapping sums are not additive wall savings.

## Current mandatory-path evidence

The ID readback hypothesis was tested and rejected. The isolated fixture passed320 checked calls; pinned same-pitch2D completed in56us at2048 rows, versus585us for the wider linear alternative. In the archived trace,887.37ms of909.69ms2D API intervals overlap GPU activity; only22.32ms has no traced GPU activity. Do not attribute those long API intervals to copy setup alone.

A fresh frozen18c mandatory-source trace completed at1157.80 prompt tokens/s for8194 input and4 output tokens, CPU prefill off, ubatch2048,64 initial slots/layer, occurrence adaptation. All1547 frozen-source hashes matched this checkout before the candidate edit; all25 binary closure hashes remained unchanged. Five source-prefill programs report zero CPU calls, zero old-window/emitter calls and zero failures. Output IDs were1596,1144,4087,1156. Raw evidence: results/generic-prefill-gap-20261009/mandatory-prefill-trace-18c-02 in the external test repository. The first trace setup failed strict request validation before a model was launched and is preserved separately.

The fresh whole-request trace contains234837cudaMemcpyAsync API calls (442.77ms summed API time), including231976H2D copies. Bounded prefill reports708 projection operations,3792 waves,126106470400 copied payload bytes and134217728 staging bytes. Quantized matmul sums2065.85ms and H2D activity2642.34ms. These overlap; they are not additive or guaranteed recoverable wall time. Traced speed is not an unprofiled serving claim.

## First bounded candidate

The existing bounded-prefill copy loop submits each missing expert independently even when expert IDs, original payloads and staging destinations are consecutive. Combine those consecutive ranges only for already registered sources with a device alias. The existing registered-copy helper validates and segments the entire range. Preserve gaps, expert maps, arithmetic, bank order, two staging lanes, events and byte accounting. Leave pageable copies per expert, since their transport grows pinned scratch to the submitted size; combining those would change the host staging budget.

This is a local copy-loop change in the canonical owner, not a new execution path. It applies through tensor strides and registration capabilities without model names or expert-count assumptions. Other optimization checkout prefill functions were inspected read-only: bounded-prefill execution is identical; replay differs only in its graph-lifetime API. Existing bounded-prefill, paired-source and mandatory-source numerical CUDA fixtures are the first qualification gates. Then compare matched unprofiled controls and current copy activity. The candidate is unqualified until those pass; removing copies alone is not evidence that the1700 target is reached.

No ordinary-prefill fallback, independent residency owner, frozen specialized edits, commits, pushes or default promotion. Full target remains model-independent hybrid performance and safety, not merely a passing microprobe.

## Copy candidate qualification and correction

The registered-only candidate passed bounded-format/pageable tests, paired-source/lifetime tests and40 mandatory-source numerical replays. Its matched Nsight capture retained exactly234837cudaMemcpyAsync calls,231976H2D copies and132076056628H2D bytes, with the same output IDs. It did not affect this workload. Traced throughput was1107.81 versus1157.80; no performance gain is claimed.

The active source path also uses the existing null-catalog direct-copy branch in copy_staged_source. That branch already calls cudaMemcpyAsync without fork staging. The next candidate allows contiguous ranges through both existing direct branches (no catalog, or catalog device alias), while leaving catalog pageable-source transport at one expert per call. A missing catalog for a budget-managed buffer is still rejected by the original helper. No new pointer or registration ownership is introduced. Build and all three numerical suites are required again before measurement.

Fixture correction: the main source-prefill rows2 test intentionally waits for CPU_ADMITTED from its GPU_ENQUEUED hook. Running it with CPU_PREFILL=0 caused the same finite5s failure in candidate and frozen control. Use its required CPU-enabled overlap configuration for that existing numerical fixture; use CPU_PREFILL=0 and zeroCPU counters for actual all-GPU performance. Preserve both failed attempts and the source-backed review in external evidence. All attempt trees were empty after teardown.

## Direct canonical comparison requested by owner

Canonical local Strata HEADfb58e0dbc8399662c0e47c76578c6e878b14f6cf and clean specialized port HEAD84d0c49309f798476c4eeeae0be786af7d28a584 were inspected directly. The following is source evidence, not a new matched cross-engine benchmark.

| Contract | Canonical Strata MMQ prefill | Current generic source prefill |
| --- | --- | --- |
| Route counts | Dense expert-indexed counts, then expert/token offsets, once per MoE layer (src/prefill/prefill.cpp:3063,3148,3236) | First-seen route mapping, separately per routed projection. The tested dense route-index candidate reduced planning cost but lost end-to-end throughput and was removed. |
| Weight payload | Whole-expert blob containing its matrices, native pack offsets (3556-3583) | Original GGUF tensor banks and strides; no repacking. Direct contiguous ranges now share a copy; catalog pageable transport retains its per-expert staging budget. |
| Compute units | MMQ_GROUP=16; gather group, combined gate/up, activation, quantized hidden, down before next group (3630-3674) | Each projection walks bounded waves; separate upstream arithmetic and intermediate tensors. Existing paired fusion remains disabled after historical regressions; do not promote it from source resemblance alone. |
| Transfer lookahead | For long chunks, known nonresident expert sequence across layers; issuer fills ring ahead of routing/compute (2322-2376). Reads no future routing results. It may transfer experts later unused. | Missing experts staged after current projection IDs are read back and planned. Two64MiB lanes overlap copy and compute within a projection; no equivalent long-prompt cross-layer stream-all queue. |
| Synchronization | Group-level copy-ready/release events; input quantization and route grouping shared across the expert body. | Host completion waits and separate route preparation at every routed projection. |
| Resources | Ring slots depend on byte budget, maximum blob and pack capabilities (241-291); additional group buffers and optional cache lending. Do not claim identical staging/peak VRAM from identical initial resident payload alone. |128MiB canonical prefill staging, fixed equal initial resident payload in current controls. |
| Optional fused experts | Native fused IQ path requires STRATA_PF_FUSED=1; the saved native MMQ trace has copy16_group_kernel and no fused-expert kernels. | Full fused-kernel port is outside this focused non-fusion work; it does not explain the saved1700 baseline. |

The stream-all mode is enabled by a chunk threshold in the specialized engine (default1024); its producer queues known nonresident weight blobs before their routing is needed. This is distinct from profile-based expert prediction. The threshold,16-expert groups, model widths and512-expert geometry must not become generic support predicates.

The first unprofiled copy-coalescing ABBA completes with identical8194 prompt tokens,4 output IDs, zeroCPU prefill and clean teardown: control1230.26/1245.93 (mean1238.09), candidate1236.59/1277.93 (mean1257.26), observational+1.55%. Copy API count separately falls234837->75858 at identical132076056628H2D bytes. This is a limited local improvement, not the missing complete-expert pipeline or a1700 claim.

The route-index candidate builds and passes bounded/pageable, paired-owner and40 mandatory-source numerical replays. Its completed isolated unprofiled ABBA uses the qualified coalescing candidate as control: control1272.61/1262.67 (mean1267.64), candidate1205.04/1239.74 (mean1222.39), a3.57% decline. Planning time drops from approximately366ms to128ms, but total prompt time increases. All outputs match and all guarded trees are empty. The cause of the end-to-end decline is unproven; lower planning time does not establish an execution gain. The route-index source addition was removed, while its frozen candidate and results remain unchanged in ROUTE-INDEX-ABBA-REPORT.json. The local build still contains that rejected addition until the next build; use frozen candidate-direct-copy-02 for the retained control. Once-per-layer grouping and stream-ahead remain missing.

## Next architectural work to make reviewable

Upstream GGML graph dependencies, MUL_MAT_ID dispatch and ordinary arithmetic remain the contracts. The current source scheduler cuts every routed projection into a separate prepared region, so its replay interface cannot keep a complete expert group's activation and down work between weight copies. Derive legal expert-body groups from shared route/activation identities and graph use counts, preserve externally read intermediates, and reuse original operator arithmetic and one canonical owner. Any extension must retain first-class projection execution within hybrid for bodies that cannot be segmented safely; required hybrid must never silently enter the ordinary executor.

Before changing that interface, prepare a concrete dependency/lifetime proposal for grouped row ordering, original-output publication, source leases, staging events, cancellation, CPU sharing, MTP, staggered requests and per-device ownership. Reuse existing row-map and prepared-operation contracts where possible. Avoid copying specialized model structures, private kernels or a second scheduler/residency owner. The older goal already authorizes generalized complete-body wave scheduling, but historical paired-fusion regressions and current optimizer ownership must be respected.

First separate preparation/host-plan savings from group execution and stream-ahead in matched controls. Explicitly measure transient staging and peak VRAM as well as resident bytes. A larger ring is a budget change, not proof of a better same-resource implementation. Continue toward the1700 target; neither these microchanges nor the comparison close the goal.

The next dependency/lifetime proposal and completed bounded arithmetic evidence are in [EXPERT-BODY-WAVES-20261010.md](EXPERT-BODY-WAVES-20261010.md). Existing compact-body metadata and upstream arithmetic pass240 exact CPU cases and240 exact CUDA cases when the original mapped-MMQ consumer is preserved. Automatic compact dispatch instead changes small groups to MMVQ and fails strict quantized equivalence. Keep consumer identity during production integration. No body-wave owner integration or new serving performance claim follows from these probes.
