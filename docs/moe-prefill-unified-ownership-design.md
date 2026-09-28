# Canonical MoE prefill and decode ownership

Status: registered CUDA prefill and decode ownership now use one grouped residency map, 2026-09-27. B2 mapped-MMQ is functionally exercised for the current RCO slice. Registration-aware direct and pageable weight transport is implemented for B1/B2. Detached staging can borrow warm exact pageable original-ID auxiliary shadows. Format-specific acceptance, cold detached auxiliary materialization, physical multi-GPU coverage, and repeated performance acceptance remain open. Base: `50df2c7cf761a4647bd2555cb4e0f0fabdaeef11`. Branch: `design/prefill-unified-ownership-multigpu-20260927`.

Current implementation status:

- Certified pure target prefill can use the existing group resource and reciprocal residency map in DIRECT execution. Original prompt-row IDs still drive the existing prefill arithmetic; compact IDs are ownership bookkeeping only.
- Ordinary target ubatches that prove contiguous per-sequence spans carry a non-required MAIN/SEQUENTIAL execution certificate through scheduler splitting. This keeps output-pruned final-layer rows under their source prompt semantics; the earlier MAIN/INDEPENDENT branch still takes precedence for ordinary parallel decode.
- Decode keeps its stricter nonzero semantic-key and route validation. A valid MAIN/INDEPENDENT source certificate admits the full row count or a one-row output projection only. Malformed certificates and unproven reductions with more than one remaining row stay fail-closed. The shared private residency helper does not promote sequential prompt rows to independent decode rows.
- Mapped MMQ prefill prepares routes and quantized activations once, preserves all configured resident slots, and processes expert ranges through two owner-level 64 MiB payload lanes. Compact MMVQ and unsupported consumers retain the B1 path. `GGML_CUDA_MOE_PREFILL_BOUNDED=0` selects that control path in the same binary.
- Registered plans no longer acquire a per-bank cache or change authority for fallback execution. `PREFILL_STAGED` and `DECODE_STAGED` use detached scratch while the canonical grouped map remains unchanged. The old host-staged authority, grouped-to-legacy handoff, dirty-map reconciliation, and phase reset were removed.
- Detached staging holds source admission for the whole graph, records completion on every stream that submitted staged work, and drains those events before candidate replacement or shutdown. It stays outside CUDA graph capture.
- B1/B2 source copies resolve the registered tensor record before enqueue. Fully registered intervals are split only at registration boundaries; pageable or partially registered intervals use one bounded owner-level two-tile pinned transport. Tile completion is separate from the B2 device-lane release events.
- Grouped execution shadows pageable original-ID scales and biases in the group resource. Detached plans borrow a strong lease to a descriptor-matched warm shadow through completion on every tracked consumer stream. A cold detached plan without an existing exact shadow still rejects the unreadable auxiliary before graph enqueue.
- Generic per-bank cached MMID remains only for inactive or unregistered sources. Active registered sources are rejected at lookup, construction, and installation. This compatibility path is outside the registered ownership claim.
- The full existing `test-moe-cache` target passes, including a 128-expert multiwave prefill fixture with repeated lane reuse, the existing MMID relative squared error limit of `2e-5`, and replacement blocking until an admitted staged dispatch records and finishes its stream.
- Prefill remains DIRECT and invalidates retained decode capture conservatively. Bounded staging is not captured. Partial enqueue failures drain both copy and compute streams before the arena can be reused; an undrainable arena is disabled.
- The final Qwen3.6 RCO single, simultaneous, staggered no-MTP soak, and MTP runs completed with zero staged fallback, grouped fallback, handoff reset, rollback, prepare error, or finish error. Performance results remain single observations.

Current implementation checklist:

- [x] One registered residency and ownership map across target prefill and decode.
- [x] Bounded two-lane mapped-MMQ prefill staging with a B1 compatibility arithmetic path.
- [x] No registered per-bank cache borrowing, dirty handoff, or phase-induced residency clearing.
- [x] Dispatch-scoped detached source lifetime through CUDA completion and replacement/shutdown drain.
- [x] Pre-enqueue rejection for detached unreadable auxiliaries without an exact warm shadow.
- [x] Registration-aware B1/B2 weight copies for split registration and pageable sources under the existing host budget.
- [x] Reuse the existing exact pageable auxiliary shadow from detached `PREFILL_STAGED` and `DECODE_STAGED`.
- [ ] Materialize a cold detached auxiliary shadow only if a supported model requires it and the existing group resource cannot be warmed first.
- [ ] Combine pageable or split registration with three or more tile uses, delayed replacement/shutdown, and injected copy/event failures.
- [x] Current Qwen 35B single, simultaneous, staggered, no-MTP soak, and MTP validation.
- [ ] Physical multi-GPU, backend matrix, and repeated performance acceptance.
- [ ] Remove generic unregistered cached-MMID only after every remaining caller has a tested replacement or explicit unsupported result.

`SEQUENTIAL` describes row semantics, not phase scheduling. A sequential prompt ubatch contains successive token positions from its source sequences. With continuous batching, `tools/server/server-context.cpp` first adds generated rows from active slots, then may append pending prompt rows to the same server batch. `llama_decode_ext` submits that batch as one target graph; graph nodes and backend splits still execute in dependency order. B1 therefore allows prefill and decode requests to share an admission cycle and one ownership map. It does not run two target graphs concurrently on separate CUDA streams. Stream-level phase overlap, a fixed prefill slot subset, scheduler fairness, and dynamic VRAM lending remain separate measured designs.

## 1. Two independent hypotheses

| Hypothesis | Proposed change | Evidence needed | What success does not establish |
| --- | --- | --- | --- |
| A: warm handoff | Keep legacy prefill. At `LEGACY -> GROUPED`, preserve only residents whose expert and physical slot agree across every required bank. | Correct output, fewer actual H2D fills/bytes, and acceptable measured transition/prefill/decode cost. | One map during prefill, bounded staging, parallel decode improvement, or permission to delete legacy dispatch. |
| B: canonical ownership | Target prefill and decode use the same group resource, reciprocal residency map, transaction rules, and completion ordering. Add a bounded staging path for requests larger than residency capacity. | No duplicate resident ownership in the claimed path, safe replay and rollback, an enforced staging cap, and measured benefit for concurrent requests or validated decode-path retirement. | Faster execution merely because resets, H2D fills, or legacy counters decline. |

This document specifies B. The separate A plan is `/home/gencoolpc/llama-moe-prefill-grouped-ownership/docs/moe-prefill-warm-handoff-design.md`. B can be developed and tested without adopting A. If A is retained temporarily, remove its reconciliation for groups migrated to B; do not leave two authorities behind a common API.

Owner constraints:

- Keep the configured resident capacity as the runtime bound. Current Qwen3.6 RCO validation uses 128 slots per group; the earlier Qwen3.8 validation uses 64. No additional full-expert allocation is implied or authorized by this design.
- The current fork's prefill is already fast; the owner reports that it is faster than FreeToken. FreeToken is an optional source of ideas, not the performance target or a required architecture.
- Existing prefill kernels and arithmetic may change when needed, but every such change requires matched numerical and performance validation.
- A small measured prefill regression is acceptable only with a concrete unified-ownership benefit, validated progress toward legacy-decode retirement, or measured parallel-decode benefit. A bridge that only reduces copies does not satisfy B.
- Source changes, commits, and publication are outside this documentation task. Reuse existing test files; a new test file needs separate approval under `AGENTS.md`.

## 2. What the base already does

| Evidence | Consequence for B |
| --- | --- |
| `ggml/src/ggml-cuda/moe-cache.cu:5597-5635` stores reciprocal maps, recency/frequency state, a pending plan, completion event, and one active transaction in the group resource. | Extend this owner instead of introducing a second resident-cache service. |
| `moe-cache.cu:9553-9578` lends grouped bank allocations to legacy caches; `13984-13995` gives each bank an independent host map and recency state. `9366-9368` marks the group dirty when such a lease is acquired. | Shared payload addresses alone do not establish shared ownership. |
| `moe-cache.cu:7373-7434` clears grouped maps, counters, auxiliaries, and pending plan on dirty handoff. | B must remove this reset for supported phase changes while retaining genuine source/generation invalidation. |
| `moe-cache.cu:3954-3990` commits the previous READY admission at the next planner invocation. | A prefill writer must resolve that pending admission before inspecting or overwriting residency. |
| `moe-cache.cu:10954-10990` creates lightweight pure-prefill records. `10993-11011` conditionally adds bindings for resident auxiliaries. | A certified full prefill transaction needs complete bindings, first/last consumers, routing proof, and stream resolution; relabeling the outcome is insufficient. |
| `ggml/src/ggml-cuda/moe-cache-graph.cu:219-241` skips stream resolution for outcomes other than `DECODE_GROUPED`. | Add explicit stream handling for the new owned-prefill outcome. A null stream must never silently become an accepted ownership path. |
| `ggml/src/ggml-cuda/ggml-cuda.cu:2899-2902` allocates expert scratch proportional to all unique routed experts. `2917-2947` adds staged readiness and resident reuse. | Existing readiness waves do not bound the expert scratch allocation. Preserve the useful reuse/overlap while adding an actual cap. |
| `moe-cache.cu:15158-15209` finds, fills, and pins legacy residents before split staging. | Scratch-only prefill loses work that contributes to current performance. |
| `moe-cache.cu:10623-10627` rejects direct plans when route count exceeds slot count. | Large prefill cannot be admitted by removing one capacity check. Routes, unique experts, persistent residents, and transient tiles need separate bounds. |

Here and below, an abbreviated `moe-cache.cu` reference means `ggml/src/ggml-cuda/moe-cache.cu`.

### Historical experiments, kept separate from B

Artifacts are under `/home/gencoolpc/moe-cache-tests/results/prefill-grouped-ownership-20260926/`. They belong to the earlier experimental checkout, not this clean design checkout.

Historical final slot `eval time` and client `decode_tok_s` values below are request-lifetime wall rates. They include stalls when another request prefills and do not establish decode execution speed. The rolling/decode-only contract in section 10 governs performance acceptance.

The owner subsequently required at least 8k model tokens of real text for the long single-prompt and staggered-prefill gates. The older 7,288-token dossier is diagnostic evidence below that gate. The new `fixtures/long-real-text-8k-plus.txt` combines real documentation and was verified at 9,370 model tokens with matched `-c 32768` controls. Parent-reported single-prompt A results matched the output hash: base/candidate prompt 1,350.64/1,355.32 tok/s, TTFT 6,952.50/6,931.00 ms, final generation wall rate 141.99/142.44 tok/s, equal 12,187 MiB sampled peak, and zero errors. The first 9,370-token staggered pair measured request-2 prompt 1,241.17/1,244.30 tok/s and TTFT 7,565.24/7,547.89 ms; final generation wall rates were 42.50/43.18 tok/s for request 1 and 115.74/117.64 for request 2, with differing hashes. These initial pairs meet the input-size requirement, not repeated-performance acceptance; the staggered execution-speed claim is unproven. File bytes or an estimated token count do not pass the gate.

- `base-prefill/` and `candidate-scratch-prefill2/`: the 7,288-token scratch-only experiment preserved the recorded output hash and removed handoff resets, but prompt throughput fell from 1,383.30 to 981.11 tok/s and TTFT rose from 5.289 to 7.442 seconds. It was rejected and removed.
- Three alternating control/bridge pairs subsequently showed approximately 1.7% lower median prefill throughput and 45-116 ms higher TTFT for A with a per-group host synchronization. This motivated batching metadata uploads and draining once per distinct stream.
- Two parent-reported alternating batched-A controls measured base/candidate prompt throughput of 1,399.98/1,405.17 and 1,392.44/1,395.81 tok/s with exact output hashes; corresponding final generation wall rates were 143.18/142.47 and 143.14/142.57 tok/s. Retain both phases' metrics. These are A observations, not B acceptance.
- Three simultaneous two-request A pairs reported average per-request final generation wall-rate increases of 2.48%, 1.06%, and 2.69%. Corresponding hashes matched in the first pair; hash multisets matched with request assignments swapped in the other two. A further distinct-prompt pair matched each request hash and measured final generation wall rates 117.16/119.90 and 117.23/120.08 tok/s, with TTFT about 453/468 ms. The companion A document owns the full artifact matrix and later repetitions. These values do not establish decode execution improvement, sustained performance, or B's ownership benefit.
- Three staggered prefill-during-decode A pairs measured request-1 final generation wall rates base/candidate 49.35/50.89, 50.20/50.83, and 49.87/51.14 tok/s, while request-2 TTFT improved in two pairs and worsened in one. Resets were 280/0, fallback/rollback were zero, and peaks were equal. Hashes generally differed across schedules. These wall/interference and fill comparisons remain observational and cannot establish decode execution speed. This evidence cannot be replaced by the simultaneous-request matrix.
- Parent-reported post-batch MTP and four-request no-MTP soaks completed cleanly for A. They do not test a future B implementation.

The first 9,370-token staggered logs also yielded proposed sums `53.55 + 106.27 = 159.82` base and `54.95 + 107.45 = 162.40` bridge tok/s. These are not valid aligned decode-only aggregate measurements. In `staggered-long8k-base/server.log:23,26,29-30`, the older slot's 53.55 window spans 10.747611-13.753933 seconds while prefill is still active at 12.127600; the newer slot's window ends at 15.209416. In `staggered-long8k-bridge/server.log:183,266,349-350`, the corresponding older window spans 10.286735-13.289443 while prefill is active at 11.661416; the newer window ends at 14.742846. A line printed after prefill can still summarize a window that straddles it. Retain the raw observations and measure suitable windows before making an aggregate execution-speed claim.

### B1 live observations

Artifacts are under `/home/gencoolpc/moe-cache-tests/results/prefill-unified-ownership-b-20260926/`. Both current runs use the RCO IQ3_XXS target, 64 expert slots, 64k context, 512 physical ubatch, NP2, no MTP, and real-text prompts of 14,627 and 14,734 model tokens where applicable.

| Workload | Prompt rates | Weighted server-window decode estimate | TTFT | Peak VRAM | Ownership counters |
| --- | --- | --- | --- | --- | --- |
| Simultaneous, `simultaneous-projection-live1/` | 374.43 and 358.47 tok/s | 59.50 aggregate; 29.76 and 29.75 tok/s | 39.09 and 69.69 s | 11,641 MiB | prefill grouped 2,775; decode grouped 49,296; all legacy, fallback, reset, rollback, and errors zero |
| Staggered by 1.5 s, `staggered-projection-live1/` | 178.33 tok/s for the 158-token request; 427.25 tok/s for the 14,734-token request | 63.45 aggregate; 31.73 and 31.73 tok/s | 0.90 and 34.51 s | 11,639 MiB | prefill grouped 1,412; decode grouped 100,032; all legacy, fallback, reset, rollback, and errors zero |

The immediately preceding B1 build still rejected Qwen4exp's final one-row FFN during four simultaneous decode steps. `simultaneous-long-1/` recorded 192 decode-legacy groups, 189 handoff resets, 60.26 aligned aggregate tok/s, and 535.2 GB H2D. The explicit one-row projection rule removed those legacy groups and resets, but `simultaneous-projection-live1/` measured 59.50 tok/s and 649.0 GB H2D. It therefore establishes cleaner ownership, not a decode speedup.

The preceding staggered observation, `staggered-long-1/`, measured 429.71 tok/s for the long prefill and 60.82 aligned aggregate decode tok/s. The projection build measured 427.25 and 63.45 respectively, with H2D decreasing from 1,195.6 to 1,114.5 GB. This single pair is consistent with an acceptable small prefill loss and better staggered decode, but repeated alternating pairs are required before either is a speed claim.

Every current request completed its configured output count and teardown was clean. Exact reasoning-output hashes differ between the preceding and projection runs for both simultaneous and staggered schedules. Those runs therefore do not establish exact cross-run output parity. Fixed-schedule numerical parity remains a separate gate before a performance acceptance claim.

`mtp-projection-live1/` used the same RCO target with the shared Q8 MTP draft, draft depth one, 64 target slots, and the draft MoE cache disabled. It completed a 158-token prompt and 1,024 output tokens at 177.36 prompt tok/s and 64.03 server-window decode tok/s, with 75.3% draft acceptance, 13,469 MiB peak VRAM, clean capture/replay and teardown, 48 grouped target-prefill groups, 28,032 grouped target-decode groups, and zero target legacy, fallback, reset, rollback, or errors. This validates target ownership with MTP attached. It does not validate a cached draft owner.

### B2 mapped-MMQ observations

The B2 artifacts share the same root. The runner now aggregates persistent `prefill_staging_bytes` by maximum rather than summing repeated shutdown snapshots.

| Workload | Prompt rates | Weighted server-window decode estimate | TTFT | Peak VRAM | Ownership and staging |
| --- | --- | --- | --- | --- | --- |
| Matched single B1 control, `b2-pair-b1off-long1/` | 422.35 tok/s for 14,627 tokens | 50.11 tok/s | 34.65 s | 11,415 MiB | bounded disabled; prefill grouped 1,364; decode grouped 24,528; all legacy/fallback/reset/errors zero |
| Matched single B2, `b2-pair-bounded-long1/` | 519.64 tok/s for 14,627 tokens | 51.62 tok/s | 28.15 s | 11,297 MiB | bounded ops 4,089 in 21,051 waves; 128 MiB arena; all legacy/fallback/reset/errors zero |
| Simultaneous B2, `b2-simultaneous-long1/` | 467.62 and 259.81 tok/s for 14,627 and 14,734 tokens | 59.58 aggregate; 29.80 and 29.78 tok/s | 38.95 and 56.73 s | 11,375 MiB | bounded ops 8,319; prefill grouped 2,775; decode grouped 49,248; all legacy/fallback/reset/errors zero |
| Staggered B2 no-MTP soak, `b2-staggered-long-soak1/` | 201.10 tok/s for 158 tokens; 527.07 tok/s for 14,734 tokens | 59.99 aggregate; 30.01 and 29.98 tok/s | 0.80 and 27.99 s | 11,381 MiB | 2,048 outputs each; bounded ops 4,230; prefill grouped 1,412; decode grouped 100,320; all legacy/fallback/reset/errors zero |
| MTP B2, `b2-mtp-live1/` | 202.23 tok/s for 158 tokens | 64.90 tok/s | 0.78 s | 13,337 MiB | 76.8% acceptance; bounded target ops 144; draft cache disabled; all target legacy/fallback/reset/errors zero |

The single matched pair estimated 23.0% higher prefill throughput, 3.0% higher weighted server-window decode throughput, and 118 MiB lower sampled peak for B2. One pair does not establish repeatable speed. B2 simultaneous decode is effectively level with the B1 observation above, while the B2 staggered aggregate estimate is 5.5% lower than the preceding B1 observation despite faster prompt processing and lower TTFT. Parallel decode improvement is therefore unproven and the two workloads cannot be substituted for each other.

The matched B1/B2 reasoning hashes differ (`b7753c...` versus `7e43d6...`), as do the mixed-request hashes. Bitwise output parity is false. Every request completed its configured output count with coherent text and clean teardown. The multiwave fixture supplies the fixed-route numerical comparison; five alternating pairs and sustained `llama-benchy` remain required for a speed acceptance claim.

### Qwen3.6-35B-A3B follow-up

Artifacts are under `/home/gencoolpc/moe-cache-tests/results/qwen35b-prefill-unified-b2-20260927/`. These runs use Qwen3.6-35B-A3B Q4_K_M, no MTP, 128 expert slots, 64k context, a 4,096 logical batch, a 512 physical ubatch, f16 KV, and the same binary. `GGML_CUDA_MOE_PREFILL_BOUNDED=0` selects B1; the default selects B2. Long prompts contain 14,585 or 14,692 model tokens of real text.

| Workload | B1 | B2 | Observed change |
| --- | --- | --- | --- |
| Single 14,692-token prompt | 1,355.91 prompt tok/s; 141.15 weighted server-window decode tok/s; 10.859 s TTFT; 12,619 MiB peak | 1,527.30 prompt tok/s; 140.58 weighted server-window decode tok/s; 9.644 s TTFT; 12,607 MiB peak | Prompt +12.6%; decode -0.4%; TTFT -11.2% |
| Simultaneous long NP2 | 1,182.38 and 1,158.53 prompt tok/s; 217.68 aggregate decode estimate; 12.366 and 21.811 s TTFT; 12,783 MiB peak | 766.38 and 1,403.84 prompt tok/s; 218.05 aggregate decode estimate; 19.057 and 12.992 s TTFT; 12,771 MiB peak | Decode +0.2%; request scheduling order and prompt-rate distribution changed |
| Staggered NP2, 2,048 outputs each | Long prompt 1,401.61 tok/s; 208.33 aggregate decode estimate; 10.511 s long-request TTFT; 12,807 MiB peak | Long prompt 1,574.91 tok/s; 212.60 aggregate decode estimate; 9.354 s long-request TTFT; 12,797 MiB peak | Prompt +12.4%; decode +2.0%; long-request TTFT -11.0% |

Every arm recorded zero `PREFILL_LEGACY`, `DECODE_LEGACY`, grouped fallback, rollback, handoff reset, prepare error, and finish error. B2 recorded a 128 MiB bounded arena and the expected bounded operations; B1 recorded none. Each request completed its configured token count and produced nonempty reasoning output. B1/B2 reasoning hashes differ, so exact cross-arm output parity is false. These are one matched pair per workload and remain observations rather than repeatable speed claims.

The first follow-up used a 512-token physical ubatch and therefore did not represent the established high-throughput prefill geometry. A second matched pair used the exact historical 64,000-token prompt with `-b 8192 -ub 8192`, NP1, 64k context, f16 KV, 128 slots, no MTP, and 512 output tokens:

| Run | Prompt throughput | TTFT | Weighted server-window decode | Peak VRAM | Output hash |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical fork reference, 132 slots | 4,345.07 tok/s derived from 64,000 tokens / 14.729 s | 14.729 s | unavailable in the old summary; request lifetime 119.27 tok/s | 14,801 MiB | `060985c9...` |
| Current B1 control, 128 slots | 3,969.23 tok/s | 16.178 s | 119.24 tok/s | 14,281 MiB | `40c71b91...` |
| Current B2, 128 slots | 4,366.24 tok/s | 14.730 s | 118.78 tok/s | 14,265 MiB | `40c71b91...` |

B2 is 10.0% faster than the matched B1 control and 0.5% above the historical derived prompt rate in this observation. B1 and B2 have exact output-hash parity in the matched current pair. B2 records 960 bounded operations in 1,872 waves, a 128 MiB arena, 360 grouped prefill groups, 20,440 grouped decode groups, and zero legacy, fallback, reset, rollback, or grouped errors. The historical run used an older binary and 132 slots, so its near-equality is a reference rather than a strict A/B result.

### Final registered-ownership validation

Artifacts are under `/home/gencoolpc/moe-cache-tests/results/qwen35b-unified-retired-ub8192-20260927/`. The final server uses Qwen3.6-35B-A3B Q4_K_M, RCO, 128 slots, 64k context, `-b 8192 -ub 8192`, f16 KV, and 14,585 or 14,692 model-token real-text prompts. Decode values are aligned server `tg_3s` windows.

| Workload | Prompt rates | Weighted server-window decode estimate | TTFT | Peak VRAM |
| --- | --- | --- | --- | --- |
| Single, 1,024 outputs | 4,445.21 tok/s | 141.71 tok/s | 3.329 s | 14,263 MiB |
| Simultaneous NP2, 1,024 outputs each | 3,237.50 and 3,809.43 tok/s | 218.29 aggregate; 109.02 and 109.27 per request | 6.504 and 3.885 s | 14,909 MiB |
| Staggered NP2 by 1.5 s, 2,048 outputs each | 142.02 tok/s for 116 tokens; 4,853.32 tok/s for 14,692 tokens | 209.39 aggregate; 104.90 and 104.49 per request | 0.824 and 3.049 s | 14,935 MiB |
| MTP depth two, 2,048 outputs | 4,171.80 tok/s | 187.15 tok/s | 3.545 s | 15,427 MiB |

Every run completed its configured output count with zero registered staged prefill/decode, grouped fallback, rollback, preparation error, finish error, and phase-handoff reset. The MTP run accepted 1,213 of 1,666 proposed tokens and reused all 832 queued overlap steps. Its draft model is dense, so the grouped MoE counters cover the target owner.

The matched 512-physical-ubatch results are under `/home/gencoolpc/moe-cache-tests/results/qwen35b-unified-retired-20260927/`. Moving to `-ub 8192` increased the single long-prompt rate from 1,515.69 to 4,445.21 tok/s while aligned decode changed from 141.38 to 141.71 tok/s. Staggered long-prompt processing increased from 1,562.58 to 4,853.32 tok/s while aligned aggregate decode changed from 208.28 to 209.39 tok/s. Simultaneous aligned decode changed from 220.52 to 218.29 tok/s. Peak VRAM increased by 1,656 MiB for single and 2,138 MiB for both NP2 workloads. Reasoning hashes differ across physical ubatch geometries, so exact cross-geometry parity is false. These single runs validate the path and expose the memory/performance tradeoff; they do not establish repeatable speed.

### Registration-aware transport follow-up

Artifacts are under `/home/gencoolpc/moe-cache-tests/results/qwen35b-unified-source-transport-ub8192-20260927/`. The source transport build uses the same Qwen3.6 RCO configuration and prompts as the final registered-ownership validation. The full existing `test-moe-cache` suite also passes, including split registration, pageable sources, graph replay, detached lifetime, and MTP-domain fixtures.

| Workload | Prompt rates | Weighted server-window decode estimate | TTFT | Peak VRAM | Output comparison |
| --- | --- | --- | --- | --- | --- |
| Single, 1,024 outputs | 4,389.41 tok/s | 141.12 tok/s | 3.369 s | 14,263 MiB | Exact reasoning hash versus the preceding build |
| Simultaneous NP2, repeat with matching schedule | 3,215.78 and 3,643.16 tok/s | 215.03 aggregate; 107.58 and 107.46 per request | 6.686 and 4.057 s | 14,909 MiB | Both reasoning hashes exact versus the preceding build |
| Staggered NP2 by 1.5 s, 2,048 outputs each | 197.33 tok/s for 116 tokens; 4,752.04 tok/s for 14,692 tokens | 207.96 aggregate; 103.95 and 104.01 per request | 0.595 and 3.113 s | 14,935 MiB | Long-request hash exact; short-request schedule/hash changed |
| MTP depth two, 2,048 outputs | 4,203.87 tok/s | 188.54 tok/s | 3.520 s | 15,427 MiB | Exact reasoning hash versus the preceding build |

Relative to the preceding single observations, single prompt/decode changed by -1.26%/-0.42%, simultaneous matched-schedule prompt rates by -0.67%/-4.36% and aggregate decode by -1.49%, staggered long-prompt/aggregate-decode by -2.09%/-0.68%, and MTP prompt/decode by +0.77%/+0.75%. A first simultaneous repeat chose a different schedule and is retained separately; its aggregate decode was 214.52 tok/s. Every run completed, recorded zero staged outcome, grouped fallback, rollback, prepare error, or finish error, and tore down cleanly. MTP retained 1,213/1,666 accepted tokens and 832/832 overlap reuse. These are regression checks and observations, not repeated performance acceptance.

### Detached pageable auxiliary follow-up

Artifacts are under `/home/gencoolpc/moe-cache-tests/results/pageable-detached-aux-20260927/`. The existing CUDA suite now checks pageable bias and NVFP4 original-ID scale shadows through both `DECODE_STAGED` and `PREFILL_STAGED`. It verifies descriptor matching, device-shadow resolution, exact copied values, a strong resource lease through tracked-stream completion, budget release at shutdown, and zero legacy calls or prepare/finish errors. The complete `test-moe-cache` target passes.

A Qwen3.6 single-request compatibility run used a 256 MiB `--moe-expert-cache-host-pinned-mb` budget with the 14,692-token real-text prompt, `-ub 8192`, 128 slots, and 512 outputs. It completed cleanly at 2,934.51 prompt tok/s, 64.16 log-derived decode tok/s, 5.031 s TTFT, and 14,229 MiB sampled peak VRAM. It recorded 17,293,049,856 bounded-prefill H2D bytes, 14,626,455,552 pageable-staged source bytes, zero staged grouped outcomes, fallback, rollback, prepare error, finish error, or legacy execution. This confirms bounded host-pinned source transport coexists with unified ownership. Qwen3.6 reports zero fixed auxiliary bytes, so the synthetic exact-value fixture, rather than this model run, is the evidence for pageable auxiliary shadows. The 256 MiB run intentionally forces extensive pageable source staging and is not a performance comparison with the fully registered control.

## 3. B's ownership contract

The unit of residency is `(backend device, candidate generation, group resource generation, expert, slot)`. The same slot index must identify the same expert in every required weight bank and every slot-bound scale/bias. Source shape, strides, type, byte extent, and materialization remain part of the proof.

The canonical reciprocal map remains in the existing group resource. A transaction may hold a transient snapshot or a route/remap description. It may not create an independently writable per-bank resident map, LRU, or eviction decision. Temporary staged expert addresses are scoped to a transaction and tile; they are not additional resident-cache entries.

Required invariants:

1. A valid resident has complete payload for every required bank and auxiliary. Publishing one completed bank is insufficient.
2. Each admitted prefill group obtains one transaction and one persistent slot decision. All bank consumers use it. Their temporary tile descriptors can differ in payload layout, but agree on expert identity and output route positions.
3. Resident hits and selected admissions are protected through their last consumers. A slot cannot be reused while any permitted reader or copy still accesses its bytes.
4. The preceding READY decode plan is committed exactly once, after its fill/compute dependencies, before prefill plans from the map. Its consumed state is cleared so a later replay cannot reinstall stale entries.
5. Phase changes preserve healthy residency and clock/frequency history. Actual source replacement, registry rejection, and generation changes keep their existing invalidation semantics.
6. Invalid IDs, broken reciprocal maps, mismatched bank geometry, unproven consumers, stale leases, and wrong streams fail closed. Required grouped certificates never silently become generic cached MMID.
7. A host lease ending means that no more work may be submitted through it. GPU completion remains represented by an event; a destructor or a new writer must honor that event.
8. Keep the existing one-writer transaction per group initially. Multiple callers do not gain permission to mutate the same device plan concurrently.

## 4. Proposed transaction and failure sequence

Use the current authority barrier and resource token machinery. `moe-cache.cu:12778-12798` already closes admissions and drains active calls, maintenance, transactions, and legacy leases. Initial support is a fully certified target-prefill group without slot auxiliaries on one CUDA device and a proven stream. Expand the certificate deliberately after that slice works.

| State | Allowed work | Exit condition |
| --- | --- | --- |
| VALIDATING | Prove complete group, routes, consumers, backing generation, source access, and budget; allocate required host/control storage. | Unsupported geometry declines before payload/output mutation. |
| OWNED | Acquire the group token and staging lease; wait on the previous group completion when crossing streams. | No competing writer or stale legacy lease exists. |
| PLANNED | Resolve the prior READY admission, derive routed experts once, pin resident hits, choose at most S persistent residents, and form bounded tile descriptors. | Every bank receives the same persistent decision. |
| FILLING | Invalidate the mappings of selected victims before their bytes are overwritten; enqueue all-bank resident fills and transient tile fills. | A ready event covers every byte a consumer will read. |
| PUBLISHED | Publish complete new persistent residents once, after their all-bank fill dependency; retain the token until final consumers are submitted. | The reciprocal map, timestamps, frequency epochs, and plan state agree. |
| CONSUMING | Compute waits on ready events; MMID consumers write their original output positions; release each lane only after its last read. | All required group consumers and auxiliary operations complete in the declared order. |
| FINISHED | Record the group completion after all consumers and joined copy dependencies; release host token/leases. | Future users can order against that completion. |
| FAILED | Stop submission, retain temporary buffers, drain every stream that may use them, invalidate overwritten/uncommitted slots, and fail the graph. | Healthy-state cleanup is complete, or the resource is retired as unusable. |

These states express dependencies; transient FILLING/CONSUMING repeats per tile. Factor the existing deferred admission commit into a reusable operation; do not copy its logic into a second host map. Decode can retain its efficient deferred commit. Prefill must explicitly resolve it at entry. Newly filled prefill residents may be published on the ordered compute stream after all required bank fills; finish clears any consumed pending record so the next decode cannot commit it again. The CPU can publish an event-backed completion contract without synchronizing the GPU on every group.

Rollback rules:

- Before any CUDA or output mutation, an unsupported certificate or allocation failure can select the existing fallback for the whole group.
- After payload overwrite, old victim metadata cannot be restored unless the old bytes were preserved. Invalidate touched slots instead. Keep unaffected residents when their validity is proven.
- After any output write, do not restart generic MMID over partial results unless an explicit whole-group replay contract first restores all outputs/intermediates. The initial implementation fails the graph instead.
- CUDA enqueue/launch/completion failure is distinct from unsupported capability. Never turn it into a successful fallback. If cleanup cannot establish healthy resource state, retire the resource and invalidate its graph witnesses.
- Drain every attempted stream even after the first error. Do not short-circuit cleanup. Keep host upload vectors, scalar sources, staging lanes, and resource references alive through the drains. A failure in a later group must not free earlier groups' asynchronous upload sources.
- Record completion after the final relevant operations. Publish host completion flags and ownership state only when their event record succeeded. On failed batch publication, do not advertise partially imported groups as clean.

The existing `finish_decode()` event/token ordering at `moe-cache.cu:10801-10817` is the model. The existing legacy handoff joins copy and compute events at `14664-14678`; B must cover the same dependencies without retaining legacy per-bank ownership for the supported path.

## 5. Bounded mapped-MMQ staging

The current B2 slice implements this contract for CUDA mapped MMQ. It keeps the B1 full unique-expert scratch as an explicit compatibility path for compact MMVQ, unsupported formats/consumers, and the disabled control. MMVQ remains a first-class grouped execution strategy when matched measurements show that it is faster for the admitted shape; unified ownership does not require one arithmetic path. Replacing B1 MMVQ requires a bounded alternative that matches correctness and does not regress prefill or decode beyond the accepted workload gates. B1 operations do not count as bounded coverage. Pageable weight transport and warm detached pageable original-ID auxiliary reuse are implemented. Other consumers, cold detached auxiliary materialization, and captured prefill remain later work.

### Bounds

- Persistent residency stays at the configured `S` slots per group. Current RCO validation uses `S = 64`; the staging cap does not silently increase `S`.
- Initial device expert-payload staging cap: **128 MiB per CUDA backend owner**, implemented as two reusable lanes of at most 64 MiB each. This is one owner-level cap shared by admitted groups, not 128 MiB per layer or request. Its exact reservation must be included in the placement budget; it is not additional unaccounted VRAM.
- For a bank projection, let `P` be the per-expert weight bytes plus required tile-local auxiliary bytes, and `T` the required allocation/tail padding. A lane holds `W = min(S, floor((64 MiB - T) / P))` experts. Require `W >= 1`. A format whose minimal supported tile does not fit must decline before mutation or use a separately validated within-expert tile; it must not enlarge the cap silently.
- Route control storage is bounded by `R_max = top_k * physical_ubatch` and model expert/slot counts. Declare and account its exact array sizes in the existing size-query infrastructure before allocation. Never use logical prompt length as an allocation bound.
- Any packed input/output tile buffers are separately included in the graph's measured workspace reservation, with maximum shapes derived from `R_max` and model dimensions. Report expert-payload bytes, route/control bytes, packed-activation bytes, and ordinary scheduler/KV allocations separately so the payload cap cannot conceal memory growth elsewhere.
- Reuse existing bounded host staging and its configured/mandatory owner budget. Submit large source rows in host-tile-sized pieces when needed. Do not create a second pinned copy of all selected experts or exceed that host budget to fill a device lane.
- Start with one active staging lease per owner. Before mutation, the initial certificate requires non-overlapping staging lifetimes between groups in a submitted graph. A single graph submitter must never wait for a lease whose release requires it to submit later graph nodes. Reject such an overlap before execution, or separately prove projection-level release/reacquisition that respects all reader lifetimes. A caller may wait only when the release is already submitted or an independent submitter can make progress; otherwise select the admitted existing path before mutation. Adding independent staging leases later must divide the same cap or receive a new explicit memory/performance budget.

The 128 MiB figure is the current experimental cap, not a measured optimum. A different cap is a documented design revision with matched controls and explicit placement accounting. It must not become a hidden function of the number of routed experts.

### Execution under the cap

Choose persistent hits/admissions once for the complete group and protect them across gate/up/down consumers. Complete all required resident bank fills before publishing those admissions. Experts not selected for persistent residency use transient tiles and never enter a per-bank cache map.

The implemented path preserves graph node order and launches contiguous expert ranges through the existing mapped-MMQ kernels. It prepares expert bounds, route destinations, and quantized activations once per projection. Each range offsets the existing expert bounds and source map while retaining global route destinations, so every routed output is written once. Gate/up and down use the same canonical transaction and resident map.

Changing the expert range changes Stream-K partitioning when more than one range is required. Bitwise output parity is therefore not expected for this arithmetic variant. The existing multiwave fixture checks finite output and relative squared error below `2e-5`; end-to-end output hashes are still reported exactly.

Double-buffer ordering for each lane is `previous consumer release -> copy/packing -> ready -> compute consumer -> release`. A copy stream waits on the previous group/consumer event before overwriting borrowed bytes. The compute stream waits on ready before using the lane. Prefetch of the next lane may overlap current computation; it cannot bypass resident pins, host-tile completion, or group authority. Group finish joins outstanding lane dependencies before it hands off the resource.

If preserving current MMID bulk execution needs a different bounded tile layout, benchmark it as a separate B variant. No FreeToken buffer shape or specific kernel replacement is mandatory. Keep a measured fast fallback for unsupported groups during rollout, and report its usage honestly.

## 6. CUDA graph, stream, and address rules

Base fingerprints include resource generation, stream, reciprocal maps, clocks, plan, and descriptors at `moe-cache.cu:12286-12305`. Replay additionally checks the resource owner at `12444-12454`, route/slot bounds and `graph_clock_active` at `12460-12467`, and completion ordering at `12470-12489`.

- First implement owned target prefill in DIRECT execution. Preserve retained decode capture where the existing proof permits it (`ggml-cuda.cu:6974-6990`). DIRECT is an execution mode under canonical ownership, not a return to legacy ownership.
- Persistent bank/map/plan addresses, resource identity, and S remain stable across a supported phase change. Resetting or changing metadata does not by itself require recapture, but clearing `graph_clock_active` while retaining its graph is invalid.
- Staging lanes referenced by a capture need stable backing and lifetime. A graph that embeds transaction-local host upload pointers cannot outlive those buffers. Keep one-shot metadata uploads outside capture or give them an explicitly retained graph lifetime.
- `begin_graph_dispatch()` precedes `cudaStreamBeginCapture()` (`ggml-cuda.cu:7096,7133`); use that boundary for admission and exceptional host drains. Do not add ordinary per-layer host synchronization to the fast path.
- Capture prefill only after fixed maximum control shapes, device-driven tile counts, source access, and staging events are proven replay-safe. A direct owned-prefill path plus captured decode is an acceptable first implementation.
- A lane-capacity, source, allocator-backing, or persistent-capacity change invalidates affected graph witnesses. Rebuild before replay; do not reuse old captured bounds with a larger live map.
- A wrong-stream or external/copied-route case is rejected unless the complete producer-to-consumer chain is certified. Physical multi-GPU evidence is required; a same-device multi-stream test does not cover peer placement.

## 7. MTP and the meaning of parallel decode

Phase-aware scheduler memory and expert-cache ownership are different resources. `src/llama-context.cpp:1400-1407` selects prompt/decode reservations, `1433-1437` waits for a shared-workspace peer, and `1454-1465` invalidates graph addresses after a backing-generation change. MTP attachment is explicit at `1914-1943`. Expert memory accounting separately sums default and MTP context contributions at `src/llama-model.cpp:3555-3569`.

Preserve physical target and draft ubatches, including independently configured draft ubatch, in every comparison. Do not claim a memory or concurrency improvement by reducing either batch. A target phase switch cannot release shared scheduler bytes while MTP still uses them. A new expert staging arena is not automatically owned by that scheduler-sharing contract. The proposed payload cap is per backend owner: two owners with a full arena reserve up to 256 MiB in total unless an explicit shared lease/lifetime design proves otherwise. Include that sum in placement; the first target-only slice must also account the draft's unchanged existing allocation.

The first B design continues to serialize writers of the same group. "Parallel decode" initially means useful progress for multiple admitted requests, including independent-row decode batches and decode while another request advances through prefill chunks. It does not mean launching two mutable-plan writers concurrently against one group.

The measurable path is:

1. Remove phase-induced authority/map rebuilding for certified target prefill; preserve decode graph resources across the handoff.
2. Make route counts above S use a certified canonical staged strategy instead of `DECODE_LEGACY`, while preserving independent/sequential/speculative row semantics.
3. Measure simultaneous parallel decode and staggered prefill-during-decode independently. Both are mandatory acceptance gates. For each, compare timestamped server rolling decode-only throughput, per-request wall/interference rates and inter-token gaps, TTFT, fairness, and admission/transaction/staging wait times against the same scheduler and chunking controls; neither workload substitutes for the other.
4. Change scheduler fairness or shorten transaction hold time only in a separate measured step when telemetry identifies that bottleneck. Canonical ownership alone does not remove scheduler serialization.
5. Concurrent read-only hits or multiple same-group in-flight plans require versioned plans, reader leases, and bounded additional state. They are outside the first B slice and cannot be inferred safe from the existing single transaction token.

## 8. FreeToken as an optional reference

The local reference checkout `/home/gencoolpc/FreeToken-v0.1.2-clean-9db1a394` was inspected at actual HEAD `708b9aa93e7dbda55d4cfd0762cea0b2a2e21da2`.

- `python/freetoken/moe/offload_cache.py:147-162,301-313` uses one cross-layer map and a fixed pool per bank, with matching bank shapes across layers.
- `offload_cache.py:524-534` views a subset of those already allocated slots as prefill double buffers. `553-561` invalidates only overwritten entries in the canonical map. `575-576,613-616,746-759` order prior compute, copies, readers, and reuse with events.
- Decode sees the full pool again through `python/freetoken/moe/offload_kernels.py:19-40`. This is temporal use of existing expert-cache slots, not scheduler-workspace lending or capacity growth on each phase switch.
- Overlap prefill does not automatically publish copied buffer contents as decode residents. Optional D2D reuse reads surviving resident slots outside its temporary range (`offload_cache.py:687-709`).

Adopt the single-map and event-ordering lessons. Reserving a smaller subset of this fork's configured slots is optional and deferred: it may reduce useful resident capacity or damage the current overlap. No additional full-expert allocation, full-layer staging, cross-layer global pool, or fixed reserved prefix is required by B. A cross-layer pool would need a separate geometry, eviction, and ownership design because this fork owns payload per group. The independent bounded staging proposal above can be evaluated without adopting slot reservation.

## 9. Retirement status and remaining compatibility deletion

| Path | Current status | Remaining gate |
| --- | --- | --- |
| Registered prefill ownership | `PREFILL_LEGACY` was replaced by grouped canonical prefill or detached staging. Detached staging does not mutate canonical ownership. | Repeat the supported model/backend/source matrix and require zero unexpected staged outcomes or pre-enqueue errors by reason. |
| Registered decode ownership | `DECODE_LEGACY` was replaced by grouped decode or detached staging. No registered per-bank cache acquisition remains. | Prove route, materialization, capacity, backend, graph, draft, and MTP cases. A staged outcome is acceptable only when its source and auxiliary lifetime is certified. |
| Generic cached MMID | Retained for inactive or unregistered sources and rejected for active registered tensors. | Instrument unbound calls; prove every supported caller reaches grouped or detached execution, or an intentional rejection, before deleting this compatibility cache. Arithmetic helpers may remain. |
| Grouped DIRECT execution | Preserved when capture/replay declines and grouped execution is otherwise valid. | Keep this supported execution mode and its fail-closed preparation rules. |

The four decode reasons need distinct evidence:

- Materialization: mapped, pinned, pageable/partially registered, and staged sources require correct ownership and source lifetimes. Do not delete the only functional source mode.
- Execution/capacity: MAIN independent rows currently require `top_k * rows <= S`; required draft/MTP/speculative domains can select staged execution (`moe-cache.cu:1604-1630`). Validate the canonical staged path above S without changing physical ubatch.
- Consumer equivalence: support the actual quantization, mapping, bias/scale, and preferred consumer. A compiler flag or shared slot map does not prove numerical equivalence.
- Route: copied/external IDs and split-device argsort need producer/stream proof (`moe-cache.cu:11255-11275,11531-11533`). Validate on the physical multi-GPU path.

Descriptor/source/geometry/capability failures, missing or duplicate roles, mixed IDs, external consumers, and unproven coverage stay fail-closed (`moe-cache.cu:11439-11470`). An armed grouped preparation failure already returns failure instead of generic MMID (`ggml-cuda.cu:3749-3771`); preserve that distinction.

The registered phase ownership is removed, but global compatibility deletion is not complete. HIP, MUSA, disabled cache, empty or unregistered inventory, unsupported models, and noncached MMID keep their valid behavior until separately proven. Do not count renamed counters as coverage and do not treat an explicit fail-closed rejection as supported execution.

## 10. Staged implementation and acceptance

| Stage | Deliverable | Required evidence before proceeding |
| --- | --- | --- |
| B0 | Freeze the clean control, certificates, source-mode/capacity matrix, and exact memory budget. Add outcome/reason and unowned-call observability in existing infrastructure. | Binary/source identity, command lines, physical batches, hashes, runtime counters, and artifacts. A and B counters/results remain separate. |
| B1 | Canonical target-prefill transaction for a narrow complete group, using existing resident payload and consumers. Resolve deferred plans and preserve capture identity. Existing scratch may temporarily remain. | Reciprocal-map/all-bank consistency, zero borrowed legacy-cache acquisition and zero phase resets for claimed groups, numerical parity, replay and abort checks. Label staging incomplete. |
| B2 | Enforce the proposed staging cap with reusable lanes and bounded route/workspace descriptions. Add supported auxiliaries/source modes incrementally. | Allocation high-water marks remain within all declared caps at capacity boundaries and long prompts; no hidden all-unique expert allocation; resident reuse and overlap are retained; correctness under lane reuse/failures. |
| B3 | Extend canonical staged execution to supported multi-row decode and required draft/MTP/verification shapes. | Correctness around S, route duplicates, multiple sequences, physical multi-GPU, shared-workspace resize, and zero newly unsupported models. |
| B4 | Evaluate simultaneous parallel decode and staggered prefill-during-decode separately; validate registered ownership retirement. | Both mandatory workload matrices pass their agreed gates, with measured claimed benefit, per-reason and unowned-call counters at zero, no numerical regression, and documented coverage. |
| B5 | Extend source and auxiliary coverage, then delete the unregistered compatibility cache where unused. | Full claimed coverage has one authority and bounded staging; every removed compatibility caller has a tested replacement or explicit unsupported result. |

### Numerical and failure gate

Reuse `tests/test-moe-cache-{plans,dispatch,graphs,staging,mmid,multigpu}.cpp` and existing failure hooks. Cover an empty/hit/miss/mixed cache, duplicate routes, counts around S and W, source/generation replacement, missing banks/auxiliaries, wrong stream, outstanding copy, partial enqueue failure, abort after consumer submission, graph capture/replay, and context teardown. Validate rejection or safe handling of interleaved group-reader intervals so the single staging lease cannot deadlock its graph submitter. Use deterministic resident poison/staging poison where existing fixtures support it. Include debug-enabled and normal execution.

Metadata-only variants require deterministic exact output parity for matched scheduling. Arithmetic/kernel variants require predeclared format-specific tolerances against the current consumer, per-node or logits comparisons before sampling, complete output checks, and sustained deterministic/coherent generation. A changed mixed-request hash is investigated; it is not waived because the text appears plausible. If scheduling changes legitimately alter floating-point batching, compare a fixed schedule or forced route replay to isolate numerical correctness before making throughput claims.

### Performance and memory gate

Metric definitions are part of the gate. Final `eval time` uses `t_gen_last - t_prompt_last` (`tools/server/server-common.h:398-429`, printed at `tools/server/server-context.cpp:746-762`), so intervening prefill stalls remain in its denominator. The server's `tg_3s` uses token-count deltas over elapsed wall time (`server-context.cpp:712-729`); it is suitable for decode-only throughput only when the complete window lies in a verified decode-only interval. It is not a CUDA kernel-duration measurement.

- Preserve timestamped server `n_gen`, `tg_3s`, slot/request identity, window start/end, and prompt-phase boundaries for every arm. Separate prefill-overlap, recovery, and steady decode windows. Cross-boundary rolling windows and final eval/client rates are wall/interference observations, not decode execution-speed evidence. If an arm has no complete decode-only window, report that missing gate and lengthen the matched workload or use separately validated phase-scoped decode telemetry.
- Compute aggregate throughput from token deltas over the same concurrent interval. Sum per-request rates only when their windows have the same boundaries and account the same phase. Do not sum differently timed first-post-prefill prints or average request-lifetime rates and call the result aggregate decode execution speed. For actual device execution time, retain phase-scoped events/profiling separately from server throughput.
- Controls: clean base; accepted A if it is retained; and B. Use the same model, placement, configured slot count, source mode, KV type/context, physical ubatches, prompt text/token counts, generation lengths, graph mode, prefix-cache policy, and scheduler settings. Current Qwen3.6 RCO controls use 128 slots and the earlier Qwen3.8 controls use 64. Use the shared GPU/build lock and preserve raw commands, logs, responses, hashes, telemetry, and teardown.
- Workloads: short prefill, a long single prompt containing at least 8k verified model tokens of real text, multi-chunk long prefill, sustained decode using the selected llama-benchy workload, and MTP on/off with unchanged draft settings. Preserve the exact prompt and tokenizer count; repeated filler and file-byte counts do not satisfy the real-text requirement. Distinguish cold and warm cache/source runs. The two multi-request gates below are mandatory and separate; the historical 7,288-token dossier does not complete the long gate.
- Simultaneous parallel-decode gate: start at least two independent requests together and measure their concurrent generation. Use distinct deterministic prompts so per-request output matching cannot be replaced by an unordered hash multiset; include capacity-relevant independent-row batches. Retain each request's output, prompt-processing rate, decode-only rolling telemetry, separately labeled final wall rate, TTFT, inter-token gaps, and actual overlap interval.
- Staggered prefill-during-decode gate: admit a prompt containing at least 8k verified model tokens of real text while another request is already decoding, preserving the control's arrival timing, context capacity, chunking, and generated work. Measure timestamped decode-only windows before and after the overlap, and report overlap stalls, wall rates, and tail gaps separately alongside the new request's TTFT and prompt rate. Use phase-scoped telemetry if claiming decode execution improvement during a window that also contains prefill. Record scheduling/route differences and use a fixed-schedule or route-replay correctness check when outputs diverge. Passing the short simultaneous gate does not pass this gate.
- Run warmups and at least five alternating paired measurements for a candidate acceptance claim. More pairs are needed if control noise exceeds the proposed effect. Report paired ratios and uncertainty, not the best run.
- Record prompt-processing and decode-only tok/s for every arm and every request, including simultaneous and staggered traffic, plus final wall rates in a separate labeled field. Read per-request `prompt_tok_s` from retained `response-1.json`/`response-2.json` metrics when a summary's aggregate prompt field is unavailable; do not treat that missing summary field as missing prompt evidence or use its client `decode_tok_s` as a substitute for rolling decode-only telemetry. Also record TTFT, sustained aggregate decode-only tok/s, p50/p95 inter-token gaps, completion counts, admission/group/staging waits, graph captures/replays/invalidations, resident hits, actual H2D bytes/fills, ownership resets, fallback/rollback/errors, and sampled plus allocator-tracked peak device/pinned memory.
- Pre-register the prompt-regression ceiling and the parallel workload before choosing an implementation result. Until the owner assigns a numeric tradeoff budget, use strict non-regression as the default; report any small prompt loss with its measured benefit rather than declaring it accepted. A real B milestone such as verified decode-legacy deletion is identified by its scope and counters, not by assertion.
- A claimed parallel-decode execution improvement requires repeatable improvement in verified decode-only telemetry beyond measured control noise on its declared workload, with acceptable TTFT/tail behavior and no reduced output work. A better final wall rate can instead demonstrate less interference when output/scheduling controls support that claim; label it accordingly. Report both mandatory multi-request gates even if only one improves; acceptance requires their agreed limits, and a gain in one cannot hide a violation in the other. Fewer copies or resets alone cannot pass. Ownership simplification may be reported independently when it removes actual duplicate authority and passes correctness; it is not a speed claim.
- The expert staging cap, total placement budget, and host staging budget are hard limits. Reduced staging bytes cannot compensate for an unreported increase in packed activations, metadata, MTP allocation, or retained graph pools.
- Repeat the existing no-MTP soak after synchronization/lifetime changes, plus MTP/shared-workspace and repeated mixed-phase replay coverage. Correct completion, hashes where deterministic, zero unexpected fallback/errors, and clean teardown are required.

## 11. Decisions and open implementation risks

Implemented direction: keep the existing group owner and configured slot resource, plan residency once per complete group, resolve deferred commits explicitly, preserve valid resident bytes, and use bounded mapped-MMQ staging under that same authority.

Still to be demonstrated: repeatable performance beyond noise; the smallest useful staging cap; cold detached auxiliary materialization and other consumer coverage; physical multi-GPU behavior; combined pageable or split-registration multiwave reuse; delayed replacement/shutdown; partial CUDA copy and event failure injection for the new arena; captured prefill; and whether canonical ownership improves both mandatory concurrent workloads. The staggered observation currently does not support a decode-speed claim.

Deferred choices: reserving a subset of resident slots for prefill, pooling slots across layers, multiple same-group mutable plans, scheduler fairness changes, dynamic VRAM lending, and copying FreeToken arithmetic. They are independent hypotheses with their own review and controls.

Completion of this design document is not completion of B. The implementation is complete only when its claimed path has one authority, bounded staging, verified graph/stream/rollback behavior, the agreed performance result, and explicit compatibility/deletion coverage.
