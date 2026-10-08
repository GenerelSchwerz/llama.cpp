# Saved-trace profile comparison

2026-10-07. CPU-only evaluation, no model load, CUDA call or new serving request. Artifacts: /home/gencoolpc/moe-cache-tests/results/generic-strata-prefill-20261007/HELDOUT-STRATA-COMPARISON.json and containment-profile-comparison-01. Adjacent corpus artifacts were read without modification.

The independent bird-migration prose teacher trace has63 observed post-prefill rows,9072 source projections and90720 expert-source accesses. Its original request asks for64 teacher rows; the first row is the prefill boundary. Counts reconstruct exactly from every saved route event. All profiles admit64 original experts per group,48 groups and144 source banks. Actual resident payload is5364121600 bytes (5.364GB) for every arm, using each source's quantized geometry. Total requested source bytes are52803072000. These byte counts exclude metadata and temporary staging.

| Initial placement | Source-access hits | Access hit rate | Missed bytes | Admitted IDs replaced vs Strata |
| --- | ---: | ---: | ---: | ---: |
| Shipped Strata STRP |29346 |32.35% |35804518400 |0 |
| Earlier generated balanced-code STRP |29496 |32.51% |35468390400 |1609 |
| Recent pooled corpus, current runtime ranking |41442 |45.68% |28615680000 |1882 |
| Adjacent weighted corpus addon |42168 |46.48% |28233728000 |1914 |

The pooled corpus reduces missed bytes by20.08% relative to shipped Strata on this trace. The weighted addon reduces them by21.15%. The latter remains the other worker's uncommitted addon: the frozen released runtime in this worktree ignores its optional ranking_scores and ranks raw counts. Do not claim weighted ranking is already integrated here. Training uses four requests and204 actual decode rows; this small sample is not a representative corpus.

The current runtime arm calls ggml_moe_source_rank_statistics from the actual frozen CPU-only libggml-base, not a Python approximation of its floating-point ordering. GGUF source domains, names, shapes, formats and offsets match between training and evaluation; coupled-bank geometry and rankings were checked. STRP pair order and its historical slot-index table were validated before extracting each group's capacity-limited prefix. Layer-name mapping is an explicit bridge for this legacy Flash validation sample, not a model admission mechanism.

Full admitted-ID sets, input/probe output, source/library/script hashes and route reconstruction are retained. The ranking probe was compiled under both ordered locks,256MiB hard memory, zero swap and30s runtime. Comparison used512MiB, zero swap,60s runtime and CPU affinity16-23; it completed in1.28s with123.1MiB peak accounted memory and an empty process tree.

This measures initial static placement only. It excludes adaptation, lookahead/prefetch, MTP, concurrent requests, latency and execution throughput. The historical generated profile's training membership is not recertified by this check. One short independent prose trace cannot close the held-out Strata-or-better quality gate; broader independent code/prose/context traces and equal actual device budgets remain required. GPU/model runs remain blocked during crash investigation.

## Longer held-out counterexample

HISTORICAL-HELDOUT-COMPARISON.json evaluates the saved8194-input/2048-output prose request:798 target windows and2372 executed target rows, including rejected MTP verification rows. The original MOR1 trace, exported NPZ arrays and marginal counts reconstruct exactly. A separate committed-prefix view contains2048 rows. This is one additional prompt, not2372 independent examples; duplicate short/long exports of that prompt are not counted as separate held-out tasks.

The recent candidate corpus has four distinct47/50/49/46-token prompts, so this8194-token request is held out from that corpus. The saved pack-index fingerprint matches, each layer's original expert payload size equals the summed current GGUF bank strides, and the old/native and current calibration model paths resolve to the same live GGUF inode. Model weights were not broadly rehashed. The legacy generated profile's training membership remains unrecertified.

| Slots per group | Equal resident payload bytes | Shipped Strata hit rate | Legacy generated hit rate | Current pooled hit rate | Pooled missed-byte increase vs Strata |
| ---: | ---: | ---: | ---: | ---: | ---: |
|16 |1341030400 |9.06% |7.51% |7.05% |2.35% |
|32 |2682060800 |15.08% |13.96% |12.52% |3.33% |
|64 |5364121600 |25.20% |24.58% |22.12% |4.31% |
|96 |8046182400 |33.96% |Not enough ranks |30.71% |4.97% |

The legacy balanced-code STRP has64 ranks per layer and is excluded at96 slots without padding. Current model statistics still cover all512 experts per layer; capacity is applied only during evaluation. At64 slots, missed bytes are1483915878400 for Strata,1497094118400 for legacy generated and1547827353600 for current pooled. The current pooled profile therefore fails the requested Strata-or-better quality criterion on this trace, despite its favorable earlier short bird-migration result. The committed-prefix comparison agrees:24.92% vs21.90% hits and4.20% more missed bytes for pooled.

An explicitly noncausal best-static bound chooses each layer's most frequent experts from this held-out trace itself. It reaches60.61% source-access hits at64 slots and72.06% at96, with identical payload budgets. This shows placement headroom for this fixed trace; it is not an implemented predictor, a permitted training profile or a runtime speed claim. No policy is tuned on these held-out outcomes.

The bounded CPU evaluation used512MiB hard memory, zero swap,60s runtime and affinity16-23; it completed in1.13s with90271744 bytes peak accounted memory and a removed process tree. Two comparison-script setup failures are preserved separately: an overly strict integer-width assertion and an invalid96-slot legacy-profile comparison. Neither involved GPU execution. Broader code/prose/context coverage and adequate training generations remain required; profile quality is not qualified for default promotion.
