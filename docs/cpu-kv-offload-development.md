# CPU KV-offload progress and decision journal

The authoritative runnable setup is
[`cpu-kv-offload-current-testing.md`](cpu-kv-offload-current-testing.md). Read it
before launching any current test. This journal records protocol transitions,
durable rationale, and concise summaries of valid rejected paths. It is not an
attempt log: invalid measurements, redundant reruns, obsolete command copies,
and temporary-artifact inventories belong neither here nor in the experiment
ledger. Use Git history for forensic reconstruction of an older edition.

All command text below is an exact historical record of the experiment that
used it, including retired controls and whole-process affinity. It is not
current runnable protocol and must not be normalized or reused for new
evidence. Translate intent through the current-testing document instead.

When the live protocol changes, update the current-testing document first and
add the transition and rationale here. Do not rewrite valid old measurements to
claim new flags or geometry. The companion experiment ledger records curated
per-change evidence.

## Manifest-driven early-screen transition: 2026-08-22

The prior current protocol defined exact provenance, clean-process GPU
serialization, matched performance/resource fields, correctness, long-context,
and profiler rules, but left each derived feature agent to assemble its own
early launcher and artifact layout. That made early decisions unnecessarily
slow and made otherwise similar feature investigations harder to compare.

The current-testing protocol now requires the generic manifest runner for a
new CPU-KV-derived performance or memory feature's early exactness,
prefill/decode smoke, representative weighted screens, and resource telemetry.
Optional regression diagnostics use independent direct-target NSYS discovery
and verified filtered NCU capture for each variant; they remain explanatory
rather than benchmark or acceptance evidence. Selection is by workload,
layout, and backend capability, never by architecture or model name.

This supersedes ad hoc early-screen orchestration only. It does not relabel
historical measurements and does not relax the canonical server lifecycle,
final long-context comparison, matching-batch perplexity and output exactness,
clean-process proof, allocation accounting, or any feature-specific acceptance
gate. A single matched fail-fast pair has no confidence claim; a clear signal
authorizes deeper validation and nothing more.

## 2026-08-22: GPU-window source consolidation, milestone 1

The same-format Q8_0 mapped-host design remains the sole candidate. The
alternate worktree contributed only reusable safety and observability pieces:
the second speculative-context reset, separate body/tail backend capability
probing, strict placement validation, and complete `llama-bench` parameter and
result plumbing. Its CPU `GET_ROWS` plus H2D transport, F16/BF16 device tail,
prepacked-body switch, and competing operation-parameter meaning were not
merged.

Validation now applies to the requested window before any effective-size
normalization and requires `256 <= N < n_ctx`, operation offload, a default
single-sequence target context, Q8_0/Q8_0, standard non-SWA attention, mapped
pinned-host storage, no independent precision tail or KV-layer policy, no
tensor split, and one actual CUDA registry device 0 across every attention
layer. The Q8_0 `N=256` boundary is routed through the same packed quantized
CUDA path as `N=257`; focused validation and backend cases cover both sides of
the former indexed-small collision.

This remains a transport-correctness milestone. The temporary packed body is
still sized from the full `kv_size`; reducing it to the aligned maximum cold
region remains a separate optimization and measurement step. At initial
consolidation there was no commit, matched performance evidence,
state/prompt-cache gate, or experiment disposition; Experiment 022 now owns the
committed milestone and its pending evidence work.

Static review then exposed an accidental capability edit and gaps between the
claimed and actual test surface. KVarN CPU tail capability is again restricted
to its implemented F16/BF16 contract; Q8_0 row conversion is advertised only
by the independent standard attached-tail CPU oracle. The public standard-tail
and CUDA capability additions are explicitly Q8_0 rather than an untested
all-quantized promise. The 256/257 Q8_0 backend cases are now required on CPU
and CUDA and are described only as segmented-current arithmetic/routing
coverage.

The mapped-host allocator now resolves and caches CUDA device 0's alias before
publishing a non-empty buffer, so an expected mapping failure returns through
cache/context allocation rather than first appearing during decode. The common
backend allocator already returns a type-preserving empty buffer before a
buffer-type allocator is invoked for a zero-byte reservation; no CUDA-specific
zero-allocation workaround is needed. Source coverage now exercises that empty
reservation and defines a CUDA-only scheduled graph with a real
`CUDA_MappedHost` Q8_0 body, a device Q8_0 suffix, the GPU-window operation
parameter, preservation of the mapped K/V sources, and a CPU numerical oracle.
The first authorized Release CPU/CUDA build and focused execution pass completed
on an RTX 4070 (SM 8.9). Both full builds succeeded. The six focused parser,
placement, tail, static, and scheduled-graph CTests passed on CPU and CUDA, and
the required seeded Q8_0 segmented-current backend cases passed at both 256 and
257 rows on CPU and CUDA.

The real Qwen3.8-27B-UD-IQ2_M smoke initially exposed a scheduler gap hidden by
the synthetic graph: model K/V reach attention through bufferless view/permute
chains, so checking only `src->buffer` copied the complete mapped body to CUDA.
Mapped-owner recognition now follows the view chain, and the scheduled graph
regression reproduces that shape and asserts that no full body copy is inserted.
The backend fixture was also corrected to compare the segmented-current
attention result directly between CPU and CUDA; its former in-graph oracle used
unsupported quantized `CONCAT` helpers and could fail before reaching the op
under test.

A bounded `llama-bench` smoke with that exact local model then completed prompt
plus decode for the default-off control and windows 256, 257, and 1024. The
single-process, single-repetition throughput printed by that smoke is not
performance evidence and is intentionally not entered in the experiment
ledger. A broader practical CTest set passed 87/87 on both CPU and CUDA. It
excluded the unfiltered backend matrix (the required cases ran separately),
network/download fixtures and their dependants, the unavailable full vocab
fixture, and `test-model-resolution`: that independent loopback HTTP fixture
reproducibly faults in `httplib::ClientImpl::set_follow_location()` at normal
speed but passes under GDB, with no KV or scheduler frame in the captured core.
Full exactness/state coverage, clean matched processes at the required depths,
resource accounting, and an experiment disposition are still required before
publication.

## 2026-08-21: GPU-window prototype protocol boundary

The first chunk-residency prototype deliberately keeps the complete standard
Q8_0 KV body authoritative in pinned host RAM and adds only a compact Q8_0 suffix on
CUDA. Attention masks those suffix rows out of the body, packs only the older
host rows, evaluates both partials, and uses the existing normalized tail merge.
This tests transfer deduplication without changing prompt-cache, state, rollback,
or recovery ownership. Physical host-RAM deduplication is deferred until those
ownership paths can represent a split canonical cache.

The new `--kv-gpu-window N` control is default-off and fail-closed to the narrow
single-sequence, standard non-SWA CUDA Q8_0/Q8_0 configuration with `N >= 256`
documented in the current protocol. No build, execution, measurement, or
experiment disposition exists for the initial source edit; it must not be cited
as evidence.

## Evidence-curation transition: 2026-08-20

Earlier editions asked agents to preserve invalid attempts and every historical
command. That policy created runnable-looking conflicts and repeated results
without adding evidence. Current documentation now retains only valid,
decision-relevant measurements. A failed acceptance gate contributes no
numbers; if it exposes a reusable hazard, this journal keeps only the cause and
the corrective gate. Statistically useful repeats are aggregated into their
existing experiment, while redundant repeats are omitted. Git history remains
available when exact discarded text is needed for forensic work.

## Current protocol edition transition: 2026-08-19

Current runnable instructions moved into
[`cpu-kv-offload-current-testing.md`](cpu-kv-offload-current-testing.md) after
the phase-aware, MTP physical-ubatch, canonical quant-store, and independent
draft-residency work converged. This journal had accumulated valid commands
from several incompatible implementation eras. Keeping those commands was
necessary for interpreting prior evidence, but presenting them beside the live
setup made it too easy to restore removed controls or superseded geometry.

The current edition makes the supported CLI the only live interface. The
removed `GGML_KV_CPU_PINNED` and `GGML_RECURRENT_STATE_OFFLOAD` variables must
not be copied into new commands, even as defensive `env -u` entries. Historical
entries retain them exactly where they were used. MTP draft ubatch 128 is also
historical; current MTP omits the draft override or explicitly matches the
target physical ubatch.

An early post-merge `llama-benchy` pair failed prompt-identity validation:
`--no-cache` varied the request and the stock prompt generator used an unseeded
NumPy RNG. No measurements from that attempt are evidence. The replacement
protocol seeds NumPy, omits `--no-cache`, sends `cache_prompt=false`, and
requires identical observed prompt-token counts before accepting a pair.

The corrected short and 5,000-token live screens matched prompt, output, and
MTP work across draft-KV residency. They support a small possible decode
benefit at a modest VRAM cost, but not a stable speed claim without alternating
repetitions. Experiment 019 owns the measurements and resource accounting; they
are not duplicated in this journal.

The ranked memory backlog is maintained in
[`cpu-kv-offload-vram-roadmap.md`](cpu-kv-offload-vram-roadmap.md). Read it
alongside this journal before starting a VRAM experiment.

This document reconstructs the development process for the
`exp/kv-cpu-offload` branch. It explains the goal, evidence, decisions,
implementation hypotheses, and intended next steps. The companion
`cpu-kv-offload-experiments.md` file is the authoritative per-change benchmark
ledger. Git commits are the source of truth for exact code changes.

## Objective

Improve long-context generation when model weights remain on the GPU but the
target model's KV cache is placed in system RAM. The immediate target workload
is the Qwen3.8 27B hybrid IQ4_XS/IQ3_S GGUF on an RTX 5070 Ti and Intel Core
Ultra 9 285K.

The optimization is not solely a smallest-cache exercise. Available system RAM
is ample, so cache quality may be traded for capacity only when the performance
benefit justifies it. Q8_0 K/V is the current working format. KVarN5/KVarN5 was
the minimum acceptable KVarN quality level, but KVarN is not currently a viable
CPU-offload format on this machine.

Success requires all of the following:

- Higher decode throughput, especially as context length grows.
- No material prefill regression.
- Minimize GPU memory after throughput is protected, so larger contexts and
  auxiliary features fit without giving back the CPU-KV performance gains.
- Explicit accounting for GPU memory, system memory, and pinned memory.
- Correct output and unchanged KV-cache semantics.
- Small, reviewable changes that remain aligned with current llama.cpp
  abstractions.
- Every retained experiment isolated in its own commit and documented well
  enough to reproduce or revert.

## Repository layout and branch policy

- Known BeeLlama baseline: `../beellama.cpp`, branch `main`, commit
  `ba27edad2a84ff045a556df06661e821285c2fab` (`v0.4.3`).
- Experimental worktree: `../beellama-kv-offload`, branch
  `exp/kv-cpu-offload`.
- Clean llama.cpp reference: `../llama.cpp`, observed at `af5172627` during the
  initial comparison.

The baseline worktree must remain usable as the known upstream-aligned Bee
build. Experimental changes belong only in the experimental worktree. Do not
commit unrelated files or reintroduce systems removed from BeeLlama v0.4.0.

Each valid, independently testable experiment should be one commit containing
its implementation and an update to `cpu-kv-offload-experiments.md`. A valid
rejected experiment should normally be reverted and recorded only when it tests
a distinct hypothesis or prevents likely repeated work. Invalid and redundant
runs are omitted. No benchmark result is portable unless its model, protocol,
hardware, settings, and commit are recorded.

## Hardware and workload

- CPU: Intel Core Ultra 9 285K, 24 physical cores, one hardware thread per core.
- CPU topology: logical CPUs 0-7 are P-cores; CPUs 8-23 are E-cores.
- Fast P-cores used for controlled decode: CPUs 0-2.
- GPU: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB, compute capability 12.0.
- RAM: approximately 62 GiB, one NUMA node.
- Model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.

The active serving/benchmark configuration uses CPU-resident Q8_0 K and V,
CUDA FlashAttention, three decode threads pinned to CPUs 0-2, poll 100, and GPU
weight offload. Batch work may use more cores, but decode measurements must not
silently mix P-cores and E-cores.

## How the investigation reached the current design

### 1. Initial symptom

The original CPU-KV configuration produced roughly 11 tokens/second, well below
expectations. Changing K/V to F16 improved performance only slightly, showing
that quantization overhead alone did not explain the result.

### 2. Thread topology was a major confounder

The 285K is heterogeneous. Allowing decode workers to range over all cores or
using too many threads substantially reduced performance. At depth 4096 with
F16 CPU KV and FlashAttention, the observed decode scaling was:

| Threads | Decode t/s |
| ---: | ---: |
| 1 | 16.74 |
| 2 | 16.64 |
| 3 | 17.16 |
| 4 | 15.42 |
| 5 | 13.62 |
| 6 | 15.25 |
| 8 | 11.36 |
| 16 | 7.95 |
| 24 | 7.08 |

Three threads pinned to distinct P-cores became the controlled decode setup.
For Q8_0 at depth 4096, CPUs 0-2 achieved `18.14 +/- 0.03` t/s. A mask using
CPUs 0, 1, and 7 achieved `17.90 +/- 0.12` t/s. This established that affinity
and physical-core selection must be held constant in every comparison.

### 3. FlashAttention remained beneficial

With F16 CPU KV, four threads, and depth 4096, FlashAttention produced 16.36
t/s versus 14.69 t/s with FlashAttention disabled. The investigation therefore
continued on the native CPU FlashAttention path rather than changing to the
non-FA layout.

### 4. Standard cache formats outperformed F16

At depth 4096, three threads, and FlashAttention enabled, the measured standard
formats were:

| K/V format | Decode t/s |
| --- | ---: |
| q4_0 | 19.45 |
| iq4_nl | 19.18 |
| q5_0 | 18.82 |
| q8_0 | 18.54 |
| f16 | 17.01 |
| bf16 | 16.01 |

Lower-bit formats can reduce memory traffic enough to outweigh dequantization
cost. Q8_0 was chosen as the quality-conservative working point, not as the
fastest observed format. Q6, Q8, and other standard formats remain relevant
because their representation and CPU kernels differ from KVarN.

### 5. KVarN CPU attention was rejected for the current path

KVarN was initially preferred for quality/capacity flexibility, with
KVarN5/KVarN5 as the minimum quality target and larger KVarN widths also in
scope. Controlled results showed that the current native CPU path was far too
slow:

| Format/layout at depth 1024 | Decode t/s |
| --- | ---: |
| KVarN5, no tail | 1.82 |
| KVarN5, exact tail 1024 | 4.31 |
| KVarN6, no tail | 1.77 |

The tail improved the result but did not approach standard Q8_0 performance.
The current decision is to optimize standard Q8_0 CPU KV first. This does not
prove that KVarN is fundamentally unsuitable; it indicates that making KVarN
competitive would require a distinct kernel-level project rather than a cache
format switch.

### 6. BeeLlama was compared with clean llama.cpp

A sibling llama.cpp checkout and matching INI configuration were used to test
whether BeeLlama's standard CPU KV implementation was simply slower. Approximate
results from that comparison were:

| Implementation | Prompt t/s | Generation t/s |
| --- | ---: | ---: |
| BeeLlama | 432.6 | 7.53 |
| llama.cpp | 435.3 | 8.08 |

BeeLlama's generation average included an anomalous first sample; the remaining
behavior was effectively comparable. Source inspection also found that both
implementations force the graph region from KV store through attention output
onto CPU when `--no-kv-offload` is active. The working conclusion was that the
main bottleneck is architectural/data-movement behavior shared with upstream,
not a broad BeeLlama regression. This describes the clean baseline and the
pre-Experiment-015 Bee route. It must not be read as the current pinned-host
policy: the retained `--no-kv-offload --kv-cpu-pinned` route now separates
persistent storage from CUDA attention execution, as documented below.

### 7. Scheduler knobs did not explain the remaining gap

With Q8 decode pinned by affinity to CPUs 0-2, poll values 0, 50, and 100 were
effectively equivalent. Disabling operation offload was also neutral or slightly worse.
Representative results ranged from 17.95 to 18.14 t/s. Poll 100 and normal
operation offload were retained to keep the setup stable.

### 8. Extending CPU FlashAttention split-K to Q8 was neutral

Source inspection showed specialized split-K/tiled CPU paths for F16/F32 while
Q8 used the generic head-partitioned
`ggml_compute_forward_flash_attn_ext_f16_one_chunk` path. A local experiment
broadened split-K eligibility to quantized Q8. It produced 18.11 t/s versus an
18.14 t/s baseline and was fully reverted. Do not recreate that broad eligibility
change without a more specific hypothesis about partition size, synchronization,
or quantized dot-product behavior.

### 9. CUDA-pinned host memory produced the first material code win

Standard CPU KV allocated ordinary pageable memory through
`ggml_backend_cpu_buffer_type()`, even though CUDA exposes a compatible host
buffer type backed by `cudaMallocHost()`. Experiment 001 originally selected
that device host buffer with `GGML_KV_CPU_PINNED=1`; the supported interface is
now `--kv-cpu-pinned` (Experiment 009).

Committed result: `346ca3000` (`kv-cache: experiment with pinned CPU buffers`).

| Test | Ordinary CPU KV | Pinned CPU KV | Change |
| --- | ---: | ---: | ---: |
| Prefill, 512 tokens at depth 4096 | 1369.49 +/- 11.82 | 1399.69 +/- 10.34 | +2.2% |
| Decode, 64 tokens at depth 4096 | 18.50 +/- 0.07 | 20.51 +/- 0.03 | +10.9% |
| Decode, 64 tokens at depth 16384 | 12.62 +/- 0.03 | 17.24 +/- 0.06 | +36.6% |
| Process VRAM at depth 4096 | 13,682 MiB | 13,682 MiB | 0 MiB reported |

The increasing benefit with context depth strongly supports a memory-transfer
or pageability bottleneck. The allocation is static: the complete configured KV
capacity is pinned at context creation and released at context destruction.
This consumes page-locked system RAM and CUDA mapping resources even though
`nvidia-smi` did not report additional device memory.

### 10. ik_llama.cpp identified a faster CPU FlashAttention reference

The `ikawrakow/ik_llama.cpp` repository was cloned into `../ik_llama.cpp` at
commit `8337e4cd3861406fc04e0854b1409cd1b027fbc9` and built with CUDA, native CPU
optimization, and its IQK FlashAttention kernels. This fork is a design
reference only; its large divergent feature surface must not be merged wholesale.

Its CPU KV allocation already uses `CUDA_Host`. More importantly, it contains a
dedicated IQK CPU FlashAttention implementation, cache-specific quant formats,
repacked layouts, and single-query specializations.

A strict-affinity Q8_0 sweep used CPUs 0-2, three generation and batch threads,
FlashAttention, CPU KV, 256-token prompt increments, 64-token generation tests,
three repetitions, and a context capacity of 18432. Selected rows were:

| Populated KV | Prefill t/s (256 tokens) | Decode t/s (64 tokens) | Process VRAM delta |
| ---: | ---: | ---: | ---: |
| 0 | 1336.71 | 25.75 | 13,270 MiB |
| 4096 | 1489.46 | 23.39 | 13,296 MiB |
| 8192 | 1438.11 | 21.60 | 13,312 MiB |
| 12288 | 1347.47 | 19.96 | 13,328 MiB |
| 16384 | 1309.33 | 18.57 | 13,344 MiB |

These numbers are not a perfectly interchangeable benchmark with BeeLlama's
synthetic `llama-bench -d` tests: ik_llama's sweep populates the cache and uses a
different benchmark implementation. They nevertheless establish that its
standard Q8_0 CPU attention path is a serious optimization reference. Relative
to the closest pinned Bee measurements, the observed decode rates were about
14.1% higher at 4096 and 7.7% higher at 16384.

The same sweep with ik_llama's cache-specific `q8_KV` format did not run on this
Qwen3.5 hybrid model. Allocation completed (`585 MiB` reported KV size at a
18432-token capacity), but graph construction aborted at `ggml.c:5427` with a
tensor-view bounds assertion. Do not port or benchmark `q8_KV` until its layout
assumptions and this hybrid-model failure are understood.

### 11. MTP depth sanity check near maximum test context

Built-in MTP was tested as a serving configuration on the pinned Q8_0 CPU-KV
branch at a context capacity of 18432. The fixed stress prompt repeated
`The quick brown fox jumps over the lazy dog.` 1800 times, filling the context
near its usable maximum while reserving generation space. Generation used
greedy sampling, CPUs 0-2, three threads, FlashAttention, and 128 tokens for the
matched comparison.

| Speculation | Generation t/s | Change vs no MTP |
| --- | ---: | ---: |
| none | 16.8 | baseline |
| MTP, `--spec-draft-n-max 1` | 23.7 | +41.1% |
| MTP, `--spec-draft-n-max 3` | 27.8 | +65.5% |

Confirmation runs generating 256 tokens produced 23.9 t/s at depth 1 and 28.8
t/s at depth 3. Both MTP configurations reported 13,684 MiB process VRAM, so no
depth-dependent VRAM difference was visible through `nvidia-smi`.

This prompt and continuation are highly repetitive and therefore favorable to
speculative acceptance. Treat the speedups as a maximum-depth stress result,
not a representative general-workload claim. The CLI does not call
`common_speculative_print_stats`, so accepted-token counts and acceptance by
draft position were not captured. A server benchmark with realistic prompts is
required before choosing depth 3 as the serving default. Prompt processing also
varied materially across these runs and should be remeasured separately rather
than attributed to MTP decode behavior.

### 12. Hybrid memory placement was the dominant Bee bottleneck

A closed-book review of BeeLlama's own hybrid-memory construction found that
`--no-kv-offload` controlled two different resources through one
`offload_kqv` boolean. It correctly placed the 16 full-attention layers' long
Q8 K/V history in CPU RAM, but it also placed the fixed-size recurrent R/S
state for the other 48 layers on CPU. Because the recurrent-layer weights stay
on GPU, this introduced repeated backend boundaries unrelated to long-context
KV capacity.

Experiment 002 separates the policies in `llama_memory_hybrid`. With attention
KV still pinned in CPU RAM and recurrent state returned to the GPU, decode rose
from 20.59 to 43.10 t/s at depth 4096 and from 17.24 to 31.23 t/s at depth
16384. Prefill rose from 1388.96 to 1788.16 t/s. The matched CUDA peak increased
by 124.0 MiB in the no-MTP benchmark.

This result changes the working theory: the largest observed loss was not the
Q8 attention inner loop. It was an overly broad placement policy for a hybrid
architecture. Two isolated Q8 kernel experiments—fused V dequantization and a
vectorized two-pass softmax—were neutral and removed. Future kernel work must
start from the decoupled-placement baseline rather than trying to recover graph
boundary overhead inside the attention kernel.

The implementation remains opt-in through `--recurrent-state-offload`; the
original `GGML_RECURRENT_STATE_OFFLOAD=1` experiment switch was removed in
Experiment 009. MTP creates recurrent rollback snapshots, so its additional
GPU-state cost and correctness must still be measured before choosing a serving
default.

### 13. CPU KV does not make VRAM independent of context

Live testing showed that GPU memory can still rise when configured or populated
context grows even though the target Q8 K/V buffers report `CUDA_Host`. CPU KV
offload controls the long attention K/V storage; it does not move every
context-shaped tensor, attention compute, or auxiliary state out of CUDA
memory.

The current resource model is:

| Resource | Normal location in this experiment | Scaling behavior |
| --- | --- | --- |
| Target attention Q8 K/V | CUDA-pinned system RAM | Scales strongly with configured context |
| Recurrent R/S state | GPU with Experiment 002 | Fixed by sequence count and rollback snapshots, not attention depth |
| Target CUDA compute workspace | GPU | Can grow with reserved graph and context shapes |
| CPU/GPU boundary-copy buffers | GPU and pinned host memory | Can grow with scheduled tensor shapes |
| CUDA graph captures | GPU | Can grow as new execution shapes are captured during serving |
| MTP draft compute workspace | GPU when the draft layer is offloaded | Additional fixed/reserved cost |
| MTP draft KV | Pinned system RAM under CPU-KV placement | Scales with its configured context |

Observed examples establish the size of the non-KV costs:

- The no-MTP 4K candidate reported approximately 555 MiB of CUDA compute
  buffers.
- The 92K target server reserve reported approximately 721 MiB of CUDA compute
  buffers.
- The MTP draft context reported approximately 631 MiB of additional CUDA
  compute buffer.
- Recurrent state was approximately 299.25 MiB with one rollback snapshot and
  598.50 MiB with three snapshots in the recorded server configurations.

These categories must not be described as KV-cache VRAM. A `CUDA_Host KV
buffer` line identifies host-resident KV, while `CUDA0 RS buffer` and `CUDA0
compute buffer` are device allocations. CUDA host mappings can also consume
driver resources without appearing as ordinary process VRAM.

Reducing VRAM is now an explicit goal, subordinate to preserving performance.
Candidate reductions should target unnecessary workspace retention, redundant
boundary buffers, excessive graph variants, or avoidable MTP state. Moving the
recurrent state back to CPU is not an acceptable memory optimization unless a
new mechanism avoids the throughput regression demonstrated by Experiment 002.

At a 32K context with target batch/ubatch fixed at 1024/512, MTP-1 increased
process VRAM by 664 MiB after initialization and 682 MiB after a live 32K
request. Detailed startup logs attribute most of the fixed increase to three
categories: 234.56 MiB of MTP-head model tensors, an additional 149.63 MiB
target recurrent rollback snapshot, and a 276.27 MiB CUDA compute buffer owned
by the MTP context. Only the last category is directly addressable by reducing
the MTP context's ubatch.

Clean Bee MTP inherits target `n_batch=1024` and `n_ubatch=512` because
`common_speculative_init_result` derives both contexts from the same
`common_params` and changes only the context type and speculative memory
settings. The initial investigation assumed that a different draft ubatch was
only a workspace/performance choice because prompt synchronization chunks work
according to `llama_n_ubatch(ctx_dft)`. The later 1,000-token exactness audit
disproved that assumption: changing those chunks changes recurrent floating-
point state and later verification/acceptance batch geometry even when the
first 99 generated tokens agree. MTP must preserve the inherited physical
ubatch; workspace lifetime is reduced by phase-aware reservation instead.

Every future serving measurement should distinguish:

1. GPU memory immediately after model/context initialization.
2. GPU memory after warmup.
3. GPU memory at fixed live depths such as 4K, 16K, and 32K.
4. GPU memory after returning to an idle slot.
5. Target recurrent state, target compute, draft compute, and CUDA graph-cache
   contributions when MTP is enabled.

This sequence separates static reservation from live-context growth and from
CUDA graph caching. A graph-disabled diagnostic may be used to identify capture
growth, but it is not a retained optimization unless throughput remains
competitive.

## Current understanding of the execution path

With `--no-kv-offload`, the persistent KV tensors are placed in host memory.
That placement does not imply that attention executes on CPU. A 32K Nsight
Systems trace showed that the scheduler copies each full K and V tensor to the
GPU and executes standard Q8 attention with CUDA FlashAttention. Model weights
and the rest of the graph also remain GPU-resident.

Storage placement also must not choose the numerical converter. A later
cross-residency exactness audit found that independently converting F32 KV rows
to Q8_0 on CPU and CUDA produced sparse byte differences and changed output at
generated token 5. F16 was an exact control. Host-resident quantized standard
KV for accelerator layers now converts the current rows in a bounded device
stage and performs a same-type byte-preserving scatter into host storage. At
ubatch 512 the Qwen3.8 target uses 17.00 MiB for 16 layers of K/V stages; the
one-layer MTP context uses another 1.06 MiB, for 18.06 MiB total. Clean
GPU, candidate GPU, and candidate pinned-CPU output then matched for target-only
1K, MTP-2/MTP-6 1K, exact-tail 128, and stochastic MTP-6 5K. This adds a D2H
transfer of newly written quantized rows. The final Nsight matrix measured the
complete route: pinned CPU full added 310,224.226 MB H2D and 214.858 MB D2H
versus GPU full over the 5K request, saved 246 MiB of profiled peak VRAM, and
reduced decode from 61.001 to 55.622 t/s. The much larger H2D delta comes from
CUDA attention over the growing host cache, not the bounded new-row store.

The placement policy now represents those decisions separately. Internal
`offload_kqv` continues to select persistent KV storage, while
`offload_attn_compute` controls whether graph construction forces the complete
attention region onto CPU. Ordinary pageable `--no-kv-offload` retains the
legacy CPU-attention placement. With operation offload enabled,
`--kv-cpu-pinned`, full KV offload, or a nonzero `--kv-gpu-layers` value keeps
attention accelerator-eligible without changing which cache layers are
persistent on the device. Explicitly disabling operation offload preserves the
legacy cache-derived CPU placement constraint; individual operations can still
be selected by an accelerator when their device inputs determine placement.
This makes the established pinned-host/CUDA-attention path explicit instead of
relying on the generic scheduler's operation-offload heuristic to override a
cache-derived CPU constraint.

Exact-tail planning follows the same separation. Its persistent storage buffer
type validates writes, while its planned attention execution backend validates
math/native-attention support. A final native-tail op is explicitly scheduled
on that execution backend. Without this split, pinned CPU storage incorrectly
selected CPU tail numerics even when operation offload placed attention on
CUDA. `--no-op-offload` still retains the deliberate CPU route.

At long context, each generated token requires scanning increasingly large K/V
history. Relevant costs may include:

- CPU memory bandwidth and cache locality while reading quantized K/V.
- Q8 dequantization/dot-product throughput.
- Work partitioning across attention heads versus context tiles.
- Transfers at the CPU/GPU graph boundary.
- Synchronization and launch overhead around those boundaries.
- Pageable-memory staging or registration overhead, which pinned memory appears
  to reduce substantially.

A generation-aligned `nvidia-smi dmon` sample at 64K later measured sustained
PCIe receive traffic of approximately 34-38 GB/s while Q8 CPU KV decoded at
`15.01` t/s. PCIe transmit was approximately 2.9-3.3 GB/s, SM utilization was
98-99%, and GPU power was only approximately 158-159 W at full reported clocks.
The receive rate is consistent with full-cache-scale host-to-GPU traffic per
token. Consequently, the current pinned-host path must not be described as a
purely CPU-computed attention path with only a small result crossing to CUDA.
The subsequent Nsight Systems trace resolved the ambiguity in favor of explicit
scheduler copies. Each measured 32K decode token issued 32 host-to-device
copies of 35,094,528 bytes: one complete K or V tensor for each of 16 attention
layers, or 1,123,024,896 bytes per token before smaller transfers. It then
launched one CUDA FlashAttention kernel per attention layer. The 64K PCIe rate
is therefore explained by copying the complete Q8 attention cache every token,
not by CPU FlashAttention scanning the history or a CUDA kernel directly reading
mapped host memory.

A matched unpinned trace produced the same copy sizes and counts and the same
CUDA FlashAttention launch count. Pinned allocation therefore did not change
backend placement. It reduced the average duration of the dominant 35,094,528-
byte copy from 1,698.773 microseconds to 624.707 microseconds, increasing its
effective transfer rate from 20.0 to 56.0 GB/s in the profiler. Experiment 001
accelerates the existing scheduler-copy path; it did not create that path.

The sharp degradation from excessive CPU threads indicates that synchronization,
shared-cache pressure, heterogeneous-core scheduling, or memory-bandwidth
contention can dominate nominal parallelism.

## Active hypotheses

The hypotheses below are ordered roughly by expected value and testability.

### H1: Hybrid recurrent-state placement dominated avoidable boundaries

Experiment 002 confirmed that boundary placement was central, but not solely at
the attention KV interface. CPU placement of fixed recurrent state caused the
48 recurrent layers to participate in CPU/GPU scheduling even though only 16
layers own long attention KV. Decoupling those policies more than doubled 4K
decode throughput.

The remaining task is to count graph splits and transferred bytes after the
change, then determine which boundaries are intrinsic to CPU attention KV and
which can still be removed or overlapped.

### H2: full-cache scheduler copies dominate long-context Q8 offload

The 32K trace observed 2,048 identical 35,094,528-byte host-to-device copies,
which is exactly 32 tensors times 64 measured tokens. Eliminating, overlapping,
or reducing these full-cache copies has higher priority than optimizing the CPU
FlashAttention loop. Software prefetch in that CPU loop cannot affect the
currently selected CUDA-attention path.

The next implementation investigation should focus on whether the scheduler can
operate on bounded context tiles, pipeline transfers with CUDA attention, or
retain a deliberately sized GPU-side cache window without restoring full GPU
KV residency.

A direct mapped-host CUDA experiment then removed staging copies by allowing a
discrete CUDA backend to consume `CUDA_Host` buffers when
`GGML_KV_CUDA_ZERO_COPY=1`. Nsight confirmed that the full-cache decode copies
disappeared, but CUDA FlashAttention averaged 8.016 ms per layer at 4K while
reading host memory directly. Decode fell to `10.15` t/s, prefill fell to
`462.01` t/s, and CUDA peak allocation remained effectively unchanged. The
exception was reverted. Direct mapped-host access is therefore not a viable
substitute for staging with the current CUDA kernel and scheduler reservation
model.

### H2b: Q8 CPU FlashAttention partitioning may matter only for a forced CPU-attention path

A guarded placement experiment forced standard Q8 FlashAttention onto CPU only
for single-token decode while retaining CUDA attention for multi-token prefill.
It eliminated the 32K decode-time 35,094,528-byte K/V copies, but reached only
`18.38` t/s at 4K and `3.34` t/s at 32K with all eight P-cores. The corresponding
CUDA-copy path reached approximately `43` t/s at 4K and `21.91` t/s in the
matched 32K memory run. CUDA peak allocation was identical at 32K because the
scheduler reserved the same buffers for both graph shapes. The implementation
was reverted.

This rejects unoptimized CPU FlashAttention as a direct substitute for the
current CUDA-copy path. CPU kernel changes such as prefetching cannot be treated
as small placement fixes: they would need to recover roughly 6.6x at 32K before
matching the existing path, as well as provide a mechanism that reduces the
scheduler's reserved CUDA buffers.

The rejected broad split-K switch does not eliminate more targeted scheduling
improvements. Context-depth-aware tiles, static work assignment, or avoiding
per-token synchronization may outperform head-only partitioning without the
overhead of generic split-K.

Any new partitioning experiment must explain why it differs from the neutral
split-K attempt and must test both 4K and long context.

The later 64K thread sweep produced `15.00` to `15.04` t/s across one through
eight P-core threads. Worker-only strict affinity also matched whole-process
pinning at `15.03` t/s. Ordinary thread-count and affinity tuning therefore
cannot recover the long-context loss. A future partitioning change must alter
how context tiles are exposed to workers or reduce synchronization; merely
requesting more CPU threads is not sufficient.

The implementation process must remain Bee-first: derive and measure a Bee
candidate without consulting another fork's source, and compare externally only
after the candidate is implemented and validated. The neutral fused-V and
two-pass-softmax experiments show why profiling evidence is required before
another Q8 kernel change.

### H3: Cache layout can improve sequential access and vectorization

Current K/V tensor strides may favor generality over the exact Qwen GQA decode
access pattern. A layout that keeps the context dimension contiguous for the
inner Q8 kernels, aligns per-head blocks, or reduces strided V access may improve
bandwidth utilization. Layout changes are higher risk because cache update,
state save/restore, shifting, and all attention consumers must remain correct.

### H4: Pinned allocation should be selective rather than all-or-nothing

Pinning the full maximum context is effective but can reserve several GiB of
unswappable RAM. Alternatives include a configurable pinned prefix, chunked
growth, a pinned transfer window, or pinning only tensors that actually cross a
CUDA boundary. These may preserve most of the speedup with lower locked-memory
cost, but dynamic registration itself can become overhead.

### H5: CPU ISA/kernel selection may leave Q8 throughput unused

The build uses `GGML_NATIVE=ON`, but the exact Q8 FlashAttention inner loop may
not make optimal use of this CPU's vector units, prefetch behavior, or core
types. Kernel-level profiling or isolated microbenchmarks are needed before
changing intrinsics. `perf`, Nsight Systems, and Nsight Compute were unavailable
during the initial investigation, so instrumentation may need to be added to
ggml or external profilers installed.

### H6: Standard formats below Q8 may offer a better speed/quality point

Q4_0, IQ4_NL, and Q5_0 were faster in the initial decode sweep. Q6_0/Q6_1 and
mixed K/V pairs should be evaluated with quality measurements, prefill, decode,
and VRAM. Q8 remains the baseline until a lower-width candidate meets an
explicit quality criterion.

The no-MTP 64K sweep strengthened this hypothesis: homogeneous Q6_0, Q5_0, and
Q4_0 CPU KV reached `16.23`, `17.85`, and `20.71` t/s respectively, versus
`15.03` t/s for Q8_0. Q6_1 reached `16.09` t/s in the shorter screening
protocol. This confirms that long-cache byte traffic is material, but the
nonlinear format ordering shows that kernel implementation and fixed costs also
matter. No lower-width format is approved until matched KLD/perplexity data is
recorded.

### H7: the remaining Q8 CPU-offload cost is predictably context-linear

No-MTP Q8 CPU KV measured `49.64` t/s at depth 128, `23.97` t/s at 30K, and
`15.03` t/s at 64K. The implied incremental latency is approximately `0.726`
microseconds per cached token, consistent with the earlier 4K-to-16K estimate.
This makes the current long-context loss predictable and supplies a direct
metric for future changes: a useful optimization must reduce the slope, not
only the fixed small-context intercept.

All-GPU measurements are retained only as comparison ceilings. Q8 GPU KV
reached `44.06` t/s at 30K and did not fit at 64K; narrower GPU formats fit at
64K and reached `32.22` to `37.10` t/s. Those results quantify the placement
gap but do not change the objective of accelerating CPU-resident KV.

## Previous benchmark-protocol edition

This section records the pre-server, `llama-bench`-centered protocol used by
the early experiments. It remains useful for interpreting those measurements,
but it has been superseded for current work by
[`cpu-kv-offload-current-testing.md`](cpu-kv-offload-current-testing.md). Do not
copy this section's command lines into a new experiment without reconciling
them against the current protocol and current CLI.

Use clean baseline and candidate processes. Do not compare a warm process with a
cold process or change affinity between runs.

Any run with potentially long or uncertain preparation or execution time must
expose progress. Add `--progress` to such `llama-bench` launches and preserve it
in the recorded command. If another tool has no native progress output,
document an external progress signal before starting it; do not launch a run
whose only observable states are running and finished.

Minimum throughput suite:

```bash
# Prefill
taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m MODEL -p 512 -n 0 -d 4096 -r 10 -t 3 --poll 100 \
  -nkvo 1 -fa on -ctk q8_0 -ctv q8_0

# Decode at 4K
taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m MODEL -p 0 -n 64 -d 4096 -r 3 -t 3 --poll 100 \
  -nkvo 1 -fa on -ctk q8_0 -ctv q8_0

# Decode at long context
taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m MODEL -p 0 -n 64 -d 16384 -r 3 -t 3 --poll 100 \
  -nkvo 1 -fa on -ctk q8_0 -ctv q8_0
```

For the pinned candidate, add `--kv-cpu-pinned`. Omit it for the ordinary
pageable-CPU-buffer baseline. Add `--recurrent-state-offload` when measuring the
retained hybrid placement policy.

While each clean process is alive, sample process GPU allocation repeatedly:

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits
```

Record the maximum observed allocation. Also record full configured system KV
size and, for pinned experiments, locked/pinned host allocation. `nvidia-smi`
VRAM alone does not account for CUDA host mappings or page-locked RAM.

Before accepting a layout or kernel change, run relevant unit/regression tests
and a correctness or quality comparison. Cache-format quality comparisons must
use `llama-perplexity` with identical `-b` and `-ub`, and both values must be
recorded.

## Build history and reproducibility notes

The experimental worktree is configured in `build-cuda-all`. It was initially
configured with `GGML_CUDA_FA_ALL_QUANTS=ON`, which generated the expanded 169
standard-pair matrix and caused a long first build. That build was stopped and
reconfigured with `GGML_CUDA_FA_ALL_QUANTS=OFF`. Dedicated KVarN CUDA support
remains enabled, so the default build still compiles its standard KVarN
templates even though the runtime experiment uses Q8.

ccache is enabled. The first build in a new worktree had few or no hits because
the original cache configuration did not normalize sibling-worktree paths. The
shared configuration is now `cache_dir=/home/gencoolpc/.cache/ccache`,
`base_dir=/home/gencoolpc`, `hash_dir=false`, and `max_size=50G`. Related
worktrees therefore share compatible objects. A genuine public-header content
change still invalidates every dependent translation unit; ccache cannot turn a
new dependency hash into a hit.

For allocator and parser iteration, use the separate CPU-only Ninja directory
`build-phase-dev`. Its first `test-alloc` plus `test-arg-parser` build completed
in 24 seconds on all 24 cores. Only a state that passes those cheap gates should
be built in the matched CUDA directory. The production RTX 5070 Ti build
requests CMake architecture `120`, which current CUDA handling compiles as
`120a`.

## Proposed next sequence

The matched 32K MTP depth sweep found initialized process VRAM of 14,290 MiB at
depth 1, 14,590 MiB at depth 3, 14,888 MiB at depth 5, and 15,338 MiB at depth
8. The fixed draft workspace did not grow. The target recurrent rollback buffer
grew from 299.25 MiB to 598.50, 897.75, and 1,346.62 MiB. On the artificial
repeated-token sample, depth 5 was fastest at 68.92 t/s; depth 8 fell to 61.60
t/s as acceptance dropped to 0.632 while using another 450 MiB.

The independent draft-context ubatch control remains available to non-MTP
model-backed speculation, but its MTP use is rejected. The original 64-token
MTP-5 screen reported identical visible output and draft counts while draft
ubatch 128 reduced live process VRAM from 14,922 to 14,834 MiB. That gate was
too short: a later 1,000-token MTP-2 run diverged from inherited-512 clean Bee
output at generated token 100 for draft ubatches 128 and 32. Trace logs showed
the first acceptance-cycle regrouping at token 60 after the 149-token prompt
was synchronized as `128 + 21` rather than one call.

The current fail-closed policy accepts an omitted value or an explicit value
equal to target ubatch, rejects other values for `draft-mtp` through CLI,
environment, and rendered INI paths, and leaves DFlash/other model-backed modes
unchanged. A rebuilt placement-matched MTP-2 1K run produced the clean Bee
token hash `b1e3f667bf3a2269f9ba2b0d41ca6f1229b1780250b173ba17db3fa0a9abad9a`
and content hash `fae6d743dc309784a909e015cc46450b7bfebfafb5e2a18c92c999206d019057`
for inherited phase-off, inherited phase-on, and explicit-equal phase-on.
Phase awareness reduced sampled peak process VRAM from 14,104 to 13,870 MiB
without changing the stream. This replaces the former MTP draft-ubatch
recommendation. A fresh equal-ubatch, cross-residency MTP-6 5K gate subsequently
matched clean GPU output for candidate GPU, pinned-CPU full planes, and
pinned-CPU three planes. Decode was 63.281/63.223/57.200/55.363 t/s for clean
GPU/candidate GPU/CPU full/CPU capped, and sampled peak VRAM was
15,006/15,006/14,514/13,926 MiB. The old 140K phase-aware pair remains
historical, but the supported equal-ubatch final Nsight transfer matrix was
subsequently completed and is reported in the final theory update below.

The supported CPU-KV placement controls now follow the existing cache ownership
layout: `llama_model::create_memory()` derives attention and recurrent placement,
while `llama_kv_cache` builds one per-layer buffer-type plan used by both route
probing and allocation. Standard and KVarN caches share pinned-host buffer
selection. This replaced the first partial-residency implementation's parallel
layer bitmap and duplicated placement branches without changing the measured
4K allocation split.

That ownership plan is now also exposed independently for a separate draft
context. `--spec-draft-kv-gpu-layers N` (alias
`--kv-gpu-layers-draft`) overrides the inherited target KV policy only while
`common_base_params_to_speculative()` derives the draft parameters. Omission
continues to inherit the target policy; an explicit count enables the same
partial-placement path for draft-owned layers. Shared layers remain excluded
from the count by `llama_kv_cache`, so this is architecture-neutral and does
not create a second placement mechanism. The option is independent of draft
weight offload and of the phase-aware target/draft compute-backing group.

After merging the output-exactness work, the independent policy passed both
the maintained 1,000-token greedy and 5,000-token stochastic MTP-6 gates with
target Q8 KV in pinned host memory, draft-owned Q8 KV on CUDA, phase-aware
workspace enabled, and three recurrent planes. Both cases matched the clean
same-geometry MTP oracle token-for-token and response-byte-for-byte. The 5K
token SHA-256 was
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`.
This validates the ownership split exercised by the integrated-MTP model; it
does not generalize arbitrary internal partial-layer mixes, cache formats,
models, or backends.

The non-MTP 90K-versus-240K Nsight allocation trace isolated context-scaled
VRAM to the target scheduler's prompt graph. Model weights and the 149.62 MiB
active recurrent state were constant, while CUDA compute grew from 707.27 to
1,751.09 MiB. The exact 7,296-byte-per-cell GPU slope is 4,096 bytes of F16 K/V
materialization inside prompt FlashAttention, 1,088-byte Q8 K staging,
1,088-byte Q8 V staging, and a 1,024-byte F16 attention mask. A second
1,024-byte-per-cell mask lives in pinned host compute memory. The best complete
solution is fixed-window online-softmax KV streaming; the most isolated smaller
candidate is replacing the explicit causal mask with compact position metadata.

### Follow-on branch: configurable MTP recurrent planes

The dedicated branch `exp/mtp-recurrent-plane-cap` investigated a bounded,
configurable pool of target recurrent-state rollback planes. It branches from
the CPU KV-offload work because the motivating capacity problem is visible when
CPU-resident Q8 KV leaves recurrent state on the GPU, but the implementation is
an MTP state-management change rather than partial KV placement.

The current implementation reserves one committed recurrent state plus one
state for every possible MTP position. Each additional state costs about
149.625 MiB for the Qwen3.8 27B test model, producing allocations of 598.50 MiB
at MTP depth 3, 897.75 MiB at depth 5, and 1,346.62 MiB at depth 8.

The first candidate used the existing consecutive rollback snapshots plus a
target host checkpoint. Four total planes reduced the recurrent buffer to
598.50 MiB at every tested depth, saving 299.25 MiB at MTP-5 and 748.12 MiB at
MTP-8. It captured a checkpoint only when the actual draft exceeded the
three-token direct horizon and restored/replayed only when the rejected suffix
also exceeded it. Omitting the option preserved full-plane behavior.

That compatibility-first design was rejected. The 128-token fixed-seed outputs
matched at MTP depths 3, 5, and 8, but the 5,000-token MTP-8 orbital-sandbox
comparison diverged at byte 1,337. Two clean full-plane controls matched each
other, isolating the difference to capped replay. Replaying the accepted prefix
uses a smaller target batch than original verification, so it does not recreate
the original recurrent state bit-for-bit on CUDA. Preserving the first pass's
sampled token was insufficient because subsequent logits still consumed the
numerically different reconstructed state.

The checkpoint cost was also material. In the long capped run, 132 captures and
18 restores serialized 20.71 GB of target payload, used 2,888.89 ms capture and
167.14 ms restore wall time, and added 20.64 GB D2H plus 10.82 GB H2D in Nsight.
In the short test, four planes reduced MTP-5 decode by 34.1% and MTP-8 by 38.1%.
The exact commands and resource ledger are in Experiment 012.

This changed the roadmap: sparse GPU snapshot selection was required for
correctness, not merely as a checkpoint-traffic optimization. The second
candidate implements that path behind two explicit capabilities: the model
graph must support selected recurrent snapshots and every recurrent-state
buffer backend must advertise the corresponding operation. NVIDIA CUDA is the
current backend provider; unsupported graphs and backends fail closed without
an architecture-name check in recurrent memory. During ordinary capped
verification it retains the pre-verification input plus the latest output
boundaries. A rejection beyond the direct horizon reruns the original full
verification batch shape and writes only the selected accepted boundary. This
reproduces the first pass's state exactly without serializing a target host
checkpoint.

The second candidate passed the long correctness gate. The matched 5,000-token
MTP-8 orbital run produced identical full/capped reasoning output and identical
draft counts. Four planes reduced the recurrent allocation from 1,346.62 MiB
to 598.50 MiB. Under Nsight, decode changed from 51.38 to 50.98 t/s (-0.79%);
the cap performed 38 replay cycles containing 217 actual batch tokens and used
zero target checkpoint captures/restores. It added 3,938.563 MB H2D plus
250.206 MB D2H to the complete trace. An initial selected-boundary matcher miss
also added 4,888.461 MB D2D; accepting the one-plane fused destination removed
that traffic entirely. Experiment 013 records the exact command, short sweep,
memory ledger, and transfer totals.

The working theory is therefore revised again: deterministic sparse GPU replay
is a viable capacity/performance trade for this Qwen CUDA configuration. Host
checkpoint/replay remains a rejected diagnostic. The interface boundary is now
capability-based, leaving backend/model expansion independent of recurrent
memory. Lower cache quants, asymmetric K/V pairs, implementations of the
capability on other backends/models, and unmeasured partial target-layer or
multi-layer draft mixes remain out of scope.

The standalone
[`mtp-recurrent-plane-cap-reproduction.md`](mtp-recurrent-plane-cap-reproduction.md)
records the complete process: workspace and ccache setup, failed candidates,
exact validation and profiling commands, artifact hashes, and CPU-KV
integration procedure.

### Follow-on branch: phase-aware target and MTP workspace

The dedicated `exp/phase-aware-prefill-decode` worktree at base
`324873dc5ca44eb31727ba3bd09897841574fa3b` tested the next largest avoidable
VRAM category: retaining prompt-sized target and MTP compute reservations while
only generation graphs are active. This is a workspace-lifetime change, not an
attempt to unload MTP weights or persistent speculative/recurrent state.
The retained implementation was merged back into this branch as
`20777977d288fbb72e9541c1e982785e90d75993`.

That historical benchmark used target/draft ubatches 512/128 on both sides. A
later 1K exactness audit rejected the smaller physical MTP ubatch because it can
change the clean-Bee stream after token 100. The allocator design remains
retained, and a fresh inherited/equal-512 MTP-2 run is exact, but the 5K/140K
performance and memory rows below remain historical until the supported
equal-ubatch matrix is remeasured.

The retained opt-in `--phase-aware-workspace` policy starts each context with a
generation reservation, grows to its full physical ubatch for prompt work, and
shrinks after returning to generation. The generation bound includes parallel
sequences and the resolved speculative horizon. Fit/no-allocation measurement
still uses the full prompt geometry. Later requests on the same server context
repeat the transition safely.

Integrated MTP creates a second scheduler, so resizing each private scheduler
was insufficient: two prompt high-water allocations would still coexist. The
implementation adds a generic GGML shared-backing group below the schedulers.
Each scheduler retains its graph plan, while physical chunks are keyed by exact
backend buffer type and sized to the maximum current member requirement rather
than their sum. Target and MTP execution remains sequential and uses explicit
ownership fences. A physical-generation counter invalidates peer graph
addresses after replacement.

The initial automatic-shrink protocol exposed a performance bug. Recurrent
checkpoint replay legitimately replans graph variants, and every smaller plan
was mistaken for a phase reduction. A short request with four replay cycles
performed six target plus six draft reserves; the 5K request with 88 replays
performed 90 plus 90 and spent 728.11 ms rebuilding workspaces. An intermediate
per-member high-water permission did not solve it because every extra reserve
was caused by peer plan-generation publication. The final protocol begins one
explicit group shrink epoch only when token geometry crosses from prompt to
generation. Growth remains immediate; shrink waits until every active member
has published once. The same replay-heavy cases now perform exactly two target
and two draft reserves: one grow and one shrink per context.

The clean 140K MTP-6/Q8 comparison retained the candidate. Initialized and
steady process VRAM fell from 15,768/15,800 MiB to 14,660/14,692 MiB, saving
1,108 MiB. Peak fell from 15,800 MiB to 14,874 MiB in the 5K case and 14,898
MiB in the 138K case, saving 926 and 902 MiB. The active generation backing was
840.82 MiB CUDA plus 2.41 MiB CUDA-host memory. The uncapped baseline retained
separate target (1,054.62/157.03 MiB) and draft (892.05/39.46 MiB) plans.

Performance is an explicit trade rather than a free win. The plain 5K coding
run changed decode from 52.10 to 51.15 t/s (-1.83%) and wall time from 96.21 to
98.02 seconds. Its target/draft transition work was 32.48/9.31 ms. The matched
138K prompt changed prefill from 742.52 to 741.66 t/s (-0.12%) and total wall
time from 187.96 to 188.03 seconds; its 64-token decode tail was too short for a
stable throughput conclusion. The Nsight pair changed decode from 50.50 to
50.27 t/s (-0.44%). H2D/D2H bytes and operation counts were exactly identical,
so no hidden transfer mechanism explains the remaining small decode cost.

All short, 5K, 138K, default-off, and same-process second-turn fixed-seed output
hashes matched their baselines. Draft counts, accepted counts, replay cycles,
and replay-batch tokens also matched. The final focused suite passed 9/9 and
the broad suite passed all 93 remaining tests after the three independent
upstream/fixture failures were classified. Experiment 016 contains
the authoritative ledger, and
[`phase-aware-workspace-reproduction.md`](phase-aware-workspace-reproduction.md)
records the complete work process.

This revises the working theory: about 1.1 GiB of the apparent CPU-KV serving
floor was inactive scheduler high-water and target/draft coexistence, and can
be removed without new attention kernels or transfer traffic. The remaining
prompt peak is real active target workspace with the previously measured
context-linear F16 materialization, Q8 staging, and explicit mask. Compact mask
metadata is therefore the next contained VRAM target; direct-Q8 prompt MMA and
fixed-window streaming remain progressively larger follow-ons.

The output-exactness audit adds one constraint to that roadmap: CPU/GPU cache
residency may change storage location but must not silently change the
persistent quantized bytes when accelerator attention remains the execution
policy. The retained 17 MiB conversion stage is small compared with the
hundreds of MiB saved by phase awareness and recurrent-plane capping. Its
new-row D2H traffic is included in the final profiler ledger. See
[`mtp-output-exactness-reproduction.md`](mtp-output-exactness-reproduction.md)
and Experiment 018.

The follow-up same-process MTP-6 lifecycle matrix also passed across GPU and
pinned-CPU Q8 residency, phase awareness off/on, and the three-plane cap. It
exercised prompt-cache reuse, prefix shrink, regrowth from an earlier longer
branch, and two observed sleep/unload/wake/reload cycles. This removes those
direct-server lifecycle paths from the current list of suspected correctness
risks for the supported Q8/CUDA configuration. Explicit router
`/models/reload`, other cache formats/backends, and authoritative profiler
transfer accounting remain separate gates; the lifecycle result does not
generalize them by implication.

The explicit router gate subsequently passed as well. A preset metadata change
forced each live model from `loaded` to `unloaded`; the following identical
request autoloaded a new child process. Clean GPU, candidate phase-aware GPU,
candidate phase-aware pinned CPU, and the three-plane pinned-CPU candidate all
matched before/after within the case and matched clean Bee across cases. This
closes the supported Q8/CUDA lifecycle gate. The remaining immediate evidence
gap was authoritative Nsight transfer accounting for the final canonical-store
implementation, not another lifecycle-specific code path.

That final profiler matrix now changes the performance theory materially. With
phase awareness held constant, GPU-resident full planes recorded 17,485.510 MB
H2D and 5,700.920 MB D2H over the 5K request. Pinned-CPU full planes recorded
327,709.736 MB H2D and 5,915.778 MB D2H: 310,224.226 MB more H2D for only 246
MiB less profiled peak VRAM, with decode falling from 61.001 to 55.622 t/s.
This confirms that normal operation offload is an accelerator-attention path
over host-resident persistent KV, not a CPU-attention path. The dominant
transfer cost is repeatedly making the growing host cache available to CUDA;
the canonical new-row store's D2H direction is much smaller.

The three-plane CPU case saved another 588 MiB, but its 90 selected replay
cycles/443 replay-batch tokens added 10,045.292 MB H2D and 465.483 MB D2H and
reduced profiled decode another 2.80%. Output, acceptance, and all checkpoint
counters remained exact. This reinforces the roadmap distinction: recurrent
plane capping is a controllable VRAM/throughput trade, while materially
improving CPU-KV serving performance requires reducing accelerator access to
the full growing host cache (for example bounded GPU windows/streaming) or a
separately competitive native CPU attention path. Store-stage tuning alone
cannot remove the 310 GB residency traffic.

The final acceptance-edge audit does not change that theory, but closes the
remaining supported-Q8 correctness bias. Full-plane depths 1/2/3/5/6/8 and
2-/3-/4-/full-plane policies at depths 3/5/8 all matched clean Bee. Two
stochastic seeds on an independent number-game prompt at `p_min=0` forced
43/69 selected replay cycles and 387/621 replay-batch tokens through a
two-plane, zero-token direct horizon; GPU and pinned-CPU candidates remained
exact. A third seed and website prompt at `p_min=0.999` accepted every actual
draft and used no replay. The cap remains a workload-dependent trade: the
rejection-heavy two-plane rows were much slower, so production selection must
be based on measured acceptance/replay behavior rather than VRAM alone.

### Perplexity full-batch output contract

Phase-aware contexts default to serving-shaped output capacity: one row per
sequence. That is too small for `llama-perplexity`, which marks every scored
token in a decode slice for logits. The allocator must continue to reject an
undeclared request, so the correction belongs at the tool boundary. Perplexity
now sets `params.n_outputs_max = params.n_batch` before context creation. This
is exactly the capacity inherited by the ordinary non-phase-aware path and
does not change graph arithmetic, cache placement, or server behavior.

The focused plumbing test guards both the declaration and its ordering. A
matched default/phase-aware/default GPU-PPL sequence reproduced the same
cumulative estimate in every process; the isolated evidence is recorded in
the experiment ledger below.

1. Implement a fail-closed compact causal-mask descriptor for the supported
   single-slot contiguous layout and measure the predicted GPU and CUDA-host
   savings.
2. Reassess full prompt peak after removing explicit-mask storage; pursue
   direct-Q8 prompt MMA only if its larger kernel/tuning cost remains justified.
3. Treat bounded fixed-window GPU streaming as the long-term path when active
   prompt workspace must remain approximately flat with maximum context.
4. Measure pinned system-memory size and CUDA mapping overhead separately from
   `nvidia-smi`; `VmLck=0` does not prove CUDA-host allocations are pageable.
5. Sweep lower homogeneous cache widths only with matched quality validation;
   asymmetric K/V and token-window KV residency remain outside the current
   scope.

## 2026-08-20: profiler affinity protocol correction

A derived worktree's current-testing draft exposed that historical
whole-process `taskset` commands were being reused around Nsight Compute.
Putting `taskset` after `ncu` made the wrapper the application-only profiling
target and produced a no-kernel capture. Moving it before `ncu` attached to the
llama binary but restricted the profiler, replay machinery, helper processes,
and target to the same small CPU set. A later all-process capture could follow
the wrapper, but that was a workaround for the launch shape rather than a
reason to retain it.

The current protocol now forbids `taskset` in CPU-KV benchmark and profiler
commands. Nsight launches the llama binary directly and remains unrestricted;
only llama.cpp worker pools receive native affinity controls. For the former
logical-CPU 0--2 `llama-bench` shape, the translation is `-C 0x7
--cpu-strict 1`. Server and common-argument tools retain their explicit
`--cpu-range`, `--cpu-range-batch`, and `--cpu-strict` controls. An outer GPU
coordination lock may serialize runs but must not alter affinity.

Historical ledger and reproduction commands remain unchanged so their exact
measurements can still be reconstructed. Their `taskset` spelling is now
explicitly superseded for new work. Existing Nsight hardware counters are not
invalid merely because a profiler was CPU-restricted when it successfully
captured the intended kernel, but profiler timing is not benchmark evidence;
no-kernel captures and wrapper-dependent all-process captures require a clean
direct-target rerun before they support a current claim.

## 2026-08-21: shared VRAM documentation and publication lanes

Shared documentation was consolidated from exact local KV tip
`6a20757854395309b32248dd4109d73e99c3e675`. The published source base is now
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9` after PR 5 and PR 6 merged. The
read-only parallel snapshot `refs/codex/pre-pr4-parallel-source` at
`f52988ee150cd27a94d6897cc049326c1e77c3e2` remains forensic research history,
not an integration source. It combined multiple features in one uncommitted
tree, so cumulative performance cannot establish an isolated feature claim.

Merged support lanes are PR 5 perplexity capacity at
`8d2f8452eb140ba52d8472ecd791cc90212a9307`, merged as
`50ee5b2d765c91a0d9cd23728ac17a27ac510e3e`, and PR 6 physical/VMM telemetry
at `3bd7a088199922b1e5e20973cd8cb6d970cde111`, merged as the current base
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`. Remaining independent lanes are
PR 4 native standard-quant attention at
`72ee96bbfcf91c17a7fb5b3b32703aae812af330` and PR 8 live-context workspace at
final head `0c8df007a504f16aa35fc5982303e3e1b9883331`. PR 7 is final at
`d4183adb8b4902a125b9339cd39032a095fca013`, directly based on `4a7f9b496` and
six commits ahead. PR 8 is likewise directly based on `4a7f9b496`, six commits
ahead, draft, open, mergeable, and clean. PR 7's resource/performance evidence
remains the isolated c9 dense/Candidate-1/dense A/B/A study; the rebased source
validation at
`ae60c7321d950937a36af096112525db777ae13f` is a separate composition gate and
does not relabel those measurements.

The retained PR 7 result is fail-closed compact causal-prefix metadata with
per-consumer views that remove the measured long-lived-copy allocator hole.
Serving output and matching-batch PPL were exact, context-scaled VRAM savings
were measured through 128K, and the focused repeated 128K decode screen rejected
a material slowdown. Invalid setup/probe launches and noisy rejected timing are
not shared evidence; exact identities, commands, exclusions, and limitations
remain in PR 7's branch-owned investigation.

PR 8's final Experiment 021 composes its default-off live-context workspace
with merged W02/W06. Omitted/live/explicit-off serving output was byte-exact at
1K and across the 32K lifecycle, and matching-geometry PPL was identical. At
32K the live policy saved 200 MiB of startup process VRAM and 212 MiB on the
post-shrink next turn; repeated 4K/30K prompt and decode throughput was neutral.
Its separate synchronized CUDA device-used high-water exposed +8 MiB at 4K
prefill and +56 MiB at 30K prefill, with no decode cost. Those values are
`cudaMemGetInfo`-style device usage, not `nvidia-smi` process VRAM. Only the
accepted explicit-Bash process-substitution launches are valid provenance;
rejected version/setup probes are excluded.

PR 8 continues to own its source, presets, and every user-facing generated
`--live-context-workspace` document until the source lands. The current
protocol, roadmap, feature-isolation plan, and source-backed feature delta are
shared KV inheritance. PR 7 and PR 8 identities and all pairwise comparisons
have been refreshed at their final heads. This consolidation does not merge or
fast-forward the published KV base.

## 2026-08-20: pre-PR4 W02 telemetry isolation

The allocation-classification and CUDA VMM high-water instrumentation from the
immutable pre-PR4 source snapshot was migrated independently onto
`c9f727c1e1995c4a871a719ab05b5f2478588efd`. This migration intentionally does
not depend on the snapshot's native-Q8 reporting field and does not carry its
live-context workspace, later phase controls, causal descriptors, VMM trimming
policy, host staging, or perplexity-capacity fix.

The first ordinary-host runtime probe found that the snapshot's legacy
reconciliation assertion still subtracted all resident KV from a CUDA-owner
total that deliberately excludes ordinary CPU buffers. The migration was
revised to subtract KV only when it is device-resident or CUDA-pinned. Physical
device, accelerator-host, and ordinary-host fields remain classified by buffer
capability and owning device type rather than architecture names.

Fresh final-binary runs established a dormant zero-valued default path, neutral
source and on/off performance screens, repeatable VMM peaks, all three physical
allocation classes, and identical matched perplexity. Experiment 020 contains
the exact commands and evidence. The result remains support instrumentation,
not a VRAM optimization or permission to infer a trimming policy.

## 2026-08-21: F32 GDN recurrent snapshot-fusion regression

The old isolated feature head
`06cb666f8d07f345840881e9720d2a0756555f33` recovered one useful test from
the immutable `refs/codex/pre-pr4-parallel-source` snapshot: an F32
whole-graph backend-op fixture for the existing CUDA gated-delta-net
recurrent-cache copy fusion. The ordinary GDN matrix validates the packed
operator result but does not construct the model graph's
`GATED_DELTA_NET -> VIEW -> CPY` cache write and therefore cannot cover the
direct snapshot store.

After shared PR 9 merged, the same test patch was rebased without behavior
changes onto exact `beellama-kv-cpu-offload` base
`8e858fcec39049fa028ce6fcb144a0c08b03abd3` as source commit
`77aa9f640f97349cf04ca6a5f50a1a7910a496b0`. It continues to use the
upstream-style `run_whole_graph()` and `fusion_test_nodes()` hooks and mirrors
the cache view in `llm_build_delta_net_base::build_recurrent_attn()`. No F16
state, compact masking, native Q8, live workspace, telemetry, staging, or
unrelated snapshot code was imported.

The appended post-PR9 test-only record owns the exact build and final-identity
CPU/CUDA validation evidence. Merge-readiness validation used complete Release
CPU and CUDA builds at `4b71f6370ef220c91f3641fc1ea4bb25ecb2674a`, capped
at six build jobs. The focused fixture, existing 36-case GDN matrices, and seed
guard passed on CPU and CUDA. Broad practical CTest passed 96/96 on CPU and
95/95 on CUDA; the record identifies the excluded unavailable Git LFS vocab
data case and unrelated unfiltered FlashAttention matrix abort. This change
adds no production source, runtime behavior, performance claim, or perplexity
claim.

## 2026-08-20: live-context workspace protocol edition

The current-testing protocol now names live-context workspace sizing as an
independent default-off control. The previous edition described only
phase-aware token geometry and therefore could not reproduce W04/W05/W09 in
isolation. Current live-sizing comparisons vary only the positive/negative
live control, require a memory layout that explicitly publishes bounded
attention reservation, and keep every other optional workspace policy fixed.
Unsupported layouts retain full reservation and the established upfront
decode order. CUDA transient-pool trimming is restricted to the all-slots-idle
boundary after effective live sizing; it is not a per-decode reclamation path.

## 2026-08-21: compose W02 telemetry with live idle trimming

The published KV-offload base now contains W06 full-batch perplexity capacity
and W02 allocation/VMM telemetry. Rebasing the isolated live-workspace work
onto that base preserves both historical Experiment 020 entries as measured;
neither prior source identity nor result is relabeled as composed evidence.

The VMM pool now enrolls in W02 telemetry before an idle trim and subtracts
only the released physical mapping from the current mapped-byte counter. Live
bytes and mapped high-water remain unchanged by trim. No new benchmark trim
caller or lifecycle checkpoint is shipped; composed validation observes the
existing server all-idle boundary, and the default path remains dormant.

## 2026-08-21: require the exact PR 8 disabled-source bracket

The prior PR 8 refresh relied on production-path equivalence with its earlier
measured head. That remains useful provenance for the enabled implementation,
but it cannot prove that compiling the opt-in source leaves omission and
explicit off neutral against the exact base. The final gate therefore uses
identically configured exact-base and candidate builds with fresh base /
candidate / base processes at both 4K and long context. Omission and explicit
off are separate candidates, and every unrelated workspace/speculative option
is fixed off or absent.

The maintained exactness runner now captures a health-ready process/GPU sample
before the first request. This makes startup time, process VRAM, ordinary RSS,
shared/pinned-backed RSS, and later lifecycle peaks distinct artifacts rather
than inferring startup from the first prefill. The current preflight template
also places CUDA-linked version probes and `nvidia-smi` inside the whole-command
GPU lock; the previous unlocked example was a protocol hazard, not evidence.

The completed bracket compared exact base `8e858fcec39049fa028ce6fcb144a0c08b03abd3`
with candidate runtime source
`4cdd2d74e7acc432fcdde4a9d1e5e832fe80e148`. Both omission and explicit off
retained the same one-time upfront compute reservation as base, produced exact
1K and 32K lifecycle output, matched PPL bit-for-reported-bit, and reproduced
all W02 CUDA/VMM and physical allocation classes. Repeated 4K and 30K
throughput stayed inside the three-sample base bracket. The candidate's lower
sampled `RssShmem` is not claimed as a saving because it is outside W02's
allocator-owned classification and is not an enabled feature effect. No
disabled-path production change is justified by this evidence.

## Cross-hardware DSpark comparison and open issues

Characterizations 020-026 preserve a historical protocol run from the PR 3
recovery line. They compare baseline, `draft-mtp`, and for the first time
`draft-dspark` on a second,
weaker machine (RTX 4070 12 GiB, Qwen3.8 IQ2_M) already used for
Experiment 011, in two passes: once against `53adab814` and again after
rebasing onto `c9f727c1e` (Experiments 017-019). These measurements and their
archived command snapshots are not the current runnable protocol; current work
must follow `cpu-kv-offload-current-testing.md`, including native affinity and
the whole-lifecycle GPU lock. One bug found there is fixed by PR 3; two items
need follow-up before this branch's MTP-oriented optimizations should be
described as covering DSpark too, and one is an unresolved stability issue
independent of DSpark:

6. **Fixed.** `common_speculative_resolve_dflash_draft_n_max`'s type check
   (`common/common.cpp`) now also matches `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`,
   so `--spec-type draft-dspark` without an explicit `--spec-draft-n-max`
   resolves the drafter's actual trained block depth from
   `dflash.block_size` instead of silently using the generic default of 3
   (Characterization 024).
7. Investigate whether the recurrent-plane-cap mechanism
   (`--spec-mtp-rs-planes`, Experiments 012/013) can be generalized to
   DSpark's rollback planes, or document explicitly that DSpark is expected
   to keep paying the full uncapped `draft.n_max`-plane cost indefinitely
   (Characterization 025).
8. **Not fixed; root cause narrowed but not pinpointed.** An intermittent
   `ggml_backend_cuda_synchronize` "illegal memory access" crash was
   reproduced in 5 of 6 sustained (500+ token) MTP generations without
   `CUDA_LAUNCH_BLOCKING=1`, and 1 of 3 eligible inference runs with it,
   across both the pre- and
   post-`c9f727c1e` binaries — ruling out the MTP draft/target-ubatch
   mismatch (Experiment 017) and the recurrent-plane cap (Experiments
   012/013) as the cause. The leading hypothesis is a race between the
   target's per-step CPU-KV host-to-device staging copies and MTP's
   concurrent draft-context GPU submissions, since MTP is the only
   configuration in this session with two GPU-active contexts running
   per decode step; DSpark's CPU-resident draft (item 9) never creates
   that condition and never crashed. A `compute-sanitizer --tool memcheck`
   run was started but abandoned after 20 minutes at ~140x slowdown without
   reaching a crash. Two other launch-blocking artifacts failed during model
   loading and are not part of the inference-run denominator. A future session
   should either budget several hours
   for that tool or instrument
   `llama_context::process_ubatch`'s `can_reuse(gparams)`
   (`src/llama-context.cpp`) and the per-input `can_reuse` overrides in
   `src/llama-graph.cpp` directly. Treat MTP as not proven safe for long
   unattended generations until this is understood, independent of any
   DSpark-related work (Characterization 021).
9. Before recommending DSpark for VRAM-constrained hardware, characterize a
   GPU-resident DSpark draft on a card with more headroom above the target
   weights than this RTX 4070 had (Characterization 022 could not get a
   GPU-resident draft to load at any tested depth down to 2). The CPU-draft
   throughput regression measured here (Characterization 023) may not
   generalize to hardware where the draft model fits on-device.

## Known non-goals

- Do not restore TurboQuant/TCQ, DDTree, CopySpec, the removed fork DFlash
  ring/tape or verifier paths, or other systems removed in BeeLlama v0.4.0.
- Do not silently change cache formats or reinterpret KVarN records.
- Do not make pinned allocation the unconditional default without a resource
  policy and failure behavior appropriate for large contexts.
- Do not optimize only the 4K microbenchmark at the expense of long-context
  decode, prefill, correctness, or memory accounting.
