# CPU KV-offload development journal

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

Each experiment should be one commit containing its implementation and an
update to `cpu-kv-offload-experiments.md`. Rejected experiments should normally
be reverted; their result should still be recorded here or in the experiment
ledger. No benchmark result is portable unless its model, command, hardware,
settings, and commit are recorded.

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
not a broad BeeLlama regression.

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
buffer type backed by `cudaMallocHost()`. Experiment 001 selects that device
host buffer for CPU KV when `GGML_KV_CPU_PINNED=1`.

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

The implementation remains opt-in through
`GGML_RECURRENT_STATE_OFFLOAD=1`. The environment switch is deliberately not a
public CLI/INI promise yet. MTP creates recurrent rollback snapshots, so its
additional GPU-state cost and correctness must be measured before choosing a
serving default.

### 13. CPU KV does not make VRAM independent of context

Live testing showed that GPU memory can still rise when configured or populated
context grows even though the target Q8 K/V buffers report `CUDA_Host`. CPU KV
offload controls the long attention K/V storage and the CPU attention region;
it does not move every context-shaped tensor or every auxiliary state out of
CUDA memory.

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

With `--no-kv-offload`, the KV buffers and the graph section from KV store
through attention output are placed on CPU. Model weights and the rest of the
graph remain GPU-resident. Standard Q8 KV uses the ordinary CPU FlashAttention
implementation rather than KVarN's record-native paths.

At long context, each generated token requires scanning increasingly large K/V
history. Relevant costs may include:

- CPU memory bandwidth and cache locality while reading quantized K/V.
- Q8 dequantization/dot-product throughput.
- Work partitioning across attention heads versus context tiles.
- Transfers at the CPU/GPU graph boundary.
- Synchronization and launch overhead around those boundaries.
- Pageable-memory staging or registration overhead, which pinned memory appears
  to reduce substantially.

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

### H2: Q8 CPU FlashAttention partitioning is suboptimal for three P-cores

The rejected broad split-K switch does not eliminate more targeted scheduling
improvements. Context-depth-aware tiles, static work assignment, or avoiding
per-token synchronization may outperform head-only partitioning without the
overhead of generic split-K.

Any new partitioning experiment must explain why it differs from the neutral
split-K attempt and must test both 4K and long context.

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

## Benchmark protocol for every code change

Use clean baseline and candidate processes. Do not compare a warm process with a
cold process or change affinity between runs.

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

For the pinned candidate, prefix the command with
`GGML_KV_CPU_PINNED=1`. For the ordinary baseline, explicitly remove that
variable with `env -u GGML_KV_CPU_PINNED`.

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
path-dependent compilation prevented reuse from the baseline build. Subsequent
source-only rebuilds should reuse this worktree's cache. The RTX 5070 Ti build
uses CUDA architecture `120a` as selected by current CMake/CUDA handling.

## Proposed next sequence

1. Run output-equivalence and recurrent-state regression tests for the separated
   placement policy.
2. Measure no-MTP and MTP depth-1 serving with realistic prompts, including
   recurrent rollback VRAM, acceptance, prefill, decode, and total VRAM.
3. Record startup, post-warmup, fixed-live-depth, and post-idle VRAM; distinguish
   static workspace growth from CUDA graph-capture growth.
4. Attribute target compute, recurrent state, MTP draft compute, boundary
   buffers, and graph-cache allocations before attempting to reduce them.
5. Count graph splits and boundary-transfer bytes before and after recurrent
   offload so the remaining CPU-attention boundary cost is explicit.
6. Measure pinned system-memory size and CUDA mapping overhead at 4K, 16K, and
   the intended maximum serving context.
7. Prototype further Bee-derived Q8 scheduling or layout changes only when the
   measurements identify a specific remaining bottleneck.
8. Compare the validated Bee implementation with external implementations only
   after the Bee candidate is complete; do not use external source to design the
   candidate.
9. Evaluate selective/chunked pinning if full-context pinned memory is too costly.
10. Sweep Q6/Q8 and mixed standard K/V pairs with throughput, VRAM, pinned RAM,
   and quality measurements.
11. Replace the environment-only switches with supported CLI/INI policies only
   if their allocation behavior survives the full benchmark and resource suite.

## Known non-goals

- Do not restore TurboQuant/TCQ, DDTree, CopySpec, the removed fork DFlash
  ring/tape or verifier paths, or other systems removed in BeeLlama v0.4.0.
- Do not silently change cache formats or reinterpret KVarN records.
- Do not make pinned allocation the unconditional default without a resource
  policy and failure behavior appropriate for large contexts.
- Do not optimize only the 4K microbenchmark at the expense of long-context
  decode, prefill, correctness, or memory accounting.
