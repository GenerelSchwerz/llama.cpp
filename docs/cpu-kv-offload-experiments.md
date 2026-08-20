# CPU KV-offload experiments

For the current runnable build, server arguments, exactness oracle, benchmark
shape, progress mechanism, and required artifacts, use
[`cpu-kv-offload-current-testing.md`](cpu-kv-offload-current-testing.md). This
ledger intentionally preserves the exact commands used by each historical
experiment, including retired environment variables and execution geometries.
Those commands explain their recorded results; they are not templates for a
new run.

Candidate VRAM reductions and their complexity ordering are maintained in
[`cpu-kv-offload-vram-roadmap.md`](cpu-kv-offload-vram-roadmap.md). This ledger
remains the source of exact commands and measured acceptance or rejection.

This document records changes made on the `exp/kv-cpu-offload` branch. Each
accepted or rejected experiment should have its own commit so that its complete
diff remains inspectable. Do not treat these measurements as portable benchmark
claims; they apply only to the configuration recorded below.

## Baseline

- BeeLlama base: `ba27edad2a84ff045a556df06661e821285c2fab`
  (`v0.4.3`, merge commit)
- Clean llama.cpp architectural reference at the time of the experiment:
  `af5172627` in `../llama.cpp`
- Model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`
- Model reported by `llama-bench`: Qwen3.5 27B, 27.32B parameters,
  13.44 GiB, Q8_0
- GPU: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB, compute capability 12.0
- CPU: Intel Core Ultra 9 285K
- Decode affinity: physical P-cores 0-2 via `taskset -c 0-2`
- Build: Release, CUDA, native CPU, CUDA FlashAttention, CUDA architecture
  `120a`, default quant matrix (`GGML_CUDA_FA_ALL_QUANTS=OFF`)
- Common runtime settings: Q8_0 K and V, CPU KV (`-nkvo 1`), FlashAttention
  enabled, three decode threads, poll 100, three repetitions unless noted

The decode command shape was:

```bash
taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 0 -n 64 -d DEPTH -r 3 -t 3 --poll 100 \
  -nkvo 1 -fa on -ctk q8_0 -ctv q8_0
```

## Experiment 001: CUDA-pinned CPU KV allocation

Status: promising, retained for further evaluation.

### Change

When `GGML_KV_CPU_PINNED=1`, standard CPU-resident KV buffers use the host
buffer type belonging to the device assigned to each model layer. For CUDA
layers this is `CUDA_Host`, which allocates page-locked system memory with
`cudaMallocHost()` and releases it with `cudaFreeHost()`.

The selection is applied consistently to both the cache route probe and the
actual per-layer KV allocation. With the environment variable absent, behavior
is unchanged. If CUDA cannot allocate pinned memory, the backend falls back to
the ordinary CPU buffer type.

This does not change cache layout, cache quantization, attention kernels, or KV
capacity. The full configured cache remains allocated at context creation and
persists until context destruction; the experiment changes only its host-memory
allocation type.

### Results

| Test | Ordinary CPU KV | Pinned CPU KV | Change |
| --- | ---: | ---: | ---: |
| Prefill, 512 tokens at depth 4096 | 1369.49 +/- 11.82 t/s | 1399.69 +/- 10.34 t/s | +2.2% |
| Decode, 64 tokens at depth 4096 | 18.50 +/- 0.07 t/s | 20.51 +/- 0.03 t/s | +10.9% |
| Decode, 64 tokens at depth 16384 | 12.62 +/- 0.03 t/s | 17.24 +/- 0.06 t/s | +36.6% |
| Process VRAM at depth 4096 | 13,682 MiB | 13,682 MiB | 0 MiB |

Prefill used the same settings with `-p 512 -n 0 -d 4096 -r 10`. Process VRAM
was sampled repeatedly with `nvidia-smi
--query-compute-apps=pid,process_name,used_gpu_memory` while each clean
`llama-bench` process was active.

### Resource and implementation tradeoffs

- Device allocation reported by `nvidia-smi` did not increase in this test.
- The full KV allocation becomes page-locked system RAM and cannot be swapped
  while the context exists.
- CUDA driver mapping/page-table resources may increase even though they are not
  reported as process VRAM by `nvidia-smi`.
- The environment variable is an experimental switch, not a supported public
  CLI or INI option yet.

## Required measurements for later changes

Every subsequent optimization commit must record, against the preceding known
baseline:

1. Prefill throughput with prompt size, depth, batch settings, and repetitions.
2. Decode throughput at depth 4096 and at least one long-context depth.
3. Process GPU-memory usage for baseline and candidate from clean processes.
4. CPU affinity, thread counts, cache formats, model path, hardware, build
   configuration, and commit identifiers.
5. Any system-RAM, pinned-memory, compatibility, or allocation-lifetime cost.
6. Whether the experiment is retained, revised, or reverted.

For serving changes, GPU-memory reporting must also separate initialization,
post-warmup, fixed live-context depths, and post-idle state. Record target and
draft compute buffers, recurrent-state buffers, CPU/GPU boundary buffers, and
CUDA graph-cache growth where the backend exposes them. A host-resident KV
buffer does not imply that CUDA workspace is independent of context size.

## Experiment 002: decouple recurrent-state and attention-KV placement

Status: promising, retained behind an experimental environment switch.

### Root cause

Qwen3.5 is a hybrid model: 16 layers use full attention and 48 layers use
recurrent linear attention. Before this experiment, `--no-kv-offload` mapped to
the single context flag `offload_kqv=false`. The standard
`llama_memory_hybrid` constructor passed that same boolean to both of its
independent memories:

- `llama_kv_cache`, containing the long attention K/V history that this branch
  intentionally places in CPU RAM; and
- `llama_memory_recurrent`, containing the fixed-size R/S state used by the
  recurrent layers.

This coupling moved the recurrent state to CPU even though recurrent state is
not the long-context KV capacity being offloaded. It also caused the recurrent
portion of the hybrid graph to cross the CPU/GPU boundary while its model
weights remained on the GPU. The server log exposed this as a CPU RS buffer,
and the baseline graph reserve reported 322 graph splits.

### Change

`llama_memory_hybrid` now accepts distinct `offload_attn` and `offload_recr`
constructor arguments. They are passed only to `llama_kv_cache` and
`llama_memory_recurrent`, respectively. This keeps ownership of the two memory
implementations inside the hybrid-memory abstraction; `llama-model.cpp` retains
its named, declarative constructor call instead of manually constructing the
two child memories.

Default behavior is unchanged. For the standard non-SWA hybrid-memory path,
setting:

```bash
GGML_RECURRENT_STATE_OFFLOAD=1
```

selects the model layer's device for recurrent R/S state even when
`--no-kv-offload` keeps attention K/V on CPU. If ordinary KV offload is enabled,
recurrent state remains offloaded regardless of the environment variable.

This experiment does not change attention kernels, KV layout, cache
quantization, recurrent-state precision, model weights, or MTP algorithms. The
target Q8_0 K/V cache remains in pinned system RAM when combined with
`GGML_KV_CPU_PINNED=1`.

### Rejected sub-hypotheses

Two Bee-only Q8 attention experiments preceded the placement finding and were
fully removed before the retained implementation:

1. Fusing Q8_0 V dequantization with FP32 weighted accumulation produced
   `20.57` t/s versus a fresh `20.59` t/s baseline at depth 4096.
2. Storing the Q8 K/Q scores and applying the existing vectorized softmax in a
   second pass produced `20.55` t/s at depth 4096.

These results reject temporary V-buffer traffic and scalar online-softmax as
the dominant costs for this workload. Neither attempted kernel change remains
in the branch.

### Benchmark configuration

Base commit for the unmodified behavior and candidate source base:
`e842bc61a`. The candidate was measured from the documented uncommitted diff
that this experiment entry accompanies; the final commit ID is the commit
containing this entry.

Common command template:

```bash
GGML_KV_CPU_PINNED=1 \
taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p PROMPT -n GENERATION -d DEPTH -r REPETITIONS \
  -t 3 --poll 100 -nkvo 1 -fa on -ctk q8_0 -ctv q8_0
```

The candidate adds `GGML_RECURRENT_STATE_OFFLOAD=1`; the baseline explicitly
unsets it. Prefill used `-p 512 -n 0 -d 4096 -r 10`. Decode used
`-p 0 -n 64 -r 3` at depths 4096 and 16384. Hardware, build configuration,
affinity, and model are otherwise the baseline recorded at the top of this
document.

### Results

| Test | Coupled CPU placement | GPU recurrent state | Change |
| --- | ---: | ---: | ---: |
| Prefill, 512 tokens at depth 4096 | 1388.96 +/- 16.51 t/s | 1788.16 +/- 13.60 t/s | +28.7% |
| Decode, 64 tokens at depth 4096 | 20.59 +/- 0.11 t/s | 43.10 +/- 0.17 t/s | +109.3% |
| Decode, 64 tokens at depth 16384 | 17.24 +/- 0.10 t/s | 31.23 +/- 0.07 t/s | +81.2% |

A post-refactor smoke test using 32 generated tokens at depth 4096 produced
`42.60` t/s, confirming that moving the policy into the
`llama_memory_hybrid` constructor preserved the improvement.

### Resource measurements

Matched `llama-bench --kv-memory` runs used 32 generated tokens at depth 4096
and one repetition. Byte counts below are synchronized CUDA accounting from
the benchmark, not `nvidia-smi` process-memory estimates.

| Measurement | Coupled CPU placement | GPU recurrent state | Difference |
| --- | ---: | ---: | ---: |
| CUDA peak used | 14,369,619,968 B | 14,499,643,392 B | +130,023,424 B (+124.0 MiB) |
| CUDA context buffers | 151,519,232 B | 308,412,416 B | +156,893,184 B (+149.6 MiB) |
| CUDA compute buffers | 575,337,248 B | 555,259,936 B | -20,077,312 B (-19.1 MiB) |
| CPU Q8 KV resident | 151,519,232 B | 151,519,232 B | unchanged |

The recurrent state has fixed size with respect to attention context length,
but grows with recurrent rollback snapshots. Server logs previously showed
approximately 299.25 MiB for one MTP rollback snapshot and 598.50 MiB for three,
versus the smaller no-MTP state measured here. MTP serving therefore needs a
separate VRAM-capacity check before enabling GPU recurrent state at a 92K
context.

The target CUDA compute workspace is also context-shape dependent: it was about
555 MiB in the 4K no-MTP measurement and about 721 MiB in the recorded 92K
server reserve. MTP added an approximately 631 MiB draft compute workspace.
These are GPU allocations even when target and draft KV buffers are
`CUDA_Host`. CUDA graph captures and boundary-copy buffers may add further live
growth as new serving shapes appear. Future VRAM optimization must attribute
these categories separately and may not trade away the recurrent-placement
throughput gain without an alternative that preserves performance.

### Correctness and compatibility considerations

- The recurrent tensors retain F32 storage and identical dimensions; only
  their backend buffer changes.
- Default behavior is bit-for-bit placement-compatible because the environment
  switch is opt-in.
- Attention KV remains CPU-resident, Q8_0, and CUDA-pinned under the established
  experiment settings.
- The constructor API now represents the actual two-memory architecture and
  prevents callers from having to construct hybrid internals manually.
- The current switch is implemented for the standard non-SWA hybrid path used
  by this Qwen3.5 model. ISWA and other special memory constructors were not
  silently changed.
- Full generation-output equivalence and MTP correctness still require explicit
  regression coverage before this becomes a supported CLI/INI policy.

## Characterization 003: no-MTP context scaling and 64K cache-width sweep

Status: characterization only; no implementation change. CPU KV offload remains
the target. GPU-KV measurements below are comparison baselines, not the proposed
solution.

### Configuration

- Source implementation: `339c275c0` (`kv-cache: decouple recurrent state
  offload`); worktree HEAD was `6d156a059`, whose later changes are documentation
  only.
- Model and hardware: the baseline Qwen3.5 27B model, Core Ultra 9 285K, and RTX
  5070 Ti recorded at the top of this document.
- No speculative decoding or multimodal projector.
- Q8 CPU-KV scaling used pinned host allocation and GPU recurrent state:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 0 -n 64 -d DEPTH -r 3 -t 3 --poll 100 \
  -b 1024 -ub 512 -nkvo 1 -fa on -ctk TYPE -ctv TYPE \
  -ngl 999 -sm none
```

The 128-token Q8 result used five repetitions. The homogeneous Q6_0, Q5_0,
and Q4_0 CPU results used the full command above. The Q6_1 CPU result was a
screening run with `-n 32 -r 2` and is marked accordingly. GPU comparison runs
used the same command with both experiment environment variables removed and
`-nkvo 0`.

### Context scaling with Q8 CPU KV

| Populated depth | Decode throughput | Time per token |
| ---: | ---: | ---: |
| 128 | 49.64 +/- 0.13 t/s | 20.14 ms |
| 30,000 | 23.97 +/- 0.06 t/s | 41.72 ms |
| 64,000 | 15.03 +/- 0.02 t/s | 66.53 ms |

The additional latency is approximately 0.726 microseconds per cached token
over the 128-to-64K interval. This agrees with the earlier 4K-to-16K estimate
of approximately 0.735 microseconds and establishes a stable context-linear
cost in the current CPU-offload path.

### 64K CPU thread and affinity sweep

The coarse sweep used `-n 32 -r 2`, batch 1024, ubatch 512, CPUs 0-7, and one
through eight decode threads. It produced:

| Threads | Decode throughput |
| ---: | ---: |
| 1 | 15.00 +/- 0.05 t/s |
| 2 | 15.03 +/- 0.02 t/s |
| 3 | 15.02 +/- 0.02 t/s |
| 4 | 15.03 +/- 0.02 t/s |
| 5 | 15.02 +/- 0.03 t/s |
| 6 | 15.04 +/- 0.03 t/s |
| 7 | 15.01 +/- 0.02 t/s |
| 8 | 14.96 +/- 0.03 t/s |

A full `-n 64 -r 3` run with three workers strictly pinned by `-C 0x7
--cpu-strict 1`, while leaving the rest of the process unrestricted, also
produced `15.03 +/- 0.02` t/s. Whole-process pinning and worker-only affinity
were therefore equivalent. Three P-core decode threads remain the general
configuration; increasing the thread count is not a 64K optimization.

### 64K CPU cache-width sweep

| CPU-resident K/V format | Decode throughput | Change from CPU Q8 |
| --- | ---: | ---: |
| Q8_0 / Q8_0 | 15.03 +/- 0.02 t/s | baseline |
| Q6_1 / Q6_1 | 16.09 +/- 0.06 t/s | +7.1% (screening protocol) |
| Q6_0 / Q6_0 | 16.23 +/- 0.02 t/s | +8.0% |
| Q5_0 / Q5_0 | 17.85 +/- 0.03 t/s | +18.8% |
| Q4_0 / Q4_0 | 20.71 +/- 0.04 t/s | +37.8% |

Lower-width CPU KV materially improves 64K decode, confirming that bytes read
from the long cache are a major part of the remaining cost. Throughput does not
scale directly with nominal bit width, so fixed model execution, format-specific
kernels, work partitioning, and backend boundaries remain material. These are
performance results only: no local evaluation corpus or persisted Q8 KLD
baseline was available, so none of the lower-width formats is quality-approved.

### 64K PCIe and GPU utilization observation

A generation-aligned diagnostic extended the Q8 CPU-KV test to 384 generated
tokens with one repetition and `--no-warmup`, then sampled `nvidia-smi dmon
-s putc -d 1` during the generation interval. Decode remained `15.01` t/s.
During active generation, representative counters were:

- PCIe receive: approximately 34-38 GB/s, usually 36-38 GB/s.
- PCIe transmit: approximately 2.9-3.3 GB/s.
- SM utilization: 98-99%.
- GPU memory utilization counter: approximately 44-45%.
- GPU power: approximately 158-159 W at a 2827 MHz reported graphics clock.

The host-to-GPU traffic rate is of the same order as reading the complete 64K
Q8 attention cache once per generated token at 15 t/s. This rules out a model
where CPU attention consumes the long cache and returns only a small attention
result without substantial PCIe traffic. The operation-level trace below later
identified scheduler-inserted copies as the mechanism.

The link is heavily used but does not reach its Gen5 x16 theoretical payload
rate. The practical bottleneck is therefore best described as effective
host-to-GPU cache-delivery throughput plus the access pattern, transaction
efficiency, and synchronization around it, rather than proven saturation of
the physical PCIe link. The low power at high SM utilization is consistent
with GPU warps waiting on host-memory data instead of sustaining dense compute.

### 32K Nsight Systems trace

Nsight Systems 2026.1.3 traced an unchanged build of implementation commit
`339c275c0`. CPU sampling and context-switch tracing were disabled to reduce
measurement overhead; CUDA, NVTX, and OS runtime APIs were enabled, with CUDA
graphs recorded at graph granularity. The command was:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=graph --output=q8-cpu-kv-32k \
  taskset -c 0-2 build-cuda-all/bin/llama-bench \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 0 -n 64 -d 32000 -r 1 -t 3 --poll 100 \
  -b 1024 -ub 512 -nkvo 1 -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 999 -sm none
```

The profiled run reached `22.81` t/s. The trace recorded:

- 1,008 CUDA `flash_attn_ext_f16` launches, exactly 16 attention layers times
  63 measured decode transitions.
- 2,048 host-to-device copies of exactly 35,094,528 bytes, exactly 32 K/V
  tensors times 64 measured tokens.
- 1,123,024,896 bytes of full-cache K/V transfer per token at 32K, excluding
  smaller graph-boundary transfers.
- 123.053 GB total host-to-device traffic and 4.456 GB device-to-host traffic
  across initialization, warmup, cache population, and measured decode.
- 3.256 seconds of aggregate CUDA FlashAttention kernel time in the complete
  trace. Median FlashAttention kernel duration was 2.799 ms.
- 10,100 `cudaMemcpyAsync` calls and 23,771 `cudaStreamSynchronize` calls over
  the complete process. These whole-process API totals include initialization
  and cache population and must not be divided directly by 64 decode tokens.

This establishes that persistent KV ownership is on the CPU, but active
attention is GPU-computed after full K/V history copies. CPU-loop prefetching is
therefore not relevant to the measured serving path. The primary optimization
target is the copy extent and scheduling between host KV storage and CUDA
attention.

A second trace repeated the command with `GGML_KV_CPU_PINNED` explicitly unset
and all other settings unchanged. It reached `12.53` t/s under Nsight versus
`22.81` t/s for pinned allocation. Placement and traffic were identical:

- Both traces issued 2,048 copies of 35,094,528 bytes and 5,811 total
  host-to-device copies.
- Both transferred 123.053 GB host-to-device over the complete process.
- Both launched 1,008 CUDA FlashAttention kernels.
- The dominant 35,094,528-byte copy averaged 1,698.773 microseconds unpinned
  versus 624.707 microseconds pinned, corresponding to effective rates of 20.0
  and 56.0 GB/s inside the profiler.
- Aggregate host-to-device copy time was 5.925 seconds unpinned versus 2.668
  seconds pinned. Aggregate CUDA FlashAttention time was nearly unchanged at
  3.216 versus 3.256 seconds.

This confirms that Experiment 001 did not move attention work from CPU to GPU.
It accelerated an otherwise identical full-cache scheduler-copy path.

### GPU-KV comparison baselines

These measurements bound the attention cost when KV is not CPU-offloaded. They
do not change the branch objective.

| Depth and GPU-resident K/V | Decode throughput | Fit result |
| --- | ---: | --- |
| 128, Q8_0 / Q8_0 | 50.83 +/- 0.15 t/s | fit |
| 30,000, Q8_0 / Q8_0 | 44.06 +/- 0.12 t/s | fit |
| 32,000, Q8_0 / Q8_0 | 43.62 +/- 0.15 t/s | fit |
| 64,000, Q8_0 / Q8_0 | not measured | context allocation failed |
| 64,000, Q6_1 / Q6_1 | 34.41 +/- 0.10 t/s | fit |
| 64,000, Q6_0 / Q6_0 | 32.22 +/- 0.08 t/s | fit |
| 64,000, Q5_0 / Q5_0 | 33.22 +/- 0.08 t/s | fit |
| 64,000, Q4_0 / Q4_0 | 37.10 +/- 0.12 t/s | fit |

The 64K Q8 GPU context failed with both the benchmark's default batch 2048 and
the server-matched batch 1024, each with ubatch 512. The successful lower-width
GPU rows serve only as a hardware ceiling and format-kernel comparison. They do
not include server, multimodal, MTP, or CUDA-graph memory overhead.

### Disposition

- Retain three P-core decode threads; thread-count retuning is exhausted for
  this 64K workload.
- Continue optimizing CPU-resident KV. Do not substitute the GPU comparison
  configurations for the CPU-offload objective.
- Treat reduced cache width as a valid CPU performance lever only after matched
  KLD/perplexity validation.
- Use the 32K trace as the Q8 CPU-offload baseline: prioritize reducing or
  overlapping the 32 full-cache host-to-device copies per token before changing
  CPU attention kernels.

## Experiment 004: force single-token standard attention onto CPU

Status: rejected and reverted.

### Hypothesis and implementation

The 32K trace showed that host-resident KV was copied in full to CUDA for every
decode token. A narrow placement override tested whether executing attention on
CPU could replace those context-sized copies with only the small query and
attention-output boundary transfers.

When `GGML_CPU_KV_ATTENTION=1`, the candidate assigned the
`GGML_OP_FLASH_ATTN_EXT` result to the CPU backend only when all of the following
were true:

- `--no-kv-offload` was active;
- the query contained one token per stream;
- the cache was standard rather than KVarN; and
- no KV tail was active.

The single-token condition was added after an initial whole-path prototype made
32K depth preparation take longer than eight minutes. The phase-aware revision
kept multi-token prefill and depth population on CUDA while forcing only decode
attention onto CPU. Recurrent R/S state remained on GPU through
`GGML_RECURRENT_STATE_OFFLOAD=1`; model weights remained GPU-resident. The
candidate changed no cache representation, quantization, attention arithmetic,
or MTP behavior.

The source base and reported build commit were `6d156a059`; the candidate was
an uncommitted diff to `src/llama-graph.cpp`, rebuilt with ccache in the existing
Release CUDA build. The implementation was removed after measurement. The
experiment record is retained in the following documentation commit.

### Commands and progress reporting

All duration-uncertain runs used `llama-bench --progress`. The common candidate
environment was:

```bash
GGML_KV_CPU_PINNED=1 \
GGML_RECURRENT_STATE_OFFLOAD=1 \
GGML_CPU_KV_ATTENTION=1
```

Decode used all eight P-cores because genuine CPU FlashAttention, unlike the
CUDA-copy baseline, benefited from the additional workers:

```bash
taskset -c 0-7 build-cuda-all/bin/llama-bench --progress \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 0 -n TOKENS -d DEPTH -r REPS -t 8 --poll 100 \
  -b 1024 -ub 512 -nkvo 1 -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 999 -sm none
```

The matched prefill pair used `-p 512 -n 0 -d 4096 -r 3`. Memory accounting
used `--kv-memory -o json -p 0 -n 4 -d 32000 -r 1`. The baseline explicitly
removed `GGML_CPU_KV_ATTENTION` and otherwise used identical settings.

### Results

| Test | CUDA-copy baseline | Forced CPU decode attention | Result |
| --- | ---: | ---: | ---: |
| Prefill, 512 tokens at depth 4096 | 1777.07 +/- 22.39 t/s | 1775.73 +/- 23.06 t/s | -0.1% |
| Decode, 32 tokens at depth 4096 | approximately 43 t/s from Experiment 002 | 18.38 +/- 0.13 t/s | approximately -57% |
| Decode, 16 tokens at depth 32000 | not rerun with 16-token protocol | 3.34 +/- 0.02 t/s | screening |
| Matched memory run, 4 tokens at depth 32000 | 21.91 t/s | 3.34 t/s | -84.7% |
| CUDA peak used, depth 32000 | 14,619,181,056 B | 14,619,181,056 B | unchanged |
| CUDA context buffers, depth 32000 | 1,279,918,080 B | 1,279,918,080 B | unchanged |
| CUDA compute buffers, depth 32000 | 701,272,096 B | 701,272,096 B | unchanged |

Three-thread CPU attention reached `16.07` t/s in an eight-token 4K screening
run. Eight P-cores improved that to `18.22` t/s in the same short protocol,
which motivated the eight-core measurements above but did not make the path
competitive.

### Placement trace

Nsight Systems 2026.1.3 traced `-n 4 -d 32000 -r 1 -t 8` with native benchmark
progress enabled. The report was generated as `cpu-attn-32k.nsys-rep`.

- No 35,094,528-byte host-to-device copies occurred. The baseline issues 32 of
  these full-cache copies per decode token.
- The trace contained 1,008 CUDA FlashAttention launches, attributable to the
  63 depth-population microbatches times 16 attention layers. Decode attention
  itself was therefore successfully placed on CPU.
- Whole-process host-to-device traffic was 51.150 GB across 3,386 copies,
  primarily model initialization and CUDA depth population; it is not a
  decode-only traffic total.

The trace proves that the intended copy elimination worked. Poor throughput was
therefore caused by the current CPU Q8 FlashAttention path, not a failure to
change backend placement.

### Disposition

- Revert the placement override. It is substantially slower at both 4K and 32K.
- Do not pursue software prefetch as an easy standalone fix. Matching the
  existing 32K path would require approximately a 6.6x CPU-attention speedup.
- Do not claim a VRAM benefit: the scheduler reserved identical CUDA context and
  compute buffers in the matched memory runs.
- Preserve CUDA prefill and GPU recurrent state.
- Continue investigating bounded CUDA staging, copy/compute overlap, or other
  mechanisms that retain host KV ownership without full-cache resident VRAM.

## Experiment 005: CUDA attention directly reading mapped host KV

Status: rejected and reverted after the 4K gate.

### Hypothesis and change

Pinned host memory is mapped into CUDA's unified virtual address space. The
candidate tested whether CUDA FlashAttention could read host-resident KV
directly, eliminating scheduler staging copies without consuming VRAM for
persistent KV or moving attention compute to CPU.

Behind `GGML_KV_CUDA_ZERO_COPY=1`, the CUDA device buffer-support predicate
temporarily accepted `CUDA_Host` buffers on a discrete GPU. The experiment was
used only with `GGML_KV_CPU_PINNED=1`, standard Q8 K/V, GPU recurrent state,
and `--no-kv-offload`. Default behavior remained unchanged. This was a broad
screening hook rather than a retainable interface: it made all CUDA-host buffers
eligible for direct CUDA access, not only KV tensors. A competitive result would
have required replacement with a KV-specific mapped buffer type.

The source base and reported build commit were `2e4ba398b`; the candidate was an
uncommitted one-function diff in `ggml/src/ggml-cuda/ggml-cuda.cu`. It was built
with ccache in the existing Release CUDA build and removed after the 4K gate.

### Commands and progress

All runs enabled native progress reporting. The common environment was:

```bash
GGML_KV_CPU_PINNED=1 \
GGML_RECURRENT_STATE_OFFLOAD=1 \
GGML_KV_CUDA_ZERO_COPY=1
```

Decode and prefill used the established model and:

```bash
taskset -c 0-2 build-cuda-all/bin/llama-bench --progress \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p PROMPT -n GENERATION -d DEPTH -r REPS -t 3 --poll 100 \
  -b 1024 -ub 512 -nkvo 1 -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 999 -sm none
```

The trace used `-p 0 -n 8 -d 4096 -r 1`. Prefill used
`-p 512 -n 0 -d 4096 -r 1`. Memory accounting used
`--kv-memory -o json -p 0 -n 4 -d 4096 -r 1`.

### Results

| Test | Staged pinned baseline | Direct mapped-host CUDA | Result |
| --- | ---: | ---: | ---: |
| Decode, depth 128 screening | 49.64 t/s | 27.57 t/s | -44.5% |
| Decode, depth 4096 | approximately 43 t/s | 10.15 t/s | approximately -76% |
| Prefill, 512 tokens at depth 4096 | 1777.07 t/s | 462.01 t/s | -74.0% |
| CUDA peak used, depth 4096 | 14,499,643,392 B | 14,501,740,544 B | +2,097,152 B |
| CUDA context buffers, depth 4096 | 308,412,416 B | 308,412,416 B | unchanged |
| CUDA compute buffers, depth 4096 | 555,259,936 B | 555,257,888 B | effectively unchanged |

The baseline throughput values are the retained Experiment 002 result and the
matched prefill baseline from Experiment 004. The baseline and candidate CUDA
memory values use the same 4K Q8 CPU-KV configuration and three decode threads;
the 2 MiB peak difference is not a saving and is too small to be material.

### Trace result

Nsight Systems 2026.1.3 generated `zero-copy-4k.nsys-rep` with CUDA, NVTX, and
OS runtime tracing, graph granularity, CPU sampling disabled, and benchmark
progress enabled.

- The trace contained no per-token full-cache K/V staging-copy pattern.
- It recorded 128 CUDA FlashAttention launches for eight decode tokens across
  16 attention layers.
- CUDA FlashAttention consumed 1.026 seconds in aggregate, averaging 8.016 ms
  per layer and reaching 14.107 ms at the maximum.
- Whole-process host-to-device traffic was 13.505 GB across 850 copies. Those
  copies were dominated by model initialization and graph preparation rather
  than decode-time full-cache staging.

This confirms that direct mapped access worked mechanically. Performance was
lost inside CUDA attention reading host memory, rather than because the
scheduler silently retained the original copy path.

### Disposition

- Revert the discrete-CUDA `CUDA_Host` support exception.
- Do not build a KV-specific mapped buffer type for the current CUDA attention
  kernel; the broad probe is already far below the 4K acceptance gate.
- Do not run 32K or 64K: mapped-host attention already worsens sharply between
  depth 128 and 4096, prefill regresses by 74%, and CUDA reservation does not
  decrease.
- Continue with bounded staged transfers and explicit copy/compute overlap,
  where CUDA attention still consumes device-local tiles.

## Characterization 006: 32K no-MTP versus MTP-1 VRAM baseline

Status: characterization before draft-specific ubatch implementation.

### Objective and configuration

This sweep establishes how much VRAM MTP-1 adds before changing its context
geometry. The target model remained fixed at batch 1024 and ubatch 512. Both
configurations used standard pinned Q8 CPU KV, GPU recurrent state, three
decode threads on P-cores 0-2, CUDA FlashAttention, all model layers on GPU,
one slot, and a 32,768-token context.

The no-MTP controlled benchmark used:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
taskset -c 0-2 build-cuda-all/bin/llama-bench --progress --kv-memory -o json \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 0 -n 16 -d 32000 -r 2 -t 3 --poll 100 \
  -b 1024 -ub 512 -nkvo 1 -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 999 -sm none
```

Matched server runs used the same target settings and `--parallel 1
--cache-ram 0 --metrics`. The MTP variant added:

```bash
--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-ngl all \
--cache-type-k-draft q4_0 --cache-type-v-draft q4_0
```

Each server received an explicit 32,000-token array containing token ID 100 and
generated 16 greedy tokens. Server logs provided native prompt progress. The
repetitive synthetic prompt yielded perfect observed MTP acceptance and is not
a representative quality/workload benchmark.

### Throughput and process VRAM

| Measurement | No MTP | MTP-1 | Difference |
| --- | ---: | ---: | ---: |
| Controlled `llama-bench` decode | 22.95 +/- 0.25 t/s | not applicable | - |
| Server prompt, 32,000 tokens | 1533.24 t/s | 1473.60 t/s | -3.9% |
| Server decode, 16 tokens | 24.17 t/s | 38.55 t/s | +59.5% |
| MTP draft acceptance | - | 7/7, 1.000 | synthetic maximum |
| Process VRAM after initialization | 13,626 MiB | 14,290 MiB | +664 MiB |
| Process VRAM after live 32K request | 13,642 MiB | 14,324 MiB | +682 MiB |

The controlled no-MTP benchmark reported a synchronized CUDA peak of
14,619,181,056 bytes, 701,272,096 bytes of CUDA compute buffers, and
1,279,918,080 bytes of CUDA context buffers. These benchmark accounting values
are not directly interchangeable with server `nvidia-smi` process VRAM; both
are retained for their respective future comparisons.

### Startup allocation breakdown

Bounded ten-second server launches at verbosity 4 recorded:

| Allocation | No MTP | MTP-1 | Difference |
| --- | ---: | ---: | ---: |
| CUDA model buffer | 12,879.47 MiB | 13,114.03 MiB | +234.56 MiB |
| Target CUDA recurrent state | 149.62 MiB | 299.25 MiB | +149.63 MiB |
| Target CUDA compute buffer | 342.27 MiB | 342.27 MiB | unchanged |
| Target CUDA-host compute buffer | 52.28 MiB | 52.28 MiB | unchanged |
| Target CUDA-host Q8 KV | 1,088.00 MiB | 1,088.00 MiB | unchanged |
| MTP CUDA compute buffer | - | 276.27 MiB | +276.27 MiB |
| MTP CUDA-host compute buffer | - | 52.28 MiB | host RAM |
| MTP CUDA-host Q4 KV | - | 36.00 MiB | host RAM |

The extra model buffer is the built-in MTP head, which the no-MTP load reports
as unused. The larger target recurrent buffer is the rollback snapshot required
by MTP-1. The 276.27 MiB draft CUDA compute buffer is the primary allocation a
draft-specific ubatch can reduce. Consequently, draft ubatch tuning cannot
remove the full 664-682 MiB MTP overhead by itself; its theoretical saving is
bounded by that draft compute category plus small shape-dependent allocations.

### Why draft and target ubatch were initially equal

`common_speculative_init_result` converts the shared `common_params` into
`llama_context_params`, sets `ctx_type=LLAMA_CONTEXT_TYPE_MTP`, disables draft
recurrent snapshots, points `ctx_other` at the target, and creates the second
context. It does not override `n_batch` or `n_ubatch`, so the MTP context
inherits target 1024/512 by implementation convenience.

This is not an algorithmic equality constraint. MTP prompt synchronization
reads `llama_n_ubatch(ctx_dft)` and chunks rows to fit. MTP-1 decode uses only a
small number of rows. A smaller draft ubatch should therefore reduce reserved
draft graph workspace while increasing the number of calls needed to mirror a
long prompt. The target context and its prefill batch remain unchanged.

### Original next gate

Add an experimental draft-context ubatch override without changing target
1024/512. Sweep draft ubatch 256, 128, 64, and 32. For each value record startup
allocation categories, initialized and live-32K process VRAM, 32K prompt speed,
decode throughput, and draft acceptance. Reject any setting that materially
reduces decode throughput or changes generated output under matched sampling.

### MTP depth scaling at 32K

Before adding the draft-specific ubatch control, the same server protocol was
extended to draft maxima 3, 5, and 8. The target remained at batch 1024 and
ubatch 512. Each run used a fresh server, the same explicit 32,000-token array
of token ID 100, 16 greedy output tokens, and native five-second progress
reporting. The only changed argument was `--spec-draft-n-max`:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --ctx-size 32768 --batch-size 1024 --ubatch-size 512 \
  --threads 3 --threads-batch 24 \
  --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 \
  --n-gpu-layers all --split-mode none --parallel 1 \
  --no-kv-offload --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max DEPTH --spec-draft-ngl all \
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
  --cache-ram 0 --metrics --host 127.0.0.1 --port PORT --log-verbosity 4

jq -nc \
  '{prompt:[range(0;32000)|100],n_predict:16,temperature:0,seed:1234,cache_prompt:false,stream:false}' | \
curl -sS -o /dev/null -H 'Content-Type: application/json' \
  --data-binary @- http://127.0.0.1:PORT/completion
```

Process VRAM was sampled with `nvidia-smi` after server initialization and
again after the request completed. These are single runs intended to determine
allocation scaling, not statistically stable throughput comparisons.

| Draft maximum | Init VRAM | Live 32K VRAM | Target recurrent buffer | Prompt | Decode | Acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14,290 MiB | 14,324 MiB | 299.25 MiB | 1473.60 t/s | 38.55 t/s | 7/7, 1.000 |
| 3 | 14,590 MiB | 14,624 MiB | 598.50 MiB | 1470.27 t/s | 60.43 t/s | 11/11, 1.000 |
| 5 | 14,888 MiB | 14,922 MiB | 897.75 MiB | 1468.25 t/s | 68.92 t/s | 12/12, 1.000 |
| 8 | 15,338 MiB | 15,374 MiB | 1346.62 MiB | 1466.26 t/s | 61.60 t/s | 12/19, 0.632 |

The MTP model buffer (13,114.03 MiB), draft CUDA compute buffer (276.27 MiB),
draft CUDA-host Q4 KV (36.00 MiB), and target CUDA compute buffer (342.27 MiB)
were invariant across depths 3, 5, and 8. Depth-dependent VRAM is dominated by
the target recurrent rollback allocation. It grows by approximately 149.625
MiB per additional rollback sequence: 299.25 MiB at depth 1, 598.50 MiB at
depth 3, 897.75 MiB at depth 5, and 1346.62 MiB at depth 8. Consequently,
increasing MTP depth from 1 to 8 costs about 1,048 MiB of process VRAM on this
model and configuration.

Depth 5 was the best decode result in this artificial high-predictability
sample. Depth 8 consumed another 450 MiB relative to depth 5 while acceptance
fell to 0.632 and decode throughput fell by 10.6%. This does not establish a
production-optimal depth because the repeated-token prompt is unusually easy,
but it does show that deeper MTP is not free and that its rollback state, rather
than its draft workspace, is the principal depth-scaling VRAM cost.

## Experiment 007: independent speculative draft ubatch

Status: retained for non-MTP model-backed speculative contexts; superseded and
rejected for MTP after the longer exactness gate in Experiment 017.

### Implementation

Added `common_params_speculative_draft::n_ubatch`, defaulting to zero so
existing behavior continues to inherit the target ubatch. The new
`--spec-draft-ubatch-size N` option (aliases `--ubatch-size-draft` and `-ubd`,
environment variable `LLAMA_ARG_SPEC_DRAFT_UBATCH`) overrides `n_ubatch` only
when `common_base_params_to_speculative` creates parameters for the separate
draft context. Target batch and ubatch are not modified. Negative values are
rejected; zero preserves inheritance.

The implementation is deliberately common to model-backed speculative modes
and MTP rather than adding a second MTP-only initialization path. Parser tests
cover CLI and environment-variable propagation.

### Validation

`build-cuda-all/bin/test-arg-parser` passed. The CUDA server and parser test
targets compiled successfully. Server startup logs confirmed that target
`n_ubatch` remained 512 while the MTP context reported each requested draft
value.

At MTP maximum depth 5 and 32K target context, an allocation-only sweep gave:

| Draft ubatch | Init VRAM | Draft CUDA compute | Init saving vs 512 |
| ---: | ---: | ---: | ---: |
| 512 | 14,888 MiB | 276.27 MiB | - |
| 256 | 14,832 MiB | 220.27 MiB | 56 MiB |
| 128 | 14,804 MiB | 192.27 MiB | 84 MiB |
| 64 | 14,790 MiB | 178.27 MiB | 98 MiB |
| 32 | 14,782 MiB | 171.27 MiB | 106 MiB |

The savings diminish below 128. Full matched runs therefore compared 512,
128, and 32. Each used the Experiment 006 server command with MTP maximum 5,
added `--spec-draft-ubatch-size UB`, and requested 64 greedy tokens after the
same explicit 32,000-token prompt. Native server logs reported progress.

| Draft ubatch | Init VRAM | Live 32K VRAM | Prompt | Decode | Acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 14,888 MiB | 14,922 MiB | 1468.09 t/s | 74.82 t/s | 51/55, 0.9273 |
| 128 | 14,804 MiB | 14,834 MiB | 1454.97 t/s | 74.67 t/s | 51/55, 0.9273 |
| 32 | 14,782 MiB | 14,812 MiB | 1404.62 t/s | 74.84 t/s | 51/55, 0.9273 |

Draft ubatch 128 saved 88 MiB of live process VRAM, changed decode by -0.2%,
and reduced prompt throughput by 0.9%. Draft ubatch 32 saved 110 MiB, left
decode unchanged within single-run noise, and reduced prompt throughput by
4.3%. Output-side draft counts and acceptance were identical in all three
64-token runs. At the time, 128 was recommended as the balanced setting.

That MTP recommendation is withdrawn. Experiment 017 extended the check to
1,000 tokens and found both 128 and 32 diverging from clean Bee's inherited-512
MTP stream at generated token 100. The short screen ended before the numerical
state difference became visible. Current MTP requires an omitted draft ubatch
or one equal to target ubatch; `--phase-aware-workspace` provides the retained
workspace saving without changing recurrent prompt-synchronization geometry.
The option and measurements above remain applicable historical evidence for
other model-backed speculative modes, whose independent ubatch is not rejected.

## Characterization 008: context-scaled non-MTP CUDA allocation

Status: allocation trace; no candidate implementation yet.

### Objective and protocol

The user observed higher VRAM with a 240K configured context than with 90K
despite CPU-resident KV. Nsight Systems 2026.1.3 captured bounded server startup
and warmup at both sizes with no speculative implementation. Target batch 1024
and ubatch 512 were unchanged. The captures used pinned Q8 CPU KV, GPU recurrent
state, CUDA FlashAttention, all model layers on the GPU, and identical CPU
placement:

```bash
nsys profile --force-overwrite=true --output=TRACE \
  --trace=cuda,nvtx,osrt --sample=process-tree --backtrace=lbr \
  --cuda-memory-usage=true --cudabacktrace=memory:0 \
  timeout --signal=INT --kill-after=5s 15s env \
    GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
    build-cuda-all/bin/llama-server \
      --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
      --ctx-size CTX --batch-size 1024 --ubatch-size 512 \
      --threads 3 --threads-batch 24 \
      --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 \
      --n-gpu-layers all --main-gpu 0 --split-mode none --parallel 1 \
      --no-kv-offload --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --cache-ram 0 --metrics --host 127.0.0.1 --port PORT \
      --log-verbosity 4
```

The comparable reports are:

- `/tmp/beellama-nsys-ctx.VxAYMO/ctx90k-bt.nsys-rep`
- `/tmp/beellama-nsys-ctx.VxAYMO/ctx240k.nsys-rep`

The server self-reported build 11215 at commit `2e4ba398b`. A temporary
allocator-logging build named graph-planner tensors after Nsight identified the
single backing allocation. That diagnostic-only source change was reverted and
the normal server rebuilt before recording this result.

### Allocation comparison

Requested contexts were padded to 90,112 and 240,128 cells, a difference of
150,016 cells.

| Allocation | 90K | 240K | Difference |
| --- | ---: | ---: | ---: |
| CUDA model buffer | 12,879.47 MiB | 12,879.47 MiB | unchanged |
| CUDA recurrent state | 149.62 MiB | 149.62 MiB | unchanged |
| CUDA compute buffer | 707.27 MiB | 1,751.09 MiB | +1,043.82 MiB |
| CUDA-host Q8 KV | 2,992.00 MiB | 7,973.00 MiB | +4,981.00 MiB host RAM |
| CUDA-host compute buffer | 108.28 MiB | 254.78 MiB | +146.50 MiB host RAM |
| Graph nodes / splits | 3,847 / 34 | 3,847 / 34 | unchanged |

Nsight recorded the CUDA compute reservation as one `cudaMalloc`, with the
backtrace:

```text
ggml_backend_cuda_buffer_type_alloc_buffer
ggml_gallocr_reserve_n_impl
ggml_backend_sched_reserve
llama_context::graph_reserve
llama_context::sched_reserve
```

It is therefore a monolithic graph-planner buffer whose tensor extents grow;
the growth is not additional graphs or a GPU-resident persistent KV cache.

### Exact context-scaled CUDA composition

Allocator planning and CUDA FlashAttention allocation code account for the
GPU slope exactly:

| Component | Bytes per context cell | 90K allocation | 240K allocation | 90K -> 240K growth |
| --- | ---: | ---: | ---: | ---: |
| F16 K/V materialization for prompt FlashAttention | 4,096 | 364.00 MiB context-scaled portion | 950.00 MiB context-scaled portion | 614,465,536 B |
| Q8 K scheduler staging | 1,088 | 93.50 MiB | 249.16 MiB | 163,217,408 B |
| Q8 V scheduler staging | 1,088 | 93.50 MiB | 249.16 MiB | 163,217,408 B |
| GPU F16 attention mask for 512 prompt rows | 1,024 | 88.00 MiB | 234.50 MiB | 153,616,384 B |
| **Total GPU slope** | **7,296** | - | - | **1,094,516,736 B (1,043.82 MiB)** |

The CUDA-host compute slope is another 1,024 bytes per cell: the source F16
attention mask has shape `[n_kv, 512]`. Its 153,616,384-byte growth exactly
matches the 146.50 MiB host-compute difference.

For the quantized K and V inputs, the prompt-processing FlashAttention route
selects the tile/MMA path. `ggml_cuda_flash_attn_ext_get_alloc_size` therefore
reserves complete F16 K and V materializations after the output tensor. The Q8
cache views must first be copied from CUDA-host memory into separate device
staging tensors. The planner reuses these allocations across attention layers,
so the slope is one layer's working set rather than all 16 attention layers.

### Reduction implications

The largest individual target is full-context F16 materialization: eliminating
it with a direct quantized multi-query prompt-attention kernel would save about
950 MiB at 240K. Fixed-window, online-softmax attention would cap both that
materialization and the 498 MiB combined Q8 K/V staging instead of scaling them
with maximum context. This is the most complete solution but is a substantial
kernel and scheduling change.

The attention mask is the most isolated smaller target. A CUDA attention path
that derives causal validity from compact query/KV positions rather than an
explicit `[n_kv, 512]` F16 tensor would save approximately 234.5 MiB of GPU
memory and 234.5 MiB of pinned host memory at 240K. The existing no-mask kernel
mode is not by itself sufficient because causal/padding semantics still need to
be represented explicitly.

A decode-only scheduler shrink could release the prompt-sized reservation after
prefill because single-token quantized attention uses a different, smaller
route. It would reduce steady-state decode VRAM but not peak prefill VRAM and
would require reallocation/graph recapture for later prompt ingestion. It is a
secondary policy option, not a solution for fitting the initial 240K context.
## Experiment 009: replace the environment switches with supported CLI/server options

Status: retained. Closes "Proposed next sequence" item 11 from
`cpu-kv-offload-development.md` for the standard non-SWA hybrid path; the
scope limitation from Experiment 002 (ISWA and other special memory
constructors not covered) still applies.

### Change

`GGML_KV_CPU_PINNED` and `GGML_RECURRENT_STATE_OFFLOAD` are replaced by first-class
parameters that follow the same `common_arg` pattern as `--kv-offload`/`--no-kv-offload`:

- `--kv-cpu-pinned` / `--no-kv-cpu-pinned` (env `LLAMA_ARG_KV_CPU_PINNED`)
- `--recurrent-state-offload` / `--no-recurrent-state-offload`
  (env `LLAMA_ARG_RECURRENT_STATE_OFFLOAD`)

Both flags are available on `llama-cli`, `llama-completion`, `llama-server`, and
`llama-bench` (the latter has its own argument parser and `cmd_params`/
`cmd_params_instance` structs, so it required separate wiring). The values flow
`common_params` -> `llama_context_params` (new `kv_cpu_pinned` and
`recurrent_state_offload` bools, appended to the trailing bool block) ->
`llama_cparams` -> `llama_model::create_memory()`, replacing the `std::getenv`
reads in `llama-kv-cache.cpp` and `llama-model.cpp`.

`llama_kv_cache_cpu_buft()` now takes an explicit `cpu_pinned` bool instead of
reading the environment. The `llama_kv_cache` constructor gained a trailing
`cpu_pinned = false` default parameter (kept at the end so the ~15 existing call
sites across `llama-kv-cache-{kvarn,dsa,dsv4,msa,iswa}.cpp` do not need to change);
it is wired to `cparams.kv_cpu_pinned` at the four call sites that construct the
actual target attention-KV cache (the plain non-hybrid path, its kvarn-native-exact
variant, the MTP-draft cache, and `llama_memory_hybrid`'s `mem_attn`). The
recurrent-state getenv check in `llama-model.cpp` is replaced by
`cparams.offload_kqv || cparams.recurrent_state_offload`.

Unlike the global environment variable, which applied to every `llama_kv_cache`
instance in the process including auxiliary DSA/MSA/DSV4 indexer caches, the CLI
option only affects the primary target attention-KV cache. Those auxiliary caches
were never part of the documented pinning workflow or its benchmarks, so this is
considered a scope correction, not a regression.

### Verification

Functional equivalence against Experiment 001/002 was confirmed on different
hardware (RTX 4070, 12 GiB, compute capability 8.9; Intel host; three pinned
decode threads) using `Qwen3.8-27B-UD-IQ2_M.gguf`:

```bash
taskset -c 0-2 build/bin/llama-bench -m MODEL -ngl 99 -t 3 --poll 100 \
  -nkvo 1 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d DEPTH -r 3 \
  [--kv-cpu-pinned] [--recurrent-state-offload]
```

| Config | Prefill @4K (t/s) | Decode @4K (t/s) | Decode @16K (t/s) |
| --- | ---: | ---: | ---: |
| baseline (no flags) | 993.40 +/- 6.57 | 13.72 +/- 0.02 | 8.96 +/- 0.01 |
| `--kv-cpu-pinned` | 1011.65 +/- 2.30 | 15.26 +/- 0.02 | 11.79 +/- 0.03 |
| `--recurrent-state-offload` | 1152.98 +/- 10.94 | 26.34 +/- 0.11 | 13.21 +/- 0.02 |
| both | 1173.53 +/- 11.59 | 31.80 +/- 0.11 | 19.75 +/- 0.04 |
| both, re-measured after the CLI-flag refactor | -- | 31.92 +/- 0.11 | -- |

The last row re-ran the "both" decode@4K case through the new
`--kv-cpu-pinned --recurrent-state-offload` flags instead of the old environment
variables; the result (31.92 t/s) matches the pre-refactor value (31.80 t/s)
within run-to-run noise, confirming the parameter-plumbing change did not alter
behavior.

Process VRAM peak (decode@4K, sampled repeatedly via `nvidia-smi
--query-compute-apps`, 8 repetitions): baseline 10,002 MiB vs. both flags
enabled 10,130 MiB (+128 MiB), consistent with the +124 MiB reported for the
same comparison in Experiment 002 on different hardware.

An end-to-end smoke test with `llama-completion -nkvo --kv-cpu-pinned
--recurrent-state-offload` produced coherent output and exited cleanly,
confirming the option reaches context creation through the normal
`common/arg.cpp` parsing path (not just `llama-bench`'s separate parser).

### Remaining scope for a supported policy

- MTP rollback-state VRAM and realistic-prompt validation from Experiment 002's
  "current limitations" are still open.
- `llama_memory_hybrid_iswa` (the SWA-hybrid path) still ties recurrent-state
  and attention-KV placement to a single `offload` argument; extending the
  split there is unstarted.
- `llama-bench`'s new flags are process-wide (not swept per combination like
  `-nkvo`), matching how the documented benchmark protocol invokes them today.

## Experiment 010: lossless compression of the Q8_0 transfer stream

Status: rejected on measured entropy grounds. No implementation was written; the
measurement is the deliverable, and it closes the direction.

### Hypothesis

Experiment 005 and the Nsight traces established that decode cost at long context
is dominated by scheduler host-to-device copies of the complete Q8_0 K and V
tensors, running at the PCIe hardware limit. With Q8_0 K/V fixed as the quality
gate, the only remaining way to reduce that cost is to move fewer bytes for the
same information. The hypothesis was that the Q8_0 stream carries enough
redundancy for a GPU-decodable lossless codec to pay for itself.

### Method

Real cache contents were used rather than synthetic data. `llama-server` ran the
established model with `-c 8192 -nkvo -fa on -ctk q8_0 -ctv q8_0
--kv-cpu-pinned --recurrent-state-offload` and `--slot-save-path`. A 6,391-token
prompt of ordinary English prose was evaluated and the populated slot was written
out with `POST /slots/1?action=save`, producing 379,801,088 bytes for 6,398
cells.

Q8_0 regions were located by their structural signature: Q8_0 derives its block
scale as `amax/127`, so every 32-value block must contain an element of magnitude
exactly 127. Scanning all 34-byte phases found the attention-KV region starting
at file offset 1,048,608 and covering approximately 212 MiB, consistent with the
expected 16 layers x 2 tensors x 6,398 tokens x 1,088 bytes. Recurrent state and
headers were excluded. Measurements below used 4,000 token rows (about 4.4 MiB)
for the structured tests and 176,470 blocks (about 11.3 MiB of payload) for the
entropy tests.

### Results

Payload statistics: standard deviation 55.0; 11.1% of values with magnitude below
8, 22.6% below 16, 43.8% below 32. The distribution is close to Gaussian and
occupies the full signed 8-bit range by construction.

| Scheme | Entropy | Achievable ratio |
| --- | ---: | ---: |
| Order-0 entropy coding of the int8 payload | 7.685 bits | 1.041x |
| Delta along the context dimension, then order-0 | 7.550 bits | 1.060x |
| Delta along the channel dimension, then order-0 | 8.355 bits | 0.958x |
| `zlib` level 9 on the raw stream | -- | 1.033x |
| `lzma` preset 6 on the raw stream | -- | 1.028x |
| FP16 block scales alone (5.9% of bytes) | 10.061 of 16 bits | 1.590x |

Combining the best payload result with separately coded scales bounds a perfect
adaptive coder at approximately **1.081x**, or an 8% byte reduction.

### Disposition

Rejected. The bound is an entropy limit assuming an ideal coder; a real
GPU-decodable implementation would realise less while adding a decompression
kernel, a host-side compression step on the write path, and a variable-length
addressing scheme on top of the current fixed-stride cache layout. An 8% ceiling
does not justify that.

The underlying reason is structural and will not change with a better codec.
Q8_0 normalises every 32-value block by its own maximum, which forces the full
8-bit range in every block and leaves a near-Gaussian residual with a standard
deviation of 55. That residual carries 7.685 bits of genuine entropy per stored
byte. Q8_0 is already close to an optimal fixed-rate code for this data, so there
is no payload redundancy for a lossless codec to remove. Delta coding along the
context dimension recovers only 0.135 bits because per-block rescaling destroys
the smoothness that would otherwise exist between neighbouring tokens.

Do not revisit payload compression for any block-scaled `qN_0` cache format. The
same normalisation argument applies to all of them.

### The redundancy that does exist

The measurement redirects rather than ends the byte-reduction effort. The Q8_0
payload is incompressible, but the *transfer schedule* is massively redundant:
the KV cache is append-only, so between two decode steps exactly one row of
1,088 bytes per tensor changes, while the scheduler re-sends the entire tensor.
At 32K that is 1,123,024,896 bytes sent per token to convey roughly 34,816 bytes
of new information.

Eliminating that redundancy is exactly lossless and needs no codec, but it
requires the previously sent bytes to stay resident on the GPU, which is the
VRAM cost that offload exists to avoid. It is therefore a tunable trade rather
than a free win. Measured on the RTX 4070 with `Qwen3.8-27B-UD-IQ2_M.gguf`,
`-ngl 99 -t 3 --poll 100 -fa on -ctk q8_0 -ctv q8_0 --recurrent-state-offload`,
`-p 0 -n 32 -r 2`, three real P-cores via `taskset -c 0,2,4`:

| Depth | Host KV (`-nkvo 1 --kv-cpu-pinned`) | GPU-resident KV | Change |
| ---: | ---: | ---: | ---: |
| 0 | 39.38 +/- 0.29 t/s | 39.95 +/- 0.32 t/s | +1.4% |
| 4096 | 31.89 +/- 0.21 t/s | 39.02 +/- 0.21 t/s | +22.4% |
| 16384 | 19.76 +/- 0.05 t/s | 36.13 +/- 0.17 t/s | +82.8% |

Reported buffer sizes for the host-KV configuration were 8.50 MiB of
`CUDA_Host` KV at depth 0, 144.50 MiB at 4096, 280.50 MiB at 8192, and 552.50
MiB at 16384. The `CUDA0` compute buffer did not scale with depth in the same
way, measuring 7.14, 505.00, 562.04, and 505.00 MiB respectively; the staging
copies therefore reuse a small per-layer arena rather than reserving
cache-sized space. A persistent GPU-side mirror is consequently **not** free in
VRAM, and the 552.50 MiB at 16K is genuine additional device memory.

Since the whole 16K Q8_0 cache is only 552.50 MiB and buys 82.8%, the useful
next step is a **tunable partial GPU KV window**: keep the newest N tokens
device-resident, stream only the older remainder, and merge the two partial
attention results. FlashAttention decomposes exactly over the KV dimension via
log-sum-exp rescaling, so the merge is lossless and preserves the Q8_0 quality
gate. The fork already has a segmented merge primitive in
`ggml_kv_tail_attention_merge_segmented` for KVarN tails, which is the closest
existing reference. The dial degrades gracefully: N = 0 reproduces today's
behaviour, N = n_ctx reproduces GPU-resident KV.

## Experiment 011: tunable partial GPU KV residency

Status: retained. Implements the direction identified at the end of Experiment
010 and gives the first continuous dial between host-resident and GPU-resident
attention KV.

### Change

`--kv-gpu-layers N` (env `LLAMA_ARG_KV_GPU_LAYERS`, also accepted by
`llama-bench`) keeps the first N attention-KV layers device-resident while
`--no-kv-offload` is active; the remaining layers stay on the host exactly as
before. `N = 0` reproduces current behaviour and any `N` at or above the
attention-layer count matches `--kv-offload`.

The selection is per layer rather than per token range. That was deliberate:
`llama_kv_cache` already chooses a buffer type per layer at both the route-probe
and the allocation site, so the placement decision fits the existing structure
with no change to cell indexing, defragmentation, state serialization, or the
attention graph. Each layer's attention still reads one contiguous cache in one
buffer, so no partial-result merge and no new kernel are involved. A recency
window would have required splitting each layer's cache in two and combining the
two partial attentions with log-sum-exp rescaling for the same byte reduction.

Implementation is `[TAG_KV_PARTIAL_GPU_RESIDENCY]` in `llama-kv-cache.cpp`: a
`layer_on_device` plan is computed once from `hparams.has_kv()` and the layer
filter, then consulted at both buffer-type selection sites. The value flows
`common_params` -> `llama_context_params::kv_gpu_layers` (appended at the end of
the struct, preserving existing field offsets) -> `llama_cparams` ->
`llama_model::create_memory()`, reaching the hybrid `mem_attn` cache and the
plain non-SWA target cache.

### Configuration

RTX 4070 12 GiB (compute capability 8.9), Intel Core i5-13400F, 31 GiB RAM,
PCIe 4.0 x16. Model `Qwen3.8-27B-UD-IQ2_M.gguf` (`qwen35`, 27.32B, 9.60 GiB,
16 attention layers of 65). Release CUDA build, `GGML_NATIVE=ON`. Three distinct
P-cores via `taskset -c 0,2,4`; note that `taskset -c 0-2` on this part spans
only two physical cores, since CPUs 0 and 1 are SMT siblings. Common settings:
`-ngl 99 -t 3 --poll 100 -nkvo 1 -fa on -ctk q8_0 -ctv q8_0 --kv-cpu-pinned
--recurrent-state-offload`. Decode `-p 0 -n 32 -r 2`, prefill `-p 512 -n 0 -r 3`.

### Results

Decode, 32 tokens:

| `--kv-gpu-layers` | d16384 (t/s) | ms/token | d32768 (t/s) | ms/token |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 19.70 +/- 0.11 | 50.76 | 13.03 +/- 0.05 | 76.75 |
| 4 | 22.34 +/- 0.12 | 44.76 | -- | -- |
| 8 | 25.70 +/- 0.17 | 38.91 | 18.76 +/- 0.09 | 53.30 |
| 12 | 30.15 +/- 0.22 | 33.17 | 24.00 +/- 0.15 | 41.67 |
| 16 | 36.00 +/- 0.21 | 27.78 | 32.42 +/- 0.17 | 30.84 |

Full residency is +82.7% at 16K and **+148.8% at 32K**. The endpoints reproduce
the independent references from Experiment 010 (19.76 t/s host-resident and
36.13 t/s GPU-resident at 16K), confirming the dial is not introducing a path of
its own.

Per-token time is linear in the number of host-resident layers, which is the
signature of a pure transfer cost: about 1.44 ms per layer at 16K and 2.85 ms
per layer at 32K, roughly doubling with depth as the per-layer tensor doubles.

Prefill of 512 tokens at depth 16384 was 984.72 +/- 6.65, 1009.94 +/- 10.70, and
1033.99 +/- 17.00 t/s at N = 0, 8, and 16. Prefill improves slightly; there is no
regression to trade against the decode gain.

### Resource cost

Reported KV buffer split at depth 16384:

| `--kv-gpu-layers` | `CUDA0` KV | `CUDA_Host` KV |
| ---: | ---: | ---: |
| 0 | -- | 552.50 MiB |
| 4 | 138.12 MiB | 414.38 MiB |
| 8 | 276.25 MiB | 276.25 MiB |
| 12 | 414.38 MiB | 138.12 MiB |
| 16 | 552.50 MiB | -- |

The total is unchanged, so this moves memory rather than adding it: 34.53 MiB of
device memory per layer at 16K and 69.06 MiB at 32K. The exchange rate is
therefore 34.53 MiB per 1.44 ms/token at 16K, and 69.06 MiB per 2.85 ms/token at
32K. At 32K, full residency needs 1,105 MiB of device memory and still fits on
this 12 GiB card alongside the 9,227 MiB of weights.

### RTX 5070 Ti integration validation

The implementation was subsequently exercised at commit `b8f855761` with the
Qwen3.8 27B AtomicChat model used throughout this branch, target Q8_0/Q8_0 CPU
KV, MTP maximum depth 3, draft Q4_0/Q4_0, target ubatch 512, draft ubatch 128,
and three pinned decode threads. At 64K configured context with a 60,052-token
prompt, moving four of the sixteen target attention layers to the GPU changed
decode from 19.32 to 25.57 t/s (+32.3%), prefill from 1,164.98 to 1,188.23 t/s
(+2.0%), and peak process VRAM from 14,924 to 15,466 MiB (+542 MiB). The N=4
run left only about 414 MiB of physical VRAM margin, so it is the practical
maximum tested setting on this 15,880 MiB card at that context and MTP depth.

At 32K and the same MTP depth, N=4 improved representative decode from 30.77
to 37.60 t/s while adding about 238 MiB of peak VRAM. Those two longer samples
did not terminate at identical token counts, so they are a compatibility and
directional-performance check rather than a publication-grade matched result.

After integration onto `exp/kv-cpu-offload`, a bounded 4K startup smoke test at
commit `abb66abb1` used only `--kv-cpu-pinned`,
`--recurrent-state-offload`, and `--kv-gpu-layers 2`. It reported 17.00 MiB of
CUDA KV, 119.00 MiB of CUDA-host KV, and 149.62 MiB of CUDA recurrent state,
confirming that all three supported flags reached the intended allocation
policy without the removed `GGML_KV_CPU_PINNED` or
`GGML_RECURRENT_STATE_OFFLOAD` switches.

### Post-merge code-layout cleanup

The integration was subsequently refactored to keep placement inside the
standard cache's existing per-layer buffer-type plan. The initial implementation
maintained a parallel `vector<bool>` for partial residency and repeated the
host/device choice in tail-capability probing and tensor allocation. The cleanup
instead computes one `layer_buft` table and consumes it in both paths. Pinned
host-buffer selection is also shared by the standard and KVarN caches rather
than duplicated in both translation units.

This exposed and fixed one edge case: a layer that shares another context's KV
previously consumed the `--kv-gpu-layers` budget even though it allocated no KV
of its own. Shared-layer indices are now resolved once, excluded from placement,
and reused by tail planning and allocation. Hybrid and hybrid-iSWA constructor
arguments were regrouped by attention and recurrent ownership, the recurrent
offload decision is derived once in `llama_model::create_memory()`, and
`llama-bench` now treats `--kv-gpu-layers` as a normal numeric benchmark
dimension, including comma/range sweeps and negative-value rejection. Its CSV,
JSON, JSONL, SQL, and Markdown records now include `kv_cpu_pinned`,
`recurrent_state_offload`, and `kv_gpu_layers`, so placement sweeps remain
self-describing.

The Release CUDA build completed successfully and the focused suite passed 8/8.
The same bounded 4K allocation smoke test remained exactly 17.00 MiB CUDA KV,
119.00 MiB CUDA-host KV, and 149.62 MiB CUDA recurrent state with two resident
attention layers, confirming that the layout cleanup did not change the tested
placement policy.

### Correctness

Greedy generation (`--temp 0 --seed 1234`, 96 tokens, `-no-cnv`) was
byte-identical across `--kv-gpu-layers` 0, 8, and 16. This is expected rather
than fortunate: both placements feed the same Q8_0 bytes to the same CUDA
FlashAttention kernel, and only the source buffer differs. The Q8_0 quality gate
is untouched.

`ctest -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"` passed 8/8.

### Scope and limitations

- Wired for the hybrid `mem_attn` cache and the plain non-SWA target cache. The
  KVarN, ISWA, and `llama_kv_cache_kvarn` constructors take the new parameter's
  default of 0 and are unaffected; extending them is unstarted.
- `N` counts attention-KV layers from the lowest index. No evidence was gathered
  that the choice of which layers to keep matters for throughput, and the linear
  scaling suggests it does not, but this was not tested directly.
- There is no automatic sizing. Choosing `N` to fill available VRAM requires
  knowing the cache size per layer at the configured context, which the user must
  currently work out from the reported buffer sizes.
- `llama-bench` sweeps `--kv-gpu-layers` like its other numeric benchmark
  dimensions. `--kv-cpu-pinned` and `--recurrent-state-offload` remain
  process-wide toggles.
- Measured on one machine only. The gain is proportional to the host-to-device
  transfer cost, so a host with faster PCIe than this PCIe 4.0 x16 link will see
  a smaller relative improvement.
## Experiment 012: capped MTP recurrent planes with checkpoint/replay

Status: rejected as a merge candidate; retained uncommitted for diagnosis.

### Implementation and configuration

Base commit: `5d6441f5310625911e6e9d2699d711df74af888c`. The candidate
was measured from the uncommitted `exp/mtp-recurrent-plane-cap` worktree, so
the results must not be attributed to the base binary.

The candidate adds `--spec-mtp-rs-planes N`, environment variable
`LLAMA_ARG_SPEC_MTP_RS_PLANES`, and INI key `spec-mtp-rs-planes`. Zero keeps
the existing `draft_max + 1` total recurrent planes. Positive values from two
through `draft_max + 1` allocate exactly that many total planes, leaving
`N - 1` direct rollback positions. Draft depth is unchanged. A target host
checkpoint is captured only when the actual draft exceeds the direct horizon;
it is restored and the accepted prefix replayed only when the rejected suffix
also exceeds that horizon. Per-request timing JSON and server timing logs report
capture/restore counts and time, serialized target/draft/speculative bytes,
peak payload, replay cycles, and replay-batch tokens.

All runtime comparisons explicitly set both experimental placement switches:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --n-gpu-layers all --fit off --split-mode none --flash-attn on \
  --no-kv-offload --parallel 1 --cont-batching --kv-unified \
  --batch-size 1024 --ubatch-size 512 --spec-draft-ubatch-size 128 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --spec-type draft-mtp --spec-draft-n-max DEPTH \
  --spec-mtp-rs-planes PLANES --draft-p-min 0.85 \
  --threads 3 --threads-batch 24 --cpu-range 0-2 \
  --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --seed 1234 --cache-ram 0
```

The clean-process short test used a 4,096-token context, 128 greedy output
tokens, and prompt `Continue the sequence and explain the rule briefly: 1, 4,
9, 16, 25,`. The candidate was four total planes. All six short outputs had
SHA-256 `e87b691143030a0ac2025ab750ed5aeebbca7296734de5c24b884cbdda7f223b`.

| MTP depth | Planes | Decode | Init/live VRAM | RS buffer | Captures/restores | Capture/restore | Replay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | full (4) | 109.17 t/s | 14,142/14,164 MiB | 598.50 MiB | 0/0 | 0/0 ms | 0/0 tokens |
| 3 | 4 | 109.20 t/s | 14,142/14,164 MiB | 598.50 MiB | 0/0 | 0/0 ms | 0/0 tokens |
| 5 | full (6) | 105.56 t/s | 14,440/14,462 MiB | 897.75 MiB | 0/0 | 0/0 ms | 0/0 tokens |
| 5 | 4 | 69.57 t/s | 14,142/14,164 MiB | 598.50 MiB | 28/3 | 601.07/26.95 ms | 3/8 tokens |
| 8 | full (9) | 99.50 t/s | 14,890/14,914 MiB | 1,346.62 MiB | 0/0 | 0/0 ms | 0/0 tokens |
| 8 | 4 | 61.54 t/s | 14,142/14,164 MiB | 598.50 MiB | 22/14 | 478.15/123.86 ms | 14/65 tokens |

Four planes saved 299.25 MiB of recurrent allocation at depth 5 and 748.12
MiB at depth 8. The corresponding short decode regressions were 34.1% and
38.1%. Each target checkpoint was 156,914,888 bytes (149.6 MiB); its buffer was
reused, so cumulative serialized bytes are traffic counters rather than
simultaneously resident host memory.

### Long reasoning comparison and transfer accounting

The matched 32,000-context comparison requested 5,000 greedy tokens for the
orbital-sandbox single-file HTML prompt recorded in the benchmark artifacts.
Each clean process ran under:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 LLAMA_TRACE=1 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o TRACE \
  build-cuda-all/bin/llama-server [common arguments above] --ctx-size 32000
```

| Measurement | MTP-8 full (9 planes) | MTP-8 capped (4 planes) |
| --- | ---: | ---: |
| Prompt | 575.18 t/s | 600.55 t/s |
| Decode under Nsight | 55.66 t/s | 54.29 t/s |
| Initialized/live/peak VRAM | 15,286/15,320/15,320 MiB | 14,538/14,570/14,570 MiB |
| RS buffer | 1,346.62 MiB | 598.50 MiB |
| Captures/restores | 0/0 | 132/18 |
| Capture/restore wall time | 0/0 ms | 2,888.89/167.14 ms |
| Cumulative target payload | 0 | 20,710,056,048 bytes |
| Peak checkpoint payload | 0 | 156,914,888 bytes |
| Replay | 0 | 18 cycles / 58 batch tokens |
| Process host `VmHWM` | 15,182.9 MiB | 15,182.8 MiB |
| Nsight H2D | 347,417.603 MB | 358,234.870 MB |
| Nsight D2H | 6,678.289 MB | 27,313.481 MB |

Nsight totals, not serialized payload sizes, are the authoritative transfer
measurement. The cap added 10,817.267 MB H2D and 20,635.192 MB D2H. The latter
closely tracks cumulative target checkpoint capture traffic.

### Correctness result and disposition

The long outputs were not identical. Full planes produced SHA-256
`61634266bc12320f360f804c472479e84748d45648ef826ca7cd59f1106b528d`;
four planes produced
`9f66a45fae91a1ec0bd261c712ccf90a6b87dd4ebced20afbbfa9b3d8b1d7317`,
with the first text difference at byte 1,337. A clean-process 512-token control
confirmed that two full-plane runs were identical to each other while the
capped run diverged. Therefore this is caused by deep checkpoint/replay, not
ordinary run-to-run CUDA nondeterminism.

Replay reconstructs an omitted recurrent boundary using a smaller target batch
than the original verification. The resulting floating-point state can differ
slightly; after enough tokens a close greedy decision changes. An attempted
diagnostic that retained the first pass's sampled token still diverged because
the reconstructed recurrent state itself was different, and was removed.

The four-plane compatibility candidate fails the fixed-output acceptance gate
and must not be merged or described as output-equivalent. It also imposes
material checkpoint traffic and short-request overhead. The next implementation
must retain selected rollback boundaries directly from the original target
verification, beginning with sparse GPU snapshots for Qwen CUDA and failing
closed elsewhere. Host checkpoint/replay remains useful as instrumentation and
a compatibility experiment, not as the accepted cap mechanism.

## Experiment 013: sparse GPU replay for capped MTP recurrent planes

Status: retained. The final implementation is commit
`d743456922e6005578d5c94e74e99180c0dbe4c7`; the benchmark provenance below
continues to identify the pre-commit binary exactly as it reported itself.

### Implementation

Base commit: `5d6441f5310625911e6e9d2699d711df74af888c`. The candidate
was built as version 11217 from the uncommitted
`exp/mtp-recurrent-plane-cap` worktree, so these results must not be attributed
to the base binary.

This revision replaces target host checkpoint/replay for a true plane cap with
a sparse snapshot mode gated by model-graph and recurrent-backend capabilities.
NVIDIA CUDA currently advertises the backend capability. An ordinary capped
verification retains the exact pre-verification input in one plane and the most
recent outputs in the remaining planes. A rejected suffix of at most `N - 2`
uses a retained boundary directly. A deeper rejection restores the retained
input, reruns the same full target batch shape, and writes only the originally
accepted boundary to plane zero. Convolution state and gated-delta-net state use
the same selected-boundary policy.

The first conforming trace exposed a CUDA fusion-matcher miss: selected replay
writes one state plane, while the matcher still required a four-plane
destination. That produced 4,888.461 MB of separate D2D copies. Matching the
selected operation as a one-plane destination lets the fused GDN kernel write
the cache directly and removes that D2D traffic; the final results below use
this fused path.

The target graph reuse keys include sparse mode and selected token, preventing
a normal-verification graph from being reused for selected-boundary replay.
That omission was the cause of the initial GPU replay failure: the replay graph
was still writing the final token state. A forced-replay diagnostic with the
full nine-plane allocation confirmed that the fixed replayed recurrent state
matched the originally selected boundary byte-for-byte across 156,894,364
serialized bytes before the temporary diagnostic was removed.

Zero and an explicit full-plane value preserve ordinary consecutive-snapshot
behavior. Only `0 < N < spec-draft-n-max + 1` enables sparse replay. The cap
fails closed unless the target graph declares selected recurrent snapshots and
every recurrent buffer resides on a backend that advertises the corresponding
operation. This removes the recurrent-memory architecture whitelist while
preserving the tested NVIDIA CUDA boundary. The target checkpoint counters
remain available for other speculative policies, but the capped path records
zero target captures, restores, and serialized target payload.

### Short fixed-seed sweep

Each clean server process used the Experiment 012 common arguments, both
placement environment switches, a 4,096-token context, temperature zero, seed
1234, and 128 output tokens for prompt
`Continue the sequence and explain the rule briefly: 1, 4, 9, 16, 25,`.
Every output had SHA-256
`64ccba06fd390281d73f4bf6d55e49f21e8cbf42a5a2064693b3a883d2c6e7c3`
when hashed with the response text's trailing newline.

| MTP depth | Planes | Decode | Init/live VRAM | RS buffer | Replay cycles/tokens | Checkpoints |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | full (4) | 91.18 t/s | 14,142/14,164 MiB | 598.50 MiB | 0/0 | 0/0 |
| 3 | explicit full (4) | 90.99 t/s | 14,142/14,164 MiB | 598.50 MiB | 0/0 | 0/0 |
| 5 | full (6) | 93.67 t/s | 14,440/14,462 MiB | 897.75 MiB | 0/0 | 0/0 |
| 5 | capped (4) | 87.23 t/s | 14,142/14,170 MiB | 598.50 MiB | 3/18 | 0/0 |
| 8 | full (9) | 94.59 t/s | 14,890/14,912 MiB | 1,346.62 MiB | 0/0 | 0/0 |
| 8 | capped (4) | 85.29 t/s | 14,142/14,164 MiB | 598.50 MiB | 5/39 | 0/0 |

The short cap cost 6.9% at MTP-5 and 9.8% at MTP-8. It saved 299.25 MiB and
748.12 MiB of recurrent allocation respectively. These short samples trigger
replay often enough that their percentage overhead is not representative of a
long reasoning request.

### Matched 5,000-token MTP-8 run

The exact prompt was:

```text
Write a single self-contained HTML file: an interactive orbital mechanics sandbox.
Canvas-based, 60fps. Users can click-drag to fling new planets into the system;
gravity is simulated with velocity Verlet integration against a central star.
Include trailing orbit paths that fade, collision merging with a mass-conserving
flash, a mass/velocity readout on hover, and a pause/reset UI. No libraries,
no assets — one file, pure JS.
```

The request used the OpenAI-compatible chat endpoint, the model's default
sampling parameters, seed 1234, `max_tokens: 5000`, no prompt cache, and no
streaming. Both clean processes ran under:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 LLAMA_TRACE=1 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o TRACE \
  build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --ctx-size 32000 --parallel 1 --cont-batching \
  --kv-unified --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 8 \
  --spec-mtp-rs-planes PLANES --spec-draft-ubatch-size 128 \
  --draft-p-min 0.85 --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --threads 3 --threads-batch 24 --cpu-range 0-2 \
  --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --reasoning-loop-guard force-close --seed 1234 --cache-ram 0
```

An external ten-second poll of `/slots` exposed generated-token progress, and
`nvidia-smi` sampled process VRAM without restarting the request.

| Measurement | Full (9 planes) | Capped (4 planes) | Change |
| --- | ---: | ---: | ---: |
| Prompt | 663.26 t/s | 674.41 t/s | +1.7% |
| Decode under Nsight | 51.383 t/s | 50.979 t/s | -0.79% |
| Target RS allocation | 1,346.62 MiB | 598.50 MiB | -748.12 MiB |
| Observed process VRAM | 15,314 MiB live sample | 14,538 init / 14,576 peak | at least -738 MiB |
| Process host `VmHWM` | 15,222.6 MiB | 15,188.8 MiB | -33.8 MiB |
| Draft accepted/generated | 1,689/2,027 | 1,689/2,027 | identical |
| Replay cycles/batch tokens | 0/0 | 38/217 | +38/+217 |
| Target captures/restores | 0/0 | 0/0 | unchanged |
| Target checkpoint payload | 0 | 0 | unchanged |
| Nsight H2D | 388,825.670 MB | 392,764.233 MB | +3,938.563 MB |
| Nsight D2H | 6,646.602 MB | 6,896.808 MB | +250.206 MB |
| Nsight D2D | 0 MB | 0 MB | unchanged |

The full and capped reasoning fields were byte-identical and had SHA-256
`09ab907a3b42bb586f7714eb1327461ff853e37cf4895795ab4328f99382bdb6`
without an added newline. They also match the earlier clean full-plane MTP-8
artifact. Nsight totals, rather than serialized payload counters, are the
authoritative transfer measurement. Unlike Experiment 012, there is no
recurring roughly 150 MiB target checkpoint transfer to host. The remaining
extra H2D/D2H traffic comes from the 38 full-shape replay batches rather than a
serialized recurrent-state payload.

Artifacts:

- `/tmp/mtp8-gpu-replay-5k-full-response.json`
- `/tmp/mtp8-gpu-replay-5k-cap4-fused-response.json`
- `/tmp/mtp8-gpu-replay-5k-full-trace.nsys-rep`
- `/tmp/mtp8-gpu-replay-5k-cap4-fused-trace.nsys-rep`

The complete artifact hashes, environment/toolchain identity, build and ccache
configuration, corrected monitoring procedure, profiler SQL, implementation
chronology, and integration steps are in
[`mtp-recurrent-plane-cap-reproduction.md`](mtp-recurrent-plane-cap-reproduction.md).

### Validation and disposition

The argument parser, recurrent rollback, prompt-checkpoint, loop-guard,
sampler, and server fixture tests passed before and after the final matcher
change. The CUDA `GATED_DELTA_NET` backend-op selection passed 36/36 cases. A
startup control confirmed that explicit full MTP-3 reports a
three-token direct horizon with GPU replay disabled. The full/capped short
outputs matched at MTP-3, MTP-5, and MTP-8, and the long MTP-8 output now passes
the fixed-output acceptance gate. No hidden full-depth recurrent allocation
appears: server allocation logs and the shutdown memory breakdown show
598.50 MiB for four planes versus 1,346.62 MiB for nine.

A post-measurement interface cleanup replaced the Qwen architecture and CUDA
buffer-name checks with the model-graph/backend capability contract described
above; it did not change the replay graph or kernels. The rebuilt tree again
passed the seven focused regressions and all 36 CUDA gated-delta-net cases. A
fresh capped MTP-8 short request produced the same output hash, draft counts
(`114/89`), replay work (`5/39`), and zero checkpoint counts at 85.14 t/s. With
recurrent state deliberately left on CPU, the same configuration failed at
startup through the missing backend capability as intended.

Retain the sparse GPU candidate. Its long-run 0.79% decode cost is materially
smaller than the host-checkpoint candidate's traffic and short-request
regression, while preserving 748.12 MiB of MTP-8 recurrent VRAM. Keep the
full-plane default and the graph/backend capability fail-closed boundary.

### CPU-KV branch merge validation

The candidate was merged into `exp/kv-cpu-offload` as
`5e16db7c9182bd6fad148a3caa80fc86b11440d0`. Source merged without conflict;
the ledger conflict was resolved by preserving CPU-KV Experiments 008-011 and
renumbering the MTP entries 012-013. The committed Release CUDA server built,
the combined focused suite passed 13/13, and CUDA GDN passed 36/36.

A clean merged MTP-8/four-plane short request used
`--kv-cpu-pinned --recurrent-state-offload` in place of the historical
environment switches. It reproduced output SHA-256
`64ccba06fd390281d73f4bf6d55e49f21e8cbf42a5a2064693b3a883d2c6e7c3`,
draft/accepted 114/89, replay 5/39, checkpoints 0/0, and init/live process VRAM
14,142/14,164 MiB. Decode was 85.54 t/s. This confirms the merged placement
interface reaches the same cap/replay path; it is not a new matched 5K
performance comparison.

## Characterization 014: raw MTP-2 and MTP-6 recurrent-plane tradeoffs

Status: characterization only; no source change. The three-plane candidate is
a useful capacity/performance trade on this workload, but this single matched
run does not change the full-plane default.

### Question and correction

This comparison tests MTP maximum depth six while changing the target recurrent
allocation, not partial attention-KV placement:

```text
full: --spec-draft-n-max 6 --spec-mtp-rs-planes 0 --kv-gpu-layers 0
cap:  --spec-draft-n-max 6 --spec-mtp-rs-planes 3 --kv-gpu-layers 0
```

An initial candidate process mistakenly used `--kv-gpu-layers 2` with full
recurrent planes. It reached server health but received no request. It was
stopped, the GPU was confirmed clear, and none of its allocation or timing data
is included below.

Three is the total recurrent-plane allocation. In the retained sparse-replay
implementation, a true cap reserves one plane for the exact pre-verification
input, so three planes expose a one-token ordinary rejection horizon (`N - 2`),
not two. The remaining two planes hold the newest verification outputs. A
deeper rejection uses deterministic full-shape GPU replay.

### Provenance and fixed configuration

- Source and measured binary commit:
  `324873dc5ca44eb31727ba3bd09897841574fa3b`; server version 11228.
- Build: Release, CUDA, native CPU, CUDA FlashAttention, CUDA architecture
  `120a`, default quant matrix (`GGML_CUDA_FA_ALL_QUANTS=OFF`).
- Hardware: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB; Intel Core Ultra 9
  285K; the established strict CPU 0-2 decode and CPU 0-23 batch placement.
- Target/MTP model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.
- Projector:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf`,
  loaded with projector offload disabled.
- Context 32,000; one slot; target batch/ubatch 1,024/512; draft ubatch 128;
  all target and draft model layers on CUDA; split none; fit disabled.
- Target and draft K/V were both Q8_0/Q8_0 in CUDA-pinned host memory. The
  supported merged-branch controls `--kv-cpu-pinned` and
  `--recurrent-state-offload` were explicitly present; the removed historical
  environment switches were not used. Both rows explicitly set
  `--kv-gpu-layers 0`.
- Nsight Systems 2026.1.3 traced CUDA, NVTX, and OS runtime with CPU sampling
  and context-switch collection disabled. `LLAMA_TRACE=1` enabled per-draft
  server progress. An external ten-second loop sampled the child server's
  `nvidia-smi` allocation while the request ran.

Both clean processes used this command, changing only `PLANES`, `PORT`, `TAG`,
and the trace/log filenames:

```bash
LLAMA_TRACE=1 nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/mtp6-coding-5k-${TAG}-trace \
  build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  --kv-gpu-layers 0 --ctx-size 32000 --parallel 1 --cont-batching \
  --kv-unified --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 6 \
  --spec-mtp-rs-planes "${PLANES}" --spec-draft-ubatch-size 128 \
  --draft-p-min 0.85 --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --threads 3 --threads-batch 24 --cpu-range 0-2 \
  --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
  --alias mtp6-coding-5k --host 127.0.0.1 --port "${PORT}" \
  --log-file /tmp/mtp6-coding-5k-${TAG}.log --log-verbosity 4
```

The OpenAI-compatible request was 571 bytes, SHA-256
`6edbe87e0896f13681887883efba7b18a669e29802a161fcfa6a737bd5695995`:

```json
{"model":"mtp6-coding-5k","messages":[{"role":"user","content":"Write a single self-contained HTML file: an interactive orbital mechanics sandbox.\nCanvas-based, 60fps. Users can click-drag to fling new planets into the system;\ngravity is simulated with velocity Verlet integration against a central star.\nInclude trailing orbit paths that fade, collision merging with a mass-conserving\nflash, a mass/velocity readout on hover, and a pause/reset UI. No libraries,\nno assets — one file, pure JS."}],"max_tokens":5000,"seed":1234,"stream":false,"cache_prompt":false}
```

Temperature was intentionally omitted, selecting the model/server default. Both
rows processed 149 prompt tokens and produced 5,000 completion tokens, all
reported as reasoning, with finish reason `length`.

### Results

| Measurement | Full, 7 planes | Cap, 3 planes | Change |
|---|---:|---:|---:|
| Prompt | 619.463 t/s | 613.985 t/s | -0.88% |
| Decode under Nsight | 53.6445 t/s | 52.2560 t/s | -2.59% |
| Decode latency | 18.6413 ms/token | 19.1365 ms/token | +2.66% |
| Draft accepted/generated | 1,851/2,193 | 1,851/2,193 | identical |
| Acceptance / mean draft length | 0.84405 / 2.61 | 0.84405 / 2.61 | identical |
| Replay cycles/batch tokens | 0/0 | 88/430 | +88/+430 |
| Target captures/restores | 0/0 | 0/0 | identical |
| Target checkpoint payload | 0 | 0 | identical |
| Target recurrent allocation | 1,047.38 MiB | 448.88 MiB | -598.50 MiB |
| Init process VRAM | 14,986 MiB | 14,388 MiB | -598 MiB |
| Live/peak process VRAM | 15,018/15,018 MiB | 14,426/14,426 MiB | -592 MiB |
| Process `VmHWM` | 15,187.59 MiB | 15,186.11 MiB | -1.48 MiB |
| Request wall time | 100 s | 100 s | same one-second resolution |
| Nsight H2D | 368,271.043 MB / 196,492 copies | 378,465.886 MB / 200,628 copies | +10,194.843 MB / +4,136 |
| Nsight D2H | 6,649.862 MB / 140,044 copies | 7,145.663 MB / 143,212 copies | +495.800 MB / +3,168 |
| Nsight D2D | 0 MB | 0 MB | unchanged |

The complete-trace H2D increase was 2.77% and D2H increase was 7.46%.
Aggregate H2D copy duration increased from 7,302.897 to 7,509.101 ms;
aggregate D2H duration increased from 249.904 to 256.016 ms. These are
whole-process Nsight totals, including initialization, and are authoritative
transfer measurements. The server's replay-batch counter identifies the added
work; serialized checkpoint payload remains zero and is not used as a transfer
proxy.

The reasoning fields were byte-identical and had SHA-256
`2f994f600563b04a0a8ce172b59b8f04515edae27664f0b15cab003ad257f44e`
without an added newline. Draft totals, acceptance at every position, and token
counts were also identical. The full response JSON hashes differ because their
timing and replay fields differ.

### Artifacts

`/tmp` is ephemeral. The exact artifacts were:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `/tmp/mtp6-coding-5k-request.json` | 571 | `6edbe87e0896f13681887883efba7b18a669e29802a161fcfa6a737bd5695995` |
| `/tmp/mtp6-coding-5k-raw-response.json` | 19,739 | `cd1bee84af7d3f70b730e7e8f76f3def08ed7a105388733217cff7ad6939c306` |
| `/tmp/mtp6-coding-5k-rs3-response.json` | 19,742 | `58cb86a9438b50e59dd91e91d4e9500e14b440ab2885c6a9b8550498519322f7` |
| `/tmp/mtp6-coding-5k-raw.log` | 245,465 | `87479e50352f0fd07d1a107f7db4ce0ac45eded066f0a7cd3da15e1a42c7c6c2` |
| `/tmp/mtp6-coding-5k-rs3.log` | 247,686 | `1cecb8ee1a42b27d902ffd6120ec184c822c6770916d573184a9c2a70378c3b0` |
| `/tmp/mtp6-coding-5k-raw-trace.nsys-rep` | 249,288,079 | `21af80c91a026a788dad4b6e9fd037f96f6cfbda1cc4dcd43fda2ded4c7d55cb` |
| `/tmp/mtp6-coding-5k-rs3-trace.nsys-rep` | 245,608,445 | `ef76598fd937a32371bbe68e4ec11f9622c8e1d2f1fb368111c33f2d37b4f421` |
| `/tmp/mtp6-coding-5k-raw-trace.sqlite` | 737,636,352 | `dcbe8681dffd99e5c2b5c6f5e20d61a9576297beb84563c6dc19ae787f9f4ef3` |
| `/tmp/mtp6-coding-5k-rs3-trace.sqlite` | 718,307,328 | `3ba210d7191fbc154918bb775503a2868dbff0300ef188066a7bb6ce1de8d771` |

### Disposition

For this 5K coding request, MTP-6 with three total recurrent planes exchanges
2.59% decode throughput for 592 MiB lower observed peak process VRAM and
598.50 MiB lower recurrent allocation, while preserving exact output. It is a
credible capacity-first setting. Full planes remain the compatibility and
maximum-throughput default; a four-plane MTP-6 comparison would be the next
useful point if a gentler memory/performance trade is desired.

### Raw MTP-2 equal-memory follow-up

A third clean process tested raw MTP-2 after the MTP-6 comparison. It used the
same commit, binary, model/projector, request file, default sampling, Nsight
settings, placement flags, affinity, Q8_0/Q8_0 target and draft KV, and external
progress monitor. Only these server values and artifact names changed:

```text
--spec-draft-n-max 2 --spec-mtp-rs-planes 0
--host 127.0.0.1 --port 8109
/tmp/mtp2-coding-5k-raw-{log,response,trace}
```

Raw MTP-2 allocates its default three total planes. Its 448.88 MiB recurrent
buffer and 14,388 MiB initialized process VRAM exactly matched capped MTP-6
with three total planes, making this an equal-recurrent-memory speed comparison.
It has a two-token direct rollback horizon and does not use selected replay.

| Measurement | Raw MTP-2, 3 planes | Capped MTP-6, 3 planes | Full MTP-6, 7 planes |
|---|---:|---:|---:|
| Prompt | 626.232 t/s | 613.985 t/s | 619.463 t/s |
| Decode under Nsight | 50.8641 t/s | 52.2560 t/s | 53.6445 t/s |
| Decode latency | 19.6602 ms/token | 19.1365 ms/token | 18.6413 ms/token |
| Draft accepted/generated | 1,527/1,727 | 1,851/2,193 | 1,851/2,193 |
| Acceptance / mean draft length | 0.88419 / 2.27 | 0.84405 / 2.61 | 0.84405 / 2.61 |
| Replay cycles/batch tokens | 0/0 | 88/430 | 0/0 |
| Recurrent allocation | 448.88 MiB | 448.88 MiB | 1,047.38 MiB |
| Init VRAM | 14,388 MiB | 14,388 MiB | 14,986 MiB |
| Live/peak VRAM | 14,422/14,422 MiB | 14,426/14,426 MiB | 15,018/15,018 MiB |
| Process `VmHWM` | 15,187.77 MiB | 15,186.11 MiB | 15,187.59 MiB |
| Nsight H2D | 399,518.732 MB / 206,302 copies | 378,465.886 MB / 200,628 copies | 368,271.043 MB / 196,492 copies |
| Nsight D2H | 6,468.824 MB / 148,698 copies | 7,145.663 MB / 143,212 copies | 6,649.862 MB / 140,044 copies |
| Nsight D2D | 0 MB | 0 MB | 0 MB |

At effectively equal VRAM, capped MTP-6 was 2.74% faster than raw MTP-2.
Full MTP-6 was 5.47% faster than raw MTP-2, at a 596 MiB higher observed peak
and 598.50 MiB higher recurrent allocation. Raw MTP-2 generated fewer accepted
draft tokens and therefore performed more target-side work; its complete trace
transferred 21,052.846 MB more H2D than capped MTP-6 despite avoiding replay.

Raw MTP-2 produced a different default-sampling reasoning stream, SHA-256
`f1e577d9f5fb2218aae850846f0bb1c59ac3c852dc03e797271b2f1af3194cb8`
without a newline, versus the common full/capped MTP-6 hash. Changing MTP depth
changes verification batch shapes and can change fixed-seed stochastic output;
unlike the full-versus-capped MTP-6 pair, this follow-up is therefore a matched
prompt/token-count serving comparison rather than an output-equivalence test.
Both responses still contain 149 prompt and 5,000 reasoning tokens and finish
at the requested length.

Raw MTP-2 artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `/tmp/mtp2-coding-5k-raw-response.json` | 20,732 | `380658990c32e4b271a7c87898c504541b2df938cbbb1628313c6bffd8436b3b` |
| `/tmp/mtp2-coding-5k-raw.log` | 250,047 | `93931efcdbf301c1e380b219dc3cff6f4e7aecf14e5eaae4a1f9271e0da55ed7` |
| `/tmp/mtp2-coding-5k-raw-trace.nsys-rep` | 241,242,124 | `b3cbc981fbad22cae175fd2b927918a86c72ccc3f93e65449a8481117c3d0a83` |
| `/tmp/mtp2-coding-5k-raw-trace.sqlite` | 699,625,472 | `b8bfdca34b7002b001d9debb06f9b467a7ae3b3088ffb2bfcd9cc245f5c49cba` |

Disposition: for this long coding workload and the same three-plane recurrent
budget, retain capped MTP-6 as the better-performing option. Raw MTP-2 avoids
replay but gives back more throughput than replay costs at MTP-6.

## Experiment 015: separate attention-compute and KV-storage placement

Status: retained architectural cleanup. This change makes the established
pinned-host-KV/CUDA-attention policy explicit; it is not presented as a
throughput optimization.

### Question and implementation

The graph builder historically used internal `offload_kqv` for two different
decisions: persistent attention-KV placement and whether to attach a CPU
backend constraint to the completed attention region. That coupling no longer
described the pinned CPU-KV path observed in Experiment 011, where the generic
scheduler selected CUDA FlashAttention while K/V remained in `CUDA_Host`.

The candidate adds internal `offload_attn_compute`. `offload_kqv` continues to
control persistent cache placement. With operation offload enabled, pinned CPU
KV or any partial device-KV layer keeps attention accelerator-eligible without
changing cache residency. Plain pageable CPU KV retains the existing CPU
constraint. This introduces no public option or serialized state and does not
change the `llama_context_params` layout.

- Base commit: `30054cb450fe20c4594ae86698d2683b5a7bc3ea`.
- Measured candidate commit:
  `d20fe04f769daeca807ae52bb0571f2b17dfc28e`; server version 11230.
- The final commit amends only this ledger onto the measured candidate; its
  compiled source is identical.
- Build: Release, CUDA, native CPU, CUDA FlashAttention, CUDA architecture
  `120a`, expanded quant matrix (`GGML_CUDA_FA_ALL_QUANTS=ON`).
- Hardware: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB; Intel Core Ultra 9
  285K; strict CPU 0-2 decode and CPU 0-23 batch placement.
- Model and projector are the same Qwen3.8 27B files used in Characterization
  014. Target and draft K/V are Q8_0/Q8_0 in pinned host memory.

### Placement smoke test

An allocation-only `llama-bench` smoke test used the target model, prompt 32,
one repetition, no warmup, `--progress`, `--no-kv-offload`,
`--kv-cpu-pinned`, `--recurrent-state-offload`, Q8_0/Q8_0 KV, and scheduler
debug output. It retained 34 graph splits, placed all 16 unique
`FLASH_ATTN` nodes on CUDA0, and kept the cache-store splits on CPU. The target
KV buffer was 8.50 MiB in `CUDA_Host`. A non-debug repetition completed at
256.41 t/s. The debug run's 239.28 t/s is diagnostic only.

### Deep-context protocol

The server command was identical for the clean base and candidate processes
apart from log name and binary commit:

```bash
build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  --kv-gpu-layers 0 --ctx-size 140000 --parallel 1 --cont-batching \
  --kv-unified --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 6 \
  --spec-mtp-rs-planes 3 --spec-draft-ubatch-size 128 \
  --draft-p-min 0.85 --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --threads 3 --threads-batch 24 --cpu-range 0-2 \
  --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
  --alias qwen3.8-27b-mtp6-rs3 --host 127.0.0.1 --port 8080 \
  --log-file LOG --log-verbosity 3
```

`llama-benchy` 0.4.0 fetched the official tokenizer itself and used cached
text. Native JSON progress was emitted to the terminal:

```bash
llama-benchy --base-url http://127.0.0.1:8080/v1 \
  --model Qwen/Qwen3.5-27B \
  --served-model-name qwen3.8-27b-mtp6-rs3 \
  --pp 4096 --tg 1 --depth 128000 --runs 1 \
  --enable-prefix-caching --no-warmup --no-adapt-prompt \
  --skip-coherence --latency-mode none --format json \
  --save-result RESULT --emit-progress -
```

Each process first evaluated 128,054 context-load tokens. Prefix restoration
then retained 128,043 tokens and the server evaluated 4,106 actual
chat-formatted tokens for the nominal 4,096-token deep prefill. A 500 ms
`nvidia-smi` process monitor recorded initialized and peak VRAM.

### Results

| Measurement | Base | Candidate | Change |
|---|---:|---:|---:|
| Context load, server | 778.64 t/s | 772.35 t/s | -0.81% |
| Context load, Benchy | 778.016 t/s | 771.755 t/s | -0.80% |
| Deep prefill, server | 446.88 t/s | 436.67 t/s | -2.28% |
| Deep prefill, Benchy | 439.631 t/s | 429.575 t/s | -2.29% |
| Deep prefill server time | 9,188.10 ms | 9,402.98 ms | +2.34% |
| Init process VRAM | 15,768 MiB | 15,768 MiB | unchanged |
| Peak process VRAM | 15,792 MiB | 15,792 MiB | unchanged |

These are single runs, as required for the expensive 128K-depth protocol. The
candidate's 2.3% lower deep-prefill result is therefore a cautionary signal,
not a statistically supported regression claim. The unchanged graph split,
CUDA FlashAttention placement, and memory footprint support the intended
architectural equivalence. This cleanup should not be used to claim a speedup;
a repeated A/B run is required before making a performance conclusion.

One earlier candidate attempt is excluded. A separate worktree was compiling
CUDA and C++ with up to 24 jobs throughout that attempt; multiple `ptxas` and
`fatbinary` processes saturated cores and system load exceeded 14. Its
720.02 t/s context load and 422.38 t/s deep prefill are invalid and are not
included in the table. The server was stopped, all build and CUDA processes
were confirmed absent, and the candidate row above came from the subsequent
clean process.

### Validation and artifacts

The expanded-matrix CUDA server, `llama-bench`, and `test-arg-parser` built
successfully. The argument-parser suite passed. `git diff --check` passed.
The placement smoke and deep run completed without an inference error.

`/tmp` is ephemeral. The retained exact timing artifacts are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `/tmp/qwen38-prefill-128k-benchy.json` | 2,762 | `0d6e26c0f2674fc0d5a6d75827e8795ef42cd4f58f465f5e2502da7b97d27500` |
| `/tmp/qwen38-prefill-128k-server.log` | 62,405 | `322693a52a4e9deb3858f1883c9394288d17d96ca1cb967574cf429e4b22691b` |
| `/tmp/qwen38-prefill-128k-placement-clean-benchy.json` | 2,762 | `a4e336fb42e114eb1be15dc957d33c79c46c243a2e585806184ecc01dd266ba2` |
| `/tmp/qwen38-prefill-128k-placement-clean-server.log` | 22,403 | `e69577e15ecd0806eeb017d92edbf6aad45153f512fe93db746d64a5bcb689b4` |
| `/tmp/attn-placement-pinned-debug.log` | 4,089,270 | `db141a5b48ac7d80c20ffb8053594d99751d7a8cf845850b49cc6b8f188866e2` |

Disposition: retain the separation as an internal architectural cleanup. It
removes a misleading cache-derived attention constraint while preserving
plain CPU-KV behavior and leaving actual backend selection to the scheduler.
Treat performance as neutral/uncertain pending repeated A/B evidence; the
single deep run provides no basis for a speed claim.

## Experiment 016: phase-aware target/MTP compute workspace

Status: retained as an opt-in candidate on the dedicated
`exp/phase-aware-prefill-decode` branch. The source was benchmarked before
commit as a source-only diff with SHA-256
`f6407cfd0987a37835a4c428ba6a63581928614a9efa691a20957b7e9bfcfec8`
against base `324873dc5ca44eb31727ba3bd09897841574fa3b`, then committed with
its evidence as `44474cd8668de56f9bf77a0682366351867f96f5` without a source
change. It was merged into `exp/kv-cpu-offload` as
`20777977d288fbb72e9541c1e982785e90d75993`; the ledger entry was renumbered
from 014 to 016 so Characterization 014 and Experiment 015 from the destination
branch remain intact. The candidate server SHA-256 is
`6b7169ca2141a606613527deecf7134c530e27884187de0a75b8c72e26ea54b2`;
the detached-base server SHA-256 is
`7b63ef24b1cfae76738793a47c8b96e5087307b627ef0d13302f8cddf89ae89b`.
The source-diff hash ties the pre-commit binary to the published implementation;
no measurement below is attributed to a differently configured binary.

### Objective and retained design

With CPU-resident target KV, active model weights and compute reservations
dominate process VRAM. The baseline retains a prompt-high-water target
scheduler and a separate prompt-high-water integrated-MTP scheduler throughout
generation, even though target and MTP graphs execute sequentially.

`--phase-aware-workspace` makes that transient lifetime explicit:

1. Each context initially reserves the generation geometry. The server derives
   its bound as `parallel * (1 + resolved speculative draft maximum)`, capped
   by `batch-size`; this experiment therefore uses seven tokens for MTP-6.
2. A submitted batch above the generation bound grows the scheduler to its
   configured physical ubatch: 512 target tokens and 128 draft tokens here.
3. Returning to generation starts one group shrink epoch. Every active member
   publishes its current allocation plan once before the physical backing can
   shrink.
4. Target and MTP schedulers retain independent graph plans but join a generic
   backing group. Chunks are matched by exact backend buffer-type identity and
   allocated to the maximum current member requirement, not their sum.
5. Physical generation counters invalidate peer graph addresses after backing
   replacement. Target/MTP ownership handoffs synchronize before the next
   sequential user executes.

The policy does not unload weights or the MTP head and does not move or change
KV, recurrent state, rollback planes, checkpoints, samplers, cache formats, or
model outputs. Fit/no-allocation probing still measures the full prompt
geometry. The option defaults off and is available through CLI,
`LLAMA_ARG_PHASE_AWARE_WORKSPACE`, and the INI key
`phase-aware-workspace`.

### Rejected implementations and the replay failure

A scheduler-destruction/model-unload approach was rejected before benchmark
because it mixed persistent state lifetime with transient graph allocation and
made later prompts unsafe. The allocator-sharing layer preserves the existing
scheduler plans and execution model instead.

The first shared-backing version automatically treated every smaller
allocation plan as permission to shrink. That was incorrect: recurrent
rollback replay builds legitimate graph variants inside the same generation
phase. The 128-token short request, with four replay cycles, performed six
target plus six draft reserves. The 5,000-token request, with 88 replay cycles,
performed 90 plus 90 reserves and spent 462.110 ms in target plus 265.997 ms in
draft reservation, or 728.107 ms total. An intermediate per-member high-water
permission still produced six plus six short-request reserves; temporary
instrumentation showed every extra replacement came from a peer plan-generation
publication rather than a local geometry transition.

The retained explicit group epoch fixes the cause. Only a real
prompt-to-generation token-geometry transition permits shrink; replay replans
within generation cannot begin another epoch. The same short and 5K cases now
perform exactly two target and two draft reserves per request: one grow and one
shrink for each context.

### Hardware, build, model, and environment

- GPU: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB usable process capacity,
  compute capability 12.0, driver 610.57.04.
- CPU: Intel Core Ultra 9 285K, 24 cores without SMT, one NUMA node; 62 GiB
  system RAM.
- Compiler: GCC 16.2.1 20260810; CUDA 13.3; Nsight Systems
  2026.1.3.425-261338342291v0.
- Build: Release, CUDA, native CPU, CUDA FlashAttention, effective CUDA
  architecture `120a`, KVarN enabled, default quant matrix
  (`GGML_CUDA_FA_ALL_QUANTS=OFF`).
- Model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.
- Multimodal projector:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf`.

The clean baseline was the exact detached base in
`/home/gencoolpc/beellama-phase-baseline`; the candidate was
`/home/gencoolpc/beellama-prefill-decode`. The shared ccache uses
`cache_dir=/home/gencoolpc/.cache/ccache`, `base_dir=/home/gencoolpc`,
`hash_dir=false`, and a 50 GiB limit. A public-header change caused 847 genuine
misses after the statistics reset; this is dependency invalidation, not a
failure to share cache objects across worktrees.

Every process was launched after explicitly unsetting
`GGML_KV_CPU_PINNED`, `GGML_RECURRENT_STATE_OFFLOAD`, `LLAMA_TRACE`,
`LLAMA_ARG_PHASE_AWARE_WORKSPACE`, and `CUDA_VISIBLE_DEVICES`. Placement and
phase selection therefore came only from the command below; no old experiment
environment flag could contaminate one side of the comparison.

### Matched server command and progress collection

The command template was identical on both sides except that `SERVER` selected
the exact baseline or candidate binary and candidate runs appended
`--phase-aware-workspace`:

```bash
env -u GGML_KV_CPU_PINNED \
    -u GGML_RECURRENT_STATE_OFFLOAD \
    -u LLAMA_TRACE \
    -u LLAMA_ARG_PHASE_AWARE_WORKSPACE \
    -u CUDA_VISIBLE_DEVICES \
SERVER \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  --ctx-size 140000 --parallel 1 --cont-batching --kv-unified \
  --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 6 \
  --spec-mtp-rs-planes 3 --spec-draft-ubatch-size 128 \
  --draft-p-min 0.85 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --threads 3 --threads-batch 24 \
  --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
  --host 127.0.0.1 --port PORT --verbosity 4 [--phase-aware-workspace]
```

Each measurement used a new process and an idle GPU. The harness polled
`/slots` every ten seconds to expose request progress and sampled
`nvidia-smi` process memory plus `/proc/PID/status` RSS, high-water RSS, and
locked memory every 0.5 seconds. It recorded initialization and post-request
checkpoints, the exact escaped server command, relevant environment, request
hash, response, output hash, logs, and summaries. The plain harness SHA-256 is
`64d7e49d020b8323bff5fdea45f111f21cd2b7970da9f691faae63b41f1bcc3d`.
The profiler harness SHA-256 is
`fe48787f6ab25c8d053111133808e3c49d2f557eeae99b8ae93c59e80c0f6d07`.

The short `/completion` request generated 128 greedy tokens. The coding
`/v1/chat/completions` request used the orbital-sandbox HTML prompt and
generated 5,000 tokens with seed 1234 and the model's default sampling
temperature. The long `/completion` request contained 138,000 explicit token
IDs repeating `[1000, 1001, 1002, 1003]`, followed by 64 greedy tokens. Prompt
cache was disabled in every request. Their request SHA-256 values were,
respectively, `408fc8936598794b092fef75283311a3f0ff9df35318416ebe4f4ffa3cbe011a`,
`6edbe87e0896f13681887883efba7b18a669e29802a161fcfa6a737bd5695995`,
and `b60a55d02f8cf01f53b73ab80bf46b22d8d5886ff970c09a4f660475a0e0f546`.

### Throughput, transition, and speculative results

| Workload | Metric | Baseline | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| Short, 26 prompt + 128 output | Prefill | 197.65 t/s | 149.25 t/s | cold 26-token sample; not used for acceptance |
| Short | Decode | 84.10 t/s | 82.78 t/s | -1.57% |
| Short | Wall | 1.655 s | 1.722 s | +4.04% |
| Coding, 149 prompt + 5,000 output | Prefill | 633.10 t/s | 571.95 t/s | cold 149-token sample; not used for acceptance |
| Coding | Decode | 52.098 t/s | 51.146 t/s | -1.83% |
| Coding | Wall | 96.211 s | 98.023 s | +1.88% |
| Long, 138,000 prompt + 64 output | Prefill | 742.519 t/s | 741.661 t/s | -0.12% |
| Long | Decode | 30.67 t/s | 33.02 t/s | 64-token tail; too short for a stable claim |
| Long | Wall | 187.959 s | 188.030 s | +0.04% |
| Nsight coding pair | Decode | 50.498 t/s | 50.275 t/s | -0.44% |
| Nsight coding pair | Wall | 99.259 s | 99.794 s | +0.54% |

| Candidate workload | Target reserves | Draft reserves | Target time | Draft time | Total transition time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 2 (1 grow, 1 shrink) | 2 (1 grow, 1 shrink) | 35.302 ms | 9.429 ms | 44.731 ms |
| Coding 5K | 2 (1 grow, 1 shrink) | 2 (1 grow, 1 shrink) | 32.482 ms | 9.313 ms | 41.795 ms |
| Long 138K | 2 (1 grow, 1 shrink) | 2 (1 grow, 1 shrink) | 36.074 ms | 9.929 ms | 46.003 ms |
| Nsight coding | 2 (1 grow, 1 shrink) | 2 (1 grow, 1 shrink) | 31.681 ms | 10.137 ms | 41.818 ms |

The coding baseline and candidate both generated 2,193 draft tokens, accepted
1,851, and performed 88 replay cycles carrying 430 actual replay-batch tokens.
The long pair both generated and accepted 54 drafts with zero replay. Every
target checkpoint counter, serialized checkpoint payload counter, capture time,
and restore time was zero. Thus the candidate neither changed speculation nor
hid allocation work in the compatibility checkpoint path.

### Device and host memory

| Workload and point | Baseline VRAM | Candidate VRAM | Saving |
| --- | ---: | ---: | ---: |
| Initialized | 15,768 MiB | 14,660 MiB | 1,108 MiB |
| Coding 5K complete/steady | 15,800 MiB | 14,692 MiB | 1,108 MiB |
| Coding 5K sampled peak | 15,800 MiB | 14,874 MiB | 926 MiB |
| Long 138K complete/steady | 15,800 MiB | 14,692 MiB | 1,108 MiB |
| Long 138K sampled peak | 15,800 MiB | 14,898 MiB | 902 MiB |

At baseline startup, the target scheduler reported 1,054.62 MiB CUDA plus
157.03 MiB CUDA-host compute and the draft scheduler reported 892.05 MiB plus
39.46 MiB. The candidate's stable shared generation backing was 840.82 MiB
CUDA plus 2.41 MiB CUDA-host. During prompt processing the shared backing grew
to the active target maximum, 1,054.62/157.03 MiB, rather than the sum of both
scheduler maxima.

The 5K initialization/completion RSS changed from 7,480,900/7,929,392 KiB to
7,237,808/7,686,608 KiB, about 237 MiB lower. The 138K values changed from
7,480,912/8,551,064 KiB to 7,235,776/8,306,416 KiB. Process `VmHWM` is
dominated by model load and was 43.6-45.4 MiB lower in the candidate. `VmLck`
was zero on both sides, but Linux `VmLck` is not a measurement of CUDA pinned
allocations or CUDA mapping resources; the logged CUDA-host buffer sizes are
the relevant allocator accounting here.

The shutdown breakdown for the 5K baseline was approximately
`CUDA: 15880 = 46 + (14617 = 13114 model + 448 context + 1054 compute) + 1216`
and `Host: 5450 = 644 + 4649 + 157`. The candidate was
`CUDA: 15880 = 1154 + (14403 = 13114 + 448 + 840) + 322` and
`Host: 5296 = 644 + 4649 + 2`. The CUDA driver reports free/unaccounted terms
differently after replacing allocations, so the direct process samples and
named compute components are the acceptance measurements.

### Nsight transfer accounting

The clean 5K pair used:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=TRACE SERVER [matched arguments]
nsys stats --force-export=true --report cuda_gpu_mem_size_sum --format csv TRACE.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum --format csv TRACE.nsys-rep
```

| Operation | Baseline total/count | Candidate total/count | Difference |
| --- | ---: | ---: | ---: |
| H2D | 378,465.886 MB / 200,628 | 378,465.886 MB / 200,628 | exactly zero |
| D2H | 7,145.663 MB / 143,212 | 7,145.663 MB / 143,212 | exactly zero |
| Memset | 969.646 MB / 3,159 | 969.646 MB / 3,159 | exactly zero |

H2D medians/averages were 1.114/1.886 MB and D2H medians/averages were
0.004/0.050 MB on both sides. Baseline/candidate H2D time was 8.520/8.164 s and
D2H time was 0.274/0.263 s; timing variation under profiling is not treated as
a benefit. Identical byte totals and call counts are the authoritative result:
replacing the compute backing adds no host/device traffic. Serialized payload
counters are not PCIe measurements.

### Correctness, compatibility, and regression coverage

All fixed-output pairs were byte-identical:

- short output SHA-256:
  `e87b691143030a0ac2025ab750ed5aeebbca7296734de5c24b884cbdda7f223b`;
- coding 5K output SHA-256:
  `2f994f600563b04a0a8ce172b59b8f04515edae27664f0b15cab003ad257f44e`;
- long 138K output SHA-256:
  `02ba58ab1c1df7dec6df58c435c3fa9eeb5c42fb04c549c82552639aaebf54dc`.

The candidate binary with the new option omitted reproduced the baseline's
15,768/15,790 MiB initialization/completion values, the same short output hash,
and the same 103/88 draft/accept and 4/26 replay statistics. This confirms the
default-off compatibility path retains full reservations.

Two identical short requests were also issued to one live candidate process.
Both output hashes matched the baseline. Each request independently performed
one target/draft grow and one target/draft shrink; the second transition took
33.396/9.427 ms. This validates prompt regrowth after generation and safe reuse
of the live context. The second run's acceptance/replay path differed slightly
because the server restored/invalidated its retained prompt state, but the
fixed output did not.

The final CUDA binary passed `test-alloc`. A focused CTest selection passed 9/9:
argument parsing, allocator sharing, recurrent rollback, prompt checkpoint,
loop-guard checkpoint/static and runtime, sampler rollback, model fixture
generation, and KVarN rollback static coverage. The broad suite, excluding the
three independently classified environment/upstream failures below, passed
93/93:

- `test-upstream-merge-keepers-static` fails identically on the exact detached
  base;
- `test-tokenizers-ggml-vocabs` sees a Git LFS pointer instead of its external
  tokenizer fixture;
- CUDA `test-backend-ops` aborts at the existing `fattn.cu:380`; this patch
  changes no CUDA kernel or attention dispatch code.

### Disposition and next work

Retain the option. It removes 1,108 MiB of steady process VRAM and 902-926 MiB
of peak VRAM in the intended MTP-6 CPU-Q8-KV configuration, with identical
outputs, replay work, and PCIe byte totals. The measured trade is a 42-46 ms
phase-transition cost and a 0.44-1.83% 5K decode regression. The allocator is
backend-type based and has no Qwen or CUDA architecture gate; integrated MTP is
the current consumer because it has two sequential schedulers to share.

This result changes the working theory: roughly 1.1 GiB of the CPU-KV serving
floor was inactive prompt-high-water/coexistence allocation, not irreducible
model or KV residency. The next contained target is compact causal-mask
metadata. Direct-Q8 prompt MMA and bounded fixed-window GPU streaming remain
larger follow-ons. The complete commands, chronology, artifact map, and
integration procedure are in
[`phase-aware-workspace-reproduction.md`](phase-aware-workspace-reproduction.md).

## Experiment 017: placement-matched MTP exactness and draft-ubatch gate

Status: retained. MTP now preserves clean Bee's inherited physical ubatch;
non-MTP model-backed speculative modes retain the independent control.

### Why the oracle and gate changed

The original correctness request compared MTP with one-token target decoding.
That is not the upstream contract: clean BeeLlama and clean llama.cpp produce
the same corresponding MTP streams, but MTP verification can differ from
target-only decoding because it evaluates multiple target rows together. The
correct oracle is therefore clean Bee MTP at the same MTP depth, sampling,
cache placement, and execution geometry.

Clean Bee commit `ba27edad2a84ff045a556df06661e821285c2fab` does not expose a
draft-specific ubatch. Integrated MTP inherits target `n_ubatch`. Experiment
007 added a common draft-context override after that baseline. Its 64-token
MTP-5 screen was insufficient: output and aggregate acceptance counts agreed,
but it ended before a changed recurrent state affected a sampled token.

A 1,000-token CPU-Q8 MTP-2 diagnostic used the canonical 149-token coding
prompt, greedy sampling, seed 1234, target ubatch 512, and candidate draft
ubatches 512, 128, and 32. The inherited/equal-512 case matched clean Bee. Both
128 cases (phase awareness off and on) and the 32 case first differed at
generated token 100 and shared the same divergent stream. Phase awareness was
therefore neither the cause nor a correction for changed physical geometry.

`LLAMA_TRACE=1` reran 130 output tokens at 512 and 128. The first acceptance
cycle regrouping occurred at output token 60: the 512 case accepted one of one
draft rows and advanced 60 to 62, while 128 accepted two of two and advanced 60
to 63. The preceding 149-token prompt had reached the draft recurrent context
in one physical microbatch at 512 versus `128 + 21` at 128. Later acceptance
cycles regrouped again around tokens 91-103, and the visible stream changed at
100. This is a recurrent/model floating-point and verification-batch-geometry
effect, not sampler corruption or a phase allocator bug.

### Retained implementation

The complete speculative configuration validator now receives target
`n_ubatch`. If `draft-mtp` is active and an explicit nonzero draft ubatch differs
from target ubatch, parsing fails with:

```text
draft-mtp requires spec-draft-ubatch-size (128) to match the target ubatch (512) for output-stable recurrent prompt synchronization; omit the draft override or use --phase-aware-workspace to reduce decode workspace
```

Validation remains order-independent and is called by both common argument
parsing and server model loading. The MTP restriction applies through CLI,
`LLAMA_ARG_SPEC_DRAFT_UBATCH`, and rendered INI/preset arguments. Omitted/zero
and explicit-equal values pass. DFlash and other non-MTP model-backed modes
remain able to use an independent draft ubatch. No model, architecture, prompt,
token position, or MTP-depth check was added.

The exactness runner now requires `effective_draft_ubatch` in every maintained
MTP manifest. It independently resolves target and draft ubatch from merged
CLI/environment settings and rejects a manifest whose declared identity does
not match its actual command. This prevents a mismatched-geometry run from
being reported as contract-exact merely because a descriptive field was
omitted.

### Source, build, hardware, and inputs

- Clean Bee source: `ba27edad2a84ff045a556df06661e821285c2fab`;
  server SHA-256
  `9af36fc55d7b2e6bee806485d698ccbddee7970cff4ca106d1df0dda6ef4fcab`.
- Candidate branch/base: `exp/mtp-bit-exact` at
  `7febdc06a795002bf9e82f4b84026fd3740a3a12`, with uncommitted source/test diff
  SHA-256
  `a30fc3b48a158317742f7b3ce4923c3baf72b3041423215a2679ae6a4bec6666`
  at measurement time. The retained validation was later committed as
  `3693c6119` without changing its runtime policy.
- Candidate server SHA-256:
  `da7374a9c26c4ddf89136c67c5927fc6f27d09dcee36fe365c04c59eb07f6be3`.
- Release CUDA build, native CPU, CUDA FlashAttention, effective architecture
  `120a`, default quant matrix, shared ccache.
- NVIDIA GeForce RTX 5070 Ti, 15,880 MiB usable process capacity, compute
  capability 12.0, driver 610.57.04; Intel Core Ultra 9 285K; CUDA 13.3.
- Model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`,
  14,437,471,712 bytes, SHA-256
  `ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.
- Request: originally
  `/tmp/mtp-target-1k-audit-20260819/request-greedy.json`, now preserved
  byte-for-byte as
  `scripts/mtp-exactness-manifests/requests/qwen38-orbital-greedy-1k.json`
  (SHA-256
  `c91ffa3ff2ca9958a7835004e3d5a6ffa7a242a9f74bfce3e7935d52161dce4c`);
  149 prompt tokens, 1,000 output tokens, temperature zero, seed 1234, prompt
  cache off.
- Target/draft K/V: Q8_0/Q8_0; context 4,096; batch/ubatch 1,024/512; MTP-2,
  `p_min=0.85`, full recurrent planes; three strict decode workers on CPUs 0-2.

The maintained exact command is:

```bash
cd /home/gencoolpc/beellama-mtp-exact
python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-cpu-q8-mtp2-phase-1k.json
```

The manifest contains the exact sterile environment and every server argument.
It launches a fresh live clean-Bee golden followed by inherited phase-off,
inherited phase-on, and explicit-equal phase-on candidates. `/slots` polling
every two seconds provides request progress and the harness samples process
VRAM throughout.

### Post-fix exactness and resource result

| Case | Decode | Draft accepted/generated | Peak VRAM | Exact tokens/content |
|---|---:|---:|---:|---|
| Clean Bee CPU MTP-2 | 17.096 t/s | 354/382 | 13,692 MiB | golden |
| Candidate, inherited, phase off | 58.883 t/s | 354/382 | 14,104 MiB | yes |
| Candidate, inherited, phase on | 58.610 t/s | 354/382 | 13,870 MiB | yes |
| Candidate, explicit 512, phase on | 58.499 t/s | 354/382 | 13,870 MiB | yes |

All four streams contain 1,000 tokens with token SHA-256
`b1e3f667bf3a2269f9ba2b0d41ca6f1229b1780250b173ba17db3fa0a9abad9a`
and generated-content SHA-256
`fae6d743dc309784a909e015cc46450b7bfebfafb5e2a18c92c999206d019057`.
Prompt-token, request-semantic, required-identity, token, and content comparisons
all passed. Both phase-aware cases performed exactly one target/draft grow and
one shrink, used zero speculative checkpoint captures/restores and zero replay,
and saved 234 MiB of sampled peak VRAM versus candidate phase-off. The small
decode difference is a single-run observation, not a performance claim.

Artifacts:

- `/tmp/mtp-cpu-q8-mtp2-postfix-v2-1k-20260819`
- manifest SHA-256:
  `015f134500063b08701b66a234aa4f417f56e8df9c6aa31019c5153ae60efee9`
- provenance SHA-256:
  `578c8aa0ac39d16b2475a5a9c2abeda3cea60c1acaf4986960f0df85a921d6cc`
- comparisons SHA-256:
  `d6cb6d06b5405538c944bd6bcf58ef1803aec5d02228866fab4e333ee80bb67c`
- pre-gate mismatch artifact:
  `/tmp/mtp-cpu-q8-mtp2-draft-ubatch-1k-20260819`
- trace artifact:
  `/tmp/qwen38-cpu-mtp2-ubatch-trace-130-20260819`

### Tests and disposition

`build-mtp-exact/bin/test-arg-parser` passes and covers inherited, equal,
mismatched CLI, mismatched environment, valid non-MTP, and valid/invalid INI
configurations. `scripts/test-mtp-exactness.py` passes 20/20 tests, including
CLI and environment disagreement between declared and actual ubatch geometry;
all four maintained manifests pass structural validation. The CUDA server and
parser test rebuilt successfully with ccache, and `git diff --check` passes.

Retain the fail-closed MTP validation and the strengthened manifest contract.
Withdraw the MTP-specific draft-ubatch recommendation from current docs while
preserving Experiment 007 as historical evidence and retaining the option for
other speculative modes. Use inherited/equal physical ubatch plus phase-aware
workspace for MTP.

### Equal-ubatch recurrent-plane matrix

The next clean-process run exercised the cap at MTP depths 3, 5, and 8 with
two, three, four, and explicit-full plane values where valid. All contexts used
target/effective-draft ubatch 512, GPU-resident Q8_0/Q8_0 target and draft KV,
phase awareness off, the same 1K greedy request, and a fresh live clean-Bee
golden for each MTP depth. Two capped planes have a zero-token direct rejection
horizon, so every rejection requires deterministic original-full-shape sparse
GPU replay. Three and four planes provide one- and two-token direct horizons.

Exact command:

```bash
cd /home/gencoolpc/beellama-mtp-exact
python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-gpu-q8-plane-cap-depths-1k.json
```

| MTP depth | Planes | Direct horizon | Decode t/s | Replay cycles/tokens | Peak VRAM | Exact to clean Bee |
|---:|---:|---:|---:|---:|---:|---|
| 3 | clean full | 3 | 65.400 | 0/0 | 14,406 MiB | golden |
| 3 | 2 | 0 | 62.457 | 27/95 | 14,106 MiB | yes |
| 3 | 3 | 1 | 63.818 | 11/42 | 14,256 MiB | yes |
| 3 | explicit 4 | 3 | 65.474 | 0/0 | 14,406 MiB | yes |
| 5 | clean full | 5 | 65.927 | 0/0 | 14,704 MiB | golden |
| 5 | 2 | 0 | 63.333 | 27/117 | 14,106 MiB | yes |
| 5 | 3 | 1 | 64.190 | 16/80 | 14,256 MiB | yes |
| 5 | 4 | 2 | 64.561 | 10/59 | 14,406 MiB | yes |
| 5 | explicit 6 | 5 | 65.924 | 0/0 | 14,704 MiB | yes |
| 8 | clean full | 8 | 63.927 | 0/0 | 15,154 MiB | golden |
| 8 | 2 | 0 | 62.071 | 23/127 | 14,106 MiB | yes |
| 8 | 3 | 1 | 62.883 | 12/82 | 14,256 MiB | yes |
| 8 | 4 | 2 | 63.042 | 8/66 | 14,406 MiB | yes |
| 8 | explicit 9 | 8 | 63.858 | 0/0 | 15,154 MiB | yes |

Every candidate matched its golden token IDs and generated content bytes for
all 1,000 tokens. Draft generated/accepted counts also matched within each
depth: 457/417 at MTP-3, 498/439 at MTP-5, and 479/419 at MTP-8. Every capped
case recorded zero target checkpoint captures/restores and zero serialized
checkpoint payload; all recovery work used sparse GPU replay. Explicit-full
values matched clean Bee performance, allocation, output, and zero-replay
behavior.

Peak allocation increased by approximately 150 MiB per plane and never hid a
full-depth snapshot. Relative to full allocation, two planes saved 300 MiB at
MTP-3, 598 MiB at MTP-5, and 1,048 MiB at MTP-8. Four planes saved 748 MiB at
MTP-8. The corresponding MTP-8 single-run decode changes were -2.90% for two
planes, -1.63% for three, and -1.38% for four. These throughput values are
recorded as matched single-run characterization, not confidence intervals.

Artifacts:

- `/tmp/mtp-gpu-q8-plane-cap-depths-1k-20260819`
- manifest SHA-256:
  `df6847f50bdbbe779b2a94f43166de5a2bcc5500b8d3c96107402df704b43aef`
- provenance SHA-256:
  `740cdc2f75408430fc6804930dba5009f65d771d894885273905956f86a3c684`
- comparisons SHA-256:
  `7119dbd4283b6013e0c1d3a0e17b6feb9f5215e07ff1b2aac00d318b96ff4965`

The equal-ubatch 1K cap matrix is accepted. Experiment 018 adds a supported
equal-ubatch MTP-6 CPU/GPU full-versus-three-plane 5K result. The historical
MTP-8 5K, 140K phase-aware, and Nsight matrices remain to be rerun before their
128-draft-ubatch numbers can be replaced.

## Experiment 018: canonical Q8 KV stores across CPU/GPU residency

Status: retained. Host-resident standard quantized KV now preserves the model
layer accelerator's conversion semantics, and exact-tail planning separates
storage ownership from attention execution placement.

### Failure and controls

The first placement-crossing exactness matrix used clean Bee GPU Q8 KV as the
golden and ran the candidate with GPU and pinned-CPU Q8 KV. Target-only output
first differed in the CPU case at generated token 5. MTP inherited the same
body-cache difference; enabling the exact-tail option did not correct it.

The corresponding F16 target-only control matched across CPU and GPU
residency. Raw state comparison then found sparse byte differences in Q8 rows
written by the CPU and CUDA converters. This isolated the issue to persistent
quantized-KV construction rather than the sampler, MTP scheduler, phase-aware
allocator, recurrent rollback, or prompt cache. CPU and CUDA quantization are
both valid approximate converters, but placement-dependent converter bytes are
not an acceptable basis for an output-invariance contract.

Two low-level approaches were rejected:

- changing CPU or CUDA rounding in an attempt to make independently maintained
  converters converge; this was backend-specific, fragile across types, and
  did not express which implementation owned the numerical contract;
- a device-side indexed `SET_ROWS` into a full persistent-sized stage; partial
  microbatches left unwritten rows and coupled scratch size to the cache rather
  than the current write batch.

Both experiments were fully reverted before the retained implementation.

### Retained implementation

For a standard quantized KV tensor that is persistent in host memory while its
model layer executes on an accelerator, `llama_kv_cache` now allocates a small
per-layer quantized store stage on that layer's device. The graph performs:

1. the normal accelerator `F32 -> cache type` conversion into the stage;
2. a transfer of only the newly written quantized rows to the host backend;
3. a same-type `SET_ROWS` scatter into persistent host KV.

CPU same-type `SET_ROWS` is a byte-preserving row copy. It does not dequantize
and requantize. Construction capability-probes both the accelerator conversion
and store operations for each layer/type and fails closed for an accelerator
layer if canonical conversion cannot be represented. There is no model-family,
prompt, token-position, or Qwen enum check.

The stage is bounded by the maximum of physical ubatch and the largest fused-op
probe shape times sequence count. The Qwen Gated Delta Net resolver builds a
synthetic 16-token graph even when a one-token benchmark requests ubatch one;
using the shared 16-token probe contract prevents a stage-shape failure without
hard-coding that behavior inside the cache implementation. At the production
ubatch of 512, the Qwen3.8 test case allocates stages for 16 host-resident
attention layers, 512 rows each, totaling 17.00 MiB of device memory.

Exact-tail routing required a separate correction. The original route planner
treated the persistent buffer owner as the execution device. With pinned host
KV and operation offload enabled, that incorrectly planned the final tail
attention for CPU even though the scheduler runs attention on CUDA. The route
descriptor now carries storage buffer type and execution backend independently:
storage capability validates writes, execution capability validates attention
and math, and a native final-tail op is assigned explicitly to the matching
scheduler backend. `--no-op-offload` still selects and successfully runs the
CPU route.

### Source, build, hardware, and environment

- Candidate branch/base: `exp/mtp-bit-exact` at
  `7febdc06a795002bf9e82f4b84026fd3740a3a12` with an uncommitted retained
  source/test patch. The runtime-bearing diff SHA-256 at the final server
  rebuild was
  `28c5eedaf35c2c776b574ffeb8a35f76346e11e71b430478a15514a026f98f32`.
  After replacing a stale syntactic upstream-merge guard with semantic
  storage/execution assertions, the final audited
  `git diff -- common ggml src tests tools/server` SHA-256 is
  `12b51655d51ab26c16ac527eeef1cdf3cc4d884d8b1bfbe06e208801406bb49c`;
  that test-only change did not require a server rebuild. The retained source
  was later committed as `3693c6119` and `d4b50c5cc`, and the harness and
  maintained manifests as `85eef4a89`.
- Candidate server SHA-256:
  `da7374a9c26c4ddf89136c67c5927fc6f27d09dcee36fe365c04c59eb07f6be3`.
- Clean Bee: `ba27edad2a84ff045a556df06661e821285c2fab`; server SHA-256
  `9af36fc55d7b2e6bee806485d698ccbddee7970cff4ca106d1df0dda6ef4fcab`.
- Clean llama.cpp control: `af5172627d3513a7efed526b206dca9cd6536452`;
  server SHA-256
  `710538e3219625f43531037128e0357621b5ba6c6898e5bbd4ec5682ed094b27`.
  Its 1K target-only, MTP-2, and MTP-6 token hashes equal the clean Bee hashes
  used below.
- Release CUDA build, native CPU, CUDA FlashAttention, effective architecture
  `120a`, default quant matrix, CUDA 13.3, shared ccache.
- NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, driver 610.57.04; Intel
  Core Ultra 9 285K, 24 cores without SMT.
- Qwen3.8 27B model path and SHA-256 are the same as Experiment 017.

The runner does not inherit the launching shell's `LLAMA_*`, `GGML_*`, CUDA
visibility, tuning, or preset variables. It creates a sterile environment and
adds only the manifest's explicit `CUDA_PATH=/opt/cuda`; every behavior-changing
server choice is a command-line argument. This prevents an ambient environment
flag from making a CPU/GPU comparison asymmetric.

### Exact commands and progress

Each command launches fresh servers sequentially, polls health and `/slots`,
samples process VRAM, and terminates a case before starting the next:

```bash
cd /home/gencoolpc/beellama-mtp-exact

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-tail-cross-residency-q8-128.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-target-cross-residency-q8-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp-cross-residency-q8-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-gpu-q8-full-depths-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-gpu-q8-plane-cap-depths-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-cross-residency-q8-5k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-lifecycle-q8.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-router-reload-q8.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp8-rejection-seeds-q8.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp8-high-confidence-q8.json
```

All final target/draft caches are homogeneous Q8_0/Q8_0, target and effective
draft ubatch are 512, batch is 1,024, FlashAttention is on, prompt cache is
disabled, seed is 1234, strict decode affinity is CPUs 0-2, and batch affinity
is CPUs 0-23. The 1K request is greedy. The 5K orbital-sandbox request uses
temperature 0.8 and MTP `p_min=0.85` in an 8,192-token context.

### Final target-only and exact-tail controls

| 1K target-only case | Prefill | Decode | Peak VRAM | Exact to clean GPU |
|---|---:|---:|---:|---|
| Clean Bee, GPU Q8 | 1,015.914 t/s | 50.055 t/s | 13,560 MiB | golden |
| Candidate, GPU Q8 | 977.607 t/s | 50.021 t/s | 13,560 MiB | yes |
| Candidate, pinned CPU Q8 | 879.466 t/s | 47.532 t/s | 13,442 MiB | yes |

All three target-only cases produced token SHA-256
`cd8d20d1270ee556a5035994abe08e55fab1a38600a89208becbc9e348e8d283`.
The 128-token exact-tail control also matched token-for-token and byte-for-byte:

| Exact-tail case | Prefill | Decode | Peak VRAM | Exact |
|---|---:|---:|---:|---|
| Clean Bee, GPU Q8 | 975.201 t/s | 47.773 t/s | 13,584 MiB | golden |
| Candidate, GPU Q8 | 953.027 t/s | 47.649 t/s | 13,584 MiB | yes |
| Candidate, pinned CPU Q8 | 820.516 t/s | 45.632 t/s | 13,464 MiB | yes |

The exact-tail token SHA-256 is
`31e64405e471ffe000382e76ddc4a6ba75b82ed19f0f52c057b9967764f9d5b0`.
Verbose logs identify CUDA0 as the CPU-resident case's native-tail execution
device; a separate `--no-op-offload` smoke test identifies CPU and completes.

### Final 1K MTP CPU/GPU matrix

| Case | Prefill | Decode | Accepted/generated | Replay cycles/tokens | Peak VRAM | Exact |
|---|---:|---:|---:|---:|---:|---|
| Clean GPU MTP-2 | 848.548 t/s | 61.918 t/s | 365/390 | 0/0 | 14,256 MiB | golden |
| Candidate GPU MTP-2 | 848.273 t/s | 62.047 t/s | 365/390 | 0/0 | 14,256 MiB | yes |
| Candidate CPU MTP-2, phase off | 808.140 t/s | 58.836 t/s | 365/390 | 0/0 | 14,124 MiB | yes |
| Candidate CPU MTP-2, phase on | 771.713 t/s | 58.416 t/s | 365/390 | 0/0 | 13,890 MiB | yes |
| Clean GPU MTP-6 | 835.586 t/s | 65.321 t/s | 428/470 | 0/0 | 14,854 MiB | golden |
| Candidate GPU MTP-6 | 840.549 t/s | 65.335 t/s | 428/470 | 0/0 | 14,854 MiB | yes |
| Candidate CPU MTP-6, phase off | 804.809 t/s | 62.091 t/s | 428/470 | 0/0 | 14,722 MiB | yes |
| Candidate CPU MTP-6, phase on | 751.091 t/s | 61.812 t/s | 428/470 | 0/0 | 14,488 MiB | yes |
| Candidate CPU MTP-6, phase on, 3 planes | 756.602 t/s | 60.790 t/s | 428/470 | 10/56 | 13,890 MiB | yes |

MTP-2 token SHA-256 is
`14e47fb5c35897bfe818339bd163ef1be1f630b558055fe0d323cad922c36217`;
MTP-6 is
`842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00`.
All full-plane rows used zero checkpoint captures/restores and zero replay. The
capped row used selected GPU replay only and still used zero checkpoints.

### Final 5K stochastic MTP-6 CPU/GPU gate

| Case | Prefill | Decode | Accepted/generated | Replay cycles/tokens | Peak VRAM | Exact |
|---|---:|---:|---:|---:|---:|---|
| Clean GPU, full planes | 847.371 t/s | 63.281 t/s | 2,077/2,435 | 0/0 | 15,006 MiB | golden |
| Candidate GPU, full planes | 841.466 t/s | 63.223 t/s | 2,077/2,435 | 0/0 | 15,006 MiB | yes |
| Candidate CPU, phase-aware, full planes | 751.182 t/s | 57.200 t/s | 2,077/2,435 | 0/0 | 14,514 MiB | yes |
| Candidate CPU, phase-aware, 3 planes | 750.641 t/s | 55.363 t/s | 2,077/2,435 | 90/443 | 13,926 MiB | yes |

All four emitted 5,000 tokens with token SHA-256
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`
and identical generated bytes. Acceptance was 85.298%. Every case recorded
zero checkpoint captures/restores, zero checkpoint payload, and zero
capture/restore wall time. CPU full saved 492 MiB of sampled peak VRAM versus
GPU full; CPU with three planes saved 1,080 MiB. The capped CPU row was 3.21%
slower in decode than CPU full in these single runs and did 443 actual replay
batch tokens. The matched final Nsight matrix below measures the transfer cost;
serialized checkpoint payload remains distinct from profiler totals.

### Same-process prompt-cache, sleep/wake, and regrowth gate

The lifecycle manifest keeps each case's server alive for four completions. It
starts from the 149-token coding prompt, generates 96 tokens, waits until
`GET /props` reports `is_sleeping=true`, wakes with a 255-token continuation,
generates 96 tokens, shrinks to a 192-token cached branch for 32 tokens, sleeps
again, and wakes/regrows a 365-token branch for 64 tokens. Continuation prompts
are assembled from recorded token IDs and an unspecialized suffix, and the
harness compares every step's prompt IDs, request semantics, output IDs, and
response bytes independently.

| Case | Peak VRAM | Exact to clean GPU at all four completions |
|---|---:|---|
| Clean GPU, phase off | 14,854 MiB | golden |
| Candidate GPU, phase off | 14,854 MiB | yes |
| Candidate GPU, phase on | 14,614 MiB | yes |
| Candidate pinned CPU, phase off | 14,722 MiB | yes |
| Candidate pinned CPU, phase on | 14,488 MiB | yes |
| Candidate pinned CPU, phase on, three planes | 13,896 MiB | yes |

Both sleep transitions were observed in every case, and server logs show model
context creation after each wake. Common per-completion token SHA-256 values
were `83388a4200b4dc6292a1a88f4f1e5cdcb223d853f6c851ee63342670d5bd9447`,
`0c68b2382550f4e53b662682e5f2900cda42cfad24f4abd8b5717b0354b6d0e9`,
`d5fe0a774d226db31b79a29843a2e3e6639d7a078c72b302edcc49a43360f31e`,
and `b23508c341f7afd50bf72c089c2011de52a45e1e0aaa852e69e8fcb5b8738287`.
Full-plane candidates accepted 160/183 draft tokens with zero checkpoints and
zero replay. The capped row accepted the same tokens, performed five selected
GPU replay cycles with 30 replay-batch tokens, and still captured/restored no
host checkpoint.

The explicit router matrix separately replaced the harness-owned preset with a
hashed fixture whose informational tag changed, called `POST /models/reload`,
required the live model to transition from `loaded` to `unloaded`, and sent an
identical request that autoloaded a new child process. The before/after outputs
within each case and both outputs across cases were token- and byte-exact:

| Router reload case | Peak child VRAM | Before/after token SHA-256 | Exact |
|---|---:|---|---|
| Clean GPU, phase off | 14,854 MiB | `83388a4200b4dc6292a1a88f4f1e5cdcb223d853f6c851ee63342670d5bd9447` | golden |
| Candidate GPU, phase on | 14,614 MiB | same | yes |
| Candidate pinned CPU, phase on | 14,488 MiB | same | yes |
| Candidate pinned CPU, phase on, three planes | 13,890 MiB | same | yes |

Router logs identify the unload reason as `source updated or removed` and show
different child PIDs loading the model before and after. The harness now sums
GPU use across the router process tree, so the table reports child allocations.
The capped case repeated 3 replay cycles/20 replay-batch tokens on both sides of
the reload; all candidate checkpoint counters and payloads remained zero.

### Final depth, cap, stochastic-seed, and acceptance-edge gates

The final server binary was also the candidate in two earlier clean-process
depth matrices. Full-plane GPU MTP matched clean Bee at depths 1, 2, 3, 5, 6,
and 8. At depths 3, 5, and 8, all tested 2-, 3-, 4-, and full-plane policies
matched. Capped rows exercised 8-27 selected GPU replay cycles and 42-127
replay-batch tokens; full-plane rows used none.

Two new 256-token stochastic matrices removed the remaining single-prompt,
single-seed, and middle-acceptance bias:

| Prompt / threshold / seed | Placement | Accepted/generated | Replay cycles/tokens | Decode | Peak VRAM | Exact |
|---|---|---:|---:|---:|---:|---|
| number game / `p_min=0` / 7 | clean GPU full | 202/424 | 0/0 | 106.376 t/s | 15,154 MiB | golden |
| same | candidate GPU, 2 planes | 202/424 | 43/387 | 65.117 t/s | 13,860 MiB | yes |
| same | candidate pinned CPU, 2 planes | 202/424 | 43/387 | 64.420 t/s | 13,754 MiB | yes |
| number game / `p_min=0` / 42 | clean GPU full | 178/614 | 0/0 | 73.679 t/s | 15,154 MiB | golden |
| same | candidate GPU, 2 planes | 178/614 | 69/621 | 43.273 t/s | 13,860 MiB | yes |
| same | candidate pinned CPU, 2 planes | 178/614 | 69/621 | 42.635 t/s | 13,754 MiB | yes |
| website / `p_min=0.999` / 2026 | clean GPU full | 31/31 | 0/0 | 49.185 t/s | 15,154 MiB | golden |
| same | candidate GPU full | 31/31 | 0/0 | 48.545 t/s | 14,920 MiB | yes |
| same | candidate pinned CPU full | 31/31 | 0/0 | 46.982 t/s | 14,794 MiB | yes |
| same | candidate pinned CPU, 3 planes | 31/31 | 0/0 | 47.073 t/s | 13,892 MiB | yes |

`p_min=0` forced full-depth MTP-8 drafting and frequent rejection beyond the
two-plane policy's zero-token direct horizon. Both seeds remained exactly
equal to their clean Bee oracle despite 387 and 621 actual replay-batch tokens.
At `p_min=0.999`, every actual draft was accepted and the capped policy did no
replay. Common token hashes were
`b1cdcec6763a8335d171825bc4a33d73feb96956dc4090599ed6cbc28763b028`
(seed 7),
`5a5450968f1862765c77170595dda7632f242fa9747e62f67ea7242c1e6cc4d0`
(seed 42), and
`f8e70e80e321d5d31efe9b47dc69a87f7d120886f67b6d3c19b2ed68b4386a13`
(website seed 2026). Every checkpoint capture/restore count, time, and payload
was zero. These are correctness and single-run characterization results, not
throughput confidence intervals.

### Final Nsight CPU/GPU and recurrent-cap accounting

The maintained profiler command is:

```bash
cd /home/gencoolpc/beellama-mtp-exact

scripts/mtp-nsys-profile.sh gpu-full gpu 0
scripts/mtp-nsys-profile.sh cpu-full cpu 0
scripts/mtp-nsys-profile.sh cpu-planes3 cpu 3
```

It uses a sterile environment, wraps the final candidate server with Nsight
Systems 2026.1.3 (`cuda,nvtx,osrt`, no CPU sampling/context switches), polls
health and `/slots`, samples process VRAM/RSS every five seconds, gracefully
stops the server, waits for report finalization, and emits both
`cuda_gpu_mem_size_sum` and `cuda_gpu_mem_time_sum`. At measurement time the
harness SHA-256 was
`3f53b7698e1d2621c679aa9c12738ab6e5bbc02c7c51a4104c5659047ac74df5`.
All rows used phase-aware workspace, MTP-6, the 5K stochastic request,
Q8_0/Q8_0 target/draft KV, context 8,192, batch/ubatch/effective draft ubatch
1,024/512/512, and otherwise identical arguments.

| Measurement | GPU full | Pinned CPU full | Pinned CPU, 3 planes |
|---|---:|---:|---:|
| Prefill | 751.827 t/s | 719.122 t/s | 726.574 t/s |
| Decode under Nsight | 61.001 t/s | 55.622 t/s | 54.065 t/s |
| Request wall | 82.167 s | 90.103 s | 92.688 s |
| Sampled peak VRAM | 14,778 MiB | 14,532 MiB | 13,944 MiB |
| H2D total/count | 17,485.510 MB / 97,840 | 327,709.736 MB / 185,590 | 337,755.028 MB / 189,820 |
| H2D aggregate time | 0.886 s | 6.549 s | 6.710 s |
| D2H total/count | 5,700.920 MB / 21,699 | 5,915.778 MB / 131,647 | 6,381.261 MB / 134,887 |
| D2H aggregate time | 0.156 s | 0.213 s | 0.223 s |
| Replay cycles/batch tokens | 0/0 | 0/0 | 90/443 |

CPU full saved 246 MiB versus same-phase GPU full under the profiler, while it
added 310,224.226 MB H2D and 214.858 MB D2H. Decode fell 8.82% and wall time
rose 9.66%. This is not a hidden full-depth GPU snapshot and not a correctness
bug: pinned CPU is persistent KV storage, while normal operation offload keeps
attention on CUDA, so the growing host KV participates in accelerator
attention. The bounded 17 MiB canonical store stage accounts for newly written
quantized rows in the D2H direction; the whole-case delta is reported rather
than mislabeled as a stage-only counter.

The three-plane cap saved another 588 MiB, added 10,045.292 MB H2D and 465.483
MB D2H, and reduced decode 2.80% versus CPU full. These deltas accompany exactly
90 selected GPU replay cycles/443 replay-batch tokens. All three traces had
token SHA-256
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`,
content SHA-256
`afd0208aaaf57cd003c1b0a8d8f29a83c73fa8a8264b547fc6cba0093e1cbe5c`,
acceptance 2,077/2,435, and zero checkpoint counts/times/payloads.

| Case | Artifact | Trace / provenance SHA-256 |
|---|---|---|
| GPU full | `/tmp/mtp-exact-nsys-gpu-full-20260819` | `2a89d5977705e5502c8cbb38026ba875f41105a989f0afa8e8fc0ce57fab4ee7` / `ecf675c279b9426c927dc1b296fa6065c5ae3087997e33d2d887cc2091bc065b` |
| pinned CPU full | `/tmp/mtp-exact-nsys-cpu-full-20260819` | `6552e47910792a2c375d8d6f8e7987eb30074b2661793f47ce3c453de624191c` / `98373df9e9ac171015490fa39304909ab37333dc8f921d642858af2883fdd093` |
| pinned CPU, 3 planes | `/tmp/mtp-exact-nsys-cpu-planes3-20260819` | `b8b42f0e05fdc231ebb82bc365546919d40aee130338c243af5cb9179ffa65b0` / `a4152610ed7491ac52d43b206f1abab8ac0cf59a9328ab307859f2a38eb69141` |

### Artifacts, hashes, tests, and disposition

| Matrix | Artifact | Manifest / provenance / comparisons / summary SHA-256 |
|---|---|---|
| exact tail | `/tmp/qwen38-tail-cross-residency-q8-128-v3-20260819` | `e2f1cbe8524bd60541019b99297b711b10ffd51337e083082425b4a3b25b6dad` / `52c523c90ea5e499c9f9bee1ba76431afcc37e50cb6ab7a94326d560303955ca` / `d43fad0f228c61ce85d7d99d23534209a0db06eb811cd2290ecedf5162d287c3` / `f8d37c1f6137aa28137482113f650907a306abcee924013aaa86b7378b02a7a1` |
| target 1K | `/tmp/qwen38-target-cross-residency-q8-1k-final-v2-20260819` | `72ce3de204c9fee4b13c8c9b28d13413b7bc80fc541de9fcee97c56949ae4735` / `8656d1a1539aafbd920894f55854c30619db641631a6f7d950e746885bc4d7d0` / `e8118f4f690ffa519b5396c57fd5c14842a7067c24524830d21e4bb83eaf4b33` / `4d5ddb2f81861005df31b8a77fe3c867131911481e002a02ec51cd83115d974e` |
| MTP 1K | `/tmp/qwen38-mtp-cross-residency-q8-1k-final-v2-20260819` | `fa056f5fea141e627fb0d8101c1a24b204e4cde2ee0ae3e625d78cb6fdddfe2a` / `92b6ee20ed9832d0a44d2de7c08c4a7fe0276fd152c89f483671658232ee8a10` / `eb6d3f275310a5855f12067e4a15931cae6abb00d923468125f9af4695733cab` / `fe4d12e2691962f82cf886846f821942dca9d375be78043142c1d1ed04aebe5f` |
| full-plane depth sweep | `/tmp/mtp-golden-gpu-q8-full-depths-1k-20260819` | `0cd15951f029869a5497e7ebba169259d262757c73eb140fcc61252e77f77700` / `fda16144ae164c8399036b5361d7251f6d04e3a072101034457540023c2190f8` / `bbc72c9aabb1e9808b03f3032017180b3714e7a0275f324ef32b2dda16c086c6` / `f3d51aae99e3b74aa392ffd71ce95c8835b6f46af4ec46c4f4ec8363dcc02951` |
| recurrent-plane depth sweep | `/tmp/mtp-gpu-q8-plane-cap-depths-1k-20260819` | `df6847f50bdbbe779b2a94f43166de5a2bcc5500b8d3c96107402df704b43aef` / `740cdc2f75408430fc6804930dba5009f65d771d894885273905956f86a3c684` / `7119dbd4283b6013e0c1d3a0e17b6feb9f5215e07ff1b2aac00d318b96ff4965` / `2607c124009105180d6d59c547b6760140c2df1db98806aee01352ed0f9f3514` |
| MTP-8 rejection/seeds | `/tmp/qwen38-mtp8-rejection-seeds-q8-20260819` | `0f605d28c782439bfbe16b42e824b9d6769def5ab5c8ce7383f0ceea3f28b148` / `312613aa2803222bbfcc4591845460e249896386c8d0213daf0e7de68ce644ba` / `e2f177efe29a30bb0ee9ec16e204d0aa527816784ab2031c3f47c824cb713bd9` / `03176b06fdd2816833c93286cef679d294539c357575017ad85e42c52fe209dd` |
| MTP-8 high confidence | `/tmp/qwen38-mtp8-high-confidence-q8-20260819` | `7d542a5a514e6d0db24b08ba30c0dcce0ff6833b3f02888dd4d35932f38dba35` / `8283338a3a734770c1e3a3bddf57fab46aaee0faf3347b3e5e9489c12fc9b398` / `15c28dd9db58bec25f1b79e80846c1153dcdfda0bfcc757e050b680844a1d4fd` / `b1c9bb20bc0cb33476e821fad5c2a9d03ae380ec7aacba8b18067f963997610c` |
| MTP-6 5K | `/tmp/qwen38-mtp6-cross-residency-q8-5k-v2-20260819` | `7bb54c0bbabcf16b44be11ae1016f55880b55a4f47d08f5113b160dd38477105` / `63385922b8fb63be331a9e467350bdaa155803fc1ddac07be00dae1edec7a0de` / `5f38d35db1a63b979c355115978d469f1d0dac0f8928e7ea099c4331073595bd` / `23490d7c5a2b69b79bb1386fead97976069b434e420bb7d349f129ae851b83cb` |
| MTP-6 lifecycle | `/tmp/qwen38-mtp6-lifecycle-q8-20260819` | `bd974faf394447b3d71eacf3ac2a3e4597dc9f9f75377b15e21ffff39387cf69` / `fe3a6781a81daf4d3e1b1a4da58921100c1f3eb74a57c83943f17b17fe18dc2d` / `4acf083d85b1ea64c1504ab2c043a5ec5ae481c0389da9cb6d61e4b195c43556` / `77d84efddc79f702978ecdeb4e2bf660941dc0aa970703a700e6b59e89d71723` |
| MTP-6 router reload | `/tmp/qwen38-mtp6-router-reload-q8-20260819` | `94b47d3eac42696fb288124131703036ffad082d879976188e82e9e7ab683b0f` / `12f4f5744521a753633fedd2417e16903184fdf948b059c02dff8ffd875eeb90` / `5d194ac098b7aed6f63dab80d9832a06b58492771df79f4d9ca4b7ea328d2983` / `f6942b8ec35fac297f881eee50557e3970ce84dc79634505bc8484f4b0c0d40e` |

The full hashes and complete chronology are in
[`mtp-output-exactness-reproduction.md`](mtp-output-exactness-reproduction.md).
The first 5K attempt used an obsolete chat-message request shape that the
runner intentionally rejected before generation; it is a setup error, not a
benchmark result.

After the final source freeze, `llama-server`, `llama-bench`,
`test-arg-parser`, `test-backend-ops`, and `test-kv-cache-tail` rebuilt. CPU
same-type/standard `SET_ROWS` passed 721/721 backend cases; CUDA passed 267/267.
The exactness-runner unit suite passed 35/35, including typed lifecycle,
request-to-sampler-identity binding, and
later-step comparison coverage. Focused parser, recurrent
rollback, prompt-checkpoint, sampler/loop-guard rollback, allocator-sharing,
and server regression results are recorded in the reproduction document.

The final focused CTest rerun passed 11/11. A stale upstream-merge guard first
counted copies of the old CPU-owner assignment and failed after the intentional
storage/execution descriptor split. It was revised to assert the actual
invariant—CPU ownership for host storage, storage-backend write probing, and
execution-backend attention probing—and then passed. The unfiltered generic
backend-op CTest was stopped after it expanded into unrelated operations; the
two relevant selections passed 721/721 CPU and 267/267 CUDA cases.

Retain the capability-probed store stage, byte-copy scatter, storage/execution
tail split, and explicit native-tail scheduling. The measured MTP trade is
17.00 MiB of bounded target device staging plus 1.06 MiB for the one-layer MTP
context, and a D2H transfer of each newly written quantized row, in exchange for
placement-independent persistent Q8 bytes. The much larger
310,224.226 MB CPU-versus-GPU H2D delta belongs to CUDA attention over the
growing host-resident cache and cannot be removed by changing the row scatter.
KVarN, lower or
asymmetric standard cache types, multi-GPU/meta placement, Vulkan/HIP, and
non-Qwen model families require their own output and profiler matrices before
equivalent claims are made.

## Experiment 019: independent draft-owned KV residency after exactness merge

Status: retained. The draft context can override inherited target KV residency
through the shared per-owned-layer placement plan. The tested split is exact
against same-geometry MTP after the physical-ubatch and canonical host-store
fixes; it is not compared with target-only decoding.

### Integration, source, hardware, and inputs

The implementation was committed as `7febdc06a` and its user documentation as
`9be4be34b`. Phase-aware workspace was already present in the branch. The MTP
output-exactness work was then merged at
`db2a119a3487f59e585e1bed8bcb155decd41069`, so the validation uses:

- target/effective-draft physical ubatch 512; the obsolete 128 override is not
  present;
- the canonical accelerator quant-store path for host-resident standard KV;
- phase-aware target/draft compute backing and three recurrent planes; and
- a clean same-depth MTP golden artifact with matching prompt, sampler, seed,
  cache formats, and execution geometry.

The measured server was the Release CUDA build at `db2a119a3`, SHA-256
`6895bb103078c43832a1ff9715a449b94f3554ba0a72804c8bee4c7f0d7f461d`,
built with native CPU optimization, CUDA FlashAttention, architecture 120, and
GNU 16.1.1. Hardware was an NVIDIA GeForce RTX 5070 Ti with 15,880 MiB usable
process capacity, compute capability 12.0, driver 610.57.04, and an Intel Core
Ultra 9 285K. The model was
`/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`,
14,437,471,712 bytes, SHA-256
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.

Both candidate configurations used that same binary. The baseline omitted the
draft override, inheriting target host residency. The candidate added
`--kv-gpu-layers-draft 1`; the value is this model's owned-cache count, not a
model-specific policy in the implementation. Target KV remained pinned-host
Q8_0/Q8_0 under `--no-kv-offload --kv-gpu-layers 0`. Draft weights remained
offloaded on both sides, so the comparison changes only draft-owned KV
residency.

### Maintained exactness gates

The exact commands were:

```bash
cd /home/gencoolpc/beellama-kv-offload

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-partial-draft-residency-q8-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-partial-draft-residency-q8-5k.json
```

Each live case used a fresh server process. Native `/slots` polling reported
decoded-token progress every two seconds for 1K and every five seconds for 5K;
the runner sampled process VRAM throughout. The artifact-backed golden is the
merged clean-GPU MTP-6 result at the same target/effective-draft ubatch of 512.

| Request and case | Accepted/generated | Replay cycles/tokens | Peak VRAM | Exact tokens/content |
|---|---:|---:|---:|---|
| 1K clean GPU, full planes | 428/470 | 0/0 | 14,854 MiB | golden |
| 1K inherited CPU draft, three planes | 428/470 | 10/56 | 13,890 MiB | yes |
| 1K explicit CPU draft, three planes | 428/470 | 10/56 | 13,890 MiB | yes |
| 1K GPU draft, three planes | 428/470 | 10/56 | 13,898 MiB | yes |
| 5K clean GPU, full planes | 2,077/2,435 | 0/0 | 15,006 MiB | golden |
| 5K inherited CPU draft, three planes | 2,077/2,435 | 90/443 | 13,926 MiB | yes |
| 5K GPU draft, three planes | 2,077/2,435 | 90/443 | 13,942 MiB | yes |

All 1K cases produced token SHA-256
`842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00`
and content SHA-256
`41dd817295f51516f3750049cfe3ecd2b5de9ae0f4d08df7a58a1408318f3bb2`.
All 5K cases produced token SHA-256
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`
and content SHA-256
`afd0208aaaf57cd003c1b0a8d8f29a83c73fa8a8264b547fc6cba0093e1cbe5c`.
Prompt-token, request-semantic, required-identity, token-ID, and response-byte
comparisons all passed with no first mismatch. Acceptance and replay work were
also unchanged within each request.

At context 4,096, inherited/explicit CPU draft residency logged 8.50 MiB of
draft `CUDA_Host` KV plus a 1.06 MiB bounded CUDA store stage; the override
logged 8.50 MiB of draft CUDA KV and no draft host KV. At context 8,192 those
values were 17.00 MiB host plus the same 1.06 MiB stage versus 17.00 MiB CUDA.
The corresponding sampled process-VRAM increases were 8 and 16 MiB. Moving the
cache to CUDA therefore releases the context-linear pinned-host allocation;
`nvidia-smi` does not measure that host-memory saving.

The exactness pass overlapped a user-owned graphics workload. Its token and
response comparisons and per-process allocation samples remain valid, but its
throughput is intentionally excluded. Matched performance uses a separate idle
GPU run with the original multimodal layout.

### Matched original-layout performance screen

An idle-GPU, one-run screen compared a fresh inherited-host-draft server with a
fresh draft-owned-GPU server. Both used the canonical server command in
[`cpu-kv-offload-current-testing.md`](cpu-kv-offload-current-testing.md) at this
experiment's commit: context 32,768, the original F16 multimodal projector in
host memory, target Q8_0/Q8_0 KV pinned on the host, target KV GPU layers zero,
MTP depth 6, physical target/effective-draft ubatch 512, phase-aware workspace,
three recurrent planes, and alias `qwen38-kv-test`. The baseline omitted the
draft residency override. The candidate changed only one server argument by
adding `--spec-draft-kv-gpu-layers 1`.

The benchmark command below was run once against each fresh server, changing
only the result filename from `cpu-draft.json` to `gpu-draft.json`:

```bash
python3 -c \
  'import numpy as np; np.random.seed(1234); from llama_benchy.__main__ import main; main()' \
  --base-url http://127.0.0.1:8080/v1 \
  --model Qwen/Qwen3.5-27B \
  --served-model-name qwen38-kv-test \
  --book-url https://www.gutenberg.org/files/1661/1661-0.txt \
  --pp 512 --tg 64 --depth 4096 30000 \
  --runs 1 --no-warmup --skip-coherence --no-adapt-prompt \
  --latency-mode none --exact-tg \
  --extra-body temperature=0,seed=1234,cache_prompt=false \
  --emit-progress - \
  --save-result /tmp/draft-kv-residency-perf-matched-20260819/cpu-draft.json \
  --format json
```

This explicit NumPy seed is required by `llama-benchy` 0.4.0 because its stock
CLI does not seed corpus-offset selection. `--no-cache` was intentionally
omitted because that option appends a per-process UUID and invalidates a
cross-process prompt match. Prompt caching was instead disabled by the common
request body and each server used `--cache-ram 0`. The cached 606,662-byte
corpus SHA-256 was
`8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`.
JSONL events exposed request progress, and timestamped `nvidia-smi` samples
recorded process VRAM once per second.

Server timings are authoritative below. The API client's independently timed
values remain in the artifacts.

| Depth | Draft residency | Prompt tokens | Prefill | Decode | Acceptance | Replay | Peak process VRAM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4,096 | inherited host | 4,661 | 1,545.66 t/s | 62.01 t/s | 35/39 | 1 cycle / 4 tokens | 14,216 MiB |
| 4,096 | owned GPU | 4,661 | 1,558.17 t/s | 63.43 t/s | 35/39 | 1 cycle / 4 tokens | 14,282 MiB |
| 30,000 | inherited host | 30,565 | 1,479.82 t/s | 29.29 t/s | 23/25 | 0 / 0 | 14,216 MiB |
| 30,000 | owned GPU | 30,565 | 1,473.09 t/s | 30.00 t/s | 23/25 | 0 / 0 | 14,282 MiB |

The candidate delta was +0.81% prefill and +2.29% decode at 4K, then -0.45%
prefill and +2.42% decode at 30K. Both requests produced the same normalized
stream-token SHA-256,
`1a42b5ce580eb774367e4e4d096ef6ddc7c1695956967b699269ca9c24231621`,
and had identical acceptance and replay work. The prefill movement is within
the ambiguity of a one-run pair. The consistent approximately 2.4% decode
direction is promising but must be repeated in alternating clean pairs before
being claimed as stable.

Peak sampled process VRAM increased by 66 MiB. The exactness allocation logs
established 17.00 MiB of draft pinned-host KV at context 8,192 and a 1.06 MiB
bounded CUDA host-store stage. Scaling the context-linear Q8 draft KV to 32,768
gives 68.00 MiB: the candidate moves that allocation to CUDA while eliminating
the store stage, consistent with the sampled delta after MiB accounting. The
trade therefore releases approximately 68 MiB of pinned system memory for
approximately 66 MiB of process VRAM in this layout.

Matched-performance artifacts and SHA-256 values:

| Artifact | Inherited host draft | Draft-owned GPU |
|---|---|---|
| Result JSON | `ca2a312a88fa03af8716eb099bbeb803c87c886d1d1ca9851209bb8911faef83` | `833f521a3b5e61661d009742bbc9dee934f0b7a8b3d54b489601b014dc811068` |
| Server log | `4e56ea3e9267b1d43d64a47cbdb78097007f3f482babea42be6f2244baf8fd96` | `04282377c73b9600562361cc8856f46b14f6e8f28d025f1ace72989fdf9bd99b` |
| Client/progress log | `e7501a0b56e41c6f044f697adf6012d517ed6950552c5c8bf44a5234c6b7e8ed` | `f3131c66e2f3b8ad6299b0351ff2edab32ed3b5ef1e74b84293cc5c9f035c5dc` |
| Timestamped VRAM log | `e9e6c8fa805cd256d8c561e4bb95b431c5b9be94f720d6916917ee8ad26fe683` | `dcb8f2b4c2c111f44ecf1881c27ac3141b112602aef9e3b23e6c8cd5d8b65874` |

All files are under
`/tmp/draft-kv-residency-perf-matched-20260819`. The earlier mismatched
`--no-cache` attempt is preserved separately under
`/tmp/draft-kv-residency-perf-invalid-no-cache-20260819` and is not evidence
for a performance delta.

### Full 5K stochastic live decode

The short-generation screen was followed by one full live CPU/GPU pair using
the maintained matrix below:

```bash
cd /home/gencoolpc/beellama-kv-offload
python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-draft-residency-live-mmproj-q8-5k.json
```

Both fresh servers retained the original F16 multimodal projector in host
memory and used context 8,192, target and draft Q8_0/Q8_0 caches, target host
KV, MTP depth 6, p-min 0.85, target/effective-draft ubatch 512, phase-aware
workspace, three recurrent planes, three decode threads on CPUs 0-2, and the
same stochastic 5,000-token request at temperature 0.8 and seed 1234. The only
configuration difference was that the candidate added
`--spec-draft-kv-gpu-layers 1`. The runner launched each server in a sterile
environment and reported `/slots` progress plus process VRAM every five
seconds.

The source HEAD during the run was documentation-only commit `61216c968`; the
measured binary remained build `11241` from `db2a119a3`, SHA-256
`6895bb103078c43832a1ff9715a449b94f3554ba0a72804c8bee4c7f0d7f461d`.
The manifest itself was untracked during execution but is committed unchanged
with this evidence; its recorded SHA-256 is
`854aa520122ad8320ba1f0e72a2fbb82d684a3b3cbfff69c1fad637cf5363f79`.

| Draft residency | Prompt | Decode | Total server time | Acceptance | Replay | Peak process VRAM |
|---|---:|---:|---:|---:|---:|---:|
| inherited host | 149 tokens at 758.63 t/s | 5,000 tokens at 54.56 t/s | 91,842.61 ms | 2,077/2,435 | 90 cycles / 443 tokens | 13,926 MiB |
| owned GPU | 149 tokens at 755.18 t/s | 5,000 tokens at 55.16 t/s | 90,839.74 ms | 2,077/2,435 | 90 cycles / 443 tokens | 13,942 MiB |

GPU draft residency improved server-reported live decode by 1.11% and reduced
total server time by 1,002.87 ms. Prefill changed by -0.46%. Both cases
produced token SHA-256
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`
and content SHA-256
`afd0208aaaf57cd003c1b0a8d8f29a83c73fa8a8264b547fc6cba0093e1cbe5c`;
the comparison contract, prompt tokens, request semantics, token IDs, and
response bytes were all exact. Acceptance and replay work were identical.

The CPU-draft allocation contained 17.00 MiB of pinned-host draft KV and a
1.06 MiB CUDA store stage. The GPU-draft allocation contained 17.00 MiB of
device draft KV and no store stage. Sampled process VRAM rose by 16 MiB, which
matches that exchange after MiB accounting. This sustained decode confirms no
performance regression and points in the same direction as the 64-token
screen, but 1.11% remains below the threshold for a stable claim from one
ordered pair.

Artifacts are under
`/tmp/qwen38-mtp6-draft-residency-live-mmproj-q8-5k-20260819`:

| Artifact | SHA-256 |
|---|---|
| Manifest | `854aa520122ad8320ba1f0e72a2fbb82d684a3b3cbfff69c1fad637cf5363f79` |
| Provenance | `aca071d17efb3cc3121c1e1398540dc5bbc93ddaf10378d73cb88e58cb0dfeb0` |
| Comparisons | `e60eaecd1e4a0b1a14a96ae20510f2960087e60551c730516d5a70d2a797357a` |
| Summary | `3c4bb19c920861d9b181a843bacfffb0e08488cb1bbfaa44291106c62c9d4abb` |
| CPU-draft server log | `a8833a278def595eb4be0f078d22301e9e24f6291ded5c58b2bc7f821abf62f4` |
| GPU-draft server log | `cd23ef55b812e0695c1304f6da90cce377061014c3df8f0ff90f64fef025100c` |

Artifacts and hashes:

| Matrix | Artifact | Manifest / provenance / comparisons / summary SHA-256 |
|---|---|---|
| 1K greedy | `/tmp/qwen38-mtp6-partial-draft-residency-q8-1k-20260819` | `4bb855b72343ae25f24c848a419cfe7fc381cb723a6d113cbfe04ded07ddc01f` / `634facaa592cc90b344d328412031861fae5feed1b6e6cbc6255d1611b48f9eb` / `8814390ac38f4afe6fa7694a7303d46968538668e934ae46295042a3a41913a3` / `e2525105d7ad9390d4a87fbcd43bffa2669905fc481c78660b120f58ef5d7219` |
| 5K stochastic | `/tmp/qwen38-mtp6-partial-draft-residency-q8-5k-20260819` | `bd5172c96d4498a4a24ca007634408aaba92e896093e9a503085f8c847af7f42` / `43dd71e11dc162059f2a34725e77014fafdb72fe190b823c0ca9ad1cf99fffe8` / `acf3b3c638a777cd8a8e9020018768b63ef35892444a22de9d4924f27d51a4b3` / `b4b18a6a8163e40c38569624e320c786563d639ec01e211ef963e3f7b205cd08` |

### Regression coverage and disposition

The final merged binary passed 12/12 focused CTest cases covering argument
parsing, batch allocation, recurrent rollback, KV-tail policy, prompt
checkpointing, loop-guard/sampler rollback, and the relevant static guards.
The exactness runner passed 35/35 unit tests. Same-type/standard `SET_ROWS`
passed 721/721 CPU cases and 267/267 CUDA cases. Both maintained manifests pass
JSON validation and `git diff --check` passes.

Retain the independent draft residency option. Its default path remains exact
inheritance, explicit zero is equivalent to inherited full host residency in
the tested layout, and the GPU-draft split is output-exact after the merged MTP
fixes. The empirical claim remains limited to the ownership split above;
arbitrary partial target-layer mixes, lower/asymmetric formats, other models,
multi-GPU, and other backends need separate gates.

## W06: declare perplexity's full-batch output capacity

**Date:** 2026-08-20

**Base:** `c9f727c1e1995c4a871a719ab05b5f2478588efd`

**Candidate:** this isolated migration commit

**Disposition:** retained correctness fix

### Implementation and scope

Phase-aware contexts without an explicit output requirement use the compact
serving maximum of `n_seq_max`. Perplexity instead requests logits for every
scored token in a slice. With one sequence and a 256-token physical batch,
that mismatch can reach the fail-closed `output_reserve()` assertion.

Immediately before context creation, `llama-perplexity` now sets
`params.n_outputs_max = params.n_batch`. The non-phase-aware default already
resolves an unspecified capacity to `n_batch`, so ordinary behavior is
unchanged. A focused source-plumbing test verifies that the declaration occurs
before `common_init_from_params(params)`. No server, live-workspace, telemetry,
causal-mask, host-staging, native-Q8, or argument-surface change is included.

### Build and identities

The candidate used a Release CUDA build with native CPU code, SM120, the
default standard quant matrix, and dedicated KVarN kernels disabled because
this tool contract does not exercise KVarN. The exact configure and focused
build commands were:

```bash
cmake -S . -B build-w06-cuda -G Ninja \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=OFF \
  -DGGML_CUDA_KVARN=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-w06-cuda \
  --target llama-perplexity test-perplexity-plumbing -j 6
```

The compiler was GNU 16.2.1 with CUDA 13.3.73; the CUDA host compiler was GNU
15.3.0. The GPU was an NVIDIA GeForce RTX 5070 Ti, compute capability 12.0,
with driver 610.57.04. Relevant identities were:

- model SHA-256: `ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`;
- corpus SHA-256: `8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`;
- `tools/perplexity/perplexity.cpp` SHA-256:
  `ae9558ebb124ff3a6db8998b57fd53e27eb956c4310674505b7c5e1a0aaa6986`;
- `llama-perplexity` SHA-256:
  `84b3140cbb0b284d297ad0119bc265f59b06e0e23bcc3e690a8ac1a5e2b3f446`;
- `libllama-perplexity-impl.so` SHA-256:
  `5d8428e7b053e8960b819555a28141e6b0b1c79139870a7f62f9fe8c9b4f83d4`.

### Matched PPL and resource validation

Every model process was fresh and wholly enclosed by
`flock /tmp/beellama-single-gpu.lock -c`. Native llama affinity exposed the
three decode CPUs and 24 batch CPUs; no external affinity wrapper was used.
The native `[1]` counter exposed progress. The exact inner command was:

```bash
build-w06-cuda/bin/llama-perplexity \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -f /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt \
  -c 4096 -b 512 -ub 256 --chunks 1 \
  -t 3 -tb 24 --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 \
  -ngl 999 -sm none -mg 0 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --no-kv-cpu-pinned --no-recurrent-state-offload --no-warmup -v \
  [--phase-aware-workspace]
```

A 200 ms sampler recorded `/proc/PID/status` separately from
`nvidia-smi --query-compute-apps=pid,used_memory`. Verbose logs recorded
allocator component sizes.

| Measurement | Default A1 | Phase-aware | Default A2 |
|---|---:|---:|---:|
| cumulative PPL | `1.9295 +/- 0.06731` | `1.9295 +/- 0.06731` | `1.9295 +/- 0.06731` |
| scoring-pass time | 23.40 s | 23.53 s | 21.26 s |
| sampled process VRAM peak | 13,682 MiB | 13,682 MiB | 13,682 MiB |
| sampled `VmHWM` | 14,594,868 KiB | 14,593,824 KiB | 14,593,752 KiB |
| sampled `RssAnon` peak | 343,536 KiB | 346,664 KiB | 345,520 KiB |

All three contexts reported `n_outputs_max = 512`, then reserved scoring graphs
with 256 outputs. CUDA compute was 252.50 MiB, CUDA-host compute was 27.78 MiB,
the CUDA-host output buffer was 0.95 MiB, device KV was 136.00 MiB, device
recurrent state was 149.62 MiB, and ordinary CPU-mapped model storage was
644.14 MiB in every run. `VmPin` and `VmLck` remained 0 KiB; those fields do
not account for the 28.73 MiB of CUDA-host buffers above. Perplexity performs
scoring passes rather than generated-token decode, so a decode throughput
measurement is not applicable.

The phase-aware process logged creation of phase-aware backing and completed
without weakening the allocator assertion. Identical A/B/A cumulative PPL
shows no numerical increase, while the repeated default rows demonstrate that
ordinary behavior remains stable. The focused `test-perplexity-plumbing` CTest
passed 1/1.
