# CPU KV-offload experiments

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

Status: retained candidate.

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
runs. Recommend 128 as the balanced setting for this hardware and workload;
32 remains an explicit capacity-first option rather than a default.
