# CPU KV-offload experiments

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
