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
