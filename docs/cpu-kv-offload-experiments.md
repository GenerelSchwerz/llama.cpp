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
