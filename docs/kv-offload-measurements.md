# Measurement record: decode with a host-resident KV cache

What was measured, on what, and what it rules in or out. This is the evidence
base for [`kv-offload-roadmap.md`](kv-offload-roadmap.md); the roadmap states
conclusions, this file states numbers.

Reproduce any of it with the scripts in [`repro/`](repro/) and the patches in
[`probes/`](probes/).

## Protocol

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070, 12,282 MiB nominal / 11,902 MiB usable, sm_89 |
| Driver / CUDA | 610.57.04 / 13.3 (`nvcc` V13.3.73) |
| CPU / RAM | Intel Core i5-13400F, 31 GiB, one socket, one NUMA node |
| IOMMU | absent (`/sys/kernel/iommu_groups` empty, no kernel command-line option) |
| THP policy | `always` |
| Source | `946c1e5b6`, fresh `build-clean` plus the stated patch |
| Model | `Qwen3.8-27B-UD-IQ2_M.gguf`, 9.60 GiB, 65 layers, 16 full-attention |
| Common flags | `-ngl 99 -sm none -mg 0 -t 3 -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --parallel 1` |
| Host residency | `-nkvo 1 --kv-cpu-pinned --recurrent-state-offload` |

Every process runs under `taskset -c 0,2,4`, one GPU job at a time behind a
lock. `llama-bench` arms use `--no-warmup` and reverse-order A/B/A/B pairs.
Server arms are greedy (`temperature 0`, `top_k 1`, `seed 1234`).

**Concurrent GPU jobs invalidate results.** Two arms that overlapped produced a
CUDA out-of-memory failure and one arm with `σ = 1.72` against a normal `σ ≈
0.06`; both were discarded and re-run. The lock in `repro/common.sh` exists for
this reason.

## 1. Where the decode token goes

`GGML_KV_TRANSPORT_STATS=1` (patches 01 + 02), differencing `tg128` against
`tg64` at fixed depth so prefill and cache population cancel:

| Depth | decode t/s | ms/token | blocking H2D | backend sync | H2D bytes/token |
|---|---:|---:|---:|---:|---:|
| 4,096 | 31.6 | 31.6 | 6.31 ms | 24.89 ms | 151.8 MB |
| 16,384 | 19.69 | 50.78 | 24.05 ms | — | 579.6 MB |
| 32,768 | 13.03 | 76.77 | 47.10 ms | 29.40 ms | 1.150 GB |

Effective bandwidth 24.1 GB/s at both 16,384 and 32,768, which is the practical
PCIe 4.0 x16 rate. **Copy efficiency has no headroom.**

Per decode token at 32,768 the graph carries **33 CUDA splits and 32 blocking
device-to-host copies**, and the same counts hold exactly at 16,384. The D2H
side totals **34.8 KB** per token -- 1,088 bytes per copy, one `q8_0` KV row --
and **0.125 ms**.

> An earlier revision of this record put the D2H side at 2.2 MB per token. That
> was wrong by a factor of about 60 and it mattered: an inflated store-side
> transfer is what made the store look like a plausible serializer.

## 2. The compute floor

Same model, cache type and depth with the KV device-resident (`repro/m4`):

| Depth | host KV | ms/tok | device KV | ms/tok | implied copy | measured H2D |
|---|---:|---:|---:|---:|---:|---:|
| 4,096 | 31.72 | 31.52 | 39.0152 | 25.63 | 5.89 ms | 6.31 ms |
| 16,384 | 19.69 | 50.78 | 36.1072 | 27.70 | 23.08 ms | 24.05 ms |
| 32,768 | 13.03 | 76.77 | 32.5832 | 30.69 | 46.08 ms | 47.10 ms |

The implied copy term, obtained by subtracting the device-resident arm, agrees
with the independently measured blocking H2D to within 4% at every depth. The
token is `copy + compute` in series with **no overlap at all**.

## 3. The overlap ceiling

Patch 04 issues the pinned-host delivery on a dedicated copy stream and returns
without recording an event the compute stream ever waits on. Same bytes, same
order; the kernels simply do not wait. **Output is incorrect by construction.**

| Depth | reps | ordered | unordered | gain | share of `max(copy,compute)` |
|---|---:|---|---|---:|---:|
| 4,096 | 5 | 31.7287, 31.7165 | 38.4860, 38.5025 | **+21.3%** | 98.6% |
| 16,384 | 3 | 19.7025, 19.6820 | 34.8800, 34.8439 | **+77.0%** | 96.5% |
| 32,768 | 3 | 13.0142, 13.0137 | 20.8258, 20.8300 | **+60.0%** | 95.9% |

The copy engine hides 96-99% of the transfer behind kernel execution. At 4,096
and 16,384 compute dominates and the unordered arm approaches the
device-resident number; at 32,768 the copy dominates and the ceiling is the copy
itself, which is why the gain is smaller there than at 16,384.

**The entire host-KV decode penalty is the ordering, not the bandwidth.**

## 4. The store side is not the serializer

Patch 03 retargets the KV store at a scratch device tensor: no device-to-host
stage copy, no CPU `set_rows`, and 32 of the 33 per-token splits collapse.
Attention still reads the body host-to-device. Output is incorrect.

| Depth | reps | store | store removed | gain |
|---|---:|---|---|---:|
| 4,096 | 5 | 31.7528, 31.7305 | 31.7514, 31.7460 | **+0.02%** |
| 32,768 | 3 | 13.0294, 13.0275 | 13.0524, 13.0518 | **+0.18%** |

Two independent routes to the same conclusion: the direct measurement above
(0.125 ms of D2H against a 76.8 ms token is 0.16%) and this structural one
(0.18%). Removing the entire store side buys nothing, because the token is
already bound on the read and the GPU has no independent work during the store
windows.

## 5. Attempts that were built and rejected

| Attempt | Result | Why |
|---|---|---|
| Async pinned-host H2D on the compute stream | +0.13% at 4,096, +0.06% at 32,768 | Route engages -- 2,520 deliveries converted, one fewer sync per conversion -- but stream ordering relocates the work instead of overlapping it. Blocking H2D 1,264 → 822 ms while backend sync rises 4,737 → 5,183 ms. |
| Cross-layer prefetch onto a copy stream | +1.38% at 32,768, **not exact** | One of four greedy server tasks diverges deterministically (`89816e58…` → `4e2dd64c…`). Slot bookkeeping is reset by a mid-split `ggml_backend_cuda_synchronize`. |
| Write-combined pinned allocation | **−0.85%** at 4,096, **−0.37%** at 32,768 | A `q8_0` block is 34 bytes, so the CPU `set_rows` never fills a 64-byte write-combine buffer on a boundary; every row is a run of partial-line flushes. Worse at shallow depth, where the store is a larger share of the token -- which is what this explanation predicts. |
| Huge-page pinned allocation | +0.16% at 32,768, ±0 at 4,096 | `MADV_HUGEPAGE` and `MADV_NOHUGEPAGE` are indistinguishable (+0.16% vs +0.18% at 32,768), so page size is not the variable; the 2 MiB alignment and `cudaHostRegister` are, and neither justifies a custom allocator. `/proc/<pid>/smaps` confirms engagement: 6 MiB of `AnonHugePages` by default, 1,188 MiB with the mode on. The IOMMU premise for this item is absent on this host. |
| NUMA-affine pinned allocation | not measurable | One socket, one NUMA node, `nvidia-smi topo -m` reports NUMA affinity 0. |

## 6. What each of these rules out

- **Bytes are not the lever.** Transfers already run at the practical link rate.
- **The store side is not the lever.** Removing it entirely is worth 0.18%.
- **Allocation properties are not the lever.** One is a regression, one is a
  null result whose premise does not hold here, one is untestable on this host.
- **Scheduling the copy differently on the compute stream is not the lever.**
  Stream ordering moves the cost rather than removing it.
- **Ordering is the lever**, and it is worth 21-77% depending on depth.

Two conclusions recorded earlier in this lane were wrong and are corrected here:
that the store side is the serializer, and that the transfer cannot be hidden
behind compute. Both were argued from structure; both fell to measurement.
