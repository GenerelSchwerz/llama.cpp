# Compact causal-mask VRAM investigation

Status: Candidate 1 accepted by the local evidence below. This record
accompanies the implementation and its reproducible two-turn resource runner.

## Scope and identities

This investigation starts from PR #7 tip
`e5cd3dfe967686ab572b16db76f9bc90d9f67624` on
`exp/compact-causal-mask-pre-pr4`. Its clean dense reference is
`c9f727c1e1995c4a871a719ab05b5f2478588efd`. Commit `591337d4d` is not an
ancestor. Native-Q8 attention, live workspace, VMM telemetry, perplexity
capacity, host staging, and every other migration remain out of scope.

The retained Release CUDA binaries used for the allocation and profiler audit
are the same artifacts as the PR #7 evidence:

| Artifact | c9 dense SHA-256 | compact SHA-256 |
|---|---|---|
| `llama-bench` | `1f6229604db5092e79b42b38579194423f0fe249f57366749f619465ec450071` | `48fe4e4e56169a26546d8f6cd8a4137744ff7dba3482cfa8d66acdd44803cb2e` |
| `libllama.so` | `92ef6eb76261ef76c56598bb0dfcb71a0b29b2f9d7542f245cb78b8991963520` | `c947fab8a134d67691bcf4bac34df96e5502a639c11b83f32c5f20011884d816` |
| `libggml-cuda.so` | `0ea097eafed04cee5f7b8221a44d5d341c3906cacf7db1c8806243d3211d4e94` | `26a012172bf181cbd44467700a22cd836a9ec19f970a5af480901d09e7ccf2d0` |

Both use Release, native CPU tuning, CUDA FlashAttention, CUDA architecture
120, default quant pairs, and tests enabled. The model is
`/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.

## Read-only root-cause audit

### Existing A/B/A disaggregated by graph shape

The published combined rows hid an important shape split:

| depth and shape | c9 dense compute buffers | compact compute buffers | compact delta |
|---|---:|---:|---:|
| 4K, `p512,n0` | 576,156,448 B | 571,441,952 B | -4,714,496 B |
| 4K, `p0,n128` | 575,337,248 B | 570,884,896 B | -4,452,352 B |
| 4K, `p512,n64` | 571,470,624 B | 593,152,032 B | +21,681,408 B |
| 30K, `p512,n0` | 592,867,360 B | 607,834,144 B | +14,966,784 B |
| 30K, `p0,n128` | 592,343,072 B | 607,834,144 B | +15,491,072 B |
| 30K, `p512,n64` | 592,867,360 B | 607,834,144 B | +14,966,784 B |

`cuda_compute_buffer_bytes` is the sum of the CUDA device and CUDA-host
compute buffers. At 30K, dense reserves 505.0020 MiB device plus 60.4004 MiB
host; compact reserves 549.2754 MiB device plus 30.4004 MiB host. The compact
representation therefore saves the expected 30 MiB host mask while causing a
44.2734 MiB device-planner penalty.

### Capability and topology

Verbose scheduler captures used fresh direct `llama-bench` processes under the
single-GPU lock. The matched command form was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'GGML_SCHED_DEBUG=2 {build}/bin/llama-bench -v \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   -p 512 -n 0 -d {4096|30000} -r 1 --no-warmup --progress --kv-memory \
   -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
   -ngl 999 -sm none -mg 0 -nkvo 1 --kv-cpu-pinned -fa on \
   -b 1024 -ub 512 -o jsonl'
```

At 30K every real compact decision was `backend=1, layout=1, reserve=0`: 59
512-token depth batches, the final 304-token batch, and the measured prompt.
The reserve graph also selected compact. Both graphs have 354 backend splits;
compact has 3,912 nodes versus dense's 3,911 because of the descriptor reshape.
No dense fallback or planner re-reservation occurred.

An earlier 30K diagnostic used long-form `--no-kv-offload` and omitted the
matched `-sm none -mg 0` arguments. This older `llama-bench` parser left KV on
GPU (`no_kv_offload=false`, `split_mode=layer`). Files
`baseline-30k-p512.stdout` and `baseline-30k-p512.stderr` in the audit artifact
are invalid and are not evidence. The `*-matched.*` files supersede them.

### Exact allocator liveness result

Two diagnostic-only builds added `-DGGML_ALLOCATOR_DEBUG` through
`CMAKE_C_FLAGS`; neither source tree was edited. They were built with at most
six jobs. The one-row execution is sufficient because the decisive gap is in
the scheduler's 512-token/512-output reserve graph:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '{alloc-debug-build}/bin/llama-bench -v \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   -p 1 -n 0 -d 30000 -r 1 --no-warmup --progress --kv-memory \
   -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
   -ngl 999 -sm none -mg 0 -nkvo 1 --kv-cpu-pinned -fa on \
   -b 1024 -ub 512 -o jsonl'
```

The final dense high-water live set is:

```text
ffn_out-63   [0x0000000, 0x0a00000)   10 MiB
result_norm  [0x0a00800, 0x1400800)   10 MiB
result_output[0x1400800, 0x1f900800) 485 MiB
max_size = 505.0020 MiB
```

The compact high-water live set is:

```text
result_norm  [0x0000000, 0x0a00000)   10 MiB
ffn_out-63   [0x3646800, 0x4046800)   10 MiB
result_output[0x4046800, 0x22546800) 485 MiB
max_size = 549.2754 MiB
```

Earlier in the compact plan, the scheduler's 4 KiB
`CUDA0#attn_inp_kq_mask_compact#0` copy takes `[0, 0x1000)` and remains live
across all full-attention consumers. GGML's best-fit allocator then builds the
activation layout around that early pin. When the descriptor finally dies,
`result_norm` reuses the small low-address hole but `ffn_out-63` remains
stranded at 54.2754 MiB, so the 485 MiB logits tensor must begin at 64.2754
MiB. Dense's 30 MiB copy cannot fit the low hole and is placed after the
attention working set, leaving the final output triplet tightly packed.

This proves that the 44 MiB device increase is allocation-order fragmentation.
It is not dense fallback, a larger logical descriptor, CUDA/VMM rounding, or a
larger FlashAttention scratch allocation. The later `nvidia-smi` process delta
reflects this already-larger compute reservation, with VMM granularity only
rounding the externally sampled value.

### Direct NSYS and NCU checks

Nsight Systems 2026.1.3 directly targeted each `llama-bench` binary (no
all-process capture) at 4K `p512,n0`, under the GPU lock:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'nsys profile --force-overwrite=true --trace=cuda,nvtx \
   --cuda-memory-usage=true --sample=none --cpuctxsw=none \
   --cuda-graph-trace=node -o {report} \
   /absolute/path/to/llama-bench {matched llama-bench arguments}'
```

Both traces contain 144 instances of the identical mangled
`flash_attn_ext_f16<256,256,8,8,false,false,GGML_TYPE_F16,GGML_TYPE_F16>`
kernel and 144 instances of the identical general stream-K fixup kernel. The
sorted FlashAttention kernel-name files have the same SHA-256,
`672a2b20461abe8cdc544621ae66274738ced2a978d4e8ad62b649fb194f380f`.
Compact reduced traced H2D traffic from 15,839.844 MB to 15,816.288 MB
(-23.556 MB over the full depth-fill and prompt sequence).

Nsight Compute 2026.2.1 directly targeted the same binaries and captured one
matching F16 FlashAttention launch:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'ncu --target-processes application-only --replay-mode kernel \
   --kernel-name-base function --kernel-name regex:flash_attn_ext_f16 \
   --launch-count 1 --set basic --force-overwrite --export {report} \
   --page details --csv /absolute/path/to/llama-bench \
   {matched llama-bench arguments}'
```

Both launches used grid 140, block 128, 34,432 B dynamic shared memory, one
wave/SM, and 16.67% theoretical occupancy. The dense binary used 243
registers/thread and the compact binary 246 because the compact-aware kernel
body adds dynamic descriptor handling even though the mangled specialization
is unchanged. One profiled launch is not a performance estimate; the prior
clean throughput bracket remains the applicable no-regression screen.

All raw audit files are under
`/tmp/beellama-compact-root-audit-20260820`. The allocator logs are
`alloc-{baseline,candidate}-30k-p1.*`; matched scheduler logs are
`{baseline,candidate}-30k-p512-matched.*`; profiler reports begin with `nsys-`
and `ncu-`.

## Candidate ranking

1. **Per-consumer descriptor views (selected first).** Give each
   FlashAttention consumer a distinct view of the shared I64 descriptor. The
   scheduler then gives each tiny cross-backend copy the lifetime of one
   attention split instead of pinning one copy across all full-attention
   layers. This is graph-visible and CUDA-Graph replay safe, changes no public
   control, and leaves allocator policy unchanged for unrelated graphs. The
   expected extra transfer is only one descriptor per consuming layer.
2. **General long-lived split-input placement.** Teach the scheduler/allocator
   to place shared cross-split copies at the arena tail or in a separate
   lifetime class. This directly addresses the general planner weakness but is
   a much broader policy change with risk to unrelated MoE, multi-backend, and
   hybrid graphs. Defer unless the local graph expression cannot solve the
   problem.
3. **Dedicated mapped-host descriptor consumption.** Avoid the device copy by
   making CUDA consume a specifically supported host-mapped metadata buffer.
   This needs new backend plumbing and is inseparable from host-staging policy,
   which is out of this branch's scope.
4. **Implicit bound in dynamic op parameters.** Rejected. The write position
   changes between CUDA Graph replays; graph-visible input memory is required.
5. **Derive the bound from logical token positions.** Rejected. Logical
   positions are not a generic substitute for proved physical KV write
   indices after shifts or other valid position transformations.
6. **Global best-fit thresholds or context cutovers.** Rejected. Size/context
   thresholds would be silent workload-specific behavior and would not be an
   upstream-style correctness or capability abstraction.

## The output-floor constraint

At the 30K reserve shape, dense's 505.0020 MiB device high-water is already the
tightly packed 10 MiB final activation, 10 MiB normalization, and 485 MiB
512-row logits tensor. The mask is not live at that high-water. A mask-only
change therefore cannot make this particular process-VRAM peak lower without
also changing logits/output capacity, which is explicitly out of scope. A
successful planner fix can remove compact's 44.2734 MiB regression and retain
the 30 MiB CUDA-host saving, but 30K process VRAM is expected to become neutral
at this shape. This is a measured lower-bound fact for this graph; whether a
different model or a much deeper context has a mask-dominated device
high-water remains unproven.

## Candidate 1 ledger

### Implementation

Candidate 1 is one graph-expression change in `build_attn_mha`. When the mask
input is an I64 compact descriptor, every FlashAttention consumer receives a
named `ggml_view_tensor` of the shared descriptor. Dense F16 masks are
unchanged. The shared graph-visible backing remains the replay-safe source of
the changing physical KV write-index bound; only the lifetime of the
scheduler's cross-backend copy changes. There is no public control, build
option, context threshold, allocator-policy change, kernel change, or
architecture-name check.

The Release CUDA candidate was configured with the same options as the clean
c9 build and built with at most six jobs:

```bash
cmake -S . -B build-compact-consumer-view-cuda \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build-compact-consumer-view-cuda \
  --target llama-bench llama-perplexity llama-cli llama-server \
           test-backend-ops -j 6
```

| Artifact | c9 dense SHA-256 | Candidate 1 SHA-256 |
|---|---|---|
| `llama-bench` | `1f6229604db5092e79b42b38579194423f0fe249f57366749f619465ec450071` | `44e9eebbb2a764330cda49d1d1b5e92e57dbabf4c55aa67e3de3f531902c8756` |
| `llama-perplexity` | `453966fcb89e105e0a3e9e8fb39935801b4b1df616ea5611482cde897dda7aff` | `1b5d066be943ca62f0e15cd896efdff42bd4b9aa461a1ea8a7791e3598478381` |
| `llama-cli` | `ce4676a1c976fd7bf52b0bfcb7b0ec63e80ad2540e1ac58a2e53742ece5a506a` | `6ec2b385ee8e960267c0cbecb174eeb4ce9011fe9203d6ca6d3992cb6c4da028` |
| `llama-server` | `9b0283fb10f12e5d5317a79ddab3262602e1b218490123e2b9315092b6277346` | `eb6dd9e702078f5d1250a997da4449f8e842f670c9c8f0baa175023ead3f3660` |
| `test-backend-ops` | n/a | `64b2bce276e629310a47f814a2cf9e50d4a58bfb1b3d29ba8e6cb5c3730471b3` |
| `libllama.so` | `92ef6eb76261ef76c56598bb0dfcb71a0b29b2f9d7542f245cb78b8991963520` | `99ec0be13a5b3aad8026bd684e7baefe8c1a66c331ac8e1362542d79eb405f63` |
| `libggml-cuda.so` | `0ea097eafed04cee5f7b8221a44d5d341c3906cacf7db1c8806243d3211d4e94` | `79b268b9346885635d159fd214c83f2ee90952ae063dc12fe75cf6b0d7629904` |

### Planner result

The 30K 512-output reserve now uses 505.0059 MiB on CUDA versus dense's
505.0020 MiB and the original compact path's 549.2754 MiB. CUDA-host compute
remains 30.4004 MiB versus dense's 60.4004 MiB. Each full-attention layer has
its own named descriptor copy, and that copy dies at the end of its scheduler
split. The 44.2734 MiB stranded `ffn_out-63` hole is gone. The remaining 4 KiB
device difference is view/copy bookkeeping and rounds away in process VRAM.

This retains the compact logical mask's host-buffer saving while repairing the
allocation-order fragmentation. It does not change the output-floor result:
`llama-bench` still cannot report lower long-context process VRAM when its
512-row final logits tensor sets the high-water mark.

### Direct CUDA allocation and pool evidence

The final direct NSYS run used the command shown in the profiler section, but
targeted Candidate 1's absolute `llama-bench` path. At 4K `p512,n0`, the clean
dense and Candidate 1 traces reported these device allocations:

| Allocation | dense | Candidate 1 |
|---|---:|---:|
| model | 13,505,105,920 B | 13,505,105,920 B |
| KV | 17,825,792 B | 17,825,792 B |
| scheduler compute | 539,560,704 B | 539,564,800 B |
| CUDA VMM scratch pool sequence | 4+4+2+4 MiB | 4+4+2+4 MiB |
| total device-allocation high-water | 14,077,172,480 B | 14,077,176,576 B |

Thus CUDA scratch-pool high-water is exactly unchanged; Candidate 1 adds only
the scheduler's 4 KiB page. NSYS pinned allocations fell from 198,021,152 B to
193,302,560 B, exactly 4,718,592 B. H2D traffic fell from 15,839.844 MB / 2,052
copies to 15,816.841 MB / 2,187 copies. Relative to the original one-copy
compact graph, per-consumer views add 135 tiny transfers (15 extra views over
nine executed full-attention graph instances), but total H2D remains 23.003 MB
below dense. D2H was identical at 10,636.249 MB / 1,157 copies.

The trace again contains 144 launches of the same F16 FlashAttention kernel as
dense. Dense used 243 registers/thread and Candidate 1 used 246, as already
explained by the compact-aware kernel body; grid, shared memory, and kernel
specialization were otherwise unchanged. Report hashes and the SQLite export
are in `/tmp/beellama-compact-consumer-view-20260820`.

### Clean `llama-bench` A/B/A

Every row was a fresh process under the GPU lock. The exact command template
was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'TIMEFMT="elapsed=%E user=%U sys=%S maxrss_kib=%M"; \
   time {build}/bin/llama-bench \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   -pg 512,64 -d {4096|30000|34000} -r 1 --no-warmup --progress \
   --kv-memory -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 \
   --poll 100 -ngl 999 -sm none -mg 0 -nkvo 1 --kv-cpu-pinned \
   -fa on -b 1024 -ub 512 -o jsonl'
```

Each depth ran c9 dense A1, Candidate 1, then c9 dense A2. Performance deltas
below compare Candidate 1 with the arithmetic midpoint of the two baselines;
positive means faster:

| depth / graph | Candidate 1 throughput delta | dense compute | Candidate 1 compute | process peak delta |
|---|---:|---:|---:|---:|
| 4K `p512,n0` | +0.410% | 576,156,448 B | 571,441,952 B | 0 MiB |
| 4K `p0,n128` | +0.048% | 575,337,248 B | 570,884,896 B | 0 MiB |
| 4K `p512,n64` | -0.264% | 571,470,624 B | 561,408,032 B | -4 MiB |
| 30K `p512,n0` | +2.841% | 592,867,360 B | 561,414,176 B | 0 MiB |
| 30K `p0,n128` | -0.452% | 592,343,072 B | 561,414,176 B | 0 MiB |
| 30K `p512,n64` | -0.671% | 592,867,360 B | 561,414,176 B | 0 MiB |
| 34K `p512,n0` | +1.804% | 596,799,520 B | 561,414,176 B | 0 MiB |
| 34K `p0,n128` | -0.511% | 596,537,376 B | 561,414,176 B | 0 MiB |
| 34K `p512,n64` | -0.415% | 597,061,664 B | 561,414,176 B | 0 MiB |

At 4K the combined row's context and peak both fell exactly 4,194,304 B in
both baseline comparisons. At 30K and 34K all process checkpoints were exactly
equal, as predicted by the measured 512-row output floor. Model, KV resident,
post-context, and staging fields matched; `kv_staging_bytes` was zero. Maximum
RSS varied by less than approximately 2.5 MiB across each bracket.

### Fresh two-turn server A/B/A

`llama-bench` deliberately uses a 512-output reserve. A serving graph with one
requested output does not have that artificial logits floor, so it is the
relevant next-turn resource check. The uncommitted reproducible runner is
`scripts/compact-mask-next-turn.sh`. Each entire runner invocation was inside
the GPU lock and directly launched the selected `llama-server`. It used normal
warmup, context/batch/ubatch `{depth}/1024/512`, pinned CPU Q8_0/Q8_0 KV,
single GPU/no split, native llama CPU ranges, `--cache-ram 0`, and 0.25-second
`nvidia-smi` plus `/proc/PID/status` sampling. It sent the identical request
twice, polled `/slots`, and verified `cache_source=none` plus full prompt
reprocessing on both turns.

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'bash /home/gencoolpc/beellama-causal-pre-pr4/scripts/compact-mask-next-turn.sh \
   {label} /absolute/path/to/llama-server {port} {4096|30000|34000} \
   {request.json} {new-output-directory}'
```

The prompts were repeated ordinary English followed by a continuation request;
`llama-tokenize --show-count --no-bos` independently measured 3,494, 29,399,
and 33,403 tokens. Server evaluation excludes the terminal newline and reports
3,493, 29,398, and 33,402 evaluated tokens. Each request forced 64 greedy
tokens with `ignore_eos=true`, disabled prompt cache, and set
`reasoning_format=none`. Request SHA-256 values are:

- 4K: `890442a7dc2112216d3a91a134961eeb90f4f905bbc9e132a2070fd5229207c4`
- 30K: `b175b00ce6ae3c8ee01aec94bcd42bced7231dd49f0bc910e3dc327c146b89f9`
- 34K: `ed666bf2f48181b53e035765450d24d2d71af29570567777e1a0ecbb2be3ab7f`

Sampled process VRAM was exact in both baseline brackets:

| context | dense startup | Candidate startup | dense prefill/decode peak | Candidate peak | dense post turn 2 | Candidate post turn 2 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 13,302 MiB | 13,282 MiB | 13,336 MiB | 13,316 MiB | 13,336 MiB | 13,316 MiB |
| 30K | 13,442 MiB | 13,412 MiB | 13,476 MiB | 13,446 MiB | 13,476 MiB | 13,446 MiB |
| 34K | 13,468 MiB | 13,434 MiB | 13,502 MiB | 13,468 MiB | 13,502 MiB | 13,468 MiB |

The direct reservation explanation matches those sampled savings:

| context | dense CUDA compute | Candidate CUDA compute | dense CUDA-host compute | Candidate CUDA-host compute | pinned KV |
|---|---:|---:|---:|---:|---:|
| 4K | 150.27 MiB | 130.50 MiB | 34.40 MiB | 30.40 MiB | 136.00 MiB |
| 30K | 290.46 MiB | 260.96 MiB | 59.90 MiB | 30.40 MiB | 1,003.00 MiB |
| 34K | 317.18 MiB | 283.93 MiB | 63.65 MiB | 30.40 MiB | 1,130.50 MiB |

`CUDA_Host` is page-locked backend memory. `/proc/PID/status` reported
`VmLck=0` because driver-pinned CUDA allocations are not reflected there; the
backend buffer accounting and the NSYS pinned-allocation events are the pinned
memory evidence. Ordinary sampled current-RSS peak versus the midpoint dense
baseline was +4,088 KiB at 4K, -25,840 KiB at 30K, and -31,864 KiB at 34K.
The approximately 14,598 MiB `VmHWM` in every process was the transient mapped
model-loading high-water and differed only by noise; it is not page-locked
memory or steady serving RSS.

All 18 final server responses evaluated the expected prompt length and
generated 64 tokens. Complete content was exact across both turns and all
three A/B/A processes at each depth. Content hashes were
`e23dd26c83e5452f05ecaa7a722b31db7133e18a0500b2a739932a9f09fd0277`
at 4K, `4d9511304ecffc3d735214172378b938f1ac0311f4d37b35628b49a53b23384c`
at 30K, and the same `e23dd26...` hash at 34K.

Server throughput also showed no decode regression. Candidate 1 versus the
midpoint dense baseline was:

| context | turn | prefill delta | decode delta |
|---|---:|---:|---:|
| 4K | 1 / 2 | +0.617% / +0.144% | +1.230% / +1.119% |
| 30K | 1 / 2 | +0.765% / +0.575% | +0.621% / +1.860% |
| 34K | 1 / 2 | +1.189% / +0.655% | +0.553% / +1.109% |

These are single A/B/A screens, not confidence intervals. Together with the
independent `llama-bench` bracket, they reject a material decode slowdown.

### Exactness, PPL, and focused regression gates

The compact-versus-dense backend oracle used the fixed seed and selected the
generic CPU plus CUDA F16 vector, tile, and MMA cases. Both passed 3/3:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-consumer-view-cuda/bin/test-backend-ops test -b CPU \
   -o COMPACT_CAUSAL_DESCRIPTOR_EQUIVALENCE -j 1 \
   --seed 0x6a09e667f3bcc909'
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-consumer-view-cuda/bin/test-backend-ops test -b CUDA0 \
   -o COMPACT_CAUSAL_DESCRIPTOR_EQUIVALENCE -j 1 \
   --seed 0x6a09e667f3bcc909'
```

A fresh deterministic CLI A/B/A generated 16 tokens from prompt
`Write one concise sentence about causal attention.`, seed 1234, temperature
zero, context 4096, `--single-turn --simple-io`, and the same placement. After
removing only build identity and timing lines, all three streams had SHA-256
`cd35be773ca9520e5f79474c94ee1d07784433199b1c3c4f531dffbbcc353c6c`.

Matching-batch PPL used the accepted command from the original migration:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '{build}/bin/llama-perplexity \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   --file /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt \
   --ctx-size 4096 --batch-size 512 --ubatch-size 256 --chunks 4 \
   --cache-type-k q8_0 --cache-type-v q8_0 --threads 3 --threads-batch 24 \
   --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
   --n-gpu-layers 999 --split-mode none --main-gpu 0 \
   --no-kv-offload --kv-cpu-pinned --flash-attn on'
```

The clean dense A1, Candidate 1, and clean dense A2 processes all printed the
identical sequence `1.9315, 2.1279, 2.2498, 2.1674` and final
`PPL = 2.1674 +/- 0.03849`. There is no PPL increase.

Seven adjacent graph/allocator/attention regressions passed under the GPU lock:
backend-op seed stability, CUDA graph source properties, generated F16 vector
dispatch, batch allocation, CUDA FlashAttention route policy, CUDA
FlashAttention vector policy, and allocator tests. The normal compact CPU and
CUDA oracle plus these tests are the proportional focused coverage for this
one graph-lifetime change.

### Invalid and preliminary launches

- The earlier unmatched 30K scheduler diagnostic is already identified above.
- An initial CLI bracket used `-C 0-2`, which `llama-cli` correctly rejected as
  a malformed hexadecimal mask before model loading. It is not evidence; the
  accepted bracket used `--cpu-range 0-2`.
- An initial timing wrapper referenced absent `/usr/bin/time`; all nine
  processes failed before model loading. The accepted A/B/A used zsh's `time`
  and overwrote none of the final evidence files.
- Repeated raw token ID 100 produced invalid UTF-8-like content and HTTP 500
  after generation. A first ordinary-text screen stopped at EOS after 13
  tokens. Neither is next-turn evidence. The final ordinary-text requests used
  matched `ignore_eos=true` to force all 64 decode tokens and completed twice
  in every process.

### Disposition

Retain Candidate 1 on the compact causal-mask branch. It converts the original
compact path's measured +20 MiB (4K) and +44 MiB (30K) process-VRAM regressions
into a strict serving win of 20, 30, and 34 MiB at 4K, 30K, and 34K,
respectively. The 512-output benchmark is 4 MiB better at 4K and neutral at
long context for a separately proven output-floor reason. Output, compact
oracle, and PPL are exact; no material decode slowdown is present. No broader
allocator policy or second candidate is justified by the evidence.

Final candidate artifacts are under
`/tmp/beellama-compact-consumer-view-20260820`; read-only root-cause artifacts
remain under `/tmp/beellama-compact-root-audit-20260820`. Source and working
documentation are published together at the Candidate 1 checkpoint. The
candidate artifact hash manifest is `final-artifact-sha256.txt`, SHA-256
`bbc68c69ee1f557f89a6f15f32b0b36a1f34d1116e462beb22324ee6c36114cf`.
