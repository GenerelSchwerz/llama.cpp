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
relevant next-turn resource check. The reproducible runner is
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

## Deep-context scaling after Candidate 1 publication

### Isolated identities and scope

This follow-up was run after Candidate 1 and the reproduction runner were
published together at
`b3ce3a5c23f5ce3213d0ddb735a7e3bcd5b490e5`. Every baseline process still
used the independently built dense source
`c9f727c1e1995c4a871a719ab05b5f2478588efd`; every candidate process used the
committed `b3ce3a5c2` source. There were no source changes after that commit,
so the accepted exact PPL gate above remains the applicable quality result.

The coordinator subsequently reported that PR 5 and PR 6 moved the remote
`beellama-kv-cpu-offload` base to `4a7f9b496`. That happened after this study
was launched. None of the following measurements use that composed base, and
they must not be relabeled as composed evidence when PR 7 is refreshed.

The Release CUDA build configuration remained native CPU tuning, CUDA
FlashAttention, CUDA architecture 120, default quant pairs, and tests enabled.
The clean relink reported version `11247 (b3ce3a5c2)`. Artifact identities were:

| Artifact | c9 dense SHA-256 | Candidate 1 SHA-256 |
|---|---|---|
| `llama-bench` | `1f6229604db5092e79b42b38579194423f0fe249f57366749f619465ec450071` | `44e9eebbb2a764330cda49d1d1b5e92e57dbabf4c55aa67e3de3f531902c8756` |
| `llama-server` | `9b0283fb10f12e5d5317a79ddab3262602e1b218490123e2b9315092b6277346` | `eb6dd9e702078f5d1250a997da4449f8e842f670c9c8f0baa175023ead3f3660` |

The model SHA-256 remained
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.
All GPU-capable commands and entire runner invocations were fresh processes
wholly inside `flock /tmp/beellama-single-gpu.lock -c`. Affinity used only
llama controls (`-C 0x7 --cpu-strict 1` for `llama-bench`, and
`--cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1` for the server).
There was no `taskset`.

### Context and request geometry

The study used binary context capacities and ordinary repeated English. An
independent committed-build `llama-tokenize --show-count --no-bos` process
counted the JSON prompt plus its extraction newline; the server reports one
fewer evaluated token because the JSON string has no terminal newline.

| Context capacity | tokenizer count | server evaluated | headroom before 64 generated tokens | Request SHA-256 |
|---:|---:|---:|---:|---|
| 49,152 | 48,550 | 48,549 | 603 | `3ca474dd0e29b7f932679c2785f3f43595e3af2a2764f2fe9fde121bac0a6ff8` |
| 65,536 | 64,929 | 64,928 | 608 | `a6f11cf3996d4d3c2afdd195ecebe86237bc29937b504e8cb94e5dc66ac01a42` |
| 98,304 | 97,698 | 97,697 | 607 | `0eda5345b395bcb4488b1a5a053b9f54e1ff9ba892d25c11b4f76640a3f76eb0` |
| 131,072 | 130,467 | 130,466 | 606 | `d590bb421096a647e1bcccb811cbc370bed078ca69150a0b622fb25f770dd952` |

Each server request used `n_predict=64`, seed 1234, temperature zero,
`cache_prompt=false`, `ignore_eos=true`, and `reasoning_format=none`. Each
fresh server received the identical request twice. The runner polled `/slots`,
sampled `nvidia-smi` and `/proc/PID/status` every 0.25 seconds, and captured
startup plus explicit post-turn checkpoints. Every one of the 24 canonical
final responses evaluated the expected prompt, generated 64 tokens, and was
content-exact across both turns and all A/B/A processes at its depth. Content
hashes were:

- 49K and 64K: `1c9a72139ed1454f11f18575dd2b254769c7f584c0ae5b40e486f1d58ab988fa`
- 98K: `1457824d8fa500e50f2606c7224cda9925380e857ac25a1b37056e556c59b593`
- 128K: `b432e780dcf6c1f133a21c550c78b5e36308503b08dd5318bc556be01f3559f9`

### Exact extension of the published runner

The first server series deliberately preserved the published runner's exact
runtime geometry, including its omission of
`--recurrent-state-offload`. That provided a direct extension of the accepted
4K/30K/34K evidence rather than silently changing placement. The command was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'bash /tmp/beellama-compact-deep-scaling-20260820/run-server-depth-aba.sh \
   {49152|65536|98304} {port}'
```

Each depth ran dense A1, Candidate 1, then dense A2. Both dense anchors had
identical sampled process VRAM. Candidate values and deltas were also exact at
startup, prefill/decode peak, and post-turn 2:

| Context | dense startup / peak / post-2 | Candidate startup / peak / post-2 | process VRAM saving | dense / Candidate CUDA compute | dense / Candidate CUDA-host compute | pinned KV both |
|---:|---:|---:|---:|---:|---:|---:|
| 49K | 13,574 / 13,608 / 13,608 MiB | 13,526 / 13,560 / 13,560 MiB | 48 MiB | 422.27 / 374.27 MiB | 78.40 / 30.40 MiB | 1,632 MiB |
| 64K | 13,688 / 13,722 / 13,722 MiB | 13,624 / 13,658 / 13,658 MiB | 64 MiB | 536.27 / 472.27 MiB | 94.40 / 30.40 MiB | 2,176 MiB |
| 98K | 13,916 / 13,950 / 13,950 MiB | 13,820 / 13,854 / 13,854 MiB | 96 MiB | 764.27 / 668.27 MiB | 126.40 / 30.40 MiB | 3,264 MiB |

The candidate's post-turn-2 current RSS was 40,900, 56,878, and 95,146 KiB
below the corresponding dense midpoint. `VmHWM` remained the approximately
14,596 MiB transient mapped-model loading high-water in every process, and
`VmLck` remained zero because CUDA driver pinning is not represented there.
Pinned KV was byte-identical and staging remained zero.

Candidate prefill was faster on all six turns (+1.10% to +3.43%). Decode was
between -2.28% and +1.95%; only the 64K second turn fell below -2%, while its
first turn was neutral and the independent benchmark below was -0.70%.

This series stopped before a no-recurrent 128K server bracket after review of
the authoritative current protocol exposed that the published runner is not
the canonical faster serving placement for this hybrid model. Its completed
49K/64K/98K evidence remains valid and is retained. The separate
`llama-bench` series did complete at 128K, and the canonical serving series
below includes every requested deep point through 128K.

### Canonical recurrent-state serving A/B/A

The current CPU-KV protocol keeps attention KV pinned on the host but supplies
`--recurrent-state-offload` so supported hybrid recurrent state remains on the
GPU. A temporary, hashed copy of the published runner added only that flag:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'bash /tmp/beellama-compact-deep-scaling-20260820/run-server-recurrent-depth-aba.sh \
   {49152|65536|98304|131072} {port}'
```

The runner SHA-256 was
`86a4296ce853a652bbb96fb206ec444dab18b1a0231a72bbe5a82db4e353fba6`.
All other model, batch/ubatch, cache, affinity, request, sampling, and runtime
settings matched the prior series. No requested canonical depth was omitted.

| Context | dense startup / peak / post-2 | Candidate startup / peak / post-2 | process VRAM saving | dense / Candidate CUDA compute | dense / Candidate CUDA-host compute | pinned KV both |
|---:|---:|---:|---:|---:|---:|---:|
| 49K | 13,774 / 13,792 / 13,792 MiB | 13,676 / 13,694 / 13,694 MiB | **98 MiB** | 473.27 / 374.27 MiB | 68.28 / 20.28 MiB | 1,632 MiB |
| 64K | 13,838 / 13,856 / 13,856 MiB | 13,774 / 13,792 / 13,792 MiB | **64 MiB** | 536.27 / 472.27 MiB | 84.28 / 20.28 MiB | 2,176 MiB |
| 98K | 14,066 / 14,084 / 14,084 MiB | 13,970 / 13,988 / 13,988 MiB | **96 MiB** | 764.27 / 668.27 MiB | 116.28 / 20.28 MiB | 3,264 MiB |
| 128K | 14,294 / 14,312 / 14,312 MiB | 14,166 / 14,184 / 14,184 MiB | **128 MiB** | 992.27 / 864.27 MiB | 148.28 / 20.28 MiB | 4,352 MiB |

The strict 64/96/128 MiB points follow the dense F16 logical-mask size and
demonstrate a real scaling serving benefit, not a fixed allocator artifact.
The 49K point crosses an additional best-fit allocation bin: CUDA-host compute
saves the expected 48 MiB, while CUDA compute saves 99 MiB and sampled process
VRAM saves 98 MiB. The next 64K point returns to the logical-mask slope, so the
extra 49K saving is correctly treated as a local allocator discontinuity, not
extrapolated.

Ordinary current RSS also fell rather than moving the cost to pageable host
memory:

| Context | Candidate startup RSS delta vs dense midpoint | Candidate post-turn-2 RSS delta |
|---:|---:|---:|
| 49K | -45,836 KiB | -43,032 KiB |
| 64K | -62,166 KiB | -59,342 KiB |
| 98K | -93,988 KiB | -91,270 KiB |
| 128K | -127,594 KiB | -124,866 KiB |

As before, `VmHWM` is dominated by transient model mapping and `VmLck=0` is
not a CUDA-pinning counter. Backend allocation logs and the direct NSYS trace
below are the pinned-memory evidence. Pinned KV is unchanged at every depth;
only CUDA-host compute falls.

Canonical serving performance versus the arithmetic midpoint of dense A1/A2
was:

| Context | turn | prefill delta | decode delta |
|---:|---:|---:|---:|
| 49K | 1 / 2 | +0.654% / +0.639% | -1.032% / -2.138% |
| 64K | 1 / 2 | +0.857% / +1.176% | +0.332% / +1.001% |
| 98K | 1 / 2 | +1.390% / +1.694% | -0.177% / -0.817% |
| 128K | 1 / 2 | +2.017% / +1.839% | +1.787% / -2.339% |

The two isolated decode regressions beyond 2% did not form a monotonic trend, but
the 128K second turn crossed the materiality screen. It was therefore not
accepted on its own. A focused fresh-process A/B/A used the canonical
placement, `p0,n128,d131072`, three timed repetitions, and the same remaining
arguments:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'bash /tmp/beellama-compact-deep-scaling-20260820/run-recurrent-decode-128k-aba.sh'
```

| process | decode throughput | raw samples (t/s) |
|---|---:|---|
| dense A1 | 8.6562 +/- 0.0045 | 8.65104, 8.65921, 8.65822 |
| Candidate 1 | 8.6036 +/- 0.0059 | 8.59687, 8.60610, 8.60780 |
| dense A2 | 8.6312 +/- 0.0430 | 8.65106, 8.58187, 8.66080 |

Candidate 1 is 0.46% below the dense midpoint in this higher-confidence
screen. That is not a material decode regression. The graph also saved 130 MiB
of measured process peak while retaining identical KV residency and zero
staging.

### Synthetic graph-shape scaling

The direct `llama-bench` A/B/A command was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'bash /tmp/beellama-compact-deep-scaling-20260820/run-bench-aba.sh'
```

It used `-pg 512,64 -d {49152,65536,98304,131072} -r 1 --no-warmup
--progress --kv-memory`, pinned CPU Q8_0/Q8_0 KV, batch/ubatch 1024/512,
single GPU/no split, and the native affinity above. The 512-output logits
floor still hid process savings at 49K. At deeper points the mask allocation
grew beyond that floor:

| Context / graph | process-peak delta | Candidate throughput delta | disposition |
|---:|---:|---:|---|
| 49K `p512,n0` / `p0,n128` / `p512,n64` | 0 / 0 / 0 MiB | +3.61% / -0.70% / -0.43% | output floor |
| 64K `p512,n0` / `p0,n128` / `p512,n64` | -34 / -34 / -36 MiB | +5.39% / -0.70% / +0.21% | strict device win |
| 98K `p512,n0` / `p0,n128` / `p512,n64` | -96 / -98 / -96 MiB | +3.72% / +2.91% / +3.80% | strict device win |
| 128K `p512,n0` / `p0,n128` / `p512,n64` | -128 / -130 / -128 MiB | noisy | strict device win; performance rejected as noisy |

The 128K memory counters were exact in both dense anchors, but dense A1 had
severe host-side timing interference: its rows were 346.8, 1.84, and 15.27
t/s versus dense A2's 475.2, 5.94, and 48.90 t/s. Those timings are not
converted into a candidate performance delta. The independent canonical
server and focused decode brackets are the accepted 128K performance evidence.

### Deep direct CUDA allocation and VMM evidence

Nsight Systems 2026.1.3 directly targeted each `llama-bench` binary at
`p512,n0,d131072` with canonical recurrent placement. It did not use an
all-process capture or wrapper target:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'bash /tmp/beellama-compact-deep-scaling-20260820/run-nsys-recurrent-128k.sh'
```

The script expands to `nsys profile --trace=cuda,nvtx
--cuda-memory-usage=true --sample=none --cpuctxsw=none
--cuda-graph-trace=node -o REPORT /absolute/path/to/llama-bench ...` for each
binary. Profiler throughput is not used as performance evidence.

| traced allocation | c9 dense | Candidate 1 | Candidate delta |
|---|---:|---:|---:|
| model device allocation | 13,505,105,920 B | 13,505,105,920 B | 0 |
| device KV allocation | 17,825,792 B | 17,825,792 B | 0 |
| recurrent/device context allocation | 156,893,184 B | 156,893,184 B | 0 |
| scheduler device allocation | 1,044,210,560 B | 909,468,544 B | -134,742,016 B |
| CUDA VMM scratch sequence | 4+4+2+4 MiB | 4+4+2+4 MiB | 0 |
| dynamic device-allocation high-water | 14,738,715,520 B | 14,603,973,504 B | -134,742,016 B (-128.5 MiB) |
| scheduler pinned allocation | 156,012,576 B | 21,270,560 B | -134,742,016 B |
| total traced pinned high-water | 4,738,234,400 B | 4,603,492,384 B | -134,742,016 B (-128.5 MiB) |

Thus the deep scratch pool is byte-identical; the win is the dense mask's
device and pinned scheduler backing, not a smaller CUDA scratch pool. D2H was
identical at 4,866,254,848 B / 8,449 copies. H2D fell from 624,633,830,920 B /
10,745 copies to 607,268,953,608 B / 14,600 copies. Per-consumer descriptors
add tiny copies, but removing dense mask transfers reduces total H2D by
17,364,877,312 B.

The complete sorted kernel-name/count manifests are byte-identical, SHA-256
`a029e149a89b4013e705fb8d74630f9a01a269adc8b01c9369101a074ce40beb`.
Both contain 4,112 launches of the same F16 FlashAttention kernel. The earlier
NCU register result remains applicable to the unchanged kernel source; a
second NCU capture was unnecessary for this allocation-only depth check.

### Deep disposition

Candidate 1 has a defensible real serving benefit that grows with supported
context: 64 MiB at 64K, 96 MiB at 98K, and 128 MiB at 128K in the canonical
placement, plus a locally larger 98 MiB win at the measured 49K allocator
discontinuity. The saving persists through startup, prefill/decode peak, and a
second fully reprocessed next turn. It does not move bytes into KV staging,
pinned KV, pageable RSS, or CUDA VMM scratch. Outputs remain exact; the source
is unchanged from the exact PPL commit; and repeated 128K decode rejects a
material slowdown.

All deep artifacts are under
`/tmp/beellama-compact-deep-scaling-20260820`. The manifest contains 1,912
files and verified all 1,912 before documentation. Its SHA-256 is
`83ac78c7ea47b64a332c7dee441f4f74f8e800d301b216c5d2093275123af1ed`.
