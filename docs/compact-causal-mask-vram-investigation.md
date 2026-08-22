# Compact causal-mask VRAM investigation

Status: Candidate 1 was accepted by the historical evidence below. A later
merge-readiness audit rejected its decode disposition, and the historical stop
rule records that state. A user-authorized narrow direct-operator phase then
superseded that stop rule. Its tile-relative uniform candidate is now frozen
pending the final quality, compatibility, profiler, and composition gates. PR
7 remains draft.

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

## Composed validation on the merged KV-offload base

### Identity and evidence boundary

After the isolated c9 study above was committed and published, PR 5 and PR 6
were merged into `beellama-kv-cpu-offload`. The feature commits were rebased
onto exact base `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`. The composed source-validation
tip was `ae60c7321d950937a36af096112525db777ae13f`; it reported build 11254 and
was five commits ahead of, and directly based on, `4a7f9b496`.

This is a composition gate, not a new baseline measurement. In particular,
none of the c9 dense/Candidate 1/dense A/B/A resource or performance rows above
are relabeled as measurements of the rebased commits. Their source identities
remain c9 `c9f727c1e`, Candidate 1 `b3ce3a5c`, and isolated evidence tip
`a71c9f6f`. The code-equivalent per-consumer-view commit after rebase is
`94480dc99`, but no old result is attributed to that new hash.

The rebase had one append-at-end textual conflict in
`docs/cpu-kv-offload-experiments.md`. Resolution retained the merged-base W02
CUDA VMM telemetry and W06 perplexity-capacity records, followed by the
pre-existing compact-mask isolation record. No source conflict was hidden.

### PR 5 and PR 6 semantic composition

PR 5 changes the perplexity tool's output-capacity declaration and its focused
plumbing test. The compact feature does not modify that tool. PR 6 adds dormant,
opted-in VMM allocation counters in `ggml-cuda.cu`, exposes get/reset callbacks
through backend proc lookup, and consumes them only from `llama-bench
--kv-memory`. It does not change allocation, mapping, reuse, release, graph, or
kernel policy.

The only shared source file between PR 6 and the compact feature is
`ggml/src/ggml-cuda/ggml-cuda.cu`. Relative to the merged base, the compact
diff in that file is exactly 12 added lines: a
`ggml_backend_cuda_flash_attn_causal_prefix_supported` capability callback and
its proc-address entry. The merged VMM get/reset entries remain present and
unchanged. The names and callers are disjoint: PR 6 measures the allocator when
the benchmark explicitly resets telemetry, while compact selection queries a
FlashAttention capability while building a representable graph. Neither path
enables, disables, or conditions the other.

### Composed build and focused commands

The existing Unix Makefiles build directory was reconfigured and rebuilt with
at most six jobs. Tests were already enabled in its cache:

```bash
cmake -S . -B build-compact-consumer-view-cuda \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_KVARN=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_COMMIT=ae60c7321d950937a36af096112525db777ae13f \
  -DLLAMA_BUILD_NUMBER=11254
cmake --build build-compact-consumer-view-cuda -j 6 \
  --target llama-bench llama-perplexity llama-cli llama-server \
           test-backend-ops test-perplexity-plumbing test-batch-alloc \
           test-cuda-fattn-route-policy test-cuda-fattn-vec-policy test-alloc
```

The focused static/plumbing set passed 4/4, including both newly composed
features:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'ctest --test-dir build-compact-consumer-view-cuda --output-on-failure \
   -R "^(test-backend-ops-seed-static|test-cuda-graph-source-properties-static|test-vmm-allocation-telemetry-static|test-perplexity-plumbing)$"'
```

The fixed-seed compact-versus-dense oracle passed 3/3 on CPU and 3/3 on CUDA:

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

The seven adjacent graph, allocator, and FlashAttention guards passed 7/7:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'ctest --test-dir build-compact-consumer-view-cuda --output-on-failure \
   -R "^(test-backend-ops-seed-static|test-cuda-graph-source-properties-static|test-fattn-vec-dispatch-generated|test-batch-alloc|test-cuda-fattn-route-policy|test-cuda-fattn-vec-policy|test-alloc)$"'
```

The exact matching-batch PPL command was rerun because the composed source now
contains PR 5:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-consumer-view-cuda/bin/llama-perplexity \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   --file /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt \
   --ctx-size 4096 --batch-size 512 --ubatch-size 256 --chunks 4 \
   --cache-type-k q8_0 --cache-type-v q8_0 --threads 3 --threads-batch 24 \
   --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
   --n-gpu-layers 999 --split-mode none --main-gpu 0 \
   --no-kv-offload --kv-cpu-pinned --flash-attn on'
```

It printed the same cumulative sequence `1.9315, 2.1279, 2.2498, 2.1674` and
the same final `PPL = 2.1674 +/- 0.03849`; there is no measured increase. A
fresh deterministic 16-token `llama-cli` run used the same prompt, seed 1234,
temperature zero, context 4096, batch/ubatch 512/256, Q8_0/Q8_0 pinned CPU KV,
native affinity, single-turn, and simple-I/O settings as the isolated bracket.
After removing only its build and timing lines, its SHA-256 remained
`cd35be773ca9520e5f79474c94ee1d07784433199b1c3c4f531dffbbcc353c6c`.

Finally, one non-performance telemetry smoke used a fresh process:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-consumer-view-cuda/bin/llama-bench \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   -p 128 -n 16 -d 4096 -r 1 -b 512 -ub 256 \
   -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
   -ngl 999 -sm none -mg 0 -nkvo 1 --kv-cpu-pinned \
   -fa on -ctk q8_0 -ctv q8_0 --no-warmup --progress --kv-memory -o jsonl'
```

The prompt row reported VMM live/mapped peaks of 9,619,456/10,485,760 bytes;
the decode row reported 421,120/2,097,152 bytes. Both reported zero VMM live,
mapped, and active-pool state after context destruction. This proves the
merged telemetry and compact graph coexist and reconcile; its one repetition
is deliberately not used as performance or A/B memory evidence.

### Artifacts, invalid probes, and disposition

Relevant composed hashes are:

| Artifact | SHA-256 |
|---|---|
| `libllama.so` | `99ec0be13a5b3aad8026bd684e7baefe8c1a66c331ac8e1362542d79eb405f63` |
| `libggml-cuda.so` | `0fec68cd98a2985743b7970a28835d276a2023072a0c5772497b87081e708320` |
| `libllama-bench-impl.so` | `68cd63ab53d92e64dcfc140c019c7ffde4ccf27468591468f3e740dbc4` |
| `libllama-perplexity-impl.so` | `4e24cc9ddefa8033f975f9f77235b39d4b52135055f2d386acf8947011030635` |
| `test-backend-ops` | `64b2bce276e629310a47f814a2cf9e50d4a58bfb1b3d29ba8e6cb5c3730471b3` |

Nine composed logs are under `/tmp/beellama-compact-composed-ae60c7321`.
Their manifest SHA-256 is
`b346da4de0d14d8379f68465f0913827f9f1ad059a95fc174a888fcfcc99a5bf`.

Two setup mistakes produced no evidence. A first reconfigure requested Ninja
against the existing Unix Makefiles build directory and was rejected before
generation; the accepted reconfigure omitted `-G`. A first build request named
the Python CTest `test-fattn-vec-dispatch-generated` as a build target and was
rejected before building; the accepted target list contains only executables,
and the Python guard ran through CTest. In addition, an identity probe invoked
`llama-bench --version` outside the GPU lock; that unsupported argument was
rejected immediately after CUDA discovery and loaded no model or workload. It
is invalid and unused. The accepted identity probe used `llama-cli --version`
inside the lock and reported the exact composed SHA above.

Disposition: the merged KV-offload base composes cleanly with the isolated
compact causal-mask implementation. Exactness and PPL remain unchanged, PR 6
telemetry remains functional and non-owning, and no new source change was
needed. These checks establish source compatibility only; the isolated c9
A/B/A and deep-scaling study remain the resource and performance evidence.

## Documentation-base refresh at 8e858fcec

PR 9 merged its documentation consolidation into
`beellama-kv-cpu-offload` as exact commit
`8e858fcec39049fa028ce6fcb144a0c08b03abd3`. PR 7's previously published head
was `d4183adb8b4902a125b9339cd39032a095fca013`; its preserved composed-source
checkpoint remains `ae60c7321d950937a36af096112525db777ae13f`. The clean
post-rebase, pre-reconciliation checkpoint was
`ed9f889481621acd5de12d5db6b0dc010862c88a`, six commits directly ahead of the
new documentation base.

This refresh changes documentation inheritance and publication metadata only.
The accepted read-only proof used:

```bash
git diff --name-status 4a7f9b496..8e858fcec
for tree_path in CMakeLists.txt cmake ggml include src common tools examples tests; do
  git rev-parse "4a7f9b496:$tree_path" "8e858fcec:$tree_path"
done
git diff --binary --full-index 4a7f9b496..d4183adb8 \
  -- . ':(exclude)docs/**' | sha256sum
git diff --binary --full-index 8e858fcec..ed9f88948 \
  -- . ':(exclude)docs/**' | sha256sum
git diff --name-status d4183adb8..ed9f88948
git rev-parse d4183adb8:docs/compact-causal-mask-vram-investigation.md \
  ed9f88948:docs/compact-causal-mask-vram-investigation.md \
  d4183adb8:scripts/compact-mask-next-turn.sh \
  ed9f88948:scripts/compact-mask-next-turn.sh
```

The old source-bearing base `4a7f9b496` and new documentation base `8e858fcec`
have identical Git tree objects for `CMakeLists.txt`, `cmake`, `ggml`,
`include`, `src`, `common`, `tools`, `examples`, and `tests`. Their only
non-document changes are `AGENTS.md` and three added explanatory comment lines
in `scripts/mtp-nsys-profile.sh`; removing comment-only lines from the old and
new script yields identical SHA-256
`7ba3085abce3d9b47faef75ae7725da9fd87e119ff7d5661e3c863b075df715d`.
Neither path is built, linked, parsed by the llama CLI, or used by the compact
runner.

The complete full-index binary diff for all non-document paths, including the
compact reproduction runner, has SHA-256
`ffaeca4eff34e6d449184d724c6579828c91cbe070a464ba8db943c5c3bc4f92`
both for `4a7f9b496..d4183adb8` and for
`8e858fcec..ed9f88948`. The old and rebased trees also have identical blobs for
the compact runner (`0c5aea501dfa85669c55882e437f721bf73c9d12`) and this
investigation before the present append
(`d04b0090c0900aba215898eef6ec68e31f48f93e`). The original 47,204 bytes of
this investigation have content SHA-256
`6a9f0738592988deb5ec5cfcabacc625f5ea9ff79eaf043c3fcbdb543ca8b79c`;
therefore every isolated, deep, and composed measurement, command, exclusion,
artifact identity, and manifest above is retained byte-for-byte.

The sole rebase conflict was the independently appended experiment record. Its
resolution keeps the complete shared PR 9 Experiments 001-020/W06 protocol and
post-KV identity index once, followed by the branch-owned compact isolation
record once. The compact record has unchanged content SHA-256
`94f98fca92342cdb145c255c55e2f74ddbbe5fdc8c658e7fe656e4636506f72e`;
both section headings occur exactly once. The only later edits to shared
documents state the genuinely changed PR 9 merge identity, inherited
documentation base, and completed PR 7 reconciliation status.

Because the production/build/CLI/test trees and the complete causal source
delta are identical, rebuilding or rerunning a CUDA binary could add no source
coverage and would risk mislabeling a documentation-only refresh as new
performance evidence. No GPU command was run. The isolated c9 A/B/A and the
`ae60c7321` composed checks remain the accepted evidence boundaries; this
section is a path/tree/diff proof only.

## Merge-readiness decode audit and kernel stop rule

### Identity and corrected evidence boundary

This audit started from exact published PR 7 head
`565233f79faebb5bace9e41f0e2d0ba9c70930cf` on exact base
`8e858fcec39049fa028ce6fcb144a0c08b03abd3`. No commit, push, PR-body edit,
or draft-state change was made. The clean dense build used source
`8e858fcec`; the compact build used source `565233f79f`. Native-Q8 attention,
phase/live workspace, speculative decoding, and other unrelated policies were
not enabled.

The evidence classification is stricter than earlier sections of this ledger.
Every model command containing `--no-kv-offload`, `--kv-cpu-pinned`, or
`--recurrent-state-offload` is CPU-KV **composition evidence only**. It is not
causal-only proof of compact-mask performance, even where NCU or NSYS directly
targeted the llama executable. Those profiler captures do measure the selected
kernel and its counters, but the enclosing model configuration remains a
CPU-KV composition.

Exact-head correctness and quality gates remained valid: Release CPU passed
574/574 tests, Release CUDA passed 804/804 tests, the fixed-seed compact oracle
passed 9/9 on CPU and 9/9 on CUDA, the seven adjacent focused checks passed,
deterministic normalized output retained SHA-256
`cd35be773ca9520e5f79474c94ee1d07784433199b1c3c4f531dffbbcc353c6c`,
and dense/compact/dense matching-batch PPL was exactly
`2.1674 +/- 0.03849` in all three processes. The generic broad backend abort
was reproduced on the exact base and remains inherited, not a compact-mask
regression.

### Published-head CPU-KV composition result

Fresh clean-process dense/compact/dense serving at 4K, 30K, and 64K retained
the memory win and exact two-turn output, but exposed a small monotonic decode
loss. The compact decode rate relative to the midpoint of the two dense
processes was approximately -0.31%/-0.29% on turns 1/2 at 4K,
-0.40%/-0.37% at 30K, and -0.64%/-0.60% at 64K. The independent five-repeat
`llama-bench -p 0 -n 128` screen was -0.17%, -0.47%, and -0.60% at the same
depths. These are CPU-KV composition measurements, not a direct-operator
benchmark, but their A/B/A direction and growth reject the prior claim that
the implicit replacement has no reproducible decode cost.

The corresponding serving memory result remains real composition evidence:
the compact process saved 20 MiB at 4K, 30 MiB at 30K, and 64 MiB at 64K,
with lower CUDA/CUDA-host compute reservation and unchanged pinned KV. This
audit did not invalidate those resource measurements; it invalidated the
performance disposition.

### Direct-input allocation repair and profiler allocation

A graph/layout experiment passed the existing I64 write-index input directly
to `FLASH_ATTN_EXT`, removing the per-consumer views while preserving the
same capability and layout checks. NSYS reported 6,090 H2D copies for both
dense and compact forms, so transfer-count differences no longer explained
the remaining decode gap. The clean five-repeat 64K composition bracket was:

| form | process 1 | dense anchor | process 2 | result |
|---|---:|---:|---:|---:|
| direct compact | 14.660429 tok/s | 14.743589 tok/s | 14.663808 tok/s | -0.56% / -0.54% |

Targeted NCU captures then isolated the selected Q8_0/Q8_0, `ncols=1`,
head-dimension-256 vector dispatch. The directly targeted llama processes used
the same CPU-KV composition, so the table is a kernel-counter observation
rather than a standalone operator benchmark:

| metric | dense | direct compact |
|---|---:|---:|
| kernel duration | 460.61 us | 482.78 us |
| registers/thread | 254 | 255 |
| executed instructions | 127,572,480 | 128,307,744 |
| branch efficiency | 100% | 97.08% |
| average divergent branches | 0 | 5.83 |

Measured fact: the compact specialization executes more instructions, uses one
more register, and introduces divergent predicate handling in this dispatch.
The simplest causal hypothesis is that the descriptor-bound load and
per-key visibility predicate extend live state through the vector reduction;
that is an inference from the counters, not proof that any one generated
instruction accounts for the full end-to-end loss.

The direct-input source diff had SHA-256
`a64d29f658036369db484ca84a712aedb42c691af230e5ae19c6eecc27fe5564`.
Artifacts are under
`/tmp/beellama-pr7-merge-ready-20260821/{direct-reverse-decode,nsys-direct-*,ncu-direct-*}`.
The NCU report hashes are
`56ae8d17939871683afebaafe7fd63ccae63a9f289e807091197db4f786b281b`
(dense) and
`a0f2b9755a3301db8289278368e98a602202aef7a3f363575d46fdd0345892da`
(compact).

### Isolated kernel candidates

Each candidate first used the fixed-seed CUDA compact-versus-dense oracle.
All GPU commands were fresh whole commands inside
`flock /tmp/beellama-single-gpu.lock -c`; no `taskset` was used. The model
screen used the exact inner command captured in each manifest:

```bash
llama-bench -m "$model" -p 0 -n 128 -d 65536 -r 5 \
  --no-warmup --progress --kv-memory -ctk q8_0 -ctv q8_0 \
  -t 3 -C 0x7 --cpu-strict 1 --poll 100 -ngl 999 -sm none -mg 0 \
  -nkvo 1 --kv-cpu-pinned --recurrent-state-offload -fa on \
  -b 1024 -ub 512 -o jsonl
```

Because this command uses CPU-KV placement flags, all throughput rows below
are composition screens.

| candidate | source-diff SHA-256 | compact process(es) | dense | disposition |
|---|---|---:|---:|---|
| direct input only | `a64d29f...` | 14.660429 / 14.663808 | 14.743589 | rejected, -0.56% / -0.54% |
| loop bound | `b95ec15f...` | 14.817501 / 14.815612 | 14.743541 | rejected despite +0.50% / +0.49%: 64K turn-2 output diverged |
| branchless select | `3cbf6dba...` | 14.707185 / 14.711500 | 14.744411 | rejected, -0.25% / -0.22% |
| explicit bit mask | `80c33467...` | 14.670834 / 14.669919 | 14.747374 | rejected, -0.52% / -0.53% |
| tile-uniform | `9a668a9f...` | 11.978935 | 14.746913 | rejected, **-18.7699%** |

The loop-bound variant reduced the profiled vector launch to 437.02 us, 249
registers/thread, and 121,808,064 instructions, but it did not preserve the
two-turn exact-output oracle. That is an unconditional rejection regardless of
throughput. Its NCU report is
`/tmp/beellama-pr7-merge-ready-20260821/ncu-kernel-bound-compact.ncu-rep`,
SHA-256
`1bbdeea26af240927849e179973b1d5f1f121f946521cd9befcb1c919885ffd8`.

The tile-uniform fixed-seed oracle passed before the model screen. Its compact
process contained five tightly grouped samples, 11.9658--11.9841 tok/s; the
dense process contained five samples, 14.7279--14.7537 tok/s. This zero-overlap
18.77% loss is reproducible within the completed processes and far outside
allocator or timing noise. A redundant second compact process had started but
was interrupted before producing a JSON row; its empty stdout and partial
stderr are preserved and explicitly excluded. Key artifact hashes are:

- compact stdout: `891b323a8c7907f914eb4d5b7e694cd1e5cd1fb935f73e6b8114a87157416aac`
- dense stdout: `41e5184e860b5b977b94a4d862abeb96ea40b2801a80e6aa0d2197edf966b3d4`
- empty second-compact stdout: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- partial second-compact stderr: `5d21805a9a590add9677309897a0fb7f9d58a11065d6fc99248db18dbcd70e56`

A subsequent shared-bound experiment failed the fixed-seed CUDA oracle with a
`misaligned address` and was rejected before performance testing. A low-word
load experiment was then staged but never completed its build and was never
run; the user stop rule superseded it. Its complete uncommitted source snapshot
is preserved at
`/tmp/beellama-pr7-merge-ready-20260821/rejected-low-word-source.patch`,
SHA-256
`3f235d21607d3778611c16f033ddd3705392f7d16141c0d5edbd76951a63ef66`.
The temporary reverse-decode runner is preserved at
`/tmp/beellama-pr7-merge-ready-20260821/runtime/rejected-variant-reverse-decode.sh`,
SHA-256
`5d3cf83153b91e8024143fa6220eeaf8046c8a6d1aedda1644964a3addf92e12`.

### Stop-rule disposition

The tile-uniform candidate passed fixed-seed equivalence but failed the 64K
decode gate decisively. Per the explicit stop rule, no further kernel variant
was proposed or benchmarked. The planned test-only 30K/64K standalone
`FLASH_ATTN_EXT` performance harness and plain GPU-resident model gate would
normally precede freeze, but they were not retroactively constructed after the
already-running tile candidate met the stop condition. No candidate was
frozen, so the final matched PPL rerun, full default-KVarN compatibility build,
final NCU capture, one-time 128K CPU-KV composition study, and integration of
the later `f6341a` KV-offload base were not performed. None of these gates can
be silently waived or used to make this candidate ready.

All rejected source changes were removed with patch edits after being
archived. Production source is again byte-identical to published head
`565233f79f`. PR 7 remains draft and unmodified remotely. The maintainable
per-consumer-view allocation fix still proves a growing memory benefit in
CPU-KV composition, but the implicit compact replacement does not satisfy the
required strict decode-performance gate. It is therefore not merge-ready.

## Narrow direct-operator investigation

This phase started from published head
`565233f79faebb5bace9e41f0e2d0ba9c70930cf` and exact dense base
`8e858fcec39049fa028ce6fcb144a0c08b03abd3`. It supersedes only the historical
kernel stop rule above. The rejected source and runtime artifacts were first
preserved, then every production/build/test/script path was restored to the
published head. The rejected-phase ledger patch is
`/tmp/beellama-pr7-merge-ready-20260821/rejected-phase-ledger.patch`
(`628ad370e580aa47f5cd6de6e9e45677d258f89767a73b5648f353866be713df`),
and the rejected low-word snapshot and runner retain the hashes recorded
above. The interrupted second tile-uniform process remains explicitly excluded.

### Test-only direct harness

The ignored local harness is under `tmp/compact-fattn-screen`. It instantiates
only the production `FLASH_ATTN_EXT` Q8_0/Q8_0 vector pair for `D=256`,
`ncols=1`, grid `[1,17,24]`, block `[32,4,1]`, and SM120a. Its copied
`fattn-vec-screen.cuh` began byte-identical to production and is updated with
each candidate before compilation. It uses deterministic Q8 blocks and query
data, a dense F16 mask and compact I64 bound, raw output/metadata bit comparison,
PDL launch attributes, and CUDA-event ABBA timing. There is no shipped flag,
compile option, dependency, or architecture-name routing in the production
feature.

The first non-aligned 30K oracle failed because the harness allocated only the
logical K and mask extents even though the vector kernel reads a complete final
128-key tile. That result is a harness failure, not feature evidence. Adding
allocation-only tail guards while retaining logical tensor strides fixed the
oracle: 30K and 64K, with hidden tails 0 and 113, became bit-exact. The final
harness source SHA-256 is
`42df44887537cccf68b37e8f344d1e3d1835346a30d7af70a7dec343e5c7bb44`;
the build-script SHA-256 is
`a943414e14262e417bec90408da5bceb919f5111614106d1f200b1993138a4bb`.

The narrow build command is:

```bash
tmp/compact-fattn-screen/build.sh
```

It expands to direct `nvcc -O3 -DNDEBUG -std=c++17 --use_fast_math
--extended-lambda --generate-code=arch=compute_120a,code=sm_120a -lineinfo`
with only the harness translation unit and ordinary ggml/CUDA include paths.
`GGML_CUDA_KVARN` is absent. Candidate compile times were 2.61--2.71 seconds.
The selected compact kernel consistently used 255 registers, 4,352 B shared
memory, and no spills; dense used 254 registers and the same shared memory.

Every invocation below was a fresh process wholly inside the GPU lock:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'tmp/compact-fattn-screen/compact-fattn-screen \
   --mode equivalence --depth {30000|65536} --hidden-tail {0|1|113|208|2177} \
   --seed 305419896'

flock /tmp/beellama-single-gpu.lock -c \
  'tmp/compact-fattn-screen/compact-fattn-screen \
   --mode timing --depth {30000|65536} --hidden-tail N \
   --warmup-cycles 50 --measure-cycles 200 --seed 305419896'
```

Each timing process collected 400 samples per form in dense/compact/compact/
dense order. Three fresh processes were used at each reported depth.

### Bounded candidate table

| hypothesis | minimal diff | fixed-seed result | 30K / 64K compact median delta vs dense | disposition |
|---|---|---|---|---|
| published descriptor predicate | none | exact after harness tail repair | +2.54--2.89% / +4.53--4.59% | causal baseline; regressive |
| low-word descriptor load | load only the proven int32 low word | exact at tails 0/113 | +2.63--3.07% / +4.51--4.61% | rejected before model load |
| direct selected assignment | select `sum` or `-inf`, avoiding add of selected zero | exact at tails 0/113 | +0.18--0.38% / +1.92--2.03% | rejected before model load |
| tile-relative selected assignment | subtract `k_VKQ_0` once per tile, retain per-key select | exact at tails 0/1/113/2177 | **-2.09--2.14% / -0.88--0.92%** | direct pass; whole-model estimate remained slightly negative |
| tile-relative uniform fast path | previous row plus bypass select for a proven fully visible tile | exact at tails 0/1/113/2177 | **-2.68--3.11% / -1.79--1.84%** | frozen after production gate |

Negative deltas are speedups. The frozen candidate also remained faster with
hidden tails 113, 208, and 2,177: 30K medians improved 2.68--3.05%, and 64K
medians improved 1.72--1.88%. This rejects the theory that only an all-visible
synthetic span produced the direct win. Its test-only overlay SHA-256 is
`6b494d6aa256b45b9fae4153601ccf59d67604658a976d409a70b95774615193`;
the complete diff against the published header is
`99714fa04e6bc6191e4a18623f6ff0f06bed7ac23b327da5b24d2d22a8501dcf`.

The maintainability boundary is small: all routing remains the existing
compact-mask capability/type specialization. The candidate converts the I64
exclusive bound to a tile-relative int32 value once per outer iteration and
skips only the select when that value proves every key in the tile visible.
Boundary and fully masked tiles retain the exact predicate. It adds no phase,
depth, model, or architecture check and does not change loop count, reduction
order, descriptor layout, or fallback policy.

### Initial tile-uniform plain-GPU production gate

This preliminary screening build combined the tile-uniform vector kernel with
the then-current per-consumer graph views. It used Release, SM120,
FlashAttention on, native CPU tuning, tests enabled, and the existing
`GGML_CUDA_KVARN=OFF` option. It built only `llama-bench` and `llama-server`, at
six jobs. The captured `libggml-cuda.so` SHA-256 is
`b7ba12d140e2e10f490362029f9676219a0b06426ab0a68c0b0735520e6e3253`.
The source/header SHA-256 is
`6b494d6aa256b45b9fae4153601ccf59d67604658a976d409a70b95774615193`.

The deepest practical plain GPU-resident point was 30K. Dense used
15,553,658,880 B at peak against 16,652,042,240 B total; the 64K Q8_0/Q8_0 KV
does not fit beside the 14,426,476,544 B model. Every unrelated placement or
workspace opt-in was omitted. The exact benchmark command was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '{dense-or-compact}/bin/llama-bench \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   -p 0 -n 128 -d 30000 -r 5 --no-warmup --progress --kv-memory \
   -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
   -ngl 999 -sm none -mg 0 -fa on -b 1024 -ub 512 -o jsonl'
```

The preliminary A/B/A was 43.9266, 43.9304, and 43.9227 tok/s: compact is
+0.013% relative to the dense midpoint. All rows report
`no_kv_offload=false`, `kv_cpu_pinned=false`, and
`recurrent_state_offload=false`. The CUDA peak was byte-identical. Compact
reduced CUDA-host compute from 52,203,552 to 21,270,560 B; its device compute
was 362,624 B larger, but the sampled serving result below retained the
expected process saving.

The temporary plain-GPU two-turn runner differed from the published runner only
by omitting `--no-kv-offload` and `--kv-cpu-pinned`. Its SHA-256 is
`e163b03be278b7aa8304bd31cbbd18d817a198940bf60f68dad1fec8be13c882`.
It used the 29,398-token fixed request
`b175b00ce6ae3c8ee01aec94bcd42bced7231dd49f0bc910e3dc327c146b89f9`
and ran dense/compact/dense followed by compact/dense/compact. These compact
processes used the tile-uniform kernel with the old per-consumer graph views;
they are not repetitions of the later combined canonical-input candidate. All
twelve timing-stripped responses were byte-identical with SHA-256
`6af7967a4929e5d58b09a969e10f2ceef7a782b2168ba1571dad8bfee0eeb958`.
Dense sampled 14,532 MiB process VRAM; every compact process sampled 14,504
MiB, a 28 MiB saving. Compact current RSS was also lower; `VmHWM` remained the
transient mapped-model high-water and `VmLck` remained zero.

The three-process server means were:

| turn | dense mean | compact mean | compact delta | approximate 95% delta interval |
|---:|---:|---:|---:|---:|
| 1 | 43.5295 tok/s | 43.4497 tok/s | -0.183% | about -0.48% to +0.11% |
| 2 | 43.7419 tok/s | 43.7064 tok/s | -0.081% | about -0.36% to +0.20% |

These server point estimates are negative. Their intervals overlap zero, but
that does not make either point estimate non-negative. They are retained under
their exact earlier source identity rather than relabeled or pooled with the
combined candidate below. A later 4K model repetition also found a reproducible
negative point estimate for this source, so its freeze was revoked before the
canonical direct-input repair. No kernel variant followed the combined
candidate described below.

All narrow-phase artifacts are under
`/tmp/beellama-pr7-direct-screen-20260821`. The SHA-256 of its sorted file
manifest at this checkpoint is
`a8c3014daf16b626fe3c0acffca28a447085a08eb7ed61e943c10bd36f653535`.

## Frozen combined candidate: final validation and strict acceptance

### Source and evidence boundary

The final frozen candidate combines two upstream-style changes:

1. the compact causal descriptor is passed once in its canonical one-dimensional
   I64 layout and consumed directly, rather than materializing a graph view for
   every attention consumer; and
2. the existing compact-capable CUDA vector specialization converts the
   exclusive bound to a tile-relative value once per tile and bypasses the
   per-key select only when the complete tile is provably visible.

There is still no phase, depth, model, or architecture-name switch; no public
runtime flag; no new CMake option or dependency; and no dense/compact routing
threshold. Capability, tensor type, and layout checks remain the only compact
route boundary. The final `fattn-vec.cuh` SHA-256 is
`6b494d6aa256b45b9fae4153601ccf59d67604658a976d409a70b95774615193`.
The KVarN-off screening `libggml-cuda.so.0.19.0` SHA-256 is
`689704e98594d365936d9fc9022d057a63443f3c962efcbcf36dda145effbfb6`;
the exact f634 dense library is
`b6a035b252ee1b1765261e2e079967994075065dd08117d6a1e8e02a9553b196`.
The complete production/test/reproduction-runner diff against exact f634 is
archived at
`/tmp/beellama-pr7-direct-screen-20260821/frozen-combined-complete-vs-f634.patch`
with SHA-256
`77617de5810167b3abba13443a967be3f008e148fecac32298fc48a83052e09d`.
A static added-line audit found no context constants, model-family routing, or
architecture-name routing. `GGML_CUDA_COMPACT_CAUSAL_MASK` remains an
unconditional internal CUDA translation-unit definition, not a user CMake
option.

The earlier tile-uniform/per-consumer 64-token server bracket remains valid
negative evidence for that earlier source: -0.183% on turn 1 and -0.081% on
turn 2. The f634 composed 4K screen likewise measured -0.33%, and its focused
repeat measured about -0.216%. None of those rows is pooled with the combined
candidate. The current combined-source plain-GPU evidence is:

| case | dense process result(s) | combined compact | point estimate | evidence limit |
|---|---:|---:|---:|---|
| 4K, 10 repeats, no warmup | 49.7502 / 49.7164 tok/s | 49.7636 tok/s | +0.061% vs dense midpoint | one compact process |
| 4K, warmed, 30 repeats | 49.7765 tok/s | 49.8291 / 49.8001 tok/s | +0.0765% | sample-level approximate 95% interval +0.031% to +0.122%; processes are the proper unit |
| 30K, 10 repeats, no warmup | 43.9404 / 43.9109 tok/s | 44.1059 tok/s | +0.4105% | one compact process |

All of these rows omit CPU-KV placement and every unrelated opt-in:
`no_kv_offload=false`, `kv_cpu_pinned=false`, and
`recurrent_state_offload=false`. At 4K the combined candidate reduced sampled
process peak by 4 MiB. At 30K both forms landed in the same process allocation
bin, although CUDA-host compute fell from 52,203,552 B to 21,270,560 B. The
device compute reservation at 30K rose by 297,088 B, so no process-VRAM saving
is claimed for that `llama-bench` graph. Both forms had identical VMM
high-water: 14,632,960 B live and 14,680,064 B mapped.

### Direct operator, exactness, PPL, profiler, and compatibility gates

The final narrow Q8_0/Q8_0, D=256, `ncols=1` direct screen was bit-exact and
measured compact median kernel speedups of 2.649% at 30K and 1.782% at 64K.
The fixed-seed production CPU and CUDA compact-versus-dense oracles passed 3/3;
their log hashes are respectively
`fd2ea5c339d7f81c6d65ba17d53264116a689305c2f80709a3d6f4e144914aca`
and
`84151f6f19516d9ad19863150ebd4fc6c738fcd11dce8861e2d7787a86ddd664`.

The clean direct-target 64K NCU report is
`/tmp/beellama-pr7-direct-screen-20260821/ncu-frozen-combined-direct-64k.ncu-rep`
(SHA-256
`7d28208da7e6f501f6be46f6be1b8670c963145a320da36b824b2a7920245ca3`).
The selected compact launch used 255 registers, executed 125,063,040
instructions, reported 97.07% branch efficiency, and took 454.72 us under the
41-pass profiler capture. This is profiler evidence, not an end-to-end timing
substitute.

The matching-batch PPL command remained the documented Q8_0/Q8_0,
`-c 4096 -b 512 -ub 256` command. Dense A1, combined compact, and dense A2 all
printed `[1]1.9315,[2]2.1279,[3]2.2498,[4]2.1674` and final
`PPL = 2.1674 +/- 0.03849`. The combined log SHA-256 is
`a668a99fdccc92919d4a5115d48642360c81a22db98672bbd861be1a2794f496`;
there is no measured PPL increase.

The final default-KVarN Release build completed at six jobs. Its
`libggml-cuda.so.0.19.0` SHA-256 is
`27c6d2dbbc5789e937c7e910764c5467b389232b9dd693198c84ab486f6486b8`.
Nine focused static/CPU/CUDA tests passed, followed by the CUDA 3/3 oracle;
the logs hash to
`613995b860e0bf5a48952f2424246116d98e7736a16f44fe05ee44e6c745800a`
and
`84151f6f19516d9ad19863150ebd4fc6c738fcd11dce8861e2d7787a86ddd664`.

### One-time 128K CPU-KV composition bracket

The requested 128K dense/compact/dense bracket used clean processes and this
exact whole-command lock template:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '{dense-or-combined}/bin/llama-bench \
   -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
   -p 512 -n 128 -pg 512,64 -d 131072 -r 3 --no-warmup \
   --progress --kv-memory -ctk q8_0 -ctv q8_0 \
   -t 3 -C 0x7 --cpu-strict 1 --poll 100 -ngl 999 -sm none -mg 0 \
   -nkvo 1 --kv-cpu-pinned --recurrent-state-offload \
   -fa on -b 1024 -ub 512 -o jsonl'
```

Those three CPU-KV placement options make every row composition evidence, not
causal-only proof. No live workspace, native-Q8 permission, speculation,
phase policy, or other optional feature was enabled.

| graph | dense A1 / combined / dense A2 | compact delta vs dense midpoint | dense / compact sampled peak | dense / compact device compute | dense / compact CUDA-host compute |
|---|---:|---:|---:|---:|---:|
| 128K `p512,n0` | 509.337 / 523.771 / 506.267 tok/s | +3.145% | 14,347.125 / 14,229.125 MiB | 995.837 / 877.337 MiB | 148.785 / 20.285 MiB |
| 128K `p0,n128` | 8.35184 / 8.21260 / 8.18615 tok/s | **-0.682%** | 14,335.125 / 14,215.125 MiB | 994.056 / 875.806 MiB | 148.535 / 20.285 MiB |
| 128K `p512,n64` | 65.2279 / 65.5275 / 65.1023 tok/s | +0.556% | 14,349.125 / 14,231.125 MiB | 997.618 / 878.868 MiB | 149.035 / 20.285 MiB |

The sampled process saving was 118 MiB for prefill/mixed and 120 MiB for the
decode graph. Dense A1 decode was visibly noisy (8.155--8.477 tok/s), while
dense A2 and compact were tightly grouped; nevertheless the midpoint point
estimate is negative and is recorded as negative. One compact process cannot
establish a process-level interval. VMM high-water was identical for dense and
compact (13.955 MiB live / 14 MiB mapped for prefill and mixed; 0.402 MiB live /
2 MiB mapped for decode). Pinned CUDA-host context and resident KV were also
identical; ordinary host context and compute buffers were zero. This benchmark
did not sample total OS RSS, so it makes no total ordinary-RAM claim.

Artifacts are under
`/tmp/beellama-pr7-direct-screen-20260821/frozen-128k-composition-bench`.
The JSONL SHA-256 values are dense A1
`00b4dab2c48934ba3ce2c31dc382b306c462d0a7803cc99dff5d71f8fb9aa5d0`,
combined
`425305b189f57b15448137b5c2a641e4f99a085a3ec68dc7bab06bf9e0fb0e76`,
and dense A2
`bb7fad80b04b6ce3ccc0a95609da259c038bd840852623071b220cb75335e172`.

### Strict 30K clean-process end-to-end decision

Because an overlapping-zero interval does not rehabilitate the earlier
negative server point estimates, the final decision used a new exact-source,
balanced `A B B A A B` server bracket. Each of the six clean processes served
two fully reprocessed copies of the same 29,398-token prompt and generated 512
deterministic tokens. The longer decode reduced timing noise without another
model suite. Adjacent pairs were `A1/B1`, `B2/A2`, and `A3/B3`, balancing
ordering. The runner SHA-256 is
`3353094b95a675277e0217a48761b8675b989d6b5e31e44d8890d6b51b801b8e`;
the request SHA-256 is
`1d33b0089ec7122fcbc3ad575726cf4d4c229d1ff64920d411107ac0740a395e`.
Every whole runner invocation was independently locked.

The command recorded in each manifest expands to:

```bash
llama-server --model "$model" --ctx-size 30000 --parallel 1 \
  --cont-batching --kv-unified --batch-size 1024 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 --threads 3 --threads-batch 24 \
  --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --n-gpu-layers 999 --fit off --split-mode none --main-gpu 0 \
  --flash-attn on --cache-ram 0 --seed 1234 --slots --metrics \
  --host 127.0.0.1 --port "$port" --verbosity 4
```

No CPU-KV, live-workspace, speculative, native-Q8, or other optional control
was present. The process means, treating each clean server as the unit, were:

| source | process 1 | process 2 | process 3 | mean +/- sample SD |
|---|---:|---:|---:|---:|
| f634 dense | 43.2533 | 43.2328 | 43.2161 | 43.2341 +/- 0.0186 tok/s |
| combined compact | 43.3807 | 43.3423 | 43.3774 | 43.3668 +/- 0.0213 tok/s |

The compact point estimate is **+0.3070%**. A two-sided 95% Welch interval on
the process means is **+0.2015% to +0.4126%**. The order-balanced adjacent-pair
sensitivity interval is **+0.1559% to +0.4581%**. Both intervals are strictly
positive. This is the clean targeted end-to-end repetition required by the
acceptance clarification; it does not rewrite the earlier negative source
results or the negative 128K composition point estimate.

All twelve timing-stripped responses were byte-identical with SHA-256
`f304d92512a513379101486c04c1cf12c63a805037224214b55841d8128e9021`.
Every dense process sampled 14,532 MiB peak process VRAM and every compact
process sampled 14,504 MiB, a strict 28 MiB saving through startup, both turns,
and post-turn residence. Startup was 14,514 versus 14,486 MiB. Mean final RSS
was 1,702,109 versus 1,678,041 KiB, a 23.50 MiB compact reduction; `VmLck` was
zero throughout. Server reservations were 227.78 versus 198.28 MiB CUDA
compute and 49.78 versus 20.28 MiB CUDA-host compute. Prompt throughput's point
estimate was -0.261%; it is reported rather than hidden, and is not a decode
regression.

The sorted artifact-manifest SHA-256 is
`b116f9b477d5aac2d61e79c3276cb2f8c92f0db9f82d6d96156104e83cd727bc`;
artifacts are under
`/tmp/beellama-pr7-direct-screen-20260821/strict-30k-server-512g`.

## Frozen-candidate unresolved-gate validation and production profiling

### Evidence boundary and exact identities

This phase made no production, test, build, or public-interface change. The
working production delta against exact f634 remained byte-for-byte identical to
`frozen-combined-complete-vs-f634.patch` (SHA-256
`77617de5810167b3abba13443a967be3f008e148fecac32298fc48a83052e09d`),
and the final `fattn-vec.cuh` remained
`6b494d6aa256b45b9fae4153601ccf59d67604658a976d409a70b95774615193`.
The comparison binaries were the existing configuration-matched Release,
SM120, CUDA-FA-on, native-on, **KVarN-off** builds:

| source | build | `llama-bench` SHA-256 | CUDA library SHA-256 | CMake cache SHA-256 |
|---|---|---|---|---|
| exact `f6341a15779eb58fe6ad9e1b890e331c32b676c7` dense archive | `/tmp/beellama-pr7-f634-base-build.VCr0BF` | `497de09c880d0c0c9539f6e8347ce789023b3de76a25bcf45c211672f43ad1d2` | `b6a035b252ee1b1765261e2e079967994075065dd08117d6a1e8e02a9553b196` | `4daf76513c5fc367db1439b5f013de19ce9675747a0aa874dcb9ebb6dcae04fb` |
| frozen combined compact working source | `build-compact-fast-screen-cuda` | `990139351acda670b8867e919401245d1086ade3d553e6a04e4f95c4e611e4d7` | `689704e98594d365936d9fc9022d057a63443f3c962efcbcf36dda145effbfb6` | `12eee7b2959ae8db22d8ecce7be3156a520e672464ce4875fbd95c1bfadd665b` |

The ignored local resource sampler and process-level Welch/paired analyzer hash
to `9ba921f80da87ae10f1030b5b1dfed74a256d1f26f72d625cc1556e5e95374b3`
and `26b7072885c3436291f21bad660dce726b49deb664b87004781b21b0c80b2f65`.
Every benchmark and profiler target was a fresh whole command inside
`flock /tmp/beellama-single-gpu.lock -c`; all affinity was native llama
affinity and no command used `taskset`. Ordinary timing runs did not run under
a profiler.

The source did not change after the already accepted quality gates. The
fixed-seed CPU and CUDA equivalence logs remain exact at
`fd2ea5c339d7f81c6d65ba17d53264116a689305c2f80709a3d6f4e144914aca`
and `84151f6f19516d9ad19863150ebd4fc6c738fcd11dce8861e2d7787a86ddd664`.
The matching-batch PPL remains byte-identical at final
`2.1674 +/- 0.03849`; its compact log is
`a668a99fdccc92919d4a5115d48642360c81a22db98672bbd861be1a2794f496`.
These llama-bench gates generated no text, so they add no new output oracle and
reuse the unchanged-source two-turn and fixed-seed exactness evidence above.

### Gate A: plain-GPU 29,398-token full prefill

Five independent processes per source ran in order `A B B A A B A B B A`.
Each process included five full-prefill samples. CPU-KV placement, live
workspace, speculation, native-Q8 permission, and every other optional policy
were omitted. The exact workload after the sampler arguments was:

```bash
llama-bench \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 29398 -n 0 -r 5 --progress --kv-memory \
  -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
  -ngl 999 -sm none -mg 0 -fa on -b 1024 -ub 512 -o jsonl
```

| source | independent process means (tok/s) | mean +/- process SD | point estimate | process-level 95% interval |
|---|---|---:|---:|---:|
| f634 dense | 1608.796, 1601.764, 1602.105, 1603.809, 1601.128 | 1603.520 +/- 3.112 | reference | -- |
| compact | 1600.558, 1600.163, 1599.463, 1599.732, 1599.360 | 1599.855 +/- 0.501 | **-0.229%** | Welch **-0.468% to +0.011%** |

All five adjacent order-balanced compact effects were negative: -0.512%,
-0.100%, -0.165%, -0.254%, and -0.110%. Their paired mean was -0.228%
with a paired 95% interval of **-0.439% to -0.017%**. The result therefore
reproduces, rather than reverses, the prior -0.261% server-prompt point. The
unpaired interval narrowly overlaps zero because dense A1 was unusually fast;
overlap is not treated as a win.

Both forms reported 13,125.125 MiB model/startup CUDA use,
14,761.125 MiB context use, 14,803.125 MiB synchronized prefill/peak use,
13,155.125 MiB after-context residence, and a 14,780 MiB sampled process
maximum. VMM high-water was byte-identical at 13.955 MiB live / 14 MiB mapped.
Compact changed device compute from 505.000 to 505.283 MiB and CUDA-host
compute from 49.035 to 20.285 MiB; the allocator bin did not change. Mean
maximum `VmRSS`/`VmHWM` was 14,585,343/14,585,343 KiB dense and
14,584,984/14,584,984 KiB compact; `VmLck` was zero in every process. Those
RSS high-waters include transient model mappings and are not pinned-memory
measurements. Mean clean-process wall time was 112.470 s dense and 112.650 s
compact.

Separate direct-target NSYS captures selected the same production
`flash_attn_ext_f16<256,256,8,8,...F16,F16>` consumer 928 times. Aggregate
attention CUDA time was 2.708 s dense and 2.751 s compact. Grouping the 32
layer launches by prefill ubatch showed compact slower through the early and
middle spans, crossing near 21K, then faster at the deepest full spans.

Targeted NCU used only that kernel with `--launch-count 1`; skips 272 and 880
selected representative ~8K and ~28K full-prefill launches. The full section
set showed:

| span/source | profiled duration | registers | achieved occupancy | executed instructions | branch uniformity / divergent targets | DRAM throughput |
|---|---:|---:|---:|---:|---:|---:|
| ~8K dense | 1.5196 ms | 243 | 14.323% | 227.679 M | 99.888% / 480 | 56.21 GB/s |
| ~8K compact | 1.6562 ms | 246 | 14.288% | 250.317 M | 100% / 0 | 41.64 GB/s |
| ~28K dense | 6.5536 ms | 243 | 16.707% | 701.159 M | 99.954% / 480 | 800.17 GB/s |
| ~28K compact | 6.4813 ms | 246 | 15.965% | 771.571 M | 100% / 0 | 764.68 GB/s |

Compact removes the sampled divergent targets but costs three registers and
about 10% more instructions. At ~8K, its math-pipe stall ratio improves
3.475 to 3.234 but its wait and short-scoreboard ratios worsen (1.716 to 1.866
and 0.309 to 0.502), and the extra work dominates. At ~28K, long-scoreboard
stall falls 4.897 to 3.392 and barrier stall 1.363 to 1.050, so tile-uniform
bypass wins despite the instruction/register cost. Full prefill pays every
early-span cost before reaching that crossover, explaining the negative
end-to-end point.

Gate-A timing artifacts have sorted-manifest SHA-256
`57e818a2cc736eda92c5ae16918ef4a3aa8c54fb65313a4407eaf34b84eeca2e`;
profile artifacts hash to
`fcf5bc1e6c52e11422f99b8bfa5c9f1ed79c773776f40063d7669c535d7de1b6`.

### Gate B: 128K CPU-KV-composed decode

Five independent processes per source used the same order. Each process filled
exactly 131,072 tokens once, restored that state between repetitions, and then
measured five pure `p0,n128` decode samples. The three placement necessities
were the only extra policies:

```bash
llama-bench \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -p 0 -n 128 -d 131072 -r 5 --no-warmup --progress --kv-memory \
  -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
  -ngl 999 -sm none -mg 0 \
  --no-kv-offload 1 --kv-cpu-pinned --recurrent-state-offload \
  -fa on -b 1024 -ub 512 -o jsonl
```

Two preliminary launches omitted the required explicit `1` after llama-bench's
`--no-kv-offload` option. The parser consumed `--kv-cpu-pinned` as that value,
lost pinned placement, and failed context creation before workload execution.
Both artifacts are retained under the two `dense-a1-failed-*` directories and
excluded from all evidence.

| source | independent process means (tok/s) | mean +/- process SD | point estimate | process-level 95% interval |
|---|---|---:|---:|---:|
| f634 dense | 8.31158, 8.62214, 8.62336, 8.59896, 8.62477 | 8.55616 +/- 0.13714 | reference | -- |
| compact | 8.29089, 8.21403, 8.61508, 8.60652, 8.64028 | 8.47336 +/- 0.20385 | **-0.968%** | Welch **-4.004% to +2.068%** |

Adjacent pair effects were -0.249%, -4.733%, -0.096%, +0.088%, and
+0.180%; their paired mean was -0.962% with interval -3.588% to +1.664%.
The final three processes per source occupied a visibly more stable state and,
as a sensitivity view only, measured +0.057% with interval -0.372% to +0.487%.
No process was dropped from the primary estimate. The prior -0.682%
one-compact result is directionally reproduced by the all-process point, but
its magnitude is not confirmed: process-state drift dominates the interval.
The new evidence cannot demonstrate strict non-regression.

Dense and compact respectively reported 14,299.125 / 14,179.125 MiB context,
14,341.125 / 14,221.125 MiB post-fill, and 14,345.125 / 14,225.125 MiB
synchronized peak CUDA use. Sampled process maxima were 14,314 / 14,194 MiB,
a repeatable **120 MiB compact saving** in all ten processes. Device compute
was 994.056 / 875.806 MiB and CUDA-host compute 148.535 / 20.285 MiB. Both
forms retained identical 4,377.5 MiB resident KV and 4,360.5 MiB pinned
CUDA-host context; ordinary host context/compute buffers were zero. VMM
high-water was identical at 13.955 MiB live / 14 MiB mapped. Mean maximum
`VmRSS`/`VmHWM` was 14,584,549/14,584,549 KiB dense and
14,584,626/14,584,626 KiB compact, with `VmLck=0` throughout. `VmLck` must not
be used as a proxy for CUDA page-locked allocations. Mean wall time including
the one-time fill was 237.433 s dense and 236.286 s compact.

The first NSYS pair used the default graph granularity and exposed only 16
graph-capture vector launches; it is retained as a scope diagnostic, not a
full-decode launch-count claim. The corrected direct-target pair used
`--cuda-graph-trace=node` and saw exactly 2,048 production
`flash_attn_ext_vec<256,1,Q8_0,Q8_0>` launches (128 tokens x 16 attention
layers). Aggregate vector time was 1.7855 s dense and 1.7524 s compact
(-1.856%); matching combine kernels were 596.0 and 581.9 ms. Thus the selected
production compact consumer is faster at 128K even though the noisy
whole-process primary point is negative.

Targeted NCU then used that proven vector name, graph-node mode,
`--launch-skip 1024`, and `--launch-count 1`. Dense versus compact was
901.824 / 884.672 us, 254 / 255 registers, 16.167% / 16.195% achieved
occupancy, and 254.041 / 249.953 M executed instructions. Compact raised DRAM
throughput 488.70 to 497.38 GB/s and reduced wait, barrier, math-pipe, and
short-scoreboard stall ratios, while slightly increasing long-scoreboard and
not-selected ratios. Unlike prefill, dense had no divergent branch targets;
compact reported 1,632 and branch uniformity 98.446%. The net sampled launch
was still 1.902% faster, consistent with the full node-level NSYS aggregate.

Gate-B timing artifacts, including the two excluded failures, have
sorted-manifest SHA-256
`edf7ea2eca93b23f267b444dcba9dd7d3c65a68f4ab794c0e663b675b9be2924`;
profile artifacts hash to
`c91e0f5164bac67719aeb816544eef841644859a734da3bfbd30da8563ab9a96`.

### Disposition

The combined candidate remains frozen; no further kernel variant is authorized
or warranted in this phase. Correctness, deterministic output, and PPL remain
exact, and the memory benefits remain real: 120 MiB at the 128K CPU-KV decode
shape. The production 128K vector kernel is itself faster. However, the
implicit replacement still has a reproducible negative 30K full-prefill point,
and the complete 128K end-to-end process aggregate also remains negative with a
wide interval. Neither overlapping-zero interval is described as a win.

Under the strict acceptance standard, these results do **not** establish a
strictly better production replacement. PR 7 must remain draft and not
merge-ready. This phase made no commit, push, PR mutation, or source change.

## 240,000-token load-to-idle allocation check

This was a deliberately narrow allocation check requested after the timed
gates. It made no inference request and performed no prefill. Each fresh
`llama-server` process loaded the model, reached the stable model-loaded and
all-slots-idle state, was sampled 16 times over several seconds, and then was
terminated cleanly. Both builds were Release, SM120, CUDA-FA-on, native-on,
and **KVarN-off**; the earlier concern that only the compact build disabled
KVarN was disproved by the preserved CMake caches. The result is therefore a
configuration-matched comparison, but it remains startup/resident evidence
only.

The common runtime shape was one slot, `--ctx-size 240000` (internally rounded
to 240128), Q8_0/Q8_0, all model layers on one GPU, and the three CPU-KV
placement necessities `--no-kv-offload --kv-cpu-pinned
--recurrent-state-offload`. Live/phase workspace, native-Q8 permission,
speculation, and unrelated policies were omitted. The exact command and binary
identities are preserved in the manifests. The important identities are:

| source | server SHA-256 | CUDA library SHA-256 | CMake cache SHA-256 |
|---|---|---|---|
| exact `f6341a15779eb58fe6ad9e1b890e331c32b676c7` dense archive | `2152baa65085baf9f5df419a42c772a82e1f6f4f7295af59628124bc3d2c25c6` | `b6a035b252ee1b1765261e2e079967994075065dd08117d6a1e8e02a9553b196` | `4daf76513c5fc367db1439b5f013de19ce9675747a0aa874dcb9ebb6dcae04fb` |
| frozen combined compact working source | `6c1aec477e7f8aecded4a1582daa0228a2cc417e0b013370babfeb36c74daf5a` | `689704e98594d365936d9fc9022d057a63443f3c962efcbcf36dda145effbfb6` | `12eee7b2959ae8db22d8ecce7be3156a520e672464ce4875fbd95c1bfadd665b` |

| stable idle measurement | dense | compact | compact delta |
|---|---:|---:|---:|
| sampled process VRAM | 15,052 MiB | 14,828 MiB | **-224 MiB** |
| `VmRSS` | 9,662,604 KiB | 9,423,676 KiB | **-238,928 KiB (-233.33 MiB)** |
| `VmHWM` | 14,580,092 KiB | 14,582,796 KiB | +2,704 KiB (+2.64 MiB) |
| `VmLck` | 0 KiB | 0 KiB | 0 KiB |
| CUDA compute reservation | 1,751.09 MiB | 1,526.59 MiB | **-224.50 MiB** |
| CUDA-host compute reservation | 254.78 MiB | 20.28 MiB | **-234.50 MiB** |

Model, KV, and recurrent-state reservations were identical: 12,879.47 MiB
CUDA model, 17.00 MiB CUDA KV, 7,973.00 MiB CUDA-host KV, and 149.62 MiB CUDA
recurrent state. `VmLck=0` does not disprove CUDA page locking and must not be
used as a pinned-memory counter. The nearly exact agreement between the
224 MiB process-VRAM reduction and the 224.50 MiB CUDA compute-reservation
reduction supports attributing the idle device-memory result to removal of the
dense causal-mask allocation, not to a different model, KV placement, KVarN
matrix, or inference lifecycle.

Artifacts are under
`/tmp/beellama-pr7-direct-screen-20260821/load-idle-240k-20260821`.
`sha256sum -c artifact-hashes.sha256` validates every listed artifact; the
artifact-manifest file itself hashes to
`f67307edb0dbb6a540e86a0ad42d099d8f5267452b0e1dbe13222142ad857f21`.

## Measured conclusion and redesign charter

### Why the frozen compact consumer is not yet a default

The measurements establish two different facts that must not be conflated:

1. The compact logical representation produces real, depth-scaling allocation
   savings. The current exact-source checks include 28 MiB at the 30K server
   shape, 120 MiB at the 128K CPU-KV decode shape, and 224 MiB at 240K stable
   idle. The 240K reservation logs directly identify the removed CUDA and
   CUDA-host compute allocations.
2. The current CUDA consumer pays for representability inside the attention
   hot path. At 29,398-token full prefill it executes about 10% more
   instructions and uses three more registers in the sampled kernel. The
   process point is -0.229%, with all five balanced pairs negative. At 128K
   the selected vector kernel is about 1.9% faster, but process-state variance
   leaves the complete decode point at -0.968% with a wide interval. Therefore
   this implementation is not proven strictly non-regressive in all required
   serving phases.

The causal explanation beyond those observations is an inference. The dense
mask is expensive to allocate but cheap for the existing FlashAttention
consumer to read because the consumer's mature tiling and control flow already
encode its access pattern. The frozen compact implementation removes the
allocation but reconstructs visibility with additional integer arithmetic and
selection in frequently executed code. At short and middle spans, that extra
instruction/register cost dominates. At deeper spans, reduced memory and stall
pressure can dominate instead. That crossover explains the direction of the
profiles; it does not prove identical behavior on every GPU generation.

NVIDIA's published architecture and CUDA guidance supports treating this as a
scheduling problem rather than assuming integer arithmetic is universally
free. Native arithmetic throughput varies by instruction and compute
capability, while Turing and newer designs can overlap independent integer and
floating-point instruction issue. Those facts make a small, amortized tile
classification plausible, but they do not justify an architecture-name check
or a cross-generation performance claim without measurement. References:

- <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions>
- <https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/>
- <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>

### Replacement objective

The next implementation phase remains part of the same compact causal-mask
feature and stays on this branch. It is not a new user feature. The objective
is to preserve the compact/symbolic causal description and its allocation
savings while moving visibility work out of the per-score inner loop:

- graph construction expresses a causal exclusive-prefix boundary as semantic
  operator metadata, using existing tensor/operator abstractions where
  possible;
- backend lowering classifies attention work at tile or launch granularity;
- fully visible tiles execute the existing unmasked inner loop unchanged;
- fully masked tiles are omitted or retired before score work;
- only the boundary tile performs a predicate, with the minimum representable
  arithmetic;
- unsupported operators, layouts, or backends retain the existing dense-mask
  path based on capability and layout checks.

There will be no public runtime control, user-visible CMake option, context
threshold, prefill/decode switch, GPU architecture-name allowlist, new
dependency, or import from native-Q8/live-workspace/telemetry/perplexity-
capacity branches. A layout that cannot prove the symbolic boundary remains on
the dense fallback; it is never silently reinterpreted.

### Bounded implementation and decision plan

The work proceeds through explicit stop gates so an appealing design cannot
hide a slower implementation:

1. **Coherent source baseline.** Preserve the published evidence checkpoint,
   retain the current frozen candidate and rejected experiments as artifacts,
   reconcile onto exact `f6341a15779eb58fe6ad9e1b890e331c32b676c7`, and
   prove that the only production delta is causal-mask representation and
   consumption.
2. **Minimal merit prototype.** Hoist the exclusive boundary to the outermost
   existing tile/launch scope that can prove it. Do not first redesign the
   operator ABI. Static review must show that the fully visible path contains
   no new per-score predicate and that unsupported layouts still take dense
   fallback.
3. **Fast correctness/performance screen.** Run the fixed-seed CPU/CUDA
   equivalence oracle, then warmed interleaved direct production-shape screens
   near 8K, 30K, 64K, and 128K. The unit is the actual Q8_0/Q8_0, D=256,
   `ncols=1` vector consumer where applicable. Reject immediately if 30K or
   64K is reproducibly slower or any tail differs bit-for-bit.
4. **Merit decision.** Only if the minimal prototype is exact and clearly
   non-regressive at both 30K and 64K does it justify the full internal
   redesign. Otherwise archive its diff, command, timing, and disposition and
   leave the published source unchanged.
5. **Full internal lowering, if earned.** Carry the symbolic boundary through
   graph/operator metadata and backend capability negotiation, and apply the
   same tile classification to generic CPU plus CUDA F16/vector/tile/MMA and
   KVarN-capable paths without architecture-name dispatch. Preserve dense
   fallback coverage for unsupported layouts.
6. **Production freeze gate.** Use clean order-balanced plain-GPU A/B/A for
   30K full prefill and the deepest fitting decode, followed by deterministic
   two-turn output. Freeze the first maintainable implementation whose
   process-level point estimates and confidence intervals meet the strict
   non-regression standard; do not continue tuning after a pass.
7. **Final acceptance.** Re-run matching-batch PPL, focused CPU/CUDA
   correctness, 4K/30K/deep performance and allocation/lifecycle measurements,
   final KVarN-enabled compatibility build/tests, targeted direct-target NSYS
   and NCU, and one deep CPU-KV composition study. Any PPL increase or
   reproducible decode loss is automatic rejection.

Profiler captures remain separate from timed evidence. Every CUDA-linked
process, including smoke/version probes, remains a fresh whole command under
`flock /tmp/beellama-single-gpu.lock -c`; Nsight targets llama binaries
directly, no command uses `taskset`, builds use at most six jobs, and long runs
expose native progress. PR 7 remains draft until the full acceptance gate is
met. Findings, rejected attempts, exact identities, commands, artifact hashes,
and tradeoffs will continue to be recorded in this ledger rather than
overwriting earlier evidence.

## Consecutive-bound consumer redesign

### Outcome and evidence boundary

The redesign charter above produced a maintainable implementation with merit.
The accepted change does not add a second mask format, a phase switch, or a
context threshold. It makes the consecutive-write property already proved by
`can_use_compact_causal_mask()` an explicit internal consumer contract:

- the graph passes the existing one-dimensional I64 write-index input directly
  to `FLASH_ATTN_EXT`, so no reshaped descriptor or per-consumer view is needed;
- the first exclusive bound in a query tile is loaded once, and later query
  bounds are derived as `first_bound + query_offset`;
- CUDA vector, tile, F16 MMA, and `KV_max` consumers classify work at tile
  scope and keep fully visible tiles on an unmasked inner path;
- the CPU consumers read the same one-dimensional layout;
- unsupported attention semantics or non-consecutive KV layouts never create
  this descriptor and continue to use the dense mask.

This is a full internal consumer redesign of the same feature, not a new public
feature. It has no runtime control, user CMake option, optional dependency,
architecture-name check, model-family check, prefill/decode route, or depth
route. Native-Q8, live workspace, VMM telemetry, and perplexity-capacity code
were not imported into the isolated change.

The implementation was developed from published findings checkpoint
`f6b930b138c0b0023cfee5d90b579b71f006aa21` and compared against exact dense
source `f6341a15779eb58fe6ad9e1b890e331c32b676c7`. The ordinary performance
pair was configuration-matched: Release, SM120, CUDA FlashAttention on,
`GGML_NATIVE=ON`, and `GGML_CUDA_KVARN=OFF`. KVarN is runtime-irrelevant for
the Q8_0/Q8_0 workload and was enabled separately for the final compatibility
build.

The frozen production-header hashes before commit were:

| source | SHA-256 |
|---|---|
| `fattn-common.cuh` | `b2372687d95bcdb0d293d8c48c9bc0462e8078275d88000ced45a2eed129e4d4` |
| `fattn-mma-f16.cuh` | `b749461c1891fb3a0a4a17594fb285d3ce4e1ced4bcbd5da09cafd269b7e75ec` |
| `fattn-tile.cuh` | `9e7efd98d2c7223908d552db2d270cdba17665e9e00a7faf4c02ca9cac6e7220` |
| `fattn-vec.cuh` | `e583fbb392773cd7f3e937178ac5a32e12ef4e757de348d1b696ab9e5aa2f5ba` |

No production source changed after these hashes were frozen.

### Bounded candidate decisions

The iteration loop used the local test-only `tmp/compact-fattn-screen` harness.
It directly instantiates the production Q8_0/Q8_0, D=256, `ncols=1` vector
header, checks that header's hash, uses fixed-seed dense-versus-compact
equivalence first, and then times warmed interleaved ABBA CUDA-event launches.
The harness and rejected patches remain local artifacts; they are not shipped
or committed as production code.

| hypothesis | minimal change | exactness | direct result | disposition |
|---|---|---|---|---|
| classify every MMA iteration | add a runtime compact boundary decision inside the repeated iterator | statically representable | not timed after review | rejected: repeats the cost this redesign must remove |
| register-hoist query bounds | load compact bounds once per launch instead of per K tile | fixed-seed exact | promising enough to continue | revised into the common consecutive-bound contract |
| shared-memory MMA classifier | cache the tile's first bound in existing shared padding and bypass fully visible mask work | CPU/CUDA exact | selected 29K launch +3.27%; whole prefill point -0.063% | retained as one consumer, but insufficient alone |
| compile-time compact MMA specialization | add another template dimension | exact | 252 registers; selected-object compile 22.5 s | rejected: matrix growth and no distinct end-to-end merit |
| outer vector tile split only | separate fully visible and boundary loops | fixed-seed exact | direct positive at 30K/64K | revised: needed a common layout contract and all consumers |
| consecutive-bound contract (accepted) | pass 1-D write indices; load one bound per query tile/warp; derive offsets; split visible/boundary work | fixed-seed and production exact | direct positive at 30K, 64K, and 128K | frozen and fully validated |

The accepted harness compile took 2.545 s. Its binary SHA-256 is
`d18843c8748764d7c89bd0fe95ad04913b2f065a55db540ab1d34ed4c3f3cfd2`,
and the mirrored vector header is
byte-identical to production hash
`e583fbb392773cd7f3e937178ac5a32e12ef4e757de348d1b696ab9e5aa2f5ba`.
The complete identities are in
`/tmp/beellama-pr7-redesign-20260822/symbolic-base-screen`.

The exact screening command form was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'tmp/compact-fattn-screen/compact-fattn-screen \
   --depth DEPTH --hidden-tail TAIL --warmup-cycles 40 \
   --measure-cycles 200 --seed 0x6a09e667 --mode both'
```

Depths 30,000, 65,536, and 131,072 used tails 0, 113, and 2,177 for
equivalence. All nine combinations reported zero mismatches. Three independent
timing processes at tail 113 produced 400 samples per representation:

| K span | dense median range | compact median range | compact time delta |
|---:|---:|---:|---:|
| 30,000 | 195.168-195.264 us | 188.064-188.928 us | **-3.640% to -3.245%** |
| 65,536 | 449.056-449.152 us | 436.736-436.896 us | **-2.744% to -2.729%** |
| 131,072 | 894.496-895.648 us | 869.888-871.072 us | **-2.751% to -2.737%** |

These are direct-kernel screening results, not substitutes for the ordinary
model-process gates below.

### Builds, focused correctness, and quality

The coherent KVarN-off performance build produced these identities:

| artifact | SHA-256 |
|---|---|
| `libggml-cuda.so` | `99463282a9ee08474356bee153ca9663217e94fcf357d5505b782583bcf2b58c` |
| `llama-bench` | `990139351acda670b8867e919401245d1086ade3d553e6a04e4f95c4e611e4d7` |
| `llama-server` | `6c1aec477e7f8aecded4a1582daa0228a2cc417e0b013370babfeb36c74daf5a` |
| `llama-perplexity` | `43c86f0f65813b4cd5f2a1ba7b31121ab54fa30d14bd6f3b5eb8744a0200a195` |
| `test-backend-ops` | `7b4d8bfa94d01436996429427850298826c1d938ad35df24fb0e77f62dbf0379` |
| `CMakeCache.txt` | `12eee7b2959ae8db22d8ecce7be3156a520e672464ce4875fbd95c1bfadd665b` |

The exact dense `f6341a157` comparison used `llama-bench` SHA-256
`497de09c880d0c0c9539f6e8347ce789023b3de76a25bcf45c211672f43ad1d2`,
CUDA library
`b6a035b252ee1b1765261e2e079967994075065dd08117d6a1e8e02a9553b196`,
and CMake cache
`4daf76513c5fc367db1439b5f013de19ce9675747a0aa874dcb9ebb6dcae04fb`.

The focused oracle now covers `nb=1,3,17,65`, so partial vector, tile, and MMA
query tiles exercise the derived-bound contract. CPU and CUDA both passed 4/4
with seed `0x6a09e667f3bcc909`, requiring bit-exact dense and compact F32 output:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-fast-screen-cuda/bin/test-backend-ops test -b CPU \
   -o COMPACT_CAUSAL_DESCRIPTOR_EQUIVALENCE -j 1 \
   --seed 0x6a09e667f3bcc909'
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-fast-screen-cuda/bin/test-backend-ops test -b CUDA0 \
   -o COMPACT_CAUSAL_DESCRIPTOR_EQUIVALENCE -j 1 \
   --seed 0x6a09e667f3bcc909'
```

Because production source changed, matching-batch PPL was rerun with the same
model and corpus, Q8_0/Q8_0, `-c 4096 -b 512 -ub 256`, and four chunks. It
again printed `[1]1.9315,[2]2.1279,[3]2.2498,[4]2.1674` and final
`PPL = 2.1674 +/- 0.03849`. There is no PPL increase.

### Plain-GPU 29,398-token prefill

Ordinary `llama-bench` processes ran in balanced order `A B B A A B B A`.
There were four independent processes per source and three full-prefill samples
inside each process. CPU-KV placement and every unrelated opt-in were omitted.
The workload portion of the whole-command GPU-lock invocation was:

```bash
llama-bench -m QWEN38_MODEL -p 29398 -n 0 -r 3 --progress --kv-memory \
  -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
  -ngl 999 -sm none -mg 0 -fa on -b 1024 -ub 512 -o jsonl
```

| source | independent process means (t/s) | mean |
|---|---|---:|
| dense | 1616.8823, 1605.0874, 1605.5986, 1604.2560 | 1607.9561 |
| compact | 1609.4855, 1608.6642, 1606.5692, 1607.2940 | 1608.0032 |

The compact point is **+0.00293%**. The process-level Welch 95% interval is
`[-0.5735%, +0.5794%]`; the paired point is +0.00380% with interval
`[-0.4980%, +0.5056%]`. This is neutral, not a speedup claim, and it rejects
the prior reproducible negative point for the older frozen consumer.

Dense/compact device compute reservations were 529,530,880 / 529,827,968 B;
CUDA-host compute was 51,417,120 / 21,270,560 B. VMM high-water was identical
at 14,632,960 B live and 14,680,064 B mapped. The small device-reservation
increase did not cross the sampled process-VRAM bin, so this specific
`llama-bench` graph is not presented as a device-memory saving.

The analysis JSON SHA-256 is
`8629b7f8fa9ec4c0b87e5731d660df56a72846390a8c0e1688a8e1e4b3711c75`.
Raw processes and identities are under
`/tmp/beellama-pr7-redesign-20260822/symbolic-base-prefill-abba`.

### 30K serving decode and two-turn exactness

Six clean `llama-server` processes ran `A B B A A B`. Each process handled two
fully reprocessed 29,398-token prompts followed by 512 deterministic tokens.
The server command omitted all CPU-KV placement options and unrelated policies:

```bash
llama-server --model QWEN38_MODEL --ctx-size 30000 --parallel 1 \
  --cont-batching --kv-unified --batch-size 1024 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 --threads 3 --threads-batch 24 \
  --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --n-gpu-layers 999 --fit off --split-mode none --main-gpu 0 \
  --flash-attn on --cache-ram 0 --seed 1234 --slots --metrics \
  --host 127.0.0.1 --port PORT --verbosity 4
```

| source | independent process decode means (t/s) | mean |
|---|---|---:|
| dense | 43.59390, 43.57164, 43.58577 | 43.58377 |
| compact | 43.80674, 43.80563, 43.80304 | 43.80514 |

Compact decode is **+0.50792%**, with process-level Welch 95% interval
`[+0.44607%, +0.56977%]` and paired interval
`[+0.44403%, +0.57182%]`. This satisfies the strict positive-decode gate.
All 12 timing-stripped responses were byte-identical, SHA-256
`9d84c444c73dc1b16c420366fe2f7223d561cea2a56ab7342d14b7bf995efad1`.

Peak process VRAM was 14,532 MiB dense and 14,504 MiB compact (**-28 MiB**).
Device/CUDA-host compute reservations fell from 227.78/49.78 MiB to
198.28/20.28 MiB. Peak `VmRSS` was about 20 MiB lower in compact processes;
`VmHWM` was effectively equal and `VmLck` was zero. `VmLck` is not a pinned
CUDA-host-memory counter. The decode analysis SHA-256 is
`7d71872faac74886b5aba8c353b9ce2c7bb2965cef8ccb9836546fc0c0863fd0`.

### 4K lifecycle check

One dense/compact/dense server bracket used two fully reprocessed deterministic
turns per process. Dense prompt means were 1810.8130 and 1803.6358 t/s;
compact was 1805.1778 t/s, or -0.113% versus the dense midpoint. Dense decode
means were 49.8430 and 49.8794 t/s; compact was 49.8589 t/s, or -0.0046%.
Both are neutral at this small bracket, not wins or regressions.

All six timing-stripped outputs were byte-identical, SHA-256
`0b683f56dbd574065f1011a3c43d0a46ad75eebaf8e2b134bd575e94ef6df4cf`.
Peak process VRAM was 13,564 MiB dense and 13,560 MiB compact (**-4 MiB**).
Device/CUDA-host compute fell from 126.28/24.28 MiB to 122.28/20.28 MiB.
Maximum `VmRSS` varied by about 10 MiB, `VmHWM` was similar, and `VmLck` was
zero.

### 128K CPU-KV composition and strict decode gate

The deep shape used CPU-KV composition only because it was necessary to fit.
Its only extras were `--no-kv-offload 1 --kv-cpu-pinned
--recurrent-state-offload`; live/phase workspace, native-Q8 permission,
speculation, and unrelated policies were omitted. Six processes ran
`A B B A A B`, each with five decode repetitions after a 131,072-token fill:

```bash
llama-bench -m QWEN38_MODEL -p 0 -n 128 -d 131072 -r 5 \
  --no-warmup --progress --kv-memory -ctk q8_0 -ctv q8_0 \
  -t 3 -C 0x7 --cpu-strict 1 --poll 100 -ngl 999 -sm none -mg 0 \
  --no-kv-offload 1 --kv-cpu-pinned --recurrent-state-offload \
  -fa on -b 1024 -ub 512 -o jsonl
```

| source | independent process decode means (t/s) | mean |
|---|---|---:|
| dense | 8.657314, 8.655378, 8.657346 | 8.656679 |
| compact | 8.689455, 8.688944, 8.688069 | 8.688823 |

Compact decode is **+0.37131%**, with process-level Welch 95% interval
`[+0.34472%, +0.39791%]` and paired interval
`[+0.33042%, +0.41221%]`. This reverses and supersedes the prior negative
single-compact result for the older consumer.

| resource | dense | compact | delta |
|---|---:|---:|---:|
| sampled process peak | 14,314 MiB | 14,194 MiB | **-120 MiB** |
| synchronized CUDA context | 14,993,719,296 B | 14,867,890,176 B | -120 MiB |
| synchronized CUDA peak | 15,041,953,792 B | 14,916,124,672 B | -120 MiB |
| device compute | 1,042,342,784 B | 918,348,672 B | -118.25 MiB |
| CUDA-host compute | 155,750,432 B | 21,270,560 B | -128.25 MiB |
| pinned CUDA-host context | 4,572,315,648 B | 4,572,315,648 B | 0 |
| resident KV | 4,590,141,440 B | 4,590,141,440 B | 0 |
| VMM live/mapped high-water | 14,632,960 / 14,680,064 B | same | 0 |

Peak `VmHWM` was approximately 14,586,xxx KiB for both forms and `VmLck` was
zero. Current RSS varied during teardown, so no pageable-memory transfer is
inferred from it. The analysis SHA-256 is
`7bc4d26c5f97fe4256f9f7f6cf4f8fdb63887a26015dc818e18219a308c12660`;
the verified artifact-hash manifest hashes to
`a2f302d925e160991b817a76025cf959489f9a8fe91e6da9df8d5cfd40a4bd25`.

### Source-matched profiler findings

Profiler runs were separate from every timing estimate. NCU directly targeted
the absolute `llama-bench` binaries; no wrapper or shell was its target. Both
128K captures used graph-node mode, the NSYS-proven vector name,
`--launch-skip 1024`, `--launch-count 1`, and the full metric set:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '/usr/bin/ncu --target-processes application-only --replay-mode kernel \
   --graph-profiling node --kernel-name-base function \
   --kernel-name regex:flash_attn_ext_vec --launch-skip 1024 \
   --launch-count 1 --set full --force-overwrite --export REPORT \
   --page raw --csv /absolute/path/to/llama-bench \
   -m QWEN38_MODEL -p 0 -n 128 -d 131072 -r 1 --no-warmup \
   --progress --kv-memory -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 \
   --cpu-strict 1 --poll 100 -ngl 999 -sm none -mg 0 \
   --no-kv-offload 1 --kv-cpu-pinned --recurrent-state-offload \
   -fa on -b 1024 -ub 512 -o jsonl'
```

| selected Q8_0/Q8_0 vector launch | dense | compact | compact delta |
|---|---:|---:|---:|
| duration | 898.752 us | 869.184 us | **-3.290%** |
| registers/thread | 254 | 249 | -5 |
| achieved occupancy | 16.178% | 16.173% | neutral |
| executed instructions | 254,040,576 | 243,577,440 | **-4.119%** |
| divergent branch targets | 0 | 1,632 | +1,632 |
| branch-target uniformity | 100% | 99.459% | -0.541 points |

Compact slightly increased long-scoreboard and MIO stall ratios, but reduced
short-scoreboard, wait, dispatch, and math-pipe stall ratios. The measured
fact is that the net selected launch is faster with fewer instructions and
registers. The causal interpretation is that one tile-level boundary branch
is cheaper than repeatedly loading and reconstructing bounds in every K tile;
NCU cannot prove that balance on untested GPU generations.

The dense/compact report SHA-256 values are
`8c2cbdd5aea41869b3cd6823a4f07048d77f2f05b5dd067ba88d53ae8aca75d6`
and
`62848baad55e62b7da6f55f31996814bc9296b092e0644704449c4feae6c7bea`.
Raw CSV hashes are
`947be9b989c537f46ae941d65ea86b9678aa774d50428efc3f73aa60fada6dff`
and
`056f5f2e545f3e87cfc50e00e51510443ed96f9f363ed7afc2e0c0e1f20bface`.
Artifacts are under
`/tmp/beellama-pr7-redesign-20260822/symbolic-final-ncu-128k`.

The separate source-matched 29K F16 prefill launch was 6.637 / 6.420 ms dense
versus compact (**-3.27%**), 701.16 / 686.15 M instructions (**-2.14%**),
243 / 248 registers, and 16.41% / 16.14% achieved occupancy. Compact had more
divergent branch targets (872 versus 480), but the lower instruction count won.
That one profiled launch is consistent with, but does not replace, the neutral
ordinary full-prefill bracket.

### Default KVarN compatibility

After the performance source was frozen, the default KVarN-enabled Release,
SM120, CUDA-FA build was rebuilt at `--parallel 6` for
`test-backend-ops`, `llama-bench`, `llama-server`, and `llama-perplexity`.
The build completed; the only diagnostic was the existing unused AMD helper
warning. Final identities were:

| artifact | SHA-256 |
|---|---|
| `libggml-cuda.so` | `d47486d86cddf52a35e3fe7a16edf5cd06edc4d5c4036cb84bbb7c56c6910850` |
| `test-backend-ops` | `011820a93f52af6f35e27276b4107b6db7e87dbe4e8fc0be358d72c0f16b194e` |
| `llama-bench` | `48fe4e4e56169a26546d8f6cd8a4137744ff7dba3482cfa8d66acdd44803cb2e` |
| `llama-server` | `49fa35909f782cab1bf68b08c60d5dd9fa0d77dc17d186a023b78044e24815b8` |
| `llama-perplexity` | `0a73bb071322119838c3bd77aa8b98e1fa4ddbedea3375f47139ff0657b882bc` |
| `CMakeCache.txt` | `a771ab7a14336beac002fe07b6df4c21400bb68fc438668b4958f283ec78e532` |

Its fixed-seed CUDA oracle passed 4/4. Seven adjacent guards passed 7/7:
backend-op seed stability, CUDA graph source properties, generated vector
dispatch, batch allocation, CUDA attention route policy, CUDA vector policy,
and allocator coverage.

### Final disposition and limits

The consecutive-bound redesign is accepted for publication on PR 7. It keeps
the automatic compact replacement and dense fallback policy unchanged while
removing the repeated consumer work responsible for the older implementation's
negative prefill/decode points. The strongest ordinary evidence is neutral at
4K and 29,398-token prefill, strictly positive at 30K serving decode, and
strictly positive at 128K CPU-KV decode. PPL and deterministic outputs are
exact. Device-memory savings remain depth-scaled: 4 MiB at 4K, 28 MiB at 30K,
120 MiB at 128K, and the unchanged-source 224 MiB load-only result at 240K.

The implementation is not claimed to be faster on every GPU architecture.
Only the RTX 5070 Ti/SM120 production path has source-matched performance and
profiler evidence. Older GPUs retain the same capability/layout selection and
generic dense fallback, and the change reduces integer work rather than adding
an architecture-specific instruction sequence, but cross-generation speed is
an inference until measured. No threshold routing is used to manufacture the
result.

The local test harness remains deliberately uncommitted because it mirrors a
production CUDA header and is an investigation tool, not product surface. Raw
artifacts remain outside Git. The isolated PR commit contains only the compact
causal representation/consumer changes, partial-tile exactness coverage, and
this ledger update; staged work from unrelated migrations is excluded.

## Post-redesign vector tile retirement

### Scope and decision rule

After the consecutive-bound redesign passed its final gates, the user
authorized one further evidence-guided improvement. The previously discussed
affine/operator representation was a suggestion, not prescribed guidance. It
was therefore screened against smaller changes that attack the measured cost
directly. The source comparison remained exact dense
`f6341a15779eb58fe6ad9e1b890e331c32b676c7` versus compact parent
`f92c2de62bcc11abed0cff8b10a848fc8d97f29d`. No native-Q8, live-workspace,
phase/depth routing, telemetry policy, speculative path, public runtime flag,
user CMake option, optional dependency, model-family condition, or
architecture-name condition was added.

The candidate loop remained bounded: fixed-seed equivalence, then the local
production-header Q8_0/Q8_0 D=256 `ncols=1` screen at 30K/64K/128K, then a
single production build only for a clearly non-regressive candidate. The
first production candidate that passed strict process-level 4K/30K/128K
decode, exact output, and PPL was frozen. Profiling was performed afterward in
separate processes and is not timing evidence.

| hypothesis | minimal change | exactness and direct timing | disposition |
|---|---|---|---|
| batch compact `KV_max` conversion | replace the existing one-thread-per-query-tile compact prepass with a wider/batched launch | static review only | rejected: the production `--ubatch-size 512` prefill has Q below the `Q >= 1024` prepass threshold, so it cannot address the measured prefill path |
| share the first compact bound across the vector block | load the I64 bound once per block through existing shared synchronization | bit-exact; candidate time versus frozen was -0.045% at 30K, **+0.016%** at 64K, and -0.031% at 128K | rejected as neutral; the bound load was not the remaining dominant cost |
| new affine/operator descriptor | add another internal representation/operator around the already consecutive write indices | not implemented after the prior two findings | rejected as unnecessary surface: removing/centralizing descriptor loads did not produce a direct win, while the existing compact contract already proves the needed affine relation |
| local compact vector upper bound | round the widest valid query's compact prefix to the vector K-tile size and use it as `k_VKQ_max` | bit-exact; clearly faster at all screened depths and tails | **accepted and frozen** |

The first shared-bound build command mistakenly invoked missing
`/usr/bin/time`; its pipeline retained the prior binary. That attempt is
explicitly excluded. The corrected build took 2.601 s and retained 249
registers/thread, one barrier, and 4,352 B shared memory. Its four-process
ABBA direct result above is preserved under
`/tmp/beellama-pr7-further-20260822/shared-bound-screen` as rejected evidence.

### Accepted implementation and direct screen

The vector kernel already held the final compact prefix bound for every query
in its tile. For decode (`ncols=1`), however, the optional host-side `KV_max`
prepass is intentionally not launched, so the vector loop still visited the
full allocated K capacity and predicated the causally masked tail. The
accepted change derives the widest valid compact bound locally, rounds it to
the existing vector `nthreads` K-tile, clamps it to `ne11`, and uses that as
the loop limit. The existing per-key predicate still handles the final partial
tile. It adds no kernel launch or allocation and changes no dense
specialization.

The optional compact `KV_max` path remains compatible. When Q is at least
1,024 or there are multiple sequences, its prepass derives the same final
compact prefix rounded to `FATTN_KQ_STRIDE`; the local vector limit is equal or
tighter because it rounds the same proven bound to the vector tile. Compact
eligibility continues to prove consecutive write indices and supported KV
layout before either path is selected.

The final `fattn-vec.cuh` SHA-256 is
`dc89e8254eafd9dca421ffc51f54fdf7c056dc4629d4e746c7599114955a7416`.
The exact-production-header local harness compiled in 2.583 s; its binary is
`ab62affe75e5ac028b37324a97fe17d0e04f399b93acc5903fb22ac99e50034e`.
It retained 249 compact registers/thread, 254 dense registers/thread, one
barrier, and 4,352 B shared memory. Four independent processes per source ran
800 warmed interleaved samples per form; every output and metadata comparison
was bit-exact:

| K span / hidden tail | frozen median | accepted median | accepted time delta |
|---|---:|---:|---:|
| 30,000 / 0 | 194.704 us | 193.728 us | **-0.501%** |
| 30,000 / 113 | 195.072 us | 194.232 us | **-0.431%** |
| 65,536 / 0 | 442.688 us | 440.336 us | **-0.531%** |
| 65,536 / 113 | 443.096 us | 440.832 us | **-0.511%** |
| 131,072 / 0 | 880.640 us | 876.272 us | **-0.496%** |
| 131,072 / 2,177 | 881.872 us | 862.424 us | **-2.205%** |

The larger partial-tail result is expected: the accepted limit retires every
fully masked vector tile after the causal boundary rather than paying the
predicate for allocator padding. These direct results are screening evidence,
not model-level throughput claims. Their 48-log aggregate SHA-256 is
`1d3ad20b591a9401a755686d64aa2b7277b6b7ac08765f8b1df7c5b97cfbc1a1`;
artifacts are under
`/tmp/beellama-pr7-further-20260822/local-kvmax-screen`.

### Production correctness, quality, and exact output

The configuration-matched performance build remained Release, SM120,
CUDA-FA-on, native-on, and KVarN-off. The accepted CUDA library SHA-256 is
`91bb0b23abb604178f13c2da2ffdac9aa0c033d32ebeea9fe7b1e2848127d5ee`;
`llama-bench` remained
`990139351acda670b8867e919401245d1086ade3d553e6a04e4f95c4e611e4d7`.
The exact dense f634 library and benchmark remained
`b6a035b252ee1b1765261e2e079967994075065dd08117d6a1e8e02a9553b196`
and
`497de09c880d0c0c9539f6e8347ce789023b3de76a25bcf45c211672f43ad1d2`.

The fixed-seed CPU and CUDA compact-versus-dense oracles both passed 4/4 for
`nb=1,3,17,65`. A fresh two-turn candidate server run reprocessed the exact
29,398-token request twice and generated 512 deterministic tokens each time.
Both responses match the existing dense and prior compact standardized output
SHA-256
`3c4940d2a701f831ffb639644a42db76ae92b25a1a3bffa705e629f6d679e73a`.
Peak process VRAM remained 14,504 MiB, `VmRSS` 1,689,624 KiB,
`VmHWM` 14,589,192 KiB, and `VmLck=0`; this preserves the 28 MiB serving
saving and is not a new memory-representation claim.

Because production source changed, matching-batch PPL was rerun in a fresh
locked process with Q8_0/Q8_0, `-c 4096 -b 512 -ub 256`, and four chunks:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'build-compact-fast-screen-cuda/bin/llama-perplexity \
   -m QWEN38_MODEL --file CC_CORPUS --ctx-size 4096 \
   --batch-size 512 --ubatch-size 256 --chunks 4 \
   --cache-type-k q8_0 --cache-type-v q8_0 \
   --threads 3 --threads-batch 24 --cpu-range 0-2 \
   --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
   --n-gpu-layers 999 --split-mode none --main-gpu 0 \
   --no-kv-offload --kv-cpu-pinned --flash-attn on'
```

It printed the exact established chunk sequence `1.9315, 2.1279, 2.2498,
2.1674` and final `PPL = 2.1674 +/- 0.03849`. The log SHA-256 is
`07ddd3d135f466ad26484f209b9c88786bbbaf54473dd3c90ab8cfadf8c26bb5`;
there is no PPL increase.

### Ordinary production decode

The 4K and 30K shapes used plain GPU-resident Q8_0/Q8_0 cache with every
unrelated opt-in omitted. The 128K shape used only the placement necessities
`--no-kv-offload 1 --kv-cpu-pinned --recurrent-state-offload`; it remains
CPU-KV composition evidence. All three used clean processes, native llama
affinity, and ordinary timing outside profilers.

| depth | order / repetitions | dense process means | compact process means | compact effect and 95% CI |
|---:|---|---|---|---|
| 4,096 | A B B A A B; 10 samples/process | 49.764924, 49.759558, 49.742826 | 49.828992, 49.819227, 49.803566 | **+0.12359%**, Welch `[+0.06778%, +0.17939%]`; paired `[+0.11217%, +0.13500%]` |
| 30,000 | A B B A A B; 5 samples/process | 43.960827, 43.869221, 43.864071 | 44.130984, 44.111277, 44.093114 | **+0.48693%**, Welch `[+0.21416%, +0.75970%]`; paired `[+0.26889%, +0.70511%]` |
| 131,072 | A B B A A B; 5 samples/process after fill | 8.667775, 8.667047, 8.668555 | 8.710877, 8.709349, 8.711315 | **+0.49287%**, Welch `[+0.46835%, +0.51740%]`; paired `[+0.48143%, +0.50432%]` |

The benchmark command shape was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '{dense-or-compact}/bin/llama-bench -m QWEN38_MODEL \
   -p 0 -n 128 -d DEPTH -r REPEATS --no-warmup --progress --kv-memory \
   -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
   -ngl 999 -sm none -mg 0 PLACEMENT -fa on -b 1024 -ub 512 -o jsonl'
```

The compact representation's resource deltas are unchanged by this kernel
loop-limit refinement. The accepted rows retain approximately -4 MiB at 4K,
-28 MiB at 30K serving, and -120 MiB at 128K. At 128K, dense/compact device
compute remained 1,042,342,784 / 918,348,672 B and CUDA-host compute
155,750,432 / 21,270,560 B; pinned CUDA-host context (4,572,315,648 B),
resident KV (4,590,141,440 B), ordinary host context/compute (zero), and VMM
live/mapped high-water (14,632,960 / 14,680,064 B) were identical. `VmLck`
was zero and is not used as a pinned-memory proxy. The 4K and 128K analysis
JSON hashes are
`2710a724faf3f11197b7680711f7f3154cbbdc2f8e8926f51ea21db771b346bc`
and
`a5dbcde60f72c8f3a1d278f04280f8cca53ad93143c503458a7831b6e6cecc00`.

The production 29,398-token prefill route is F16 MMA, not the modified vector
consumer. Its representation, selected kernel source, and previously neutral
four-process prefill evidence are unchanged; no prefill result is relabeled as
new evidence. The accepted source change cannot phase- or depth-switch into
that route.

### Source-matched NCU and final compatibility

After the ordinary timing was frozen, one paired full NCU capture directly
targeted each production `llama-bench` binary. NSYS had already established
the Q8_0/Q8_0 vector kernel and 2,048 launches for this workload; graph-node
mode, `--launch-skip 1024`, and `--launch-count 1` selected one matching 128K
launch. The exact direct-target form was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '/usr/bin/ncu --target-processes application-only --replay-mode kernel \
   --graph-profiling node --kernel-name-base function \
   --kernel-name regex:flash_attn_ext_vec --launch-skip 1024 \
   --launch-count 1 --set full --force-overwrite --export REPORT \
   --page raw --csv /absolute/path/to/llama-bench \
   -m QWEN38_MODEL -p 0 -n 128 -d 131072 -r 1 --no-warmup \
   --progress --kv-memory -ctk q8_0 -ctv q8_0 -t 3 -C 0x7 \
   --cpu-strict 1 --poll 100 -ngl 999 -sm none -mg 0 \
   --no-kv-offload 1 --kv-cpu-pinned --recurrent-state-offload \
   -fa on -b 1024 -ub 512 -o jsonl'
```

| selected vector launch | dense | accepted compact | compact delta |
|---|---:|---:|---:|
| duration | 899.232 us | 859.456 us | **-4.423%** |
| registers/thread | 254 | 249 | -5 |
| achieved occupancy | 16.20% | 16.21% | neutral |
| executed instructions | 254,040,576 | 243,331,296 | **-4.216%** |
| branch uniformity / divergent targets | 100% / 0 | 99.46% / 1,632 | divergence cost added |
| DRAM throughput | 490.47 GB/s | 511.73 GB/s | +4.34% |

Compact long-scoreboard and MIO stall ratios rose slightly (1.32 to 1.33 and
0.60 to 0.72), while short-scoreboard and wait fell (0.34 to 0.17 and 0.62 to
0.59). The measured causal finding is that retiring masked tail tiles removes
4.22% of executed instructions and wins despite modest added boundary
divergence. This explains the SM120 result; it does not prove speed on an
untested GPU generation. Dense/compact report SHA-256 values are
`86901d8364722899c42727ace4afa380bbc4d96b709d774c6f42584124cf0e7a`
and
`0625d2ee67c8a3bbec0c551f1c37cc6943af6ab7332aa0acae766535aa06ef72`.

Finally, the default KVarN-enabled Release/SM120/CUDA-FA build completed at
`--parallel 6`. Its CUDA library SHA-256 is
`f45adc5cc62d2c77e9953c2208a3e777047d8819ab3691cf15ae473cd1161178`;
the CMake cache remained
`a771ab7a14336beac002fe07b6df4c21400bb68fc438668b4958f283ec78e532`.
The same seven adjacent static/CPU guards passed 7/7, and the final CUDA
compact-versus-dense oracle passed 4/4. Their log hashes are
`7d4ea82b3678574f4c53cebb27d501aaac0a47db7a087eb00ae5618c35cc935d`
and
`81da5c1121b6851b7a1a28f996b01e7a40b2b1b20a66dcc3150cf65c37939b8e`.

All GPU-capable commands were fresh whole processes under
`flock /tmp/beellama-single-gpu.lock -c`; no command used `taskset`, and NCU
targeted llama binaries directly. The complete 233-file external artifact
manifest is `/tmp/beellama-pr7-further-20260822/SHA256SUMS`, SHA-256
`01d03357defd6f2603fcb1638f4ed5a2a65180b119b7bd90f5d2e80d1e833f4e`.

### Disposition

The local compact vector upper bound is accepted. It is a 12-line refinement
of the existing compact vector specialization, not a new representation or
operator. It makes 4K, 30K, and 128K decode strictly positive on the tested
RTX 5070 Ti while retaining exact PPL/output and the existing depth-scaled
memory savings. The 30K point is statistically the same performance class as
the prior accepted compact consumer rather than a claim of an additional
end-to-end gain; the direct and 128K results show the added tail retirement's
specific merit. The strongest remaining limitation is cross-generation GPU
performance, which is still unmeasured and must not be inferred from SM120
alone.

## 2026-08-22: PR 11 toolkit inheritance and final interaction gates

This section records the final PR validation and base refresh. The user first
authorized one published source-history change: merge exact shared KV-offload tip
`748c1df5bad4dae0d1f59f65997cc7e1f3f3125b` into the existing PR 7 history
without rebasing or rewriting its causal commits. The resulting merge is
`d71ddf93215be824dd3e4f7ef5c7551fbc32af37`, with parents `3fd817df9` and
`748c1df5b`, and is published as the head of PR 7.

### Base classification and source identity

The previously tested shared base was
`f6341a1572e54206a025bc376dd32958a5d33cbb`. The complete `f6341a..748c1d`
delta consists of AGENTS/protocol documentation, the manifest-driven feature
validation toolkit, its example manifest and tests, and CTest registration: 15
paths and 7,336 insertions. Direct tree comparison found no change to PR 7's
production, CUDA-kernel, public API, CLI, or build-option behavior. The
non-document causal delta against the shared base retained SHA-256
`3c4854dd89fcab72ad1c561da0be272cb1e045a5c0461ba632f53ffb4f5a1dbb`
before and after the merge. It was therefore valid to retain the exact-source
PPL, clean-process performance/resource, deterministic-output, VMM, and NCU
evidence above rather than relabeling or repeating it.

The inherited toolkit passed `py_compile`, its 64 Python unit tests, its
focused CTest registration, and the eight adjacent static regressions. The
merged Release CUDA/FA/native/SM120/KVarN-on configuration built
`llama-server`, `llama-bench`, `llama-perplexity`, and `test-backend-ops` at no
more than six parallel jobs. CPU and CUDA compact-versus-dense fixed-seed
oracles each remained 4/4 bit-exact for `nb=1,3,17,65`. Cold CUDA compilation
was serialized with `/tmp/beellama-cuda-build.lock`; GPU-linked tests and
runners owned `/tmp/beellama-single-gpu.lock` for their complete lifecycles.

The production fallback audit also remains source-proven. Compact construction
in `build_attn_inp_kq_mask()` requires FlashAttention, offloaded attention,
scheduler-wide backend support for the causal-prefix capability, and
`can_use_compact_causal_mask()` layout proof. The latter rejects non-causal
attention, ALiBi, SWA/tails, multiple streams/sequences, empty or malformed
batches, explicit attention bias, nonmonotonic positions/cells, cache holes,
wrong sequence ownership, and nonconsecutive physical writes. Failure of any
condition constructs the established dense F16/F32 mask. KVarN's context
returns false explicitly, so it also receives dense fallback. There is no
public control, context threshold, model name, or architecture-name routing.

### Same-geometry target-plus-MTP and merged live workspace

The PR-owned interaction gate compared exact dense `748c1df5b` with compact
`d71ddf932`, with the merged opt-in live-context workspace enabled equally on
both sides. It used one-slot MTP6, `p_min=0.85`, fixed seed 1234,
Q8_0/Q8_0 CPU-pinned target and draft caches, target and draft effective
ubatch 512, full recurrent planes, phase-aware workspace off, and 1,000
generated tokens. The entire exact runner invocation was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'python3 scripts/mtp-exactness.py \
   tmp/pr7-validation-20260822/mtp-live-1k-manifest.json'
```

The manifest and each case's `server-command.txt` are the exact executable and
argument record. Contract, prompt tokens, token stream, and content were exact.
Dense and compact both made 470 draft attempts and accepted 428. Both output
token SHA-256 values were
`842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00`;
content SHA-256 was
`41dd817295f51516f3750049cfe3ecd2b5de9ae0f4d08df7a58a1408318f3bb2`.

| PR-owned MTP/live case | startup / peak VRAM | RssAnon / RssFile / RssShmem | VmHWM / VmRSS / VmLck |
|---|---:|---:|---:|
| dense `748c1d` | 14,704 / 14,736 MiB | 606,480 / 878,200 / 235,024 KiB | 14,551,004 / 1,719,704 / 0 KiB |
| compact `d71d` | 14,704 / 14,734 MiB | 619,120 / 880,556 / 230,928 KiB | 14,553,616 / 1,730,604 / 0 KiB |

The compact path saved 2 MiB at the sampled peak. CUDA-host compute remained
fixed at 20.28 MiB for compact while dense grew from 20.53 to 22.28 MiB as the
live reserve crossed 256/512/1,024/2,048 KV rows. Both sides reported the same
target/draft capacity transitions. The one-process prompt/decode point values
are not performance evidence; only exactness, attempts/accepts, topology, and
resource telemetry are claimed from this screen.

### Temporary canonical pool/growth interaction

For interaction evidence only, the exact uncommitted production diffs from
`/home/gencoolpc/beellama-kv-store-stage-pool` and
`/home/gencoolpc/beellama-live-context-bounded-growth` were applied temporarily
and identically to the dense and compact build sources. Their source diff
SHA-256 values were
`017c71898e54aef9ea0b8cc8047c233374fbed5c30daabcab0d9d6035b3107af`
and `6ede48268965459654d922ea24c417473478573965b46497d7805bf59a673609`.
Neither named worktree was modified. Both borrowed diffs were removed from PR
7 afterward, and its tracked production files again match `d71ddf932` exactly.

The same MTP/live gate was rerun under the whole-command GPU lock with:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'python3 scripts/mtp-exactness.py \
   tmp/pr7-validation-20260822/mtp-combined-live-1k-manifest.json'
```

It again produced exact contract, prompt, 1,000-token stream, and content, with
470 attempts and 428 accepts on both sides. Startup was 14,688 MiB for both;
peak process VRAM was 14,720 MiB dense and 14,716 MiB compact. The single-run
prompt/decode point values are not used as performance claims.

One deeper interaction used a 97,697-token prompt at context capacity 140,000,
followed by an independent fully reprocessed short next turn. The only enabled
extras were the CPU-KV placement necessities, merged live-context workspace,
and the two canonical staging/growth candidate sources. Phase-aware workspace,
native-Q8 permission, speculation, and unrelated policies were absent. Exact
commands reside in the manifest and case `server-command.txt` files; the
whole-run command was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'python3 scripts/mtp-exactness.py \
   tmp/pr7-validation-20260822/combined-98k-manifest.json'
```

Both 64-token responses and content were exact. Long-turn token SHA-256 was
`8dc2caa05af802d4e381119257cabe2e5917269a89b84e685635d5a5f95c4181`,
next-turn SHA-256 was
`0cb5eb25dd4a69ef1e8fd89cdfdd349c6c80449508e9a1c132964f5878715b51`,
and aggregate SHA-256 was
`e4875a1fda5fcfbfe6e73c0efec901c5cb890b29d41cebeccd7f277d00ebcfd0`.

| 98K pool+growth+live case | startup / prefill peak / next-turn resident VRAM | peak RssAnon / RssFile / RssShmem | peak VmHWM / VmRSS / VmLck |
|---|---:|---:|---:|
| dense | 13,428 / 14,068 / 14,054 MiB | 735,584 / 916,472 / 4,892,452 KiB | 14,590,476 / 6,544,508 / 0 KiB |
| compact | 13,428 / 13,982 / 13,968 MiB | 743,208 / 917,432 / 4,794,148 KiB | 14,590,556 / 6,454,788 / 0 KiB |

Thus compact saved 86 MiB of process VRAM at both full-prefill peak and the
next-turn resident sample. `RssShmem`, the page-locked CUDA-host mapping here,
fell by exactly 98,304 KiB; ordinary anonymous RSS rose by 7,624 KiB and file
RSS by 960 KiB. `VmLck=0` is recorded but is not treated as a CUDA pinned-memory
counter. At the 98,304-row reserve, dense/compact device compute was
764.27/678.27 MiB and CUDA-host compute was 116.28/20.28 MiB. Both reported the
same 4,649.50 MiB CUDA-host KV buffer. The dense reserve took 96.724 ms over
eight transitions versus 22.189 ms for compact, but this one pair does not
establish a performance distribution. Prompt point values were 988.719 and
1,005.344 token/s; decode point values were 11.1070 and 11.1482 token/s.

This interaction run did not enable separate CUDA VMM instrumentation, so no
new combined-source VMM claim is made. For exact PR-owned source, the already
settled 128K evidence remains applicable: dense/compact device compute was
1,042,342,784/918,348,672 B, CUDA-host compute was
155,750,432/21,270,560 B, pinned KV was identical at 4,572,315,648 B, ordinary
host context/compute was zero, and CUDA VMM live/mapped high-water was
byte-identical at 14,632,960/14,680,064 B. The unchanged-source 240K load-only
evidence remains a startup/resident result, not a prefill peak claim.

### Matched quality after temporary composition

The temporary three-feature composition received one fresh matched PPL pair,
because this tested interaction quality rather than changing PR-owned source.
Each binary ran as a separate whole command under the GPU lock with this exact
argument shape (substitute the dense or compact build path):

```bash
{build}/bin/llama-perplexity \
  -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  -f /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt \
  -c 4096 -b 512 -ub 256 --chunks 4 -t 3 -tb 24 \
  --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  -ngl 999 -sm none -mg 0 --flash-attn on \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  --kv-gpu-layers 0 --no-phase-aware-workspace --live-context-workspace \
  --cache-type-k q8_0 --cache-type-v q8_0
```

Dense and compact printed the exact same chunk sequence
`1.9315, 2.1279, 2.2498, 2.1674` and final
`PPL = 2.1674 +/- 0.03849`. Log SHA-256 values are
`61e4ea65852aed4d31dda2beb36ade54c3f746ac624609159e12cf6a7897414b`
and `50f1e0f5bec42903d5a8c20f7c0c4ffa43d3d4549da1f6ef796aa2bc073f0f14`.
There is no PPL increase.

No profiler was launched for these gates. The PR-owned CUDA consumer and its
stable long-context effect were already directly profiled above. The two
temporary candidates change host-side staging ownership and reserve growth,
not the compact consumer kernel; exact interaction output and non-regressive
single-pair points exposed no new stable kernel question. Profiling would not
have been timed evidence.

### Artifacts and restoration

All new artifacts are under `tmp/pr7-validation-20260822`. Its 126-file
`SHA256SUMS` has SHA-256
`8f2f0ddec5bed15cca6eed7a0612a386045f87206053c17b105f471be4617db5`.
The principal manifest/provenance/summary/comparison hashes are:

| gate | manifest | provenance | summary | comparisons |
|---|---|---|---|---|
| PR-owned MTP/live | `e3482af6...` | `5d9129f0...` | `9f51b92e...` | `eafef1fc...` |
| combined MTP/live | `67c444ed...` | `98905f87...` | `f3262d7b...` | `96015dad...` |
| combined 98K | `23580e21...` | `2a818ce0...` | `6f263e03...` | `c9dd3298...` |

After restoring exact PR-owned source, the KVarN-off compact build was relinked
under `/tmp/beellama-cuda-build.lock`. Its `llama-server`, `llama-bench`,
`llama-perplexity`, `libllama.so`, `libggml-cuda.so`, and CMake cache SHA-256
values are respectively `6c1aec47...`, `99013935...`, `43c86f0f...`,
`b7bc98a7...`, `91bb0b23...`, and `12eee7b2...`; these match the accepted PR
source rather than the temporary interaction build.

### Exact `775450a6` policy refresh

At the first publication checkpoint, PR 7 was open, draft, cleanly mergeable,
based at `748c1df5b`, and headed by `d71ddf932`; its labeler check passed. The
remote base then advanced to
`775450a688437e8e9d32bb77ae466f7acb5d3577`, commit `build: make KVarN
compilation explicitly opt-in`. A later authorization closed that one-commit
gap with merge `aa6532b68c1e4b15bbd05d8f0aadef36c4979f9d`, whose parents are exact old
head `d71ddf932` and exact new base `775450a68`. Published causal commits were
not rewritten.

The incoming delta changes KVarN's fresh-cache CMake default, configure
messages, policy documentation, and two static tests. It changes no `.cpp`,
`.cu`, or `.cuh` runtime implementation. The only path modified on both sides
was `ggml/src/ggml-cuda/CMakeLists.txt`; it merged automatically. Relative to
the new base, that file differs by exactly PR 7's existing
`add_compile_definitions(GGML_CUDA_COMPACT_CAUSAL_MASK)` line. The root
`ggml/CMakeLists.txt` is byte-identical to the new base. A per-path blob audit
found every other causal production source identical to `d71ddf932`; its log
SHA-256 is
`9e723703d19920ce71384184c7d4d8a151c9a7bef1a95e7a04759f983437acd0`.

Three fresh configuration directories used Release, CUDA/FA on, explicit
SM120, native/NCCL/tests/examples/server/tools off, and no optional causal
feature controls. The default case omitted `GGML_CUDA_KVARN`; the other two
set it explicitly. All configuration compiler probes ran serially under
`/tmp/beellama-cuda-build.lock`:

| fresh build graph | cache values | native / portable / fast-decode KVarN sources | configure output |
|---|---|---:|---|
| default | `KVARN=OFF`, `ALL_QUANTS=OFF` | 0 / 0 / 0 | `OFF (minimal fresh-cache default)` |
| explicit default tier | `KVARN=ON`, `ALL_QUANTS=OFF` | 2 / 17 / 15 | `ON (15 fast-decode pairs)` |
| explicit expanded tier | `KVARN=ON`, `ALL_QUANTS=ON` | 2 / 17 / 36 | `ON (36 fast-decode pairs)` |

The graph-count record SHA-256 is
`fa6378e9daaedb38645297f7402dae029a4286512c4352ce19e1ff1522d978af`.
Default/15/36 configure-log SHA-256 values are respectively
`e235b9ed48806877d0a51b936b204de3d02e7047ac4c8242fa7fae7323e59848`,
`527968060df48718b4172c049957b90ed7c5e1682fbceedc66bf3e8bfa08dbd2`,
and `93694ec904256ed42120f86eca5083c67a61b9a715ca5e391d2be11afd475073`.

The existing isolated performance build was then reconfigured explicitly with
KVarN and all-quants off. Only the route/vector policy targets were requested,
at `--parallel 6`, under the CUDA build lock. The whole focused CTest process
ran under `/tmp/beellama-single-gpu.lock`:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'ctest --test-dir build-compact-fast-screen-cuda --output-on-failure \
   -R "^(test-cuda-fattn-route-policy|test-cuda-fattn-vec-policy|test-std-kv-tail-static|test-fattn-vec-dispatch-generated)$"'
```

All four tests passed. Configure, build, and CTest log SHA-256 values are
`990a8e097f1b84411a3e2f8b4740541245a9e38def1d12c12d0abfe16e5bb4cd`,
`f6154efbd9b2d49511d2bce4537cbde982715a41d1be5544ae58cd6ce6a40e6e`,
and `352ef4ee33bc9891556ffb1ec7f138298015a5cc7b0d995657e8d2a2f1260134`.
No CUDA device object was recompiled: the relevant object mtimes remain from
the accepted pre-refresh build. The relinked `libggml-cuda.so` and
`libllama.so` remain byte-identical at
`91bb0b23abb604178f13c2da2ffdac9aa0c033d32ebeea9fe7b1e2848127d5ee`
and `b7bc98a79d7a6bd8d40ab0f8fb45b50488fed2ba92d2a21e351d437490ee31c2`.
The CMake cache hash changed, as expected, because it now records the inherited
policy text/configuration; this is not a runtime-object change.

The 98K/128K, MTP, PPL, NCU, and NSYS campaigns were not repeated. Their exact
runtime sources and KVarN-off runtime libraries are unchanged, so another run
would duplicate settled evidence. Refresh artifacts are under
`tmp/pr7-refresh-775450a6`; the pre-merge ledger copy and patch remain there as
recovery evidence.

The evidence disposition remains positive for the tested SM120 hardware:
exact quality/output, positive repeated 4K/30K/128K decode, neutral production
prefill, depth-scaled device-memory savings, safe dense fallback, exact MTP
attempts/accepts, merged-live composition, and clean interaction with both
named future candidates. Cross-generation performance remains unmeasured, but
the exact current base is now inherited and its default-off/explicit-on CUDA
policy is verified. No remaining validation result requires PR 7 to stay
draft; draft removal is appropriate once the pushed head is cleanly mergeable
and its remote checks pass.
