# Compact causal-prefix design and evidence

Status: PR 7 is the isolated compact causal-prefix publication lane. Its final
main-based runtime gates passed at measured runtime head `222eec2e7`; review
readiness still requires the pushed head to pass GitHub checks and report
cleanly mergeable.
The complete superseded investigation transcript is available from Git at
`d86145f50:docs/compact-causal-mask-vram-investigation.md`; it is not a current
command source.

## Scope and current identity

PR 7 targets `beellama/main` and intentionally excludes native quantized
FlashAttention, KVarN implementation changes, workspace policy, allocator
policy, VMM policy, host staging, speculation, and development-branch features.
It adds no public argument, user CMake option, model or architecture check,
context threshold, or prompt/decode switch.

The official branch incorporated maintenance head
`4b86269fdf001de44dd96e9c9ae26a9e25091cca` through ordinary merge
`c542f137cbe34218aac8858016ca946e1fb4cb58`. The isolated MMA lifetime repair
is `1a3e6fcf3`; metadata consolidation is `af7393ef8`; compiled capability
coverage is `91bf37eb2`. These commits do not import PR 4's native-Q8 path.

## Representation and selection contract

Eligible graphs reuse the existing one-dimensional I64 K-write-index input as
an exclusive causal bound (`write_index + 1`). Consecutive queries therefore
have consecutive bounds. The graph passes that tensor directly to
`FLASH_ATTN_EXT`; it does not allocate a second descriptor or create
per-consumer views.

Selection is automatic and fail-closed. Compact construction requires all of
the following:

- causal FlashAttention with attention compute offloaded;
- every scheduled accelerator registering compiled causal-prefix support;
- one standard-KV stream and one sequence;
- nonempty, monotonic query positions and monotonic live cache cells;
- a hole-free physical prefix owned by the sequence;
- consecutive current writes whose physical index plus one equals the causal
  visibility boundary;
- no ALiBi, model-supplied attention bias, SWA, or exact-tail overlay; and
- a real batch that proves the layout, or the corresponding reserve topology.

Failure of any condition constructs the existing F16/F32 dense mask. KVarN
contexts explicitly reject the compact representation and retain their normal
path. HIP and MUSA do not register the capability because this PR has no
performance/correctness evidence for those backends.

The graph stores causal-prefix presence and its retained logical K span as one
optional value. Dense graphs have no prefix metadata; compact graphs cannot
carry a separate boolean/count pair that disagrees during graph reuse.

## Consumer contract

CPU and CUDA consumers interpret the same I64 descriptor. CUDA vector, tile,
F16 MMA, and optional `KV_max` preparation derive query bounds from the proven
consecutive layout.

The CUDA vector path deliberately retains compile-time dense and compact
specializations. Host dispatch selects
`ggml_cuda_flash_attn_ext_vec_case_dispatch<..., false>` or `<..., true>`;
there is no per-key runtime representation branch in the vector kernel. The
compact specialization loads bounds at tile scope, preserves an unmasked inner
path for fully visible tiles, and rounds the widest visible prefix to the
existing K-tile width so fully masked allocator padding is never visited.

F16 MMA shares one implementation for its established route. Its query-tile
first bound must outlive the dynamically shared Q/K/V/mask phases. The bound is
therefore held in an explicit block-owned `int32_t` shared object and broadcast
per warp. Borrowing mask-row padding for that scalar is forbidden because the
dynamic arena is aliased between phases.

No consumer changes cache-format eligibility. The existing standard vector
pair matrix remains available, and unsupported representations or routes fall
back rather than being reinterpreted.

## Accepted implementation decisions

The retained design followed bounded direct and production screening:

| question | decision |
|---|---|
| Separate public mask format or operator | Rejected; reuse the existing I64 write indices internally. |
| Runtime prompt/decode or depth routing | Rejected; eligibility is semantic and backend-capability based. |
| Vector runtime representation branch | Rejected; retain compile-time dense/compact specializations. |
| Per-consumer descriptor views | Rejected after allocator liveness showed that small early-lived views fragmented the compute arena. |
| Repeated per-K-tile bound reconstruction | Rejected; load once per query tile/warp and derive consecutive offsets. |
| Fully masked vector padding tiles | Retired with a local compact upper bound. |
| MMA bound in aliased dynamic padding | Rejected; use explicit block-owned shared storage. |
| Architecture-specific fast-path selection | Rejected; measurements limit claims but do not alter routing. |

The local direct-header harness used during design remains an external
investigation artifact. It is not shipped because it mirrors a production CUDA
header and would create a second maintenance surface.

## Accepted isolated evidence

These results belong to the final pre-main-refresh isolated source. They remain
historical evidence for those exact commits, not measurements of a later
source-bearing head. The final readiness section below records the fresh
main-based boundary.

### Correctness and quality

- Fixed-seed compact-versus-explicit-dense CPU and CUDA output was bit-exact
  for partial query tiles of 1, 3, 17, and 65 tokens.
- Deterministic two-turn serving output was byte-identical after timing fields
  were removed.
- Matching-batch perplexity used `-c 4096 -b 512 -ub 256 --chunks 4`. Both
  forms printed `1.9315, 2.1279, 2.2498, 2.1674` and final
  `PPL = 2.1674 +/- 0.03849`; the measured increase is zero.
- KVarN-off performance and KVarN-on compatibility builds passed their focused
  CUDA and adjacent graph/allocator tests. KVarN remained on dense fallback.

### Clean-process performance

Profiler processes are excluded from timing. Decode rows used three independent
processes per source in balanced `A B B A A B` order.

| gate | dense mean | compact mean | compact effect and process-level 95% interval |
|---|---:|---:|---|
| 4K GPU-KV decode, 10 repetitions/process | 49.75577 tok/s | 49.81726 tok/s | **+0.1236%**, Welch `[+0.0678%, +0.1794%]` |
| 30K GPU-KV decode, 5 repetitions/process | 43.89804 tok/s | 44.11179 tok/s | **+0.4869%**, Welch `[+0.2142%, +0.7597%]` |
| 128K CPU-pinned-KV decode, 5 repetitions/process | 8.66779 tok/s | 8.71051 tok/s | **+0.4929%**, Welch `[+0.4683%, +0.5174%]` |

The production 29,398-token prefill route is F16 MMA. Four processes per
source measured 1607.9561 tok/s dense and 1608.0032 tok/s compact: +0.00293%
with Welch interval `[-0.5735%, +0.5794%]`. This is neutral, not a speedup.

### Resource result

| lifecycle point | dense peak | compact peak | process-VRAM delta |
|---|---:|---:|---:|
| 4K serving | 13,564 MiB | 13,560 MiB | **-4 MiB** |
| 30K two-turn serving | 14,532 MiB | 14,504 MiB | **-28 MiB** |
| 128K CPU-pinned-KV decode | 14,314 MiB | 14,194 MiB | **-120 MiB** |
| 240K load to stable idle | 15,052 MiB | 14,828 MiB | **-224 MiB** |

At 128K, dense/compact device compute was
1,042,342,784/918,348,672 bytes and accelerator-owned host compute was
155,750,432/21,270,560 bytes. Pinned KV context and resident KV were identical
at 4,572,315,648 and 4,590,141,440 bytes. Ordinary host context/compute was
zero, and CUDA VMM live/mapped high-water was identical at
14,632,960/14,680,064 bytes. The 240K result is allocation-only startup/idle
evidence, not a prefill throughput or peak claim.

### Profiler finding

A direct-target NCU pair selected the production 128K Q8_0/Q8_0 vector launch:

| metric | dense | compact | compact effect |
|---|---:|---:|---:|
| duration | 899.232 us | 859.456 us | **-4.423%** |
| executed instructions | 254,040,576 | 243,331,296 | **-4.216%** |
| registers/thread | 254 | 249 | -5 |
| achieved occupancy | 16.20% | 16.21% | neutral |
| branch uniformity / divergent targets | 100% / 0 | 99.46% / 1,632 | modest divergence added |

Retiring masked tail tiles removes more work than the compact boundary adds on
the tested RTX 5070 Ti. Dense/compact report SHA-256 values are
`86901d8364722899c42727ace4afa380bbc4d96b709d774c6f42584124cf0e7a`
and `0625d2ee67c8a3bbec0c551f1c37cc6943af6ab7332aa0acae766535aa06ef72`.

## Interaction evidence

The isolated PR source was also composed temporarily with the then-current
target-plus-MTP and live-workspace paths. Exact 1,000-token MTP output retained
470 attempts, 428 accepts, output-token SHA-256
`842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00`,
and content SHA-256
`41dd817295f51516f3750049cfe3ecd2b5de9ae0f4d08df7a58a1408318f3bb2`
for dense and compact.

A separate temporary 98K pool/growth/live composition produced exact long and
short next-turn responses. It saved 86 MiB of sampled process VRAM at prefill
peak and next-turn residency and reduced the measured accelerator-host compute
reservation. Those borrowed features were removed afterward. This is
interaction evidence only; it does not broaden PR 7's source scope.

The verified 126-file interaction manifest is
`tmp/pr7-validation-20260822/SHA256SUMS`, whose SHA-256 is
`8f2f0ddec5bed15cca6eed7a0612a386045f87206053c17b105f471be4617db5`.

## Reproduction protocol

The current instructions in `docs/cpu-kv-offload-current-testing.md` override
older commands. Every CUDA build is serialized and capped at 12 jobs; every GPU
lifecycle is wholly owned by the single-GPU lock. Do not use `taskset` or
retired environment controls.

### Fresh isolated build

```bash
flock /tmp/beellama-cuda-build.lock -c \
  'cmake -S . -B build-pr7-official-cuda \
   -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
   -DGGML_CUDA_KVARN=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF \
   -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
   -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON \
   -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON'

flock /tmp/beellama-cuda-build.lock -c \
  'cmake --build build-pr7-official-cuda \
   --target llama-server llama-bench llama-perplexity test-backend-ops \
   test-cuda-fattn-route-policy test-cuda-fattn-vec-policy \
   --parallel 12'
```

Register each final CMake-built benchmark executable with
`scripts/feature-performance-validation.py register-build` before referencing
it from a validation manifest.

### Compiled capability and descriptor exactness

The compact operator filter requires the compiled CUDA backend to expose
`ggml_backend_flash_attn_causal_prefix_supported`; the test fails if the
capability is absent or false. It then compares compact and explicit-dense
results in one graph:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  'build-pr7-official-cuda/bin/test-backend-ops \
   -b CUDA0 -o COMPACT_CAUSAL_DESCRIPTOR_EQUIVALENCE -j 1 \
   --seed 0x6a09e667f3bcc909'
```

The current oracle includes D=256 F16 MMA shapes with and without GQA in
addition to the four partial-tile cases. Native-Q8-specific tests remain owned
by PR 4 and development composition.

### Perplexity

```bash
flock /tmp/beellama-single-gpu.lock -c '
  build-pr7-official-cuda/bin/llama-perplexity \
    -m MODEL -f CORPUS -c 4096 -b 512 -ub 256 --chunks 4 \
    -t 3 -tb 24 --cpu-range 0-2 --cpu-range-batch 0-23 \
    --cpu-strict 1 -ngl 999 -sm none -mg 0 --flash-attn on \
    --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
    --kv-gpu-layers 0 --no-phase-aware-workspace \
    --cache-type-k q8_0 --cache-type-v q8_0
'
```

Baseline and candidate must use identical `-b 512 -ub 256` values.

### Matched performance and resources

Use fresh processes, `--no-warmup --progress --kv-memory`, and identical model,
cache formats, affinity, batch/ubatch, placement, and build options. The 4K and
30K GPU-KV forms vary only depth; the 128K form adds only the CPU-pinned
placement needed to fit:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  BUILD/bin/llama-bench -m MODEL -p 0 -n 128 -d DEPTH -r REPS \
    -b 512 -ub 256 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
    -ngl 999 -sm none -mg 0 -fa on -ctk q8_0 -ctv q8_0 \
    --no-warmup --progress --kv-memory -o jsonl EXTRA_PLACEMENT
'
```

Use `DEPTH=4096` and `30000` with no placement extra. `llama-bench` requires an
explicit value for its placement option, so use `DEPTH=131072` with
`--no-kv-offload 1 --kv-cpu-pinned --recurrent-state-offload`. Prefill coverage
substitutes `-p 512 -n 0` while retaining the same depth and other arguments.
A single pair is a screen; small claims require balanced independent-process
brackets. A missing `1` lets the benchmark parser consume the following option;
such a context-creation failure is invalid setup, not evidence.

### Two-turn lifecycle runner

The retained runner has no embedded model path. Its arguments are:

```text
scripts/compact-mask-next-turn.sh \
  LABEL SERVER PORT CTX MODEL REQUEST_JSON OUTPUT_DIR
```

The entire invocation, including server, client, resource sampler, and teardown,
must be inside `flock /tmp/beellama-single-gpu.lock -c 'COMMAND'`. The runner
emits readiness and turn progress, command/binary/request identity, response
hashes, process VRAM, RSS/HWM, and `VmLck`. `VmLck` is recorded but is not a
CUDA pinned-memory counter.

As a future suggestion, if HIP or MUSA gains and validates the same compact
consumer semantics, retire the backend-private build distinction in favor of
those backends registering the capability directly. This is not current
guidance to expose an option or to claim cross-backend support: until that work
exists, the contained five-use guard is the honest capability boundary.

## Dev composition repair: native Q8 MMA shared-memory lifetime

This evidence was collected from the final working tree before its commit, so
it is not a result attributed to the clean base
`8f34a3355ca7519400b8a4d527672a7a282fac46`. The final
implementation-and-regression diff, excluding this document, has SHA-256
`1b74d7d161d3e5129affc41d12d29808faff048f6630cd2bde2ed4035e78f107`.
The build is Release, native CPU tuning, CUDA FlashAttention, SM120, native Q8
MMA on, KVarN off, and the default quant matrix. Hardware was an RTX 5070 Ti
with driver 610.57.04. Every GPU process, including every profiler process,
ran under `/tmp/beellama-single-gpu.lock`; every process was fresh.

### Cause and general repair

The compact and native-Q8 features were exact separately but produced NaNs
when composed on wide D=256 MMA tiles. The compact path cached its first causal
bound in the padding immediately after the first mask row. MMA intentionally
aliases shared-memory phases. Synchronous quantized K/V loading uses zero
pipeline stages and therefore a smaller K/V phase than staged F16; on affected
layouts, the old padding address falls inside active Q staging. Q loading
overwrote the causal bound, which could mark the entire attention tile masked
and make the softmax denominator zero. The original D=256/GQA6 oracle failed
at output index zero with CUDA NaN versus CPU `0.053088`; the four-chunk model
gate printed NaN for all four chunks.

The repair gives the scalar the lifetime it already semantically has: one
block-owned `int32_t` in static shared memory. Thread `(0,0)` loads it once.
At each use, lane zero of each warp reads the shared scalar and shuffle-
broadcasts it within that warp. The mask materializer derives its row-relative
first-masked index from that value. This is one ownership rule for all MMA tile
layouts; it does not borrow padding from any aliased Q/K/V/mask phase and has
no model, architecture-name, head-size, GQA, or layout exception. It adds no
runtime option, compile-time dependency, or new native-Q8 route.

The final source change is confined to the existing F16 MMA consumer and two
permanent regression registrations. It is 62 touched kernel lines (40 added,
22 removed) plus nine test lines. NCU reports 16 bytes of statically allocated
shared memory after hardware alignment, the same 33,792 dynamic shared bytes,
255 registers per thread, and 16.67% theoretical occupancy for compact and
dense launches. Device context allocation is not increased.

Two permanent equivalence cases cover D=256 native Q8 without and with GQA.
The final source passed all seven bitwise compact-versus-explicit-dense cases,
including CPU-oracle comparison and both new wide layouts. The final log
SHA-256 is
`bc0b9a244b970a8e3acbf02bfeac4c3008de9c8aaf346647522be78d0443c732`.
Seven adjacent allocator, graph, generated-dispatch, and CUDA route-policy
tests also passed 7/7. Temporary perf registrations were removed before this
final build and exactness run.

### Isolated performance disposition

The baseline for the wide explicit-dense case is the valid pre-redesign series
`81.15, 81.15, 81.27, 81.43, 81.43 us`, mean `81.286 us`. Measurements in the
82-us range came from rejected candidates and are not baseline evidence.

| layout | fixed dense reference | final compact | compact delta |
|---|---:|---:|---:|
| D128, GQA4, KV512, query 16 | about 12.850 us | 12.842 us (5 runs) | -0.06% |
| D256, GQA1, KV512, query 16 | 12.324 us | 12.316 us (5 runs) | -0.06% |
| D256, GQA6, KV512, query 256 | 81.286 us | 81.695 us (10 balanced standalone runs) | +0.50% |

The balanced audit alternated which clean, one-case process ran first. Its last
five compact runs averaged `81.804 us`, or +0.64% against the fixed reference.
The same compiled binary's dense route averaged `80.815 us`, making the local
route delta +1.09%; that candidate-era dense result does not replace the valid
81.286-us historical baseline. This small and clock-sensitive isolated cost is
a disclosed tradeoff, not reported as a speedup. The accepted model decode
brackets below are the regression gate.

The result is lower than the +1.14% cost of the first block-owned-scalar
implementation. Direct accumulator, global per-warp descriptor load, borrowed
phase-tail, packed-store, and other single-address variants were exact but
either perturbed unaffected D128/D256 layouts more or left a larger wide cost;
they were rejected. The initial five-case, initial 20-case, and balanced
standalone logs hash to
`ae0b9b6afb3373a29c631231cbfc925aec78264001263ce35d44f6e68de95934`
`7283e8db8fc461121384756d77cc1f260062c54cba07971f16dc82440fdff534`,
and
`e3db40ebde9a98ce938b82cd07ec41d3286934042db54b221542ddbbb98fa47d`.

Five clean direct-target NCU metric processes reported compact versus dense:

| metric | compact | dense | delta |
|---|---:|---:|---:|
| instructions | 11,849,040 | 12,135,984 | -2.364% |
| global loads | 640,256 | 648,192 | -1.224% |
| local loads / stores | 2,048 / 6,016 | 2,048 / 6,016 | unchanged |
| shared loads | 83,968 | 122,880 | -31.667% |
| shared stores | 267,008 | 270,336 | -1.231% |
| NCU duration mean | 76,057.6 ns | 76,153.6 ns | -0.126% |

Profiler duration is supporting causality only; the unprofiled measurements
above are the throughput authority. Three additional full-compute NCU
repetitions found identical resource limits. Compact reduced long-scoreboard
stall share from 26.28% to 23.08%, while short-scoreboard, barrier, MIO, and
math-pipe shares rose by 1.07, 0.22, 0.53, and 0.51 percentage points. This is
consistent with trading dense-mask shared traffic for a lane-zero shared load
and warp shuffle; the microcase remains latency-sensitive rather than limited
by allocation, occupancy, or instruction count.

NCU directly targeted `test-backend-ops`, with no wrapper between NCU and the
executable and no `taskset`:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  /usr/bin/ncu --target-processes application-only --replay-mode kernel \
    --kernel-name-base function --kernel-name regex:flash_attn_ext_f16 \
    --launch-count 2 \
    --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_local_ld.sum,smsp__inst_executed_op_local_st.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum \
    --force-overwrite --csv --log-file REPORT.csv \
    /home/gencoolpc/beellama-dev/build-dev-cuda/bin/test-backend-ops test \
    -b CUDA0 -o COMPACT_CAUSAL_DESCRIPTOR_EQUIVALENCE \
    -p "hsk=256,hsv=256,nh=4,nr23=\\[6,1\\],kv=512,nb=256"
'
```

The direct metric CSV SHA-256 values are
`47e27e73c920810fb46b3a4b0df10e936ecbb48ea302db46e9c2d6676f52021a`,
`28e6e227c7f2dc929f3d35d64f0fb1af0136fd6a4f75fe3c568a0eee028e76e6`,
`baed9b279fc13888a886fc409414fd15e518379e08b02f0ba69082a4dbf34ea7`,
`6a4fb4bd06a94619b25fdf1d79b44c8d61f0e2a4bc573158c69e89a586833442`,
and
`26e28951a26a80a31e2cb1e47a85ab05d51883f02396e43941572c7142d7e76d`.
The three compute and three stall CSVs are alongside them under
`/home/gencoolpc/vram-results/2026-08-22-compact-native-lifetime`.

### End-to-end A/B/A performance and memory

The clean-process model brackets used Q8_0 K/V, native-Q8 FlashAttention,
`-b 512 -ub 256`, native llama affinity, unrelated live-context workspace off,
and no warmup. GPU-KV used depth 4,096 with five repetitions; CPU-pinned KV
used depth 30,000 with three. Each process exposed native `--progress` and
`--kv-memory` JSONL. Compact A1 and restored compact A2 had the identical CUDA
library SHA-256
`cd7ffbbce086ec94101596747ab691e52f8c2053e29be5bff4b70a92b9e59726`;
the temporary dense-control library was
`952f3fab8b43628c2d3801339d76d15c2ae01b5506297fa834243c5355e76dd0`.
No control remains in the source.

The exact GPU-KV command was:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  /home/gencoolpc/beellama-dev/build-dev-cuda/bin/llama-bench \
    -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
    -p 512 -n 128 -d 4096 -r 5 -b 512 -ub 256 \
    -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
    -ngl 999 -sm none -mg 0 -nkvo 0 \
    -fa on -ctk q8_0 -ctv q8_0 --flash-attn-native-quants \
    --no-live-context-workspace --no-warmup --progress --kv-memory -o jsonl
'
```

The CPU-pinned command changed only the depth, repetitions, and placement:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  /home/gencoolpc/beellama-dev/build-dev-cuda/bin/llama-bench \
    -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
    -p 512 -n 128 -d 30000 -r 3 -b 512 -ub 256 \
    -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
    -ngl 999 -sm none -mg 0 -nkvo 1 --kv-cpu-pinned \
    --recurrent-state-offload --kv-gpu-layers 0 \
    -fa on -ctk q8_0 -ctv q8_0 --flash-attn-native-quants \
    --no-live-context-workspace --no-warmup --progress --kv-memory -o jsonl
'
```

| placement / depth | metric | compact A1 | dense B | compact A2 | compact bracket vs dense |
|---|---|---:|---:|---:|---:|
| GPU KV / 4K | prefill tok/s | 1,801.661 | 1,799.765 | 1,800.575 | +0.075% |
| GPU KV / 4K | decode tok/s | 49.810 | 49.792 | 49.815 | +0.040% |
| GPU KV / 30K | prefill tok/s | 1,360.784 | 1,336.815 | 1,357.342 | +1.664% |
| GPU KV / 30K | decode tok/s | 44.196 | 43.915 | 44.187 | +0.630% |
| CPU-pinned KV / 30K | prefill tok/s | 1,232.699 | 1,215.307 | 1,232.770 | +1.434% |
| CPU-pinned KV / 30K | decode tok/s | 23.951 | 23.876 | 23.946 | +0.306% |

Thus neither full prefill nor decode regressed in either placement. The small
isolated wide-kernel cost does not appear as an end-to-end decode loss.

GPU-KV compact-versus-dense resource differences were deterministic in both A
runs:

| phase | process VRAM | device compute | accelerator-host compute | ordinary host | VMM live / mapped high-water |
|---|---:|---:|---:|---:|---:|
| prefill | -2,097,152 B | -2,359,296 B | -2,359,296 B | 0 B both | 9,619,456 / 10,485,760 B both |
| decode | -2,097,152 B | -2,228,224 B | -2,228,224 B | 0 B both | 421,120 / 2,097,152 B both |

A follow-up at the deepest practical fully GPU-resident point used the same
GPU-KV command with `-d 30000 -r 3`. Process VRAM was byte-identical at every
compact/dense/compact checkpoint: prefill context/peak were
15,257,960,448/15,297,806,336 bytes and decode context/peak were
15,270,543,360/15,276,834,816 bytes. The context buffers were also identical.
The context-scaled mask reduction remained accelerator-host storage rather
than becoming VRAM in this `llama-bench` graph:

| phase | compact device compute delta | compact accelerator-host compute delta | compact total CUDA-owned compute delta | process VRAM delta |
|---|---:|---:|---:|---:|
| prefill | +287,872 B | -15,728,640 B | -15,440,768 B | 0 B |
| decode | +287,872 B | -15,466,496 B | -15,178,624 B | 0 B |

VMM live/mapped high-water was again identical. Thus the direct answer for
this benchmark is not a 16 MiB VRAM saving: it is zero process-VRAM change at
30K, plus about 15 MiB less CUDA-owned storage, almost entirely pinned
accelerator-host memory. The 4K benchmark's 2 MiB process-VRAM reduction is a
different allocator outcome; the process-VRAM delta is not monotonic with
context depth.

At 30K CPU-pinned placement, process VRAM and context buffers were identical.
Compact reduced accelerator-host compute by 15,728,640 bytes in prefill and
15,466,496 bytes in decode. Its device compute buffer was 282,752 bytes larger,
so total CUDA-owned compute storage still fell by 15,445,888 and 15,183,744
bytes respectively. Ordinary host context and compute buffers were zero for
both. VMM high-water was identical to the GPU-KV values in the table. These
host mappings are reported separately from `nvidia-smi` process VRAM.

The six source logs and their SHA-256 values are:

- GPU compact A1 / dense B / compact A2:
  `b26616568f798d1873a2702b1e1d83e7331b634aac1cca0fd5f178bf072ff714`,
  `466347efe604549b8c08a51699d7125e318d46ad48cfc8d9d893ee2416efb1b3`,
  `14ce44f6c4d4e615abda3b501b20cd84d58f32ff0f8db6c48da2dab48f9796ff`.
- CPU-pinned compact A1 / dense B / compact A2:
  `6a7863a9f461ae4d1996fe6b1c3b36f2137081938cadcca7b127ab0f5294804b`,
  `de63eae6b0b78e675bbd272bc54a28856fece4afc56a2de80c2e2f1ceb38d053`,
  `c8ce096c679eab60bdd9b8cf227a573ae45b2ffb5a58620a05c3cd487c1b1b9b`.
- 30K GPU-KV compact A1 / dense B / compact A2:
  `0dfa376bb455266e8faf09d77f4749beb685fa94c2859636dee8897387cdd354`,
  `6337cc0c2006359a05be1db29d63a4ae301b4cb8b4de1146c7075554e2d482e1`,
  `a2b2d2e65cd556acc97356ebbfc80a62a29051c1d30c3c6df49924b6677361b9`.

### Quality gate

The model SHA-256 is
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`;
the input SHA-256 is
`8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`.
Separate clean GPU-KV and CPU-pinned processes used `-c 4096 -b 512 -ub 256`
and `--chunks 4`, native Q8 K/V attention, and disabled the unrelated phase-aware
and live-context workspace options. No dataset download was needed: the local
606,662-byte corpus was already present. Both printed the exact sequence
`1.9315, 2.1279, 2.2498, 2.1674` and final
`PPL = 2.1674 +/- 0.03849`. The logs hash to
`c1636948105f75fcb7e57a4d813b449d53f38e8880c6f181e730708ae971e5a2`
and
`ab0f3620133a660e10c868c34f6407915077fc7231c9f2df94037b6948121ef7`.
There is exactly zero measured perplexity increase.

The exact GPU-KV quality command was:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  /home/gencoolpc/beellama-dev/build-dev-cuda/bin/llama-perplexity \
    -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
    -f /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt \
    -c 4096 -b 512 -ub 256 --chunks 4 -t 3 -tb 24 \
    --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 \
    -ngl 999 -sm none -mg 0 --flash-attn on \
    --no-phase-aware-workspace --no-live-context-workspace \
    --flash-attn-native-quants --cache-type-k q8_0 --cache-type-v q8_0
'
```

The CPU-pinned quality process used that command plus
`--no-kv-offload --kv-cpu-pinned --recurrent-state-offload --kv-gpu-layers 0`.

A fresh deterministic `llama-cli` compact/dense/compact bracket generated 16
tokens with seed 1234 and temperature zero. After removing only the build and
timing rows, all three streams had SHA-256
`cd35be773ca9520e5f79474c94ee1d07784433199b1c3c4f531dffbbcc353c6c`.
The restored compact library matched the pre-control library byte-for-byte.

Disposition: retain the block-owned shared scalar with per-warp lane-zero
load/shuffle broadcast and retain the two D256 native-Q8 regression cases.
The repair eliminates the NaN composition bug, keeps the earlier compact-mask
representation and its context-scaled allocation savings, has exact output and
unchanged PPL, and is neutral-to-positive in the model decode gates. Its
disclosed remaining cost is +0.50% overall / +0.64% in the later balanced
isolated short D256/GQA6 microkernel (+1.09% versus the same binary's dense
route). No additional feature, flag, dependency, architecture check, or
native-Q8 implementation was added.

## 2026-08-23: isolated 245,760-depth performance and allocation bracket

The final native-Q8 lifetime-repair source received the missing source-matched
long-context process gate. Three clean processes ran compact A1, dense B, and
restored compact A2 at depth 245,760. The temporary dense control changed only
the CUDA backend's compact-causal capability answer from `true` to `false`;
all compact-capable kernels remained compiled. This reproduced the established
dense-control CUDA-library SHA-256
`952f3fab8b43628c2d3801339d76d15c2ae01b5506297fa834243c5355e76dd0`.
Compact A1 and A2 both used
`cd7ffbbce086ec94101596747ab691e52f8c2053e29be5bff4b70a92b9e59726`.
All three libraries were 228,047,032 bytes, and all three processes used the
same `llama-bench` SHA-256
`d9cfbe682e113054441a8f37531c2a91f26b5289996f975a5cd57422c4157055`.
The compact capability and library were restored byte-for-byte after B; no
control remains in the source or build.

The common build was Release, SM120, CUDA FlashAttention on, native Q8 on,
KVarN off, and the default quant matrix. Runtime used Q8_0/Q8_0, CPU-pinned KV,
GPU recurrent state, all model layers on one GPU, `-b 512 -ub 256`, native
llama affinity, live-context workspace explicitly off, no warmup, and no
speculation. CPU-KV placement was the only non-feature necessity at this
depth. `llama-bench` has no phase-aware-workspace control, so that server
feature was unavailable and not enabled. The model SHA-256 was
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.

Each complete GPU process, including the progress-preserving capture, ran
inside the shared lock. Substitute the A1, B, or A2 artifact path for `LOG`:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  /usr/bin/script -qefc "/home/gencoolpc/beellama-dev/build-dev-cuda/bin/llama-bench \
    -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
    -p 512 -n 128 -d 245760 -r 3 -b 512 -ub 256 \
    -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
    -ngl 999 -sm none -mg 0 -nkvo 1 --kv-cpu-pinned \
    --recurrent-state-offload --kv-gpu-layers 0 \
    -fa on -ctk q8_0 -ctv q8_0 --flash-attn-native-quants \
    --no-live-context-workspace --no-warmup --progress --kv-memory -o jsonl" LOG
'
```

This is deep-context performance: the prompt rows process 512 new tokens with
the KV positioned at 245,760. It is not a complete 245,760-token prompt timing.
The first depth fill took about six minutes per clean process and was not used
as a throughput measurement; repetitions two and three reused the filled KV.

| phase | compact A1 | dense B | compact A2 | compact bracket | bracket effect |
|---|---:|---:|---:|---:|---:|
| 512-token prefill | 361.865756 tok/s | 349.950511 tok/s | 361.680525 tok/s | 361.773141 tok/s | **+3.378372%** |
| 128-token decode | 5.060519 tok/s | 5.035306 tok/s | 5.059173 tok/s | 5.059846 tok/s | **+0.487359%** |

A2 differed from A1 by only -0.051188% in prefill and -0.026598% in decode.
The bracket therefore shows no long-context prefill or decode regression and a
positive point effect in both phases. It contains one dense process, so it is
reported as an A/B/A bracket rather than an independent-process confidence
interval.

Both compact processes produced byte-identical allocation fields. Compact
deltas relative to dense were:

| phase | process VRAM at context / peak | device compute | accelerator-host compute | total CUDA-owned compute | ordinary host | VMM live / mapped high-water |
|---|---:|---:|---:|---:|---:|---:|
| prefill | -119,537,664 / -119,537,664 B (-114 / -114 MiB) | -120,848,384 B (-115.25 MiB) | -126,091,264 B (-120.25 MiB) | -246,939,648 B (-235.50 MiB) | 0 B both | 9,619,456 / 10,485,760 B both |
| decode | -121,634,816 / -121,634,816 B (-116 / -116 MiB) | -120,717,312 B (-115.125 MiB) | -125,960,192 B (-120.125 MiB) | -246,677,504 B (-235.25 MiB) | 0 B both | 421,120 / 2,097,152 B both |

Device and accelerator-host context buffers, resident KV, and VMM high-water
were identical. The decode teardown checkpoint retained a 2 MiB compact
process-VRAM advantage. These results also explain the apparent difference
from the earlier 245,760 server load result: this benchmark used ubatch 256,
so the dense mask's query-width-scaled device and host allocations are about
half the ubatch-512 server shape. The earlier server result remains 230 MiB
process VRAM and 240 MiB accelerator-host storage saved at ubatch 512; this
ubatch-256 process saves 114--116 MiB of process VRAM and about 120 MiB of
accelerator-host storage while running.

`llama-bench` is not output-bearing, so this long-context timing gate does not
create a new text-output hash. Its runtime source and restored compact library
are exactly those of the immediately preceding 7/7 compact/dense equivalence,
deterministic 16-token compact/dense/compact output, and matched GPU/CPU-KV PPL
gates. Those remain applicable: the normalized output SHA-256 is
`cd35be773ca9520e5f79474c94ee1d07784433199b1c3c4f531dffbbcc353c6c`,
and both PPL modes remain exactly `2.1674 +/- 0.03849`. No code changed for
this timing extension.

Raw artifacts are under
`/home/gencoolpc/vram-results/2026-08-23-compact-causal-245760-isolated`.
Compact A1, dense B, and compact A2 JSONL SHA-256 values are
`e08384f953a24b7f77aab5b8b5409e8b89cf9eac110c81f14e823aa32cddb3b6`,
`4987b31dfdfe55366eda58c947c3fa139d3e43b50b7f922a4f0bf80c27f151c5`,
and
`0d00d69917c1e2903f784561114483e37bbc04bf5118f0cd9b737222722d7d78`.
The artifact manifest SHA-256 is
`5a4f059b52f4a5bdaaa5869c71a400c9de56d3b7df24a01be1d8914ef36ff0b9`.

### Quick `b2048/ub2048` extension

At the user's request, the same 245,760-depth shape received a quick
large-ubatch compact/dense screen. Both `-b` and `-ub` were set to 2,048 so the
requested ubatch was effective; retaining `-b 512` would not define a valid
2,048-token physical batch capacity. All other build, model, placement,
affinity, native-Q8, workspace, progress, and telemetry settings were unchanged.
Each arm used one clean process and one timed sample per phase (`-r 1`), so
these are screening points rather than a repeated performance distribution.
Both configurations fit and completed.

| phase | compact | dense | compact effect | compact versus prior ub256 bracket |
|---|---:|---:|---:|---:|
| 512-token deep-context prefill | 401.452619 tok/s | 377.853694 tok/s | **+6.245519%** | +10.968055% |
| 128-token decode | 5.057570 tok/s | 5.032498 tok/s | **+0.498202%** | -0.044982% |

The 2,048 capacity improved the compact prefill point while leaving decode
effectively unchanged from the repeated ub256 result. Because there is only
one sample per source, no claim is made that the +10.97% ubatch comparison is
the exact repeatable gain. More importantly for the feature gate, compact did
not lose to dense in either phase.

Compact peaked at 16,201,678,848 B process VRAM during prefill; dense peaked at
16,203,776,000 B. Relative to dense, compact used 2 MiB less process VRAM in
both phases. The much larger saving was CUDA-owned host storage:

| phase | compact device-compute delta | compact accelerator-host-compute delta | compact total CUDA-owned-compute delta | ordinary host | VMM live / mapped high-water |
|---|---:|---:|---:|---:|---:|
| prefill | +311,424 B (+0.297 MiB) | **-1,008,730,112 B (-962 MiB)** | **-1,008,418,688 B (-961.703 MiB)** | 0 B both | 40,126,464 / 41,943,040 B both |
| decode | +311,424 B (+0.297 MiB) | **-1,007,681,536 B (-961 MiB)** | **-1,007,370,112 B (-960.703 MiB)** | 0 B both | 421,120 / 2,097,152 B both |

Thus increasing ubatch width makes the compact representation's host-memory
benefit much larger: the dense `[n_kv, ubatch]` mask costs roughly another
960 MiB at this depth. In this `llama-bench` graph it is accelerator-owned host
storage rather than process VRAM, so the correct result is not a 962 MiB VRAM
claim. Device/accelerator-host context buffers, resident KV, padding, and VMM
high-water were identical. The compact prefill peak left only about 430 MiB
between reported CUDA usage and total device memory, so this working point has
little additional VRAM margin despite the host saving.

The exact command was the preceding 245,760 template with `-r 1 -b 2048
-ub 2048`. Compact and dense used the same established library identities,
and the compact library was restored byte-for-byte afterward. Raw artifacts
are under
`/home/gencoolpc/vram-results/2026-08-23-compact-causal-245760-ub2048`.
Compact/dense JSONL SHA-256 values are
`696af7d78bcb6d3af7877aec8280ef4e15a963655d3aca51c74f2804d0f71d23`
and
`b07d65d0c3dad4c25f09ae1a05890633fb2d22a0deb1f70c0a864531b177cc8e`;
the artifact manifest SHA-256 is
`90e5d33caecce8cbe060c46e19b73ed09db1c659096f8388649a2708dbe660e0`.

## 2026-08-23: post-main-merge composition validation

The lifetime repair was committed as
`0a85856eeb6509881677cb745fb11745887dc1f3`; its allocation attribution was
committed separately as `77164bcd8f51090eabbd0fefe03e1ce2cf8e9710`.
Merge `df9491bb2ff5442b70c2b1e91f4502792bd5e18b` then incorporated maintenance
head `4b86269fdf001de44dd96e9c9ae26a9e25091cca` without conflicts. The incoming
source included allocator, scheduler-reserve, KV ownership, VMM telemetry,
validation-tooling, and speculative replay cleanup, so the merged head received
fresh proportional compile, exactness, quality, performance, and allocation
coverage. The committed compact source was clean for its accepted runs; the
dense controls carried only the one-line capability delta documented below.

The post-merge build retained Release, native CPU tuning, SM120, CUDA
FlashAttention, native Q8 MMA on, KVarN off, and the default quant matrix. The
complete build command was serialized by `/tmp/beellama-cuda-build.lock` and
used `--parallel 12`. `llama-bench`, `llama-perplexity`,
`test-backend-ops`, `libllama.so.0`, the restored compact CUDA library, and the
CMake cache respectively hash to
`d9cfbe682e113054441a8f37531c2a91f26b5289996f975a5cd57422c4157055`,
`1f6ae1d70479735eb9783da2835df5651213fc952084cdfb77b5dd9dbf2a106b`,
`0cb9a4f69613ebd5caa4bc8be5010f426eda63b8e0e047072f89053a9e8057f3`,
`ddd9339c7860a863344bf8d006fcc34c38aae68cd76558558a0b73b033a18258`,
`bbce271d9ced0b59ce0ae72e57ca418f9fe227a53b5d14bdb9c0b4c5b91d949f`,
and
`de994706eb1a9ea97ad6a947d5fffb0c2547f9d8a720c1943d17399449fd97d8`.
The runtime reported build `11379 (df9491bb2)`. Hardware remained the NVIDIA
GeForce RTX 5070 Ti with driver 610.57.04 and Intel Core Ultra 9 285K recorded
for the preceding campaign.

### Correctness and quality after composition

The restored source passed all seven compact-versus-explicit-dense CUDA
equivalence cases, including the two D=256 native-Q8 layouts; its captured log
SHA-256 is
`d3f602c92ced804cf65f3233e4088da3170902861f9a7759771d933a6f8a6714`.
Fourteen focused static, generated-dispatch, allocator, graph, native-route,
CPU attention-support, argument, and speculative-replay tests passed 14/14.
Their log SHA-256 is
`b23a187fbb3fe7627c599453faeb04ecf7acaddbbe3bc3ff5f4ea6746a155975`.
The initial attempt to capture that CTest group was a launcher failure because
the nested shell expanded an unquoted alternation regex; CTest never started,
the invalid log was replaced, and it contributes no test result.

The source-matched CPU-pinned quality gate used the exact preceding
`llama-perplexity` command with `-c 4096 -b 512 -ub 256 --chunks 4`. It again
printed exactly `1.9315, 2.1279, 2.2498, 2.1674` and final
`PPL = 2.1674 +/- 0.03849`. The captured log SHA-256 is
`aaca2f32c237f73c4f503d1c8b7a5b40c754ce8d36f62f86649c9f00750973f5`.
The model and corpus retain their recorded SHA-256 values
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`
and
`8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`.
There is still exactly zero measured perplexity increase.

### Source-matched performance and allocation bracket

Every benchmark was a fresh process wholly enclosed by
`/tmp/beellama-single-gpu.lock`. The commands were the exact 4K GPU-KV and 30K
CPU-pinned templates recorded above, plus a 30K GPU-KV form that changed only
`-d 30000`. All used three repetitions, `-b 512 -ub 256`, native llama
affinity, `--no-warmup --progress --kv-memory`, and unrelated live-context
workspace off. The dense control changed only the CUDA backend's compact
capability return to `false`; its one-line source diff SHA-256 is
`d4e817ebff07164040ddd822f2cc273acdb029022bb944e248f9c86cae7e73a0`
and its CUDA library SHA-256 is
`48c1dca1c82befda1394fbd5a9285e1e7297e9158dd465302d3342a3c0290887`.
The compact library was restored to the exact hash above after every dense
arm.

The stable GPU-resident results are:

| depth / phase | compact bracket | dense | compact point effect | A2 versus A1 |
|---|---:|---:|---:|---:|
| 4K / 512-token prefill | 1,801.965355 tok/s | 1,800.314908 tok/s | +0.091675% | -0.186717% |
| 4K / 128-token decode | 49.760205 tok/s | 49.743544 tok/s | +0.033495% | -0.157402% |
| 30K / 512-token prefill | 1,355.102178 tok/s | 1,335.199178 tok/s | **+1.490639%** | +0.101756% |
| 30K / 128-token decode | 44.164006 tok/s | 43.915398 tok/s | **+0.566106%** | -0.000509% |

Thus the 4K point is neutral and the long-context GPU bracket is positive.
The two 30K compact decode processes are effectively identical, closing the
post-merge throughput gate without turning the small 4K point into a speedup
claim.

The 30K CPU-pinned order was extended from A/B/A to A/B/A/B/A after the
closing A exposed host-sensitive drift. The three compact process means were
`23.973638, 23.282981, 23.354568 tok/s`; the two dense means were
`23.888814, 23.135163 tok/s`. Their aggregate point means are
`23.537062` and `23.511989 tok/s` (+0.106643%), but the between-process range
is much larger than that difference and the two local brackets disagree.
One-second telemetry for the closing dense/compact pair showed matching
sustained SM clocks near 2.827 GHz, so GPU throttling does not explain the
host-path variation. This timing series is valid but neutral/inconclusive and
supports no CPU-pinned decode speed claim. Its three compact prefill means all
exceeded both dense means; the aggregate prefill point is +1.563414%, but it is
reported only as a point effect because the process count is small.

Allocation fields were deterministic across every repeated process:

| placement / phase | peak process VRAM compact / dense | device compute delta | accelerator-host compute delta | total CUDA-owned compute delta |
|---|---:|---:|---:|---:|
| GPU KV 4K / prefill | 14,389,739,520 / 14,391,836,672 B | -2,359,296 B | -2,359,296 B | -4,718,592 B |
| GPU KV 4K / decode | 14,377,156,608 / 14,379,253,760 B | -2,228,224 B | -2,228,224 B | -4,456,448 B |
| GPU KV 30K / prefill | 15,297,806,336 / 15,297,806,336 B | +287,872 B | -15,728,640 B | -15,440,768 B |
| GPU KV 30K / decode | 15,276,834,816 / 15,276,834,816 B | +287,872 B | -15,466,496 B | -15,178,624 B |
| CPU-pinned KV 30K / prefill | 14,240,841,728 / 14,240,841,728 B | +282,752 B | -15,728,640 B | -15,445,888 B |
| CPU-pinned KV 30K / decode | 14,236,647,424 / 14,236,647,424 B | +282,752 B | -15,466,496 B | -15,183,744 B |

Device/accelerator-host context buffers, resident KV, ordinary-host buffers,
and VMM live/mapped high-water matched within every placement. The 4K graph
retains a 2 MiB process-VRAM saving. At 30K the allocator outcome keeps
process VRAM equal while compact removes about 15 MiB of CUDA-owned compute
storage, almost entirely accelerator-host memory. These post-merge values
reproduce the earlier source-matched allocation deltas exactly despite the
incoming ownership and VMM cleanup.

Disposition: retain the lifetime repair and the merged maintenance cleanup.
The merged source is exact at the operation and model levels, the stable 30K
GPU throughput gate is positive, the 4K point is neutral, the CPU-pinned decode
timing is explicitly inconclusive, and the compact allocation reduction is
unchanged. Raw artifacts are under
`/home/gencoolpc/vram-results/2026-08-23-dev-main-merge-validation`; the verified
artifact manifest SHA-256 is
`6911a3bd31fc288e6782f77db1507261b9e4a330b8caefce632f51dd62f6e442`.
## Evidence and source-identity boundaries

- The 233-file vector-retirement artifact manifest is
  `/tmp/beellama-pr7-further-20260822/SHA256SUMS`, SHA-256
  `01d03357defd6f2603fcb1638f4ed5a2a65180b119b7bd90f5d2e80d1e833f4e`.
- The consecutive-bound 128K artifact manifest has SHA-256
  `a2f302d925e160991b817a76025cf959489f9a8fe91e6da9df8d5cfd40a4bd25`.
- The interaction artifact manifest hash is recorded above.
- Historical binaries and measurements remain evidence only for their named
  source/build hashes. Documentation-only or source-identical base merges may
  reuse them only after an explicit path/tree or binary-identity proof.
- The main merge, MMA lifetime repair, and graph metadata consolidation created
  a fresh runtime-adjacent boundary. The official results below, rather than
  the historical binaries, close that boundary.

## Official main-based readiness result (2026-08-23)

### Identity and build

The clean baseline is `beellama/main` at
`4b86269fdf001de44dd96e9c9ae26a9e25091cca` (build 11318). The clean candidate
runtime head is `222eec2e7c91296b5f9335389dcff0cfc8a08129` (build 11340). Both
fresh caches used Release, native CPU tuning, CUDA FlashAttention, effective
architecture 120a, the default quant matrix, KVarN off, all-quants off, and the
same CUDA 13.3/GNU toolchain.

| artifact | baseline SHA-256 | candidate SHA-256 |
|---|---|---|
| `CMakeCache.txt` | `49f215303890f183b578d7d90b5faff198137cc7a361428d8593bc1b3455ed72` | `65bab8b8d350395ea210fed16a42eb96de25330842695dfde2b488d9a3745091` |
| `llama-server` | `7ba71678c2209d2583c593e925852de1e1fc59a49ad690c3a30be33d24901a4a` | `e80265c04cb57297703c63808fab3e4b5d74f5512eb75f196294ece0e6f59d18` |
| `llama-bench` | `07fdaefc52d2ab9221447827e39981024b004b613fa396c2f4950880ab297aac` | `762beef3acc5365ff88f928210eaaac48b6ba112b95f2e761370786a85c452a9` |
| `llama-perplexity` | `cb285a9375e7b7ed2ad1da079218c166a41470c26d8c466b075a99d413a41096` | `c9c156766a8440b4e2cde74ccb935edacfcbf33c101239c0f07b0d8de43cb47c` |
| `libllama.so` | `70fe086256df33df98740554dfdaf27662f59f6cece20240b6df58dfb0ffdd4e` | `dafc691007346601947b5eb9371bf6cf1dbf98513f16da2874466228013e7852` |
| `libggml-cuda.so` | `9617d0226a2f44e7700c35754407b656a2fbcc4e0784ee362577d296e88a7cc2` | `5db25529bef9038dffaab7316c9a820c2293b96f971d326fabc455c9af7e9be4` |

The model SHA-256 is
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`;
the 606,662-byte corpus SHA-256 is
`8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`.
Hardware was one NVIDIA GeForce RTX 5070 Ti (15,880 MiB usable, compute
capability 12.0, VMM enabled, driver 610.57.04) and an Intel Core Ultra 9 285K.

### Correctness and quality

The compiled CUDA capability and dense-versus-compact execution oracle passed
6/6, including D=256 F16 MMA with no GQA and GQA 6. Ten focused graph,
allocator, generated-vector, route, fallback, parser, and replay CTests passed
10/10. The main/candidate/main PPL bracket produced the exact chunk sequence
`1.9315, 2.1279, 2.2498, 2.1674` and
`PPL = 2.1674 +/- 0.03849` in all three processes with matching
`-b 512 -ub 256`; the increase is zero.

### Performance

Every row used fresh processes, native progress, `--kv-memory`, q8_0 K/V,
batch/ubatch 512/256, and the affinity and placement in the reproduction
template. Values are process means; profiler and resource-sampler processes do
not contribute timing.

| gate | baseline process means | candidate process means | candidate effect |
|---|---|---|---:|
| 4K GPU-KV prefill | 1810.6711, 1807.4789, 1803.6544, 1798.3230 | 1806.0622, 1803.0656, 1799.8895 | **-0.0736% time-adjusted; neutral** |
| 4K GPU-KV decode | 49.5585, 49.5890, 49.6129 | 49.5099, 49.6635, 49.6586 | **+0.0482%** |
| 30K GPU-KV prefill | 1232.2076, 1229.3053 | 1240.3132 | **+0.7765%** |
| 30K GPU-KV decode | 43.7413, 43.7491 | 43.9228 | **+0.4059%** |
| 128K CPU-pinned-KV prefill | 452.4388, 452.2785 | 458.1242 | **+1.2745%** |
| 128K CPU-pinned-KV decode | 8.1422, 8.2597, 8.1686, 8.2778 | 8.0687, 8.3076, 8.3418, 8.2919 | **+0.4923%** |

The user-requested closing A4 anchor makes the 4K prefill chronology
`A B A B A B A`. Baseline throughput declines monotonically from 1810.6711 to
1798.3230 tok/s. A linear elapsed-time/source model estimates the compact
effect as -1.3282 tok/s (-0.0736%), standard error 1.7246 tok/s, against a
baseline time slope of -8.2511 tok/s/hour. Effects against linearly
interpolated adjacent baseline anchors are -0.1665%, -0.0330%, and -0.2066%.
This is neutral within measured time/process variation, not a speedup and not a
detected regression. The 128K decode campaign was expanded after its first
candidate process ran low; all four processes per source remain in the final
mean. Its wide process variance limits the result to a no-aggregate-regression
finding rather than a strict speedup claim.

### Resources

Independent resource-only A/B prefill processes sampled `nvidia-smi`
process VRAM and `/proc` RSS every 100 ms. They are excluded from timing:

| depth | baseline peak process VRAM | candidate | delta |
|---:|---:|---:|---:|
| 4K | 13,702 MiB | 13,700 MiB | **-2 MiB** |
| 30K | 14,566 MiB | 14,566 MiB | 0 MiB; another allocation owns the peak |
| 128K CPU-pinned KV | 14,194 MiB | 14,136 MiB | **-58 MiB** |

Peak `VmHWM` was 14,559,336--14,559,560 KiB across these processes and
`VmLck` was zero. This does not imply zero CUDA-pinned memory: allocator-class
telemetry reports the driver-owned pinned buffers directly.

At 128K prefill, synchronized CUDA used high-water falls from 14,916,124,672
to 14,855,307,264 bytes (-58 MiB). Device compute falls from 934,892,416 to
872,764,288 bytes (-59.25 MiB), and accelerator-host compute falls from
78,145,568 to 10,774,560 bytes (-64.25 MiB). Pinned accelerator-host context
is identical at 4,581,228,544 bytes, resident KV is identical at 4,590,141,440
bytes, ordinary-host context/compute are zero, and CUDA VMM live/mapped
high-water is identical at 9,619,456/10,485,760 bytes. The corresponding 128K
decode high-water also saves 58 MiB; total compute backing saves 123.25 MiB.

At 4K, synchronized peak CUDA usage saves 2 MiB for prefill and decode, while
total compute backing saves 4.5 and 4.25 MiB respectively. At 30K the
synchronized peak is equal; compact saves 14.73 MiB prefill and 14.48 MiB
decode in total compute backing, primarily accelerator-host memory, while the
device-compute bin is 287,872 bytes larger. KV residency, ordinary-host
buffers, and VMM high-water are unchanged at each matched depth.

### Artifacts and disposition

The 135-file readiness manifest is
`/home/gencoolpc/vram-results/2026-08-23-pr7-official-readiness/SHA256SUMS`,
SHA-256
`d4246230238f39cf643a2a2a1037835cc4a20dd490844abc1dc976d1180206a1`.
It contains source/build identity, exact commands, JSONL, progress, PPL,
exactness, focused-test, resource-sampler, and summary artifacts. The one
128K setup failure is explicitly named invalid and contributes no result.

All local official-branch gates pass. Final publication readiness additionally
requires a clean main-relative scope audit, a normal push of the existing head,
passing GitHub checks, and a cleanly mergeable PR. PR 7 must not be merged into
`beellama/main` as part of this work.

## Limitations

- Source-matched performance and profiler evidence is currently limited to the
  RTX 5070 Ti, compute capability 12.0. Cross-generation performance remains
  unmeasured.
- Automatic compact selection is CUDA-only in this PR. Other backends retain
  dense masks; this is not a claim that the descriptor is inherently CUDA-only.
- Multi-stream, nonconsecutive, biased, SWA/tail, and KVarN layouts deliberately
  remain dense.
- The 240K value is load-to-idle allocation evidence, not a completed prefill.
- Interaction experiments do not import their borrowed feature sources or
  convert one-process timing points into performance claims.
- Git history, not this curated document, is the forensic record for rejected
  or invalid attempts. Invalid launcher/setup runs contribute no measurements.
