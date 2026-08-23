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
