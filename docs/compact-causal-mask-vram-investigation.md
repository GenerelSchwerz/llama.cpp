# Compact causal-prefix design and evidence

Status: PR 7 is the isolated compact causal-prefix publication lane. It remains
draft while the final main-based readiness gates described below are running.
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

Use `DEPTH=4096` and `30000` with no placement extra. Use `DEPTH=131072` with
`--no-kv-offload --kv-cpu-pinned --recurrent-state-offload`. Prefill coverage
substitutes `-p 512 -n 0` while retaining the same depth and other arguments.
A single pair is a screen; small claims require balanced independent-process
brackets.

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
- The main merge, MMA lifetime repair, and graph metadata consolidation are a
  fresh runtime-adjacent boundary. They require the final exactness, PPL, and
  4K/30K/128K clean-process gates below before review readiness.

## Final official-branch readiness gate

The final record must identify the exact `beellama/main` base, candidate head,
binary and cache hashes, model/corpus hashes, build options, GPU/CPU/driver,
commands plus deltas, progress artifacts, and disposition. Required coverage:

1. compiled CUDA capability and all six dense-versus-compact cases;
2. focused graph, allocator, vector dispatch, route, and fallback tests;
3. matching-batch four-chunk PPL with no increase;
4. balanced clean-process 4K and 30K GPU-KV prefill/decode;
5. balanced 128K CPU-pinned-KV prefill/decode;
6. peak process VRAM plus device, accelerator-host, ordinary-host, resident-KV,
   pinned-memory, and VMM fields; and
7. a final main-relative scope audit proving no PR 4, KVarN implementation,
   workspace, allocator-policy, or development-only source entered this branch.

PR 7 may leave draft only after those gates pass and GitHub reports the pushed
head cleanly mergeable. It must not be merged into `beellama/main` as part of
this work.

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
