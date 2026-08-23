# AGENTS.md

This file gives code assistants local context for BeeLlama.cpp. The local tree is
the source of truth for behavior; use `tmp/upstream-llama.cpp` only as the
architectural reference when rebasing fork features.

## What This Is

BeeLlama.cpp is Anbeeld's llama.cpp fork. The v0.4.0 fork surface is intentionally
small:

- Upstream speculative decoding, including `draft-dflash`, `draft-mtp`,
  EAGLE3, and n-gram modes.
- KVarN target KV-cache compression for Qwen3.6 and Gemma 4, selected with
  `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, or `kvarn8`.
- Standard low-bit KV cache formats `q2_0`, `q2_1`, `q3_0`, `q3_1`,
  `q6_0`, and `q6_1`. Bee's cache-facing `q2_0` uses the internal enum
  `GGML_TYPE_Q2_0S` so it cannot collide with upstream's serialized Q2_0 weight
  format.
- A profit-only adaptive draft-max controller for DFlash.
- Reasoning-loop detection and the opted-in realtime
  `/v1/chat/completions/control` endpoint.
- INI presets and KLD measurement support in `llama-perplexity`.

DFlash GGUFs must use upstream's `dflash` architecture, metadata, and tensor
names.

TurboQuant/TCQ, TQ3_1S/TQ4_1S, DDTree, CopySpec, the fork DFlash ring/tape and
reduced-verifier paths, the fringe controller, and their arguments and
environment variables were removed in v0.4.0. Do not reintroduce those systems
as compatibility code. The old cache names redirect to same-width KVarN presets.
Use upstream's `draft-dflash` name for the DFlash speculative type; the bare
`dflash` alias was removed in v0.4.0 and now errors.

## Build

```bash
# Linux CUDA
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows MSVC + CUDA
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS Metal
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The minimal fresh-cache CUDA FlashAttention build contains 50 standard vector
pairs and omits KVarN. The standard quant matrix follows the same bit-pair rules
as KVarN and adds homogeneous F16/F16 and BF16/BF16 tail pairs.
`GGML_CUDA_KVARN=ON` explicitly adds the dedicated CUDA/HIP KVarN kernels and 15
balanced KVarN fast-decode pairs. `GGML_CUDA_FA_ALL_QUANTS=ON` expands the
standard matrix to 169 pairs and, when KVarN is enabled, expands its matrix to
all 36 ordered bit pairs. CMake caches retain earlier selections, so pass
`-DGGML_CUDA_KVARN=OFF` when converting an existing build tree to the minimal
policy. Enable KVarN only in worktrees that build or validate that feature.
`GGML_CUDA_FA_HALF_QUANTS` no longer exists. Valid KVarN pairs outside the fast
matrix use descriptor-native MMA fallback.

`GGML_CUDA_FATTN_Q8_NATIVE=ON` independently compiles the off-by-default native
Q8_0/Q8_0 MMA FlashAttention family. `GGML_CUDA_FA_ALL_QUANTS` must not imply or
expand native MMA families. Runtime cache types still come from the graph;
`--flash-attn-native-quants` only permits a registered direct loader. Read
`docs/quantized-native-flash-attention.md` before changing or measuring it.

Use `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090 and `89` for RTX 4090 when
the build host cannot detect the target GPU.

On Windows hosts matching CUDA 13.1 and compute capability 8.6, prefer:

```powershell
powershell -File scripts/build-win-cuda-13.1-sm_86.ps1 -AllTests
powershell -File scripts/build-win-cuda-13.1-sm_86-default.ps1 -AllTests
powershell -File scripts/build-win-vulkan.ps1 -AllTests
```

The first CUDA script compiles the expanded quant matrix; the `-default`
variant compiles the default pair matrix. The Vulkan script requires a Vulkan
SDK. For other hardware or toolkits, adapt the architecture, toolkit, and
build-name parameters instead of reusing the `sm_86` artifact names.

Key binaries are `llama-server`, `llama-cli`, `llama-bench`, and
`llama-perplexity` under the configured build directory's `bin` folder.

## Architecture

### Main Directories

- `ggml/` - tensor library, quantization, and CPU/GPU backends.
- `src/` - model loading, contexts, graphs, and memory.
- `src/models/` - model-specific graph builders.
- `common/` - arguments, sampling, presets, and upstream speculative decoding.
- `tools/server/` - HTTP API, slots, speculative scheduling, and Bee server
  extensions.
- `include/llama.h` - public C API.

### Fork-Specific Files

- `src/llama-kvarn.cpp` / `.h` - KVarN descriptors, presets, and validation.
- `src/llama-kv-cache-kvarn.cpp` / `.h` - KVarN memory and state handling.
- `ggml/src/ggml-cuda/kvarn.cu` / `.cuh` - shared CUDA/HIP KVarN store and
  materialization operations.
- `ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu` and
  `fattn-kvarn-portable.cuh` - optimized CUDA and portable CUDA/HIP direct
  KVarN attention.
- `ggml/src/ggml-vulkan/vulkan-shaders/kvarn_store.comp` and
  `kvarn_materialize.comp`, `kvarn_wht.comp`, and `kvarn_flash_attn.comp` -
  Vulkan KVarN storage, fallback materialization, transforms, and direct
  attention shaders.
- `tools/server/server-adaptive-dm.h` - profit adaptive draft-max controller.
- `tools/server/server-loop-guard.cpp` / `.h` - reasoning loop detection.
### Key Docs

- `docs/beellama-features.md` - fork feature and compatibility matrix.
- `docs/beellama-args.md` - Bee arguments, aliases, and removals.
- `docs/quickstart-qwen36-dflash.md` - Qwen3.6 DFlash guide.
- `docs/quickstart-gemma-4-31b-dflash.md` - Gemma 4 DFlash guide.
- `docs/preset.md` - INI preset format.
- `docs/cpu-kv-offload-current-testing.md` - authoritative current CPU KV
  build, runtime, exactness, benchmark, progress, and artifact protocol. Start
  here before running or recording any CPU KV experiment.
- `docs/cpu-kv-offload-development.md` - progress and decision journal. It
  records protocol transitions, durable rationale, and concise summaries of
  valid rejected paths. It is not a transcript of every attempted run.
- `docs/cpu-kv-offload-experiments.md` - complete curated CPU KV Experiments
  001-020 and W06 record plus the post-KV evidence/identity index. It contains
  only valid, decision-relevant evidence; use Git history for superseded
  editions.
- `docs/cpu-kv-offload-vram-roadmap.md` - ranked shared VRAM work, including
  integrated controls, independent PR lanes, rejected paths, and later research.
- `docs/vram-feature-isolation-plan.md` - source, measurement, comparison, and
  post-composition gates for every KV-derived VRAM feature branch.
- `docs/cpu-kv-offload-feature-delta.md` - source-backed capability delta from
  BeeLlama v0.4.3, including explicit features absent from the published KV base.
- `docs/feature-performance-validation.md` - manifest-driven early A/B
  screening, telemetry, statistics, resume, and optional Nsight diagnostics for
  KV-derived feature work.

### Invariants

- KVarN is target-context only. Draft and auxiliary contexts use normal cache
  types.
- CUDA, CPU, Vulkan, and HIP/ROCm consume KVarN records directly in native
  attention paths. Vulkan native attention requires shader Int64 and
  buffer-device-address support. Materialization is an explicit fallback, not
  the normal route for these backends.
- Unsupported KVarN placements fail closed or use the explicit
  bit-width-matched fallback path; they must not silently reinterpret records.
- Custom CUDA helpers are resolved through
  `ggml_backend_cuda_reg_get_proc_address`.
- DFlash scheduling, checkpoints, verification, and multi-GPU behavior belong
  to upstream. Bee extensions must use upstream task, sampler, and checkpoint
  APIs rather than restoring fork-private verifier state.
- Benchmark claims require the exact model files, command, prompt, sampling
  settings, hardware, and commit ID.

## Test and Benchmark

```bash
# Unit and regression tests
flock /tmp/beellama-single-gpu.lock -c '
  ctest --test-dir build --output-on-failure
'

# KVarN quality at the intended serving cadence
flock /tmp/beellama-single-gpu.lock -c '
  build/bin/llama-perplexity \
    -m model.gguf -f test.txt -c 4096 -b 512 -ub 256
'

# Decode speed
flock /tmp/beellama-single-gpu.lock -c '
  build/bin/llama-bench -m model.gguf -p 0 -n 64 -t 1 --progress
'

# Upstream DFlash with recommended standard q cache
flock /tmp/beellama-single-gpu.lock -c '
  build/bin/llama-server -m target.gguf \
    --spec-type draft-dflash \
    --spec-draft-model drafter.gguf \
    --spec-draft-n-max 8 \
    --flash-attn on --cache-type-k q5_0 --cache-type-v q4_1 \
    --port 8080
'
```

KLD comparisons use matching `-b` and `-ub` values for the baseline and
candidate. Record both values with every result.

## CPU KV-offload Experiment Workflow

These instructions apply when working on local `beellama/main` or continuing its
CPU-resident KV-cache investigation.

- Read `docs/cpu-kv-offload-current-testing.md` before changing code, launching
  a test, or interpreting a current result. It is the only documentation source
  for the runnable current protocol; the local source remains authoritative if
  documentation and behavior disagree.
- For a new CPU-KV-derived performance or memory feature, preregister and run
  the manifest workflow in `docs/feature-performance-validation.md` for early
  exactness, short prefill/decode, representative weighted screens, and
  resource telemetry. A clear early signal only permits deeper validation;
  it never replaces the current long-context, perplexity, clean-process,
  allocation-lifecycle, or feature-specific acceptance gates.
- Read the relevant sections of `docs/cpu-kv-offload-development.md` and
  `docs/cpu-kv-offload-experiments.md` when historical rationale, prior
  protocol editions, rejected approaches, or old measurements are needed.
  Never promote a command from those records into a current run unless the
  current-testing document explicitly permits it. Treat Git commits as the
  source of truth for exact historical diffs.
- Do not include retired arguments or environment-variable names in a current
  command, including defensive `env -u` entries. In particular,
  `GGML_KV_CPU_PINNED` and `GGML_RECURRENT_STATE_OFFLOAD` are historical and
  must not appear in current setup or benchmark commands. Use the supported
  CLI controls documented in the current-testing document.
- Do not use `taskset` in a current CPU-KV benchmark or profiler command. In
  particular, never put it before or after `ncu` or `nsys`: the former pins
  profiler collectors and helpers, while the latter makes an affinity wrapper
  the direct profiling target. Launch the llama binary directly under the
  profiler and express target worker placement with llama.cpp's own
  `--cpu-mask`/`--cpu-range`, batch-affinity, and `--cpu-strict` controls. Any
  `taskset` command in the development journal, evidence index, or a
  reproduction document is historical evidence, not a current template.
- Keep the known BeeLlama baseline worktree unchanged. Make experimental source,
  build, profile, and documentation changes in the dedicated experimental
  worktree and branch.
- Treat the canonical `beellama-kv-cpu-offload` branch as the main integration
  line for KV-offload feature work. An assigned starting commit is provenance,
  not a permanent pin. Incorporate the latest validated integration head before
  new acceptance measurements and final handoff, and target feature PRs at that
  branch rather than general `main`. Update only at a clean measurement
  boundary; refresh provenance, rebuild binary-affecting changes, and rerun
  proportional coverage afterward. Preserve feature work and report conflicts
  instead of dropping either side.
- Give each independently testable optimization its own commit. Include the
  implementation and its corresponding update to
  `docs/cpu-kv-offload-experiments.md` in that commit so the code and evidence
  cannot drift apart.
- Update `docs/cpu-kv-offload-current-testing.md` first whenever supported
  controls, the active setup, exactness oracle, benchmark shape, progress
  mechanism, or required artifact set changes. Record why the prior edition
  was superseded in `docs/cpu-kv-offload-development.md`. Preserve durable
  rationale, not obsolete command copies or an attempt-by-attempt transcript.
  Pure result additions that do not change the protocol or working theory
  belong only in the experiment/evidence index.
- Record a rejected or neutral experiment only when its run was valid and its
  result tests a distinct hypothesis or prevents likely repeated work. Prefer a
  clean revert or a separate revert commit when preserving the exact attempted
  source diff is useful; never leave an undocumented partial implementation in
  the branch.
- Invalid runs are not evidence. A run with a wrong binary, unmatched prompt or
  configuration, contaminated hardware, setup/launcher failure, incomplete
  output, or a violated acceptance gate must not contribute measurements,
  artifact inventories, or a ledger entry. If the mistake exposes a reusable
  protocol hazard, record only a short correction in the development journal;
  Git history and temporary artifacts are sufficient for forensic detail.
- Do not create a new record for an otherwise identical rerun that adds no new
  confidence or decision. Aggregate required repetitions into the existing
  experiment with sample count and summary statistics. Record a reproduction
  separately only when it changes confidence, covers a materially different
  model/hardware/commit, closes a named correctness gate, or contradicts prior
  evidence.
- For every performance change, measure clean baseline and candidate processes
  with identical model, depth, cache formats, affinity, thread counts, build
  options, and runtime settings. At minimum record prefill, decode at depth 4096,
  decode at a long-context depth, and peak process VRAM.
- Record system-RAM and pinned-memory costs when allocation behavior changes.
  Do not equate `nvidia-smi` process VRAM with CUDA host mappings or page-locked
  system memory.
- Run correctness/regression coverage proportional to the change. Cache-format
  quality comparisons require matching `llama-perplexity` `-b` and `-ub`
  values for baseline and candidate.
- Every command that may have a long or uncertain runtime must expose progress.
  Use the tool's native progress option when available (`llama-bench
  --progress`, for example); otherwise launch it with a documented external
  progress mechanism that can be inspected without restarting or attaching a
  debugger. State the progress mechanism in the launch update and preserve it
  in the recorded command. Do not start an unbounded or duration-uncertain run
  whose only observable states are running and finished.
- On the current 24-core benchmark host, serialize every CUDA build across
  worktrees with `/tmp/beellama-cuda-build.lock` and keep the complete build
  command inside the lock's safely quoted `-c` argument. Limit the one active
  build to 12 parallel jobs. Queueing prevents builds from overlapping but does
  not remove the per-build job cap. Follow the exact current template in
  `docs/cpu-kv-offload-current-testing.md`.
- Wrap every runnable GPU test, benchmark, profiler, or server lifecycle in
  the exact whole-command form
  `flock /tmp/beellama-single-gpu.lock -c 'COMMAND'`. Do not use flock's
  direct-command form; the quoted command must own the complete lifecycle and
  use safe inner quoting.
- Every experiment entry must state its base and candidate commit IDs, the
  current protocol plus any explicit command/configuration delta, hardware,
  model path, measurements, resource tradeoffs, and disposition: retained,
  revised, neutral, or reverted. Include a full command only when the canonical
  current template plus recorded deltas cannot reproduce the valid evidence.
- Do not present measurements from an uncommitted or differently configured
  binary as results for the current commit. Rebuild or verify the binary's build
  commit before benchmarking.
- Before using a CMake-built executable with the feature-performance validation
  toolkit, run its `register-build` command at the executable's final path and
  pin the emitted sidecar path and SHA-256 in the manifest. Missing, copied, or
  stale registrations fail closed; an embedded commit label is not sufficient
  build provenance.

## Git Conventions

- Keep fork-specific changes small and aligned with current upstream
  abstractions.
- Do not treat old benchmark notes as current evidence without rerunning them.
- Do not commit unless the user explicitly asks.
