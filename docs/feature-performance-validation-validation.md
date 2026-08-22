# Feature/performance toolkit validation

This record covers the tooling implementation based on
`f6341a15779eb58fe6ad9e1b890e331c32b676c7` on
`tools/feature-performance-validation`. It is a toolkit validation, not a
feature-performance result and not long-context acceptance for CPU KV offload
or any other feature.

The current usage and evidence boundaries are in
[feature-performance-validation.md](feature-performance-validation.md).

## CPU-only validation

Static compilation and whitespace validation:

```bash
git diff --check
PYTHONWARNINGS=error::ResourceWarning python3 -m py_compile \
  scripts/feature_validation/*.py \
  scripts/feature-performance-validation.py \
  scripts/test-feature-validation.py
```

Result: pass, with no output.

Focused schema, provenance, quoting/arity, profiler, and telemetry layers:

```bash
python3 scripts/test-feature-validation.py -v \
  ManifestAndScheduleTest ProvenanceTest QuotingAndArityTest \
  ProfilerTest TelemetryTest
```

Result: 22 tests passed in 0.552 seconds at that checkpoint.

Final warning-as-error suite:

```bash
PYTHONWARNINGS=error::ResourceWarning \
  python3 scripts/test-feature-validation.py -v
```

Merge-readiness result: 44 tests passed in 8.334 seconds with
`ResourceWarning` promoted to an error. The tests use temporary Git
repositories, fake direct binaries, fake NVIDIA XML streams, fake profiler
JSONL, and fake NSYS SQLite. They cover safe quoting, zero/one option arity,
common-versus-`llama-bench` `--no-kv-offload` arity, fail-closed provenance,
same-size/mtime-restored identity changes, workload-only selectors,
controlled A/B settings, balanced fresh processes, three-to-five adaptive
repetition, all raw paired statistics, persistent telemetry cleanup and
contamination, executed-versus-passed acceptance states, explicit NSYS
graph-node support/capture checks, post-NCU report/count/shape verification,
resume, preserved failed attempts, and taskset/wrapper rejection.

CTest registration was checked without compiling a production target:

```bash
cmake -S . -B build-tools-validation \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
ctest --test-dir build-tools-validation \
  -R '^test-feature-performance-validation$' --output-on-failure
```

Final result: 1/1 test passed in 8.38 seconds with `ResourceWarning` promoted
to an error.
No build was launched; therefore no
parallel build wider than `-j6` occurred.

## Merge-readiness checkpoint

The coordinator review points are resolved as follows:

- every performance stage has an explicit acceptable-decision policy;
  `execution_status=completed` is separate from
  `status=passed|failed|unresolved`, and regression/inconclusive final-gate
  tests prove neither can emit `acceptance_complete`;
- graph-capable profiling explicitly requests NSYS graph-node tracing, checks
  exact-tool help/version support, and requires nonzero graph-node IDs in the
  export;
- the separate NCU report is required, hashed, re-imported, and checked for
  discovery-derived kernel, unique capture count, and available shapes;
  missing fields are unverifiable and cross-process launch order is never
  assumed stable;
- the local source-backed arity schema treats common-tool
  `-nkvo`/`--no-kv-offload` as bare and both `llama-bench` spellings as valued;
- large identity files are hashed once per fresh/resumed provenance capture,
  while every child checks size, mtime, ctime, device, and inode; a test changes
  content at the same size and restores mtime and confirms ctime fails closed;
- duplicate identity-spec and GPU-lock launcher implementations and the legacy
  single-executable manifest form were removed. The remaining modules have
  one contract each: CLI orchestration, manifest/provenance/statistics,
  lifecycle/state, telemetry ownership, and profiler parsing/planning.

Read-only host capability verification used:

```bash
nsys profile --help=cuda
nsys --version
ncu --help
ncu --version
```

The installed NSYS 2026.1.3 advertised `--cuda-graph-trace` with node
granularity, and the toolkit capability check returned `status=supported` for
`node:host-only`. NCU 2026.2.1 advertised graph-node profiling, report import,
raw-page CSV, demangled print names, and kernel-name filtering. No profiler
capture was claimed from these read-only checks.

### Early-screen overhead

A temporary clean Git fixture ran the complete fresh early ladder with all
early stages marked `resource=gpu`. The target itself was the tiny fake direct
binary; GPU integration here means the real flock/telemetry/clean-process
lifecycle, not a CUDA performance result. The invocation promoted
`ResourceWarning` to an error and preserved visible output in
`/tmp/beellama-fperf-overhead-review.log`.

Result: 32/32 separately locked processes passed in 38.932 seconds, inside the
30-90 second target. All 32 persistent samplers were reaped, all clean-process
checks passed, and the run retained 32 GPU XML plus 64 proc samples. Aggregate
target time was 1.221 seconds; sampler startup was 3.383 seconds (about 0.106
seconds/run on this host); per-run identity checks were 0.198 seconds; initial
full provenance preparation was 0.064 seconds; and honest wall-minus-target
overhead was 37.711 seconds. The runner now records locked-wrapper timing too,
so future studies can separate lock/interpreter launch from in-executor time.
No unmeasured cause is assigned to the remaining orchestration time.

## Small real-GPU integration smoke

The assigned worktree had no configured BeeLlama build, model, or prebuilt CUDA
sample. The host did have CUDA 13.3 `nvcc` and one NVIDIA GeForce RTX 5070 Ti,
so a temporary, uncommitted CUDA program was compiled outside Git. It performed
one 4 MiB allocation, `cudaMemset`, synchronization, and a three-second hold so
the approximately 1 Hz sampler could observe the process. The compile was a
single compiler process:

```bash
/opt/cuda/bin/nvcc -O2 -arch=sm_120 \
  /tmp/beellama-feature-validation-gpu-smoke/smoke.cu \
  -o /tmp/beellama-feature-validation-gpu-smoke/gpu-smoke
```

The final valid integration command was:

```bash
flock /tmp/beellama-single-gpu.lock -c \
  '/usr/bin/python3 /home/gencoolpc/beellama-feature-performance-validation/scripts/feature-performance-validation.py _execute-run /tmp/beellama-feature-validation-gpu-smoke/artifacts/run-spec-attempt-05.json'
```

Result: exit 0 in 3.480 seconds. The target printed
`gpu-smoke=ok bytes=4194304`. The owned sampler produced four GPU XML samples
and was reaped; the proc reader produced 14 samples. The report observed no
foreign compute process, retained the 4 MiB Xorg graphics-only context as
ambient, and recorded:

- target process VRAM peak: 226 MiB from `nvidia-smi`;
- process-tree VmRSS/VmHWM peak: 116252 KiB;
- GPU temperature: 32-33 C;
- instantaneous/average power field used by this driver: 16.90-46.07 W;
- SM clock: 180-2490 MHz;
- pinned host memory: not directly measured, `null`, and not inferred from
  VmLck, VmPin, process VRAM, or the ordinary host-memory samples.

Raw smoke artifacts remain outside Git at
`/tmp/beellama-feature-validation-gpu-smoke/artifacts`. Only attempt 05 is the
current integration evidence. No throughput, latency, exactness, model-memory,
NSYS/NCU, BeeLlama end-to-end, or long-context claim is made from this smoke.

## Coverage and limitations

Automated and manual responsibilities are inventoried in the current how-to.
Important present limitations are:

- the ~1 Hz GPU sampler can miss sub-second peaks;
- direct per-process pinned allocation bytes are unavailable;
- the NSYS parser and NCU-plan derivation have hermetic fake-log/SQLite tests,
  but no production profiler capture was appropriate without a matched
  BeeLlama build and model;
- no in-process copied CUDA A/B harness was added; a future harness must compile
  production source or pass exact source hashes and remain production-aligned;
- a final long-context campaign remains a mandatory per-feature acceptance
  activity and was intentionally not run merely to validate this toolkit.

Integration recommendation: review and integrate this as a tooling-only commit
series. Use a fresh manifest outside the source tree for the first real feature,
validate a matched clean build/model identity, tune the early stages to the
30-90 second target, then keep production confirmation and final long-context
acceptance as separate recorded gates.
