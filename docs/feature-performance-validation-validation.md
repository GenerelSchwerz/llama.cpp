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
parallel build wider than 12 jobs occurred.

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

## Fast early-screen follow-up

The first compact-causal application showed that applying the full 3-to-5
statistical policy to every smoke and every low/mid/high screen produced 32
fresh model processes and a 292.91-second cold early ladder. The runner now
supports an explicit `single_pair_fail_fast` policy only for smoke and kernel-
screen stages. It emits a raw point observation, `confidence_interval=null`,
and `confidence_claim=none`; it cannot be used on production confirmation or
long-context acceptance and cannot coexist with a statistical decision policy.

The revised causal manifest ran 12 clean locked processes in 76.27 seconds.
Its target processes used 67.81 seconds, preparation 6.20 seconds, recorded
runner overhead 1.65 seconds, and lock-wrapper handoff 0.46 seconds. A warm
resume completed in 6.24 seconds. Exactness passed, all samplers were reaped,
and the three preregistered one-pair signals cleared their 1% fail-fast
threshold. These signals are screening workflow evidence, not new confidence
intervals or acceptance results.

Final validation after this addition:

```bash
PYTHONWARNINGS=error::ResourceWarning \
  python3 scripts/test-feature-validation.py -v
```

Result: 54 tests passed in 8.757 seconds.

## Regression-triggered Nsight diagnostics follow-up

The optional agent-diagnostic path adds no work to a passing early screen. With
`--diagnose-regressions`, only a preserved preregistered
`regression_signal` starts profiling; the screening command still exits
nonzero. The diagnostic independently performs NSYS discovery and one filtered,
post-export-verified NCU capture for baseline and candidate. It writes a compact
side-by-side report while retaining raw launches, shapes, graph IDs, requested
NCU metrics, timing, commands, and artifact paths. The report is explicitly
diagnostic-only.

Focused validation used:

```bash
PYTHONWARNINGS=error::ResourceWarning \
  python3 scripts/test-feature-validation.py -v \
  ManifestAndScheduleTest ProfilerTest DiagnosticCliTest
```

Result: 34 tests passed in 0.512 seconds. This covers schema/runtime
preregistration, graph-node requirements, deterministic largest-regression
span selection, zero profiler calls after a clear result, independent
baseline/candidate orchestration, raw NCU metric exposure, agent-facing
inventory/comparison output, preservation of the failed screening exit status,
and fail-closed unverifiable capture handling.

The final warning-as-error suite used:

```bash
PYTHONWARNINGS=error::ResourceWarning \
  python3 scripts/test-feature-validation.py -v
```

Result: 64 tests passed in 9.515 seconds. Static Python compilation, all three
checked-in manifest examples, JSON schema parsing, and `git diff --check` also
passed. The registered CTest invocation also passed 1/1 in 9.68 seconds. No new
GPU or Nsight capture was run for this follow-up: doing so would
require deliberately producing a real preregistered regression in exact
matched builds. Existing real-GPU and profiler evidence is not relabeled as a
test of this new trigger. All actual NSYS/NCU target executions still use the
whole-command GPU lock and direct llama affinity path shared with the production
profiler.

## Build-provenance contamination follow-up

The attention-staging investigation exposed a preserved temporary executable
whose embedded commit, binary hash, and cache policy looked like an exact
baseline even though its nested source archive contained unrelated staging-
pool and bounded-growth patches. No contaminated timing or memory value was
accepted. This follow-up adds final-path CMake build registration and makes its
hashed sidecar mandatory for every CMake executable role.

The warning-as-error CPU-only suite used:

```bash
python3 -m json.tool scripts/feature_validation/manifest.schema.json >/dev/null
python3 -m json.tool \
  examples/feature-performance-validation/manifest.example.json >/dev/null
PYTHONWARNINGS=error::ResourceWarning python3 -m py_compile \
  scripts/feature_validation/*.py \
  scripts/feature-performance-validation.py \
  scripts/test-feature-validation.py
PYTHONWARNINGS=error::ResourceWarning \
  python3 scripts/test-feature-validation.py -v
```

Result: 76 tests passed. New tests cover successful
registration, exact Git-root enforcement, CMake-home mismatch, missing
manifest registration, missing sidecar file, altered external hash, invalid
internal fingerprint, source drift after registration, copied binary plus
copied sidecar, and rejection of a non-ignored sidecar that would change its
own source identity. CMake registration combined with a declared direct harness
is covered independently so its source-file list cannot replace the complete
worktree inventory.

A real CPU CMake tree built and registered
`test-cuda-fattn-route-policy` from the exact worktree root. Registration
hashed 3,874 source paths and produced an 808,463-byte sidecar in 0.19 seconds;
the registered executable then passed. The same configured tree ran the
checked-in CTest integration:

```bash
ctest --test-dir /tmp/beellama-build-provenance-real.CaTwAk/build \
  -R '^test-feature-performance-validation$' --output-on-failure
```

Result: 1/1 passed. No CUDA target, model, GPU process, NSYS, or NCU command ran
for this tooling-only hardening.

## Artifact/interrupt correction and NSYS memory inventory

The generic artifact/interrupt correction was independently reviewed from
`1e74eac6c513b7dd9b0445052f3ca999e9a7d28f` against canonical
`c2ea686d662ef36e0f6dcbef9195309b505cb422`. The imported files are byte-for-
byte identical to that toolkit source; the attention-staging investigation
report was not imported. The review accepted the correction because it checks
the exact artifact directory in every containing variant root, repeats the
check for actual run/profile/diagnostic directories, retains source-provenance
identity, tracks the target process-group ID after leader exit, escalates from
SIGTERM to SIGKILL, reaps the leader and descendants, shuts down telemetry, and
preserves interrupted attempts as failed for explicit retry.

The profiler follow-up adds an independent standard-library NSYS SQLite memory
inventory. A configured capture explicitly requests
`--cuda-memory-usage=true` only after checking the exact tool help. Required
categories fail closed; optional categories remain visibly incomplete. The
machine-readable inventory covers CUDA memory events, address-paired
lifetimes, captured outstanding device-allocation high-water, memcpy/memset,
CUDA runtime/driver calls, named VMM calls, and standalone NVTX ranges. Missing
tables/columns/enum meanings are unavailable rather than zero.

Focused warning-as-error checks used:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  python3 scripts/test-feature-validation.py -v \
  ProfilerTest ArtifactLifecycleTest DiagnosticCliTest

PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  python3 scripts/test-feature-validation.py -v \
  ManifestAndScheduleTest.test_ignored_synthetic_probe_does_not_authorize_unignored_root \
  ManifestAndScheduleTest.test_ignore_negation_fails_closed_for_exact_artifact_root \
  ManifestAndScheduleTest.test_nested_variant_roots_must_both_ignore_the_exact_root \
  ManifestAndScheduleTest.test_distinct_variant_roots_check_only_the_containing_root \
  ManifestAndScheduleTest.test_ignored_in_source_artifacts_do_not_change_provenance_identity \
  ArtifactLifecycleTest \
  RunnerTest.test_cleanup_errors_preserve_result_and_telemetry \
  RunnerTest.test_exited_leader_does_not_hide_sigterm_ignoring_descendant \
  RunnerTest.test_real_sigint_reaps_target_group_and_preserves_failure \
  RunnerTest.test_failed_attempt_is_preserved_and_retry_requires_opt_in
```

The first focused suite passed 26 tests in 0.580 seconds. The second passed 11
adversarial artifact/process tests in 3.791 seconds. Synthetic SQLite coverage
includes current and alternate names, present-but-empty tables, absent tables,
missing semantic columns, capture-start deallocation, pinned/device separation,
allocation pairing, copy/memset aggregation, CUDA/VMM API names, and NVTX.

The final warning-as-error suite used:

```sh
#!/bin/sh
if [ "$#" -ne 3 ] || [ "$2" != "-c" ]; then exit 64; fi
exec /bin/sh -c "$3"
```

```bash
PATH="$PWD/.feature-validation-test-bin:$PATH" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  python3 scripts/test-feature-validation.py -v
```

Result: 95 tests passed in 11.300 seconds. The temporary CPU-test `flock` shim
accepted only the exact `LOCK -c COMMAND` shape and exec'd `/bin/sh -c COMMAND`;
it preserved wrapper/target process groups and real signals without competing
with unrelated long GPU work. It was removed immediately after the run. This
does not replace the real-flock result: before the memory extension, the exact
imported correction passed all 89 tests in 11.114 seconds with `/usr/bin/flock`,
and the post-extension targeted process tests used real signals. JSON
schema/example parsing and `git diff --check` also passed before commit.

A read-only compatibility check parsed a pre-existing NSYS 2026.1 SQLite
export twice in 0.04 and 0.05 seconds (28,736 and 28,768 KiB maximum RSS). It found 332 CUDA memory
events, 3,345 copy/memset events, 25,227 CUDA API events, and 19 named VMM
events. Those counts validate parser compatibility only; they are not new
feature-performance or acceptance evidence. The same export had no NVTX table,
which the inventory correctly labelled unavailable. No GPU command or new
NSYS/NCU capture was run for this task.

Automated evidence stops at trace capability, event availability, exact raw
allocation rows, address pairing, captured allocation balances, grouped copy
and API activity, and requirement enforcement. Human investigation still owns
the meaning of unmatched live-at-exit allocations. Robust nested-NVTX
attribution, per-phase device high-water, and allocation backtraces remain
bounded follow-ups. Persistent `nvidia-smi` process VRAM and `/proc` ordinary
host memory remain separate telemetry; CUDA events labelled Pinned are not
relabeled as total process pinned memory and are never inferred from VmLck.
