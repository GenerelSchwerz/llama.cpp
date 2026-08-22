# Feature/performance validation toolkit

This is the current how-to for `scripts/feature-performance-validation.py`.
The toolkit is developer tooling only. It does not add a runtime option, a
production kernel, a model-family special case, or a build dependency. Python,
NCU, and NSYS are not required to build BeeLlama; the runner itself uses only
the Python standard library, and the NVIDIA profilers are optional runtime
tools.

The checked-in [manifest example](../examples/feature-performance-validation/manifest.example.json)
is a template, not current evidence. Replace every absolute path, hash, commit,
workload geometry, regex, threshold, and capability before running it. The
machine-readable shape is
[`scripts/feature_validation/manifest.schema.json`](../scripts/feature_validation/manifest.schema.json);
the runner's fail-closed checks are authoritative.

## Evidence boundary

The validation ladder deliberately has three boundaries:

1. **Early screening** runs output exactness first, short prefill and decode
   smoke, then low/mid/high direct-command prefill screens. It is designed so a
   typical tuned manifest can make an early decision in about 30-90 seconds.
2. **Production confirmation** adds the production binary at a preregistered
   selected context depth. NSYS discovery and the one filtered production NCU
   capture attach here, not to every repetition.
3. **Final acceptance** separately runs the mandatory long-context stage.

An early result or an Nsight capture is never reported as proof of end-to-end
performance, resource use, output exactness, or long-context behavior. A
production-confirmation result is likewise not final acceptance. `summary.json`
states which boundary was actually completed.

The selector schema accepts workload properties, tensor-layout properties,
backend capabilities, and execution mode. It rejects architecture and model
family/name selectors. This makes the same ladder usable for live workspaces,
VMM telemetry, allocation strategies, native-quant attention, causal-mask
work, CPU KV placement, and future features.

## Prepare a manifest

Keep the manifest small and preregister it before collecting measurements.
Large outputs must go to the absolute `artifact_root`, which the runner rejects
if it is inside either source repository.

For each baseline and candidate, record:

- the source root, exact commit, and either a clean-tree requirement or one
  exact dirty-tree fingerprint;
- one or more executable roles with an expected SHA-256;
- a discoverable CMake cache plus expected important options, or an explicit
  reason CMake does not apply;
- every shared model, prompt file, library-like input, or other material input
  as a hashed `inputs` entry.

The runner hashes the executable and all ELF libraries reported by `ldd`, saves
the entire discoverable CMake cache, verifies the source commit/tree before the
study and before each process, and fingerprints the inputs. A resume fails if
any stable identity changed. Host provenance records tool versions, CPU and OS
identity, and starting load. Profiler executions additionally hash the `nsys`
or `ncu` binary actually launched.

Place all matched workload settings in `command.common_args`. If the feature
requires different arguments or environment between variants, copy the exact
differences into `controlled_delta`, explain why, and list the allowed option
and environment names. The validator rejects variant deltas that change model,
prompt, repetitions, depth, batch/ubatch, threads, affinity, GPU placement,
flash-attention mode, cache layout, or seed. It also parses option arity; for
example, both of these are invalid because `--no-kv-offload` is zero-arity:

```text
--no-kv-offload 1
--no-kv-offload=1
```

Unknown options fail unless their zero/one arity is declared in `cli_schema`.
`taskset` is rejected anywhere. Shells, `env`, `flock`, `timeout`, Python, and
other opaque wrappers cannot be profiler targets.

Every stage needs a finite timeout and a visible progress description. Use the
target's native progress option where it has one. The runner also emits a
five-second heartbeat for long or unexpectedly stalled processes. If preparing
a build for a study, do not build wider than `-j6`.

## Ladder requirements

The schema requires this order:

- one exactness stage using a stdout or named-file SHA-256 comparison;
- a short prefill smoke and a short decode smoke;
- one prefill `direct_command` screen with low, mid, and high token spans;
- one selected-depth `production_binary` confirmation;
- one final mandatory `long_context_acceptance` stage.

Decode smoke must state whether CUDA-graph replay applies. When it does, its
execution mode must be `cuda_graph_replay` and the capability list must contain
`cuda_graphs`. When it does not, the manifest must say why.

Each direct prefill screen records `span_tokens`, the number of times that span
occurs in the real ubatch geometry, and the same number as `weight`. Use a
weighted harmonic aggregate for throughput and a weighted arithmetic aggregate
for latency-like quantities. Do not invent weights to favor a result; derive
them from the preregistered production prompt/ubatch geometry.

## Validate and run

Validation includes live provenance, so the untouched example intentionally
fails until its placeholder paths and hashes are replaced.

```bash
python3 scripts/feature-performance-validation.py validate /absolute/path/to/manifest.json

# Exactness, both short smokes, and the weighted direct screen.
python3 scripts/feature-performance-validation.py run \
  /absolute/path/to/manifest.json --through early

# Resume the same deterministic study and add selected-depth confirmation.
python3 scripts/feature-performance-validation.py run \
  /absolute/path/to/manifest.json --through production --resume

# The separate, mandatory final gate. This includes any earlier missing stage.
python3 scripts/feature-performance-validation.py run \
  /absolute/path/to/manifest.json --through acceptance --resume
```

The artifact directory is deterministic:
`ARTIFACT_ROOT/STUDY_ID-MANIFEST_SHA256_PREFIX`. Every target is a fresh direct
process. GPU runs are launched in the safely quoted whole-lifecycle form
`flock /tmp/beellama-single-gpu.lock -c <internal-executor-command>`; the
internal executor owns telemetry preflight, the direct target, and telemetry
cleanup. Both the exact argv/environment command and the exact flock command
are saved in each attempt.

A successful run is skipped on `--resume`. A failed, timed-out, contaminated,
or incomplete attempt is retained and stops the study. It is never dropped or
overwritten. After correcting an external prerequisite, an explicit
`--retry-failed` creates the next numbered attempt:

```bash
python3 scripts/feature-performance-validation.py run \
  /absolute/path/to/manifest.json --through early --resume --retry-failed
```

Do not delete a failed attempt to make a report look complete. If the manifest
or any provenance identity changes, start a new deterministic study rather
than combining unlike runs.

## Statistics

Performance stages start with three independent matched pairs in balanced
order `AB, BA, AB`. Each observation is a new process. The runner calculates
candidate/baseline paired log ratios, their geometric percent change, and a
two-sided Student-t 95% interval. The report includes every raw per-screen and
per-pair sample.

The manifest preregisters improvement, regression, and equivalence effects. A
three-pair result extends to pairs four and five, in `BA, AB` order, only when
the confidence interval and effect thresholds classify it as inconclusive. A
conclusive three-pair result records pairs four and five as
`not_run_by_preregistered_conclusive_rule`. No subset selection, silent run
dropping, threshold changes, or post-hoc extension is supported.

## Telemetry and clean-process evidence

Each locked GPU lifecycle starts exactly one persistent approximately 1 Hz
`nvidia-smi -q -x -l 1` process. It is not a loop that spawns `nvidia-smi` and
`awk` every 250 ms. The first XML sample is a clean-GPU preflight; an existing
compute-capable (`C`/`C+G`, or unknown-type) process aborts the attempt. Ambient
graphics-only contexts such as a display server are recorded separately and do
not masquerade as benchmark processes. Subsequent samples retain GPU/driver
identity, clocks, temperature, power, device memory, and per-process VRAM.

A separate lightweight reader samples `/proc/PID/status` for the target process
tree and relevant `/proc/meminfo` fields. Reports keep three concepts separate:

- process VRAM comes from NVIDIA's per-process used-memory field;
- ordinary host memory comes from process-tree VmRSS/VmHWM and related proc
  status plus host meminfo;
- pinned host memory is `null` unless a future direct allocation counter is
  added.

`VmLck`, `VmPin`, and process VRAM are retained only as observations and are
never converted into a pinned-memory claim. The sampler is started in its own
owned process group, terminated in every success/error/timeout path, joined,
and reported as reaped. A foreign GPU process seen during the target makes the
attempt invalid.

The roughly 1 Hz GPU cadence can miss a sub-second peak; use a named manual
allocation audit when exact transient or page-locked allocation accounting is
an acceptance requirement.

## NSYS discovery and NCU capture

The optional profiler section must target the selected-depth production stage
and sets `profile_repetitions` to one. For the established CPU 0-2
`llama-bench` shape, the direct target must carry `-C 0x7 --cpu-strict 1`.
Other direct llama tools must carry `--cpu-strict 1` plus both native
`--cpu-range` and `--cpu-range-batch` controls. Never put `taskset` before or
after a profiler.

```bash
# One locked direct-target NSYS discovery, SQLite export, parse, and NCU plan.
python3 scripts/feature-performance-validation.py profile \
  /absolute/path/to/manifest.json

# Execute the one discovery-derived filtered production NCU capture later.
python3 scripts/feature-performance-validation.py profile \
  /absolute/path/to/manifest.json --resume --execute-ncu
```

The NSYS SQLite parser inventories kernel names, graph-node IDs, grid/block
shapes, counts, chronological launch indices, and same-name occurrence indices.
The selector is applied to that inventory. The runner emits one exact-name NCU
command only when the selected occurrences form one contiguous range after the
kernel filter; otherwise it fails closed and asks for a narrower discovery or
selector. Thus `--launch-skip` and `--launch-count` are derived for the current
binary, hardware, and capture rather than copied between builds. Re-run NSYS
after any relevant identity changes.

Profiler output is a kernel investigation, not a benchmark repetition. The
intended pattern is one production NSYS discovery followed by one filtered
production NCU capture for a preregistered question.

## Direct harnesses

A normal executable role is already a pluggable direct-command path. A small
native harness may be used as a profiler target only when its executable role
declares `direct_harness.kind=native_executable`, native llama affinity, and an
expected SHA-256 for every production source file it depends on. Any mismatch
fails before execution.

This toolkit intentionally does **not** add a copied CUDA-event A/B harness.
The causal-mask investigation showed that a small copy quickly acquires
production launch, tail-padding, graph, and source-identity coupling. A bounded
follow-up is appropriate only when it can compile production source directly,
interleave A/B CUDA-event measurements, weight representative prefill spans by
real ubatch geometry, and replay representative decode CUDA graphs without a
parallel implementation. Until then, use the reusable direct-command path.

## Automated versus manual

Automated:

- schema/arity/selector validation and controlled A/B deltas;
- source, binary, ELF-library, CMake-cache, input, host, and profiler identity;
- deterministic artifacts, balanced clean processes, progress, resume, and
  preserved failures;
- exactness ordering, short smokes, weighted screens, selected-depth and final
  stage separation;
- paired statistics and the preregistered 3-to-5 extension rule;
- persistent GPU/proc telemetry and sampler cleanup;
- NSYS SQLite discovery inventory and the derived one-command NCU plan.

Manual and study-specific:

- choosing representative workloads, layouts, capability declarations,
  thresholds, context depths, and real ubatch weights;
- preparing clean matched builds and models (with build parallelism no wider
  than `-j6`);
- deciding whether direct pinned-memory instrumentation or a VMM/allocation
  trace is required;
- reviewing raw logs, contamination flags, thermals/clocks, and host load;
- running and interpreting the final long-context campaign for the feature,
  rather than treating an early toolkit smoke as acceptance.

Keep `.nsys-rep`, SQLite, NCU reports, stdout/stderr, and telemetry JSON outside
Git. Check in only a schema/inventory or a concise truthful summary of valid
evidence. Never promote a launcher failure, wrong identity, contaminated run,
historical capture, or incomplete long-context run as current evidence.
