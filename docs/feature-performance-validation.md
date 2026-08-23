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
   smoke, then low/mid/high direct-command prefill screens. Early-only stages
   may preregister one matched fail-fast pair with no confidence claim; this is
   the recommended way to keep a typical tuned manifest near 30-90 seconds.
2. **Production confirmation** adds the production binary at a preregistered
   selected context depth. NSYS discovery and the one filtered production NCU
   capture attach here, not to every repetition.
3. **Final acceptance** separately runs the mandatory long-context stage.

An early result or an Nsight capture is never reported as proof of end-to-end
performance, resource use, output exactness, or long-context behavior. A
production-confirmation result is likewise not final acceptance. `summary.json`
states which boundary was executed and which gate actually passed. Merely
completing every process is not acceptance.

The selector schema accepts workload properties, tensor-layout properties,
backend capabilities, and execution mode. It rejects architecture and model
family/name selectors. This makes the same ladder usable for live workspaces,
VMM telemetry, allocation strategies, native-quant attention, causal-mask
work, CPU KV placement, and future features.

## Prepare a manifest

Keep the manifest small and preregister it before collecting measurements.
Large outputs must go to the absolute `artifact_root`. By default the runner
rejects a root inside either source repository. A worktree-confined study may
set `artifact_root_policy` to `git_ignored_inside_sources`; this opt-in is
accepted only when the root is below (and not equal to) every containing source
root, is outside `.git`, and `git check-ignore --no-index` proves that the exact
artifact directory itself—not a synthetic child—is ignored by every containing
variant root. The runner repeats this check on each actual study, attempt,
diagnostic, and profile directory immediately before a target starts, including
on resume.

This opt-in deliberately relaxes physical isolation: ignored artifacts can
share a worktree filesystem with source and may be visible to build scripts or
other tools that traverse ignored paths. It protects Git source identity and
accidental commits, but is not equivalent to the default cross-worktree
boundary. Prefer an outside-source root unless worktree confinement is required.

Register every CMake-built executable after its final build, reconfiguration,
or copy and before adding it to a manifest:

```bash
python3 scripts/feature-performance-validation.py register-build \
  --source-root /absolute/path/to/exact/worktree \
  --executable /absolute/path/to/build/bin/llama-bench \
  --cache /absolute/path/to/build/CMakeCache.txt
```

The default output is adjacent to the executable as
`llama-bench.build-provenance.json`. Use `--output /absolute/path` only when an
explicit location is needed and `--force` only after intentionally rebuilding,
reconfiguring, or changing source. The command prints the exact sidecar path
and SHA-256 for the manifest:

```json
"provenance_sidecar": {
  "path": "/absolute/path/to/build/bin/llama-bench.build-provenance.json",
  "sha256": "..."
}
```

Registration fails unless `source_root` is the exact Git worktree root and the
cache's `CMAKE_HOME_DIRECTORY` names that same root. It binds the final binary
path and hash, resolved ELF libraries, full CMake cache, Git commit/tree/dirty
fingerprint, and content hashes for every tracked or untracked non-ignored
source path. A sidecar copied with a binary remains bound to the old path and
is rejected; re-register the copy at its final path. Every CMake executable
role requires a sidecar, and validation rejects missing, stale, altered, or
mismatched registrations before launching a process.

For each baseline and candidate, record:

- the source root, exact commit, and either a clean-tree requirement or one
  exact dirty-tree fingerprint;
- one or more executable roles with an expected SHA-256;
- a registered build-provenance sidecar plus a discoverable CMake cache and
  expected important options, or an explicit reason CMake does not apply;
- every shared model, prompt file, library-like input, or other material input
  as a hashed `inputs` entry.

The runner verifies the sidecar and hashes the executable, all ELF libraries
reported by `ldd`, inputs, harness sources, the CMake cache, the sidecar, and
`nvidia-smi` once at fresh-study or resume provenance capture. Each child then
compares size, mtime, ctime, device, and inode to that cryptographic capture,
while source commit/tree identity is rechecked for every process. This avoids
repeatedly hashing a large model and shared-library set without allowing a
changed file to pass silently; every new invocation or resume performs the
full hashes again. Host provenance records tool versions, CPU and OS identity,
and starting load. Profiler executions additionally hash the `nsys` or `ncu`
binary actually launched.

Place all matched workload settings in `command.common_args`. If the feature
requires different arguments or environment between variants, copy the exact
differences into `controlled_delta`, explain why, and list the allowed option
and environment names. The validator rejects variant deltas that change model,
prompt, repetitions, depth, batch/ubatch, threads, affinity, GPU placement,
flash-attention mode, cache layout, or seed. It also parses target-specific
option arity. Common-argument binaries such as `llama-cli`, `llama-server`, and
`llama-perplexity` use bare `-nkvo`/`--no-kv-offload`; both of these are invalid
for those targets:

```text
--no-kv-offload 1
--no-kv-offload=1
```

`llama-bench` has a separate parser: both `-nkvo` and
`--no-kv-offload` require a `0` or `1` value. The built-in llama schema derives
that distinction from the direct executable name, so the valid bench long
alias is not mistaken for the common zero-arity flag. A local harness must use
`builtin=none` and explicitly declare its option arities.

Unknown options fail unless their zero/one arity is declared in `cli_schema`.
`taskset` is rejected anywhere. Shells, `env`, `flock`, `timeout`, Python, and
other opaque wrappers cannot be profiler targets.

Every stage needs a finite timeout and a visible progress description. Use the
target's native progress option where it has one. The runner also emits a
five-second heartbeat for long or unexpectedly stalled processes. If preparing
a build for a study on the current benchmark host, hold the shared CUDA build
lock for the complete command and do not build wider than 12 parallel jobs.

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

For a fast ladder, add `screening_policy` to the two smoke stages and the
kernel screen:

```json
{
  "kind": "single_pair_fail_fast",
  "order": "AB",
  "regression_threshold_percent": 2.0,
  "confidence_claim": "none"
}
```

This executes exactly one fresh base/candidate pair. Alternate `AB` and `BA`
between adjacent early stages to limit systematic order bias. The result keeps
the raw pair, weighted per-screen values, point change, and preregistered
regression signal. It deliberately records `confidence_interval=null` and
cannot claim improvement, equivalence, or acceptance. A regression signal
stops the ladder; a clear signal only authorizes moving to the statistical
production gate. `screening_policy` is rejected on production confirmation or
long-context acceptance, and it is mutually exclusive with statistical
`decision_policy`.

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

Each attempt records identity-check, telemetry-startup, target, cleanup, and
locked-wrapper timing. A fresh `summary.json` aggregates those fields and
reports wall-minus-target overhead against the 30-90 second early target. Use
the fresh-study numbers for overhead decisions; a resume summary reuses prior
attempt artifacts and therefore labels its timing mode as `resume`.

A successful run is skipped on `--resume`. A failed, timed-out, contaminated,
or incomplete attempt is retained and stops the study. It is never dropped or
overwritten. After correcting an external prerequisite, an explicit
`--retry-failed` creates the next numbered attempt:

For every new process, the internal executor atomically records its PID plus
Linux process start time before telemetry or target creation, then records the
direct target's process-group ID. The outer launcher installs SIGINT and SIGTERM
handlers before starting `flock`. On parent interruption it signals only that
identified executor—not the whole wrapper group—so `flock` continues to hold
the GPU lock while the executor cleans up. Repeated parent signals are recorded
but do not interrupt cleanup again. The executor tracks the target group after
leader exit, escalates from SIGTERM to SIGKILL while descendants remain, reaps
the direct child, stops telemetry, records the failed result, and only then
exits the locked lifecycle. A bounded outer fallback uses the same recorded
target group before terminating an unresponsive wrapper.

On resume, the runner reconciles `state.json` with every numbered attempt
directory for the scheduled run. A directory with no state entry is never
reused, even when it contains a child `result.json` reporting success. Once no
recorded executor or target group remains, the runner records a failed recovery
with no metric and seals every preserved file in `attempt-evidence.json`.
Normally the recovery is also written as `interrupted-result.json`. If the
attempt was already sealed immediately before the parent died, its immutable
seal is verified and retained while the recovery record lives in `state.json`,
rather than adding a file that would invalidate that seal. The seal binds the
file inventory and its SHA-256s to the run/attempt, manifest SHA-256, and
provenance identity fingerprint. Tampering fails closed. The next retry is
numbered above the highest attempt in state or on disk, so repeated
interruptions preserve attempts 1, 2, and so on. If an ownership record still
identifies a live process, resume refuses to recycle or seal that directory and
asks the user to retry later.

Cleanup errors are recorded without discarding telemetry or the result. The
attempt is failed in `state.json` and is never resumable as successful evidence;
a later attempt still requires `--resume --retry-failed`.

```bash
python3 scripts/feature-performance-validation.py run \
  /absolute/path/to/manifest.json --through early --resume --retry-failed
```

Do not delete a failed attempt to make a report look complete. If the manifest
or any provenance identity changes, start a new deterministic study rather
than combining unlike runs.

Catchable parent SIGINT/SIGTERM and normal timeout/error paths have the complete
ownership contract above. No userspace runner can guarantee cleanup after
SIGKILL to the entire wrapper/executor group, kernel failure, host reset, or
power loss. After such an event, resume remains fail-closed while a recorded
process group is live; inspect it and the GPU lock before retrying. A parent-only
SIGKILL does not release the separately sessioned wrapper: the lock-owning
executor continues its finite target lifecycle, and the resulting unindexed
directory is conservatively recovered as failed afterward.

## Statistics

Statistical stages start with three independent matched pairs in balanced order
`AB, BA, AB`. This includes production confirmation and final acceptance, and
any early stage that intentionally omits `screening_policy`. Each observation
is a new process. The runner calculates
candidate/baseline paired log ratios, their geometric percent change, and a
two-sided Student-t 95% interval. The report includes every raw per-screen and
per-pair sample.

The manifest preregisters improvement, regression, and equivalence effects. A
three-pair result extends to pairs four and five, in `BA, AB` order, only when
the confidence interval and effect thresholds classify it as inconclusive. A
conclusive three-pair result records pairs four and five as
`not_run_by_preregistered_conclusive_rule`. No subset selection, silent run
dropping, threshold changes, or post-hoc extension is supported.

Every performance stage also preregisters `decision_policy`. Only
`improvement` and/or `equivalent` may be acceptable; a regression always fails,
and an inconclusive result after five pairs is explicitly unresolved and fails
closed. Stage records keep `execution_status=completed` separate from
`status=passed|failed|unresolved`. `acceptance_complete` is emitted only when
the final mandatory long-context gate passed its policy.

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

Two profiler paths share the same fail-closed direct-target implementation:

- `profiler` answers one preregistered production-stage investigation.
- `early_diagnostics` is an opt-in response to a preserved early
  `regression_signal`. It never runs after a clear screen and cannot change the
  failed screening result.

The optional production profiler must target the selected-depth production
stage and sets `profile_repetitions` to one. For the established CPU 0-2
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

For a graph-capable or graph-only stage, `cuda_graph_trace` must preregister
node tracing and its launch origin. Before launch, the runner inspects the exact
NSYS binary's version/help for `--cuda-graph-trace` node support. It then passes
the explicit option (normally `--cuda-graph-trace=node:host-only`) and rejects
an export with no nonzero graph-node IDs. Unsupported tool versions therefore
fail before or immediately after discovery instead of silently producing a
non-graph inventory.

The NSYS SQLite parser inventories kernel names, graph-node IDs, grid/block
shapes, counts, chronological launch indices, and same-name occurrence indices.
The selector is applied to that inventory. The runner emits one exact-name NCU
command only when the selected occurrences form one contiguous range after the
kernel filter; otherwise it fails closed and asks for a narrower discovery or
selector. Thus `--launch-skip` and `--launch-count` are derived for the current
binary, hardware, and capture rather than copied between builds. Re-run NSYS
after any relevant identity changes.

An optional `memory_evidence` block preregisters `mode` (`required` or
`optional`) and the evidence categories needed from this same discovery. When
configured, the runner inspects the exact NSYS help and, when supported,
explicitly passes `--cuda-memory-usage=true`. Required mode fails before launch
if the option is unsupported, and fails after export if any requested category
is unavailable or only partial. Optional mode retains the same diagnostics but
does not turn unavailable evidence into a successful memory claim.

`memory-inventory.json` is schema-tolerant across known NSYS SQLite naming
variants and separately reports:

- CUDA allocation/deallocation events by memory kind, including device,
  pinned, pageable, managed, and static kinds when NSYS supplies them, plus
  captured outstanding-allocation high-water separately for each kind;
- address-paired lifetimes plus every unmatched allocation/deallocation;
- captured outstanding device-allocation high-water by process/device, which
  is not total process VRAM or VMM reservation;
- memcpy/memset counts, bytes, duration, and memory-kind groupings;
- aggregated CUDA runtime/driver memory calls and named VMM reserve, create,
  map, access, unmap, release, and address-free calls; and
- standalone NVTX ranges when an NVTX table is present.

Each category is `available`, `partial`, or `unavailable`. An available table
with zero rows is measured zero; a missing table, required column, enum
meaning, or unresolved identity is never reported as zero. Raw allocation rows
and lifetimes remain in JSON; raw copy/API rows remain in the SQLite export and
the JSON aggregates retain their complete counts and byte totals.

The allocation-event balance is deliberately named *captured outstanding
high-water*. It does not replace the persistent roughly-1-Hz `nvidia-smi`
process-VRAM series, `/proc/PID/status` VmRSS/VmHWM ordinary-host-memory
series, or a dedicated pinned-memory audit. CUDA events labelled `Pinned` are
allocator events only; the toolkit never infers pinned bytes from VmLck or
process VRAM. Robust nested-NVTX phase attribution, per-phase device high-water,
and allocation backtraces remain bounded follow-ups because their cross-version
joins and symbolization would create a substantially larger profiling
framework.

After the separate NCU process, the runner requires the expected nonempty
`.ncu-rep`, hashes it, imports its raw CSV with the same NCU binary, and checks
the observed demangled kernel, unique capture count, and available grid/block
shapes against the NSYS-derived plan. Missing report fields are
`unverifiable`; mismatches fail. This post-capture check accounts for
cross-process launch-order drift rather than assuming the discovery skip/count
still selected the intended work.

Profiler output is a kernel investigation, not a benchmark repetition. The
intended pattern is one production NSYS discovery followed by one filtered
production NCU capture for a preregistered question.

### Regression-triggered agent diagnostics

An early diagnostic entry must name a smoke or direct-kernel stage that uses
`single_pair_fail_fast`, declare `trigger=regression_signal_only`,
`on_clear=skip`, `evidence_claim=diagnostic_only`, and target both variants in
baseline/candidate order. It also preregisters the kernel-family selector,
metrics, graph-trace applicability, and one of these screen policies:

- `fixed` names the one screen to inspect.
- `largest_observed_regression` deterministically selects the most negative
  benefit among the preserved raw matched screen observations. The report
  retains every span and the selection calculation; it does not select a
  favorable subset.

The checked-in manifest example contains the complete schema. To opt in while
running the early ladder:

```bash
python3 scripts/feature-performance-validation.py run \
  /absolute/path/to/manifest.json --through early --diagnose-regressions
```

A clear early result pays no profiler or extra provenance-hashing cost. If the
stage reports a regression, the runner remains failed and launches a diagnostic
phase. It performs an independent NSYS discovery and one discovery-derived,
verified NCU capture for baseline, then repeats that sequence independently for
candidate. Kernel spelling, shapes, graph-node IDs, occurrences, and launch
indices are never transferred between binaries. This costs more than the fast
screen by design and is reported separately from its 30-90 second budget.

If the screen was run without the flag, or a partial diagnostic must be safely
resumed, use:

```bash
python3 scripts/feature-performance-validation.py diagnose \
  /absolute/path/to/manifest.json

python3 scripts/feature-performance-validation.py diagnose \
  /absolute/path/to/manifest.json --resume
```

The deterministic
`early-diagnostics/agent-diagnostic-report.json` exposes, for each variant:

- NSYS total/matching/selected launch counts, exact kernel, discovery indices,
  same-name occurrences, shapes, and graph-node IDs;
- verified NCU capture count, raw per-capture requested metric values, and any
  verification issues;
- memory-category availability, captured allocation high-water, allocation
  lifecycle counts, copy/memset groups, and CUDA/VMM API groups when requested;
- NSYS/NCU target and locked-lifecycle timing plus paths to the raw reports,
  inventory, plan, stdout/stderr, and verification record.

Graph-replay diagnostics must request node tracing explicitly and verify
nonzero graph-node IDs under the exact NSYS version. Missing tools, unsupported
graph tracing, empty/mismatched exports, contamination, or cross-process
launch-order drift fail the diagnostic and remain visible in its report. The
compact report is intended to give an optimization agent an immediate lead; it
does not convert Nsight counters into correctness, equivalence, end-to-end
speed, memory, production, or long-context evidence.

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
- executed-versus-passed gate state and fail-closed final acceptance;
- persistent GPU/proc telemetry and sampler cleanup;
- explicit NSYS graph-node capability/capture verification, the derived
  one-command NCU plan, and post-NCU report verification.
- opt-in early-regression diagnostics with deterministic span selection,
  independent per-variant discovery/capture, and a compact agent-facing report.

Manual and study-specific:

- choosing representative workloads, layouts, capability declarations,
  thresholds, context depths, and real ubatch weights;
- preparing clean matched builds and models (with build parallelism no wider
  than 12 parallel jobs on the current benchmark host);
- deciding whether direct pinned-memory instrumentation or a VMM/allocation
  trace is required;
- reviewing raw logs, contamination flags, thermals/clocks, and host load;
- running and interpreting the final long-context campaign for the feature,
  rather than treating an early toolkit smoke as acceptance.

Keep `.nsys-rep`, SQLite, NCU reports, stdout/stderr, and telemetry JSON outside
Git. Check in only a schema/inventory or a concise truthful summary of valid
evidence. Never promote a launcher failure, wrong identity, contaminated run,
historical capture, or incomplete long-context run as current evidence.

## Maintenance boundary

The implementation is intentionally split into a thin CLI, manifest/provenance
and statistics core, clean-process runner, telemetry owner, and profiler
parser/planner. The merge-readiness pass removed duplicate provenance-spec and
GPU-lock launch implementations and removed the legacy single-executable
manifest shape; the JSON schema and runtime now use the same executable-role
contract. The remaining standard-library Python is mostly fail-closed
validation, artifact inventory, lifecycle cleanup, and parsers with separate
tests. Interrupt ownership and attempt reconciliation remain in the runner
because they extend its existing process/state contract and share its cleanup
and provenance primitives; they do not add a parallel supervisor or artifact
framework. There is no copied CUDA framework or project build dependency.
Further code should enter a new module only when it has an independent
lifecycle or artifact contract, rather than growing another parallel runner.
