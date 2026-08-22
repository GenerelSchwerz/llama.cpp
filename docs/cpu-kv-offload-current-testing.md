# CPU KV-offload current testing and setup

This is the sole runnable protocol for the published CPU-KV line. Its exact
source-bearing baseline is
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`; documentation-only descendants,
including the PR 9 consolidation, do not change that production source or its
binary identity. The retained local journal and Experiments 001-019 come from
`6a20757854395309b32248dd4109d73e99c3e675`. Source and binary identities are
authoritative if this document ever disagrees with behavior.

PR 8 is refreshed onto the documentation-only base
`8e858fcec39049fa028ce6fcb144a0c08b03abd3`. Its rebased production/source
checkpoint is `107b926e5`; path, tree, and binary-delta comparison against the
previous published head `0c8df007a504f16aa35fc5982303e3e1b9883331`
shows that the live-context implementation, CLI, generated arguments, and
tests are unchanged. That equivalence does not replace the exact source-level
disabled-path gate: final PR 8 evidence compares this exact base with fresh
candidate processes at measured runtime head
`4cdd2d74e7acc432fcdde4a9d1e5e832fe80e148`, using both omission and explicit
off. The 2026-08-21 A/B/A gate passed exact output and PPL, full upfront
reservation, identical allocation/VMM fields, and neutral repeated 4K/30K
throughput; Experiment 021 owns the exact measurements and artifacts.

The companion documents have deliberately separate roles:

- [`cpu-kv-offload-development.md`](cpu-kv-offload-development.md) records
  durable decisions and protocol transitions.
- [`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md) preserves
  valid Experiments 001-020 plus W06 and indexes exact post-KV branch
  identities.
- [`cpu-kv-offload-vram-roadmap.md`](cpu-kv-offload-vram-roadmap.md) ranks
  retained, pending, rejected, and research-only VRAM work.
- [`vram-feature-isolation-plan.md`](vram-feature-isolation-plan.md) defines
  how independently published candidates must be compared before integration.
- [`cpu-kv-offload-feature-delta.md`](cpu-kv-offload-feature-delta.md) lists
  the source-backed difference from BeeLlama v0.4.3.

Update this document first when supported controls, the exactness oracle,
benchmark shape, progress mechanism, or artifact contract changes. Old Git
revisions are historical evidence, not runnable current protocol.

## Integration-base policy for feature worktrees

Treat the canonical KV-offload integration branch (currently
`beellama-kv-cpu-offload` and its configured remote-tracking branch) as the
main integration line for this work. A feature worktree's original base commit
records where the investigation started; it is not a reason to remain pinned
there while the integration branch advances.

Feature branches intended for KV-offload should incorporate the latest
validated integration head before new acceptance measurements, final
merge-readiness validation, and handoff. Prefer updating the feature branch to
the integration line over carrying an avoidable old-base delta. Merge or rebase
according to the branch's publication policy, preserve all feature work, and
report substantive conflicts rather than resolving them by dropping either
side. Open their PRs against the KV-offload integration branch, not the
repository's general-purpose `main` branch.

Every result must still record the exact base and candidate commits actually
measured. After incorporating a newer integration head, invalidate the prior
build registration, re-register the exact source and binary identities, rebuild
when binary-affecting inputs changed, and rerun coverage proportional to the
incoming delta. Do not relabel evidence from the older base as measurements of
the updated branch. Avoid needless update churn during a running matched pair;
finish or invalidate that pair, update at the next clean process boundary, and
then collect evidence against the new base.

## Current source surface

The published KV source contains these controls:

| Control | Current meaning |
|---|---|
| `--no-kv-offload` | Keep attention KV in host memory. |
| `--kv-cpu-pinned` | Use supported accelerator-visible pinned host buffers for host KV. |
| `--recurrent-state-offload` | Keep supported hybrid recurrent state on the accelerator while attention KV remains on the host. |
| `--kv-gpu-layers N` | Keep the first `N` target-owned attention-KV layers on the accelerator. |
| `--spec-draft-kv-gpu-layers N` | Override target placement for independently owned draft KV; omission inherits and zero explicitly selects host placement. |
| `--phase-aware-workspace` | Share and resize sequential target/MTP compute backing between prompt and generation phases. |
| `--live-context-workspace` | Independently and opt-in, bound supported standard-attention graph reservations by padded live physical KV extent. |
| `--spec-mtp-rs-planes N` | Cap total target recurrent planes, including the current plane. |
| `llama-bench --kv-memory` | Capture KV/component checkpoints, physical device/accelerator-host/ordinary-host allocation classes, and CUDA VMM live/mapped/high-water telemetry. |

Live-context comparisons vary only
`--no-live-context-workspace`/`--live-context-workspace` and keep phase-aware
workspace disabled unless their interaction is the stated subject. The policy
is default-off, does not resize persistent KV, and requires standard attention
memory that advertises bounded reservation. KVarN, ISWA, recurrent-only, and
other unsupported layouts intentionally retain the full plan. The server asks
CUDA to trim transient-pool mappings only after all slots become idle and only
when bounded sizing was effective.

MTP must use the target physical ubatch. Omit
`--spec-draft-ubatch-size`, or set it equal to `--ubatch-size` only when the
explicit parser path is under test. A different MTP draft ubatch is rejected.

The published KV base does **not** contain native quantized FlashAttention,
compact causal masking, or bounded host-KV attention staging. Do not expect
those later fields or use an associated later option in a current KV-base
command. Live-context source and its preset/generated argument documentation
remain owned by this PR 8 branch until it lands. See the evidence index for
the exact PR identities.

`llama-perplexity` declares its full logical batch as the maximum output-row
requirement before context creation. Therefore matched PPL runs may enable or
disable `--phase-aware-workspace` without reducing the all-logits capacity the
tool needs. Treat an output-capacity assertion as a failed run, not as quality
evidence.

## Retired and historical controls

Do not put retired controls into a current command, including defensive
environment cleanup. `GGML_KV_CPU_PINNED` and
`GGML_RECURRENT_STATE_OFFLOAD` are historical and removed. Use the supported
CLI controls directly. Use a current `LLAMA_ARG_*` variable only when that
environment path is the subject of the test.

Do not use `taskset` in a current benchmark, server, exactness, or profiler
command. Express llama worker placement with `--cpu-range`,
`--cpu-range-batch`, `--cpu-mask`, the corresponding batch control, and
`--cpu-strict`; `llama-bench` uses `-C` for the worker mask. Historical
native-Q8-composed manifests and commands are archived with their source
branch and are not current KV-base templates.

## Current host and inputs

- Experimental worktree: `/home/gencoolpc/beellama-kv-offload`, branch
  `exp/kv-cpu-offload`.
- Known BeeLlama baseline: `/home/gencoolpc/beellama.cpp`; keep it unchanged.
- CUDA build: `build-cuda-all`, Release, native CPU, CUDA FlashAttention,
  compute architecture 120, default quant-pair matrix.
- GPU: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB usable process memory, compute
  capability 12.0.
- CPU: Intel Core Ultra 9 285K. Decode workers use CPUs 0-2; batch workers may
  use CPUs 0-23.
- Model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.
- Projector:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf`.
- Cached corpus:
  `/home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt`,
  606,662 bytes, SHA-256
  `8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`.

These paths describe the current benchmark host, not portable defaults.

## Clean-process and GPU-serialization contract

Every controlled GPU run must satisfy all of these rules:

1. Record `git status --short --branch`, `git rev-parse HEAD`, binary version
   and SHA-256, build options, model/input hashes, driver, and hardware.
2. Refuse to start while an unrelated compute process is using the GPU.
3. Acquire `/tmp/beellama-single-gpu.lock` before model load and hold it until
   the llama process has exited and samplers/profiler reports have finalized.
4. Start a fresh llama process for every A, B, and closing A case. Do not reuse
   prompt cache or allocator state across configurations.
5. Put the server, client, progress monitor, and samplers inside the same lock
   owner for a server lifecycle. Locking only the server launch is insufficient.
6. Expose progress for every duration-uncertain run and preserve the progress
   artifact.

Every runnable GPU template uses the exact outer form
`flock /tmp/beellama-single-gpu.lock -c 'COMMAND'`. Do not use flock's
direct-command form. The single-quoted command must contain the complete GPU
process, pipelines/redirections, samplers, and teardown; use safe inner double
quotes for shell variables or arguments that require quoting.

For one-process tools the supported shape is:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  build-cuda-all/bin/llama-bench BENCH_ARGS \
    -C 0x7 --cpu-strict 1 --progress
'
```

For a server lifecycle, use a harness that owns the lock, verifies the clean
GPU, starts the server, waits for health, runs the client and samplers, stops
the server gracefully, waits for every child, and only then exits. The
maintained exactness runner already follows that fresh-process lifecycle; put
the whole runner under the outer lock. It records health-ready elapsed time and
a process/GPU startup sample before the first request, then preserves the
request-level process, VRAM, timing, output, and progress artifacts.

## Shared CUDA build scheduling

Serialize CUDA builds from every worktree on this host with the dedicated
build lock. The lock and the per-build job limit address different resource
constraints: the lock prevents separate builds from overlapping, while the job
limit bounds CPU and memory pressure inside the one active build.

This 24-core, 64-GiB benchmark host uses at most 12 parallel build jobs:

```bash
flock /tmp/beellama-cuda-build.lock -c \
  'cmake --build build-cuda-all --target llama-server --parallel 12'
```

Keep the entire `cmake --build` command inside the safely quoted lock command.
Queued builds wait on the lock and must retain the same 12-job cap when they
start. Select only the targets required by the current gate. The build lock is
separate from `/tmp/beellama-single-gpu.lock`; compiling does not acquire the
GPU-run lock, and a benchmark or profiler must still follow its GPU locking
contract.

The 12-job value is host-specific. Do not copy it to a different machine
without checking its CPU topology, available memory, and peak compiler-process
memory. Historical reproduction documents may contain higher parallelism; they
reproduce their named build and are not current scheduling templates.

## Preflight and binary identity

Run before a measurement:

```bash
cd /home/gencoolpc/beellama-kv-offload
git status --short --branch
git rev-parse HEAD
llama-benchy --version
sha256sum build-cuda-all/bin/llama-server \
  build-cuda-all/bin/llama-bench \
  build-cuda-all/bin/llama-perplexity \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt
flock /tmp/beellama-single-gpu.lock -c '
  build-cuda-all/bin/llama-server --version
  build-cuda-all/bin/llama-bench --list-devices
  nvidia-smi
'
```

Do not present an uncommitted or differently configured binary as evidence for
the current commit. Documentation-only dirt does not change a binary; source
or build changes do.

Before placing any CMake-built executable in a feature-validation manifest,
register it at its final path:

```bash
python3 scripts/feature-performance-validation.py register-build \
  --source-root /absolute/path/to/source \
  --executable /absolute/path/to/build/bin/llama-bench \
  --cache /absolute/path/to/build/CMakeCache.txt
```

This writes an adjacent `.build-provenance.json` sidecar. Record that absolute
path and its SHA-256 in the executable's manifest entry. The validator requires
the sidecar for every CMake-built role and rejects a missing, copied, stale, or
mismatched sidecar. Registration also requires the declared source root to be
the exact Git worktree root and to equal the cache's `CMAKE_HOME_DIRECTORY`;
an ignored archive nested inside another worktree is not a source identity.
Re-register after rebuilding, reconfiguring, changing source, or copying the
binary to a new final path. Do not reuse a sidecar from another path.

## Manifest-driven early feature screen

Every new CPU-KV-derived performance or memory feature starts with a
preregistered manifest based on
[`feature-performance-validation.md`](feature-performance-validation.md) and
the checked-in schema/example. Select screens by workload, tensor layout, and
backend capability; do not select by model or architecture name. Before a
deeper campaign, run:

```bash
python3 scripts/feature-performance-validation.py validate MANIFEST.json
python3 scripts/feature-performance-validation.py run \
  MANIFEST.json --through early
```

The early ladder runs exactness first, then short prefill/decode and
representative low/mid/high screens with real-ubatch weighting and fresh clean
processes. A preregistered `single_pair_fail_fast` stage is a screening signal
with no confidence interval or acceptance claim. Preserve all raw samples and
stop on a regression signal.

When the manifest preregisters `early_diagnostics`, an agent may opt into
direct-target Nsight discovery and filtered capture after a regression:

```bash
python3 scripts/feature-performance-validation.py run \
  MANIFEST.json --through early --diagnose-regressions
```

This diagnostic independently discovers baseline and candidate launches and
remains explanatory evidence only. It must follow the profiler launch contract
below and cannot convert an early result into acceptance. A clear early screen
only authorizes proportional production confirmation. The canonical
long-context comparison, matching-batch perplexity and output exactness oracle,
whole-lifecycle clean-process evidence, pinned/ordinary-host and process-VRAM
accounting, allocation-lifecycle checks, and every feature-specific gate in
this document remain mandatory. Keep large raw runner and profiler artifacts
outside Git; record only reproducible manifests and concise valid summaries.

## Canonical server lifecycle

This is the current source-supported target-plus-MTP layout and matched short
serving screen. The outer shell owns the lock for clean-GPU preflight, server,
client, progress, graceful teardown, and child reaping. Replace the result/log
paths with new dated paths for each case.

```bash
flock /tmp/beellama-single-gpu.lock -c '
  set -eu
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
      rg -q "^[[:space:]]*[0-9]+[[:space:]]*$"; then
    echo "refusing to run while another GPU compute process is active" >&2
    exit 1
  fi

  server_pid=
  cleanup() {
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
      kill -INT "$server_pid"
      wait "$server_pid" || true
    fi
  }
  trap cleanup EXIT INT TERM

  build-cuda-all/bin/llama-server \
    --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
    --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
    --no-mmproj-offload \
    --n-gpu-layers 999 --n-gpu-layers-draft 999 \
    --fit off --split-mode none --main-gpu 0 --flash-attn on \
    --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
    --kv-gpu-layers 0 \
    --ctx-size 32768 --parallel 1 --cont-batching --kv-unified \
    --batch-size 1024 --ubatch-size 512 \
    --phase-aware-workspace \
    --spec-type draft-mtp --spec-draft-n-max 6 \
    --spec-mtp-rs-planes 3 --spec-draft-p-min 0.85 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
    --threads 3 --threads-batch 24 \
    --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
    --reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
    --alias qwen38-kv-test --host 127.0.0.1 --port 8080 \
    > CURRENT_CASE.server.log 2>&1 &
  server_pid=$!

  until curl -fsS http://127.0.0.1:8080/health >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      wait "$server_pid"
      exit 1
    fi
    sleep 1
  done

  python3 -c \
    "import numpy as np; np.random.seed(1234); from llama_benchy.__main__ import main; main()" \
    --base-url http://127.0.0.1:8080/v1 \
    --model Qwen/Qwen3.5-27B \
    --served-model-name qwen38-kv-test \
    --book-url https://www.gutenberg.org/files/1661/1661-0.txt \
    --pp 512 --tg 64 --depth 4096 30000 \
    --runs 1 --no-warmup --skip-coherence --no-adapt-prompt \
    --latency-mode none --exact-tg \
    --extra-body temperature=0,seed=1234,cache_prompt=false \
    --emit-progress CURRENT_CASE.progress.jsonl \
    --save-result CURRENT_CASE.result.json --format json

  kill -INT "$server_pid"
  wait "$server_pid"
  server_pid=
  trap - EXIT INT TERM
'
```

For the inherited-host draft baseline, omit a draft residency override. For a
draft-owned-GPU candidate, add `--spec-draft-kv-gpu-layers N`, where `N` is
the independently owned draft-layer count confirmed for that model. Change
only the intended variable and restart the server for every case.

## Matched performance protocol

Use the complete locked lifecycle above for each clean A/B/A process when a
difference may be small. The short screen measures 512-token prefill and
64-token exact generation at depths 4,096 and 30,000. Seed NumPy before
entering `llama-benchy`; do not use `--no-cache`, because it changes the
request. Send `cache_prompt=false` and require equal observed prompt-token
counts.

Record prefill, decode at 4K and long depth, peak process VRAM, system RAM, and
allocator-reported pinned memory. `nvidia-smi` process VRAM does not include
page-locked host allocation. A single ordered pair is a screen, not a stable
small performance claim.

## Perplexity and exactness gates

The merged W06/PR 5 fix declares the full logical batch as
`llama-perplexity`'s output-row capacity before context creation. Matched runs
may use phase-aware workspace on both sides when phase behavior is the named
variable, but both sides must use the same setting and matching `-b`/`-ub`
values. General isolated quality gates keep unrelated optional features
explicitly disabled; the canonical baseline template is:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  build-cuda-all/bin/llama-perplexity \
    -m /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
    -f /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt \
    -c 4096 -b 512 -ub 256 --chunks 4 \
    -t 3 -tb 24 --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 \
    -ngl 999 -sm none -mg 0 --flash-attn on \
    --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
    --kv-gpu-layers 0 --no-phase-aware-workspace \
    --cache-type-k q8_0 --cache-type-v q8_0
'
```

MTP correctness uses a same-MTP-geometry reference, never target-only decoding.
Match model, prompt, sampler, seed, MTP depth and threshold, cache formats,
context, batch, target/effective-draft ubatch, and request semantics. Change
only the feature under test. The maintained gates are:

```bash
cd /home/gencoolpc/beellama-kv-offload
flock /tmp/beellama-single-gpu.lock -c '
  python3 scripts/mtp-exactness.py \
    scripts/mtp-exactness-manifests/qwen38-mtp6-partial-draft-residency-q8-1k.json
'
flock /tmp/beellama-single-gpu.lock -c '
  python3 scripts/mtp-exactness.py \
    scripts/mtp-exactness-manifests/qwen38-mtp6-partial-draft-residency-q8-5k.json
'
```

The output directory must be new. Do not use `--allow-mismatch` for an
acceptance gate. Preserve prompt IDs, request semantics, output token IDs and
bytes, acceptance, replay work, logs, progress, VRAM samples, and summary.

## Profiler launch contract

Nsight must directly target the llama binary. Keep profiler helpers
unrestricted and apply affinity only through llama options:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  /usr/bin/ncu --target-processes application-only NCU_OPTIONS \
    build-cuda-all/bin/llama-bench BENCH_ARGS \
    -C 0x7 --cpu-strict 1 --progress
'

flock /tmp/beellama-single-gpu.lock -c '
  nsys profile NSYS_OPTIONS \
    build-cuda-all/bin/llama-server SERVER_ARGS_WITH_NATIVE_AFFINITY
'
```

For NCU, preflight performance-counter permission before a long replay and
bound the kernel filter, skip count, and launch count. A no-kernel capture,
wrapper-targeted capture, incomplete report, or profiler-timed throughput is
not benchmark evidence. Profiler results explain a matched unprofiled result;
they do not replace it.

## Evidence acceptance

Every retained result must identify base and candidate commits, binary hashes,
build, hardware, model/input hashes, exact protocol plus deltas, observed
prompt geometry, progress artifact, performance/resource measurements,
correctness gates, and disposition. Allocation changes require both system-RAM
and pinned-memory accounting.

Invalid runs contribute no measurements or artifact inventory. Record only a
short reusable protocol correction when an invalid attempt exposes a hazard.
Aggregate repetitions into one experiment; do not add a duplicate section for
an otherwise identical rerun. Use Git history for forensic details from a
superseded protocol edition.

### Allocation and CUDA VMM telemetry

`llama-bench --kv-memory` is the opt-in client for allocation classification
and CUDA VMM transient-pool telemetry. It reports physical device, accelerator-
owned host (normally CUDA-pinned), and ordinary-host context/compute buffers,
plus VMM live, mapped, high-water, and active-pool checkpoints. The older
CUDA-owner totals remain for result compatibility; they are not physical VRAM
because they include accelerator-owned host buffers.

The CUDA counters are dormant until this option resets them. Validate a change
with fresh processes in this order: instrumented on, instrumented off,
instrumented on. The off result must leave every new field zero, and the two on
results must reproduce the allocation classes and high-water values. Also
bracket an instrumented default-path run between two pristine-source runs.
Use `--no-warmup --progress` so VMM growth is visible and progress remains
inspectable. Every GPU invocation must remain wholly inside the single-GPU
lock, for example:

```bash
flock /tmp/beellama-single-gpu.lock -c '
  BUILD/bin/llama-bench -m MODEL -p 128 -n 16 -d 4096 -r 3 \
    -b 512 -ub 256 -t 3 -C 0x7 --cpu-strict 1 --poll 100 \
    -ngl 999 -sm none -mg 0 -nkvo 0 -fa on -ctk q8_0 -ctv q8_0 \
    --no-warmup --progress --kv-memory -o jsonl
'
```

Repeat focused `-nkvo 1` rows without and with `--kv-cpu-pinned` when host
classification changes. Record buffer bytes separately from process RSS and
from `nvidia-smi` process VRAM: pinned CUDA-host allocation is system memory,
and ordinary-host allocation is neither pinned memory nor device VRAM. This
surface is measurement support only. It does not authorize pool trimming,
workspace policy, staging changes, causal descriptors, or capacity changes.
On PR 8, W02 observes the already-established live-policy server idle trim and
reconciles mapped-current when mappings are released; it does not add a second
trim caller, benchmark lifecycle, or telemetry policy.
