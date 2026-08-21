# CPU KV-offload current testing and setup

This is the sole runnable protocol for the published CPU-KV line. Its exact
source-bearing baseline is
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`; documentation-only descendants,
including the PR 9 consolidation, do not change that production source or its
binary identity. The retained local journal and Experiments 001-019 come from
`6a20757854395309b32248dd4109d73e99c3e675`. Source and binary identities are
authoritative if this document ever disagrees with behavior.

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
| `--spec-mtp-rs-planes N` | Cap total target recurrent planes, including the current plane. |
| `llama-bench --kv-memory` | Capture KV/component checkpoints, physical device/accelerator-host/ordinary-host allocation classes, and CUDA VMM live/mapped/high-water telemetry. |

MTP must use the target physical ubatch. Omit
`--spec-draft-ubatch-size`, or set it equal to `--ubatch-size` only when the
explicit parser path is under test. A different MTP draft ubatch is rejected.

The published KV source does **not** contain native quantized FlashAttention,
compact causal masking, live-context workspace sizing, or bounded host-KV
attention staging. Do not expect those later fields or use an associated later
option in a current KV-base command. In particular,
`--live-context-workspace` and its preset/generated argument documentation
remain owned by PR 8 until its source lands. See the evidence index for the
exact PR identities.

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
the whole runner under the outer lock.

## Preflight and binary identity

Run before a measurement:

```bash
cd /home/gencoolpc/beellama-kv-offload
git status --short --branch
git rev-parse HEAD
build-cuda-all/bin/llama-server --version
build-cuda-all/bin/llama-bench --version
llama-benchy --version
sha256sum build-cuda-all/bin/llama-server \
  build-cuda-all/bin/llama-bench \
  build-cuda-all/bin/llama-perplexity \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt
nvidia-smi
```

Do not present an uncommitted or differently configured binary as evidence for
the current commit. Documentation-only dirt does not change a binary; source
or build changes do.

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
