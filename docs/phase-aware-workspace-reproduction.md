# Phase-aware target/MTP workspace: complete reproduction record

This document records the full design, implementation history, build setup,
matched commands, measurements, failures, validation, and integration notes for
the opt-in phase-aware compute-workspace experiment. It is intentionally more
detailed than the user-facing argument documentation. The authoritative result
row is also recorded as Experiment 014 in
[`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md).

## Final result

The retained `--phase-aware-workspace` candidate reduces 140K-context MTP-6
CPU-Q8-KV serving residency by resizing transient graph workspace for the
active prompt or generation geometry and sharing physical target/MTP backing.
It does not unload weights, the MTP head, KV, recurrent state, or rollback
planes.

On the matched RTX 5070 Ti workload:

| Measurement | Full-workspace baseline | Phase-aware candidate | Change |
| --- | ---: | ---: | ---: |
| Initialized process VRAM | 15,768 MiB | 14,660 MiB | -1,108 MiB |
| Post-5K steady VRAM | 15,800 MiB | 14,692 MiB | -1,108 MiB |
| 5K sampled peak VRAM | 15,800 MiB | 14,874 MiB | -926 MiB |
| 138K sampled peak VRAM | 15,800 MiB | 14,898 MiB | -902 MiB |
| 138K prefill | 742.519 t/s | 741.661 t/s | -0.12% |
| Plain 5K decode | 52.098 t/s | 51.146 t/s | -1.83% |
| Nsight 5K decode | 50.498 t/s | 50.275 t/s | -0.44% |
| Plain 5K wall time | 96.211 s | 98.023 s | +1.88% |
| 138K wall time | 187.959 s | 188.030 s | +0.04% |
| 5K target + draft transition time | none | 41.795 ms | +41.795 ms/request |
| Nsight H2D bytes/calls | 378,465.886 MB / 200,628 | identical | zero |
| Nsight D2H bytes/calls | 7,145.663 MB / 143,212 | identical | zero |

Fixed-seed short, 5K, and 138K outputs are byte-identical. Draft generation,
acceptance, replay cycles, and actual replay-batch tokens match. A second prompt
on the same live candidate context regrows and shrinks safely and produces the
same output. The option-off candidate reproduces baseline allocation and
output.

## Source identity and worktree isolation

The comparison is rooted at BeeLlama commit:

```text
324873dc5ca44eb31727ba3bd09897841574fa3b
```

The candidate is:

```text
worktree: /home/gencoolpc/beellama-prefill-decode
branch:   exp/phase-aware-prefill-decode
HEAD:     324873dc5ca44eb31727ba3bd09897841574fa3b
state:    uncommitted candidate diff
```

The matched baseline is an exact detached worktree:

```text
worktree: /home/gencoolpc/beellama-phase-baseline
HEAD:     324873dc5ca44eb31727ba3bd09897841574fa3b
state:    clean
```

The existing `/home/gencoolpc/beellama-kv-offload` worktree had other active
work and was never used as the baseline. This matters: an earlier apparent
comparison was rejected when inspection showed that the presumed base still
contained changes. The detached worktree above is the actual source control.

To recreate the isolation from a clean BeeLlama repository:

```bash
git worktree add -b exp/phase-aware-prefill-decode \
  ../beellama-prefill-decode \
  324873dc5ca44eb31727ba3bd09897841574fa3b
git worktree add --detach \
  ../beellama-phase-baseline \
  324873dc5ca44eb31727ba3bd09897841574fa3b
git -C ../beellama-phase-baseline status --short
git -C ../beellama-phase-baseline rev-parse HEAD
```

The benchmarked source-only candidate diff, excluding documentation, has
SHA-256:

```text
f6407cfd0987a37835a4c428ba6a63581928614a9efa691a20957b7e9bfcfec8
```

Recalculate it with:

```bash
git diff -- common ggml include src tests tools | sha256sum
```

Do not call the base commit the candidate commit: the source remained
deliberately uncommitted because no commit was requested. The binary and diff
hashes identify the measured state.

## Build-cache correction and build workflow

### Why sibling worktrees originally rebuilt cold

The original ccache setup hashed absolute worktree paths. Identical sources in
sibling worktrees therefore did not reliably share cached objects. Multiple
concurrent builds also made progress and ownership unclear. All active build
processes were stopped before reconfiguring the cache; the exact PIDs were
identified first rather than using a broad destructive command.

Use this safe inspection before stopping duplicate local builds:

```bash
pgrep -af 'cmake --build|ninja|make|nvcc|cc1plus'
```

Send `SIGINT`, then `SIGTERM` only to the explicitly verified build-driver
PIDs. Do not kill unrelated compiler processes on a shared host.

The common configuration is now:

```text
cache_dir     = /home/gencoolpc/.cache/ccache
base_dir      = /home/gencoolpc
hash_dir      = false
max_size      = 50.0 GB
compiler_check= mtime
```

Configure it with:

```bash
ccache --set-config=cache_dir=/home/gencoolpc/.cache/ccache
ccache --set-config=base_dir=/home/gencoolpc
ccache --set-config=hash_dir=false
ccache --set-config=max_size=50G
ccache --set-config=compiler_check=mtime
ccache --show-config
```

`base_dir` plus `hash_dir=false` normalizes sibling-worktree paths. It does not
make a changed dependency a cache hit. The final source changed public GGML and
llama headers, so the production rebuild recorded 847 cacheable calls and 847
genuine misses after a statistics reset. The resulting 2.4 GiB of objects is
now reusable by identical later branches.

### Fast iteration build

Allocator and parser work does not need CUDA. The candidate has a CPU-only
Ninja build used as the first gate:

```bash
cd /home/gencoolpc/beellama-prefill-decode
cmake -S . -B build-phase-dev -G Ninja \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-phase-dev \
  --target test-alloc test-arg-parser -j 24
build-phase-dev/bin/test-alloc
build-phase-dev/bin/test-arg-parser
```

The first targeted build completed in approximately 24 seconds. For future
iterations, keep public-header changes behind this gate and build only the
targets needed for the current test. A production CUDA rebuild is still
necessary before any performance claim.

### Matched production builds

Configure both exact source trees with the same options:

```bash
cmake -S . -B build-cuda-all \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_CUDA_KVARN=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build-cuda-all -j 24
```

Run it once from `/home/gencoolpc/beellama-phase-baseline` and once from
`/home/gencoolpc/beellama-prefill-decode`. The Makefile generator was used for
these production directories. The CMake request is architecture `120`; the
current CUDA/CMake build compiles it as `120a`.

Binary identities:

```text
baseline llama-server:
7b63ef24b1cfae76738793a47c8b96e5087307b627ef0d13302f8cddf89ae89b

candidate llama-server:
6b7169ca2141a606613527deecf7134c530e27884187de0a75b8c72e26ea54b2
```

Verify them before benchmarking:

```bash
sha256sum \
  /home/gencoolpc/beellama-phase-baseline/build-cuda-all/bin/llama-server \
  /home/gencoolpc/beellama-prefill-decode/build-cuda-all/bin/llama-server
```

## Hardware and software identity

```text
OS/kernel:  Linux cachyos-x8664 7.1.8-1-cachyos x86_64
GPU:        NVIDIA GeForce RTX 5070 Ti, compute capability 12.0
driver:     610.57.04
GPU report: 16,303 MiB driver total; 15,880 MiB llama/process capacity
CPU:        Intel Core Ultra 9 285K
topology:   24 cores, one thread/core, one socket, one NUMA node, CPUs 0-23
RAM:        approximately 62 GiB
GCC:        16.2.1 20260810
CUDA:       13.3, build cuda_13.3.r13.3/compiler.38244171_0
Nsight:     2026.1.3.425-261338342291v0
```

Useful identity commands:

```bash
uname -a
gcc --version | head -n 1
nvcc --version | tail -n 1
nsys --version
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader
lscpu
```

## Model and fixed runtime policy

```text
target/MTP model:
/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf

multimodal projector:
/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf
```

The matched server arguments are:

```bash
--model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
--mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
--no-mmproj-offload \
--n-gpu-layers 999 --n-gpu-layers-draft 999 \
--fit off --split-mode none --main-gpu 0 --flash-attn on \
--no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
--ctx-size 140000 --parallel 1 --cont-batching --kv-unified \
--batch-size 1024 --ubatch-size 512 \
--spec-type draft-mtp --spec-draft-n-max 6 \
--spec-mtp-rs-planes 3 --spec-draft-ubatch-size 128 \
--draft-p-min 0.85 \
--cache-type-k q8_0 --cache-type-v q8_0 \
--cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
--threads 3 --threads-batch 24 \
--cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
--reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
--host 127.0.0.1 --port PORT --verbosity 4
```

Only the candidate adds:

```bash
--phase-aware-workspace
```

This is MTP depth 6 with three total recurrent planes. Do not confuse the
workspace control with either depth or plane count. MTP can still draft six
tokens; recurrent-plane rollback/replay is unchanged.

### Environment isolation

Every server launch used:

```bash
env -u GGML_KV_CPU_PINNED \
    -u GGML_RECURRENT_STATE_OFFLOAD \
    -u LLAMA_TRACE \
    -u LLAMA_ARG_PHASE_AWARE_WORKSPACE \
    -u CUDA_VISIBLE_DEVICES \
    SERVER [arguments]
```

This is essential. Earlier CPU-KV experiments used environment switches for
placement, and an inherited `LLAMA_ARG_PHASE_AWARE_WORKSPACE` could silently
enable the candidate on a nominal baseline. Unsetting all five variables makes
the current public CLI flags authoritative and makes GPU visibility symmetric.

## What was implemented

### Public control and propagation

- `common/common.h` adds the default-false common parameter.
- `common/arg.cpp` adds positive and negative CLI forms and
  `LLAMA_ARG_PHASE_AWARE_WORKSPACE`.
- The existing normalized INI mechanism maps `phase-aware-workspace` to the
  same argument; parser coverage proves it.
- `common/common.cpp`, `include/llama.h`, and `src/llama-cparams.h` propagate the
  setting to the context.
- `common/speculative.cpp` derives a generation-side `n_outputs_max` large
  enough for parallel sequences and the entire configured speculative horizon.

### Phase classification and scheduler lifecycle

`llama_context::sched_reserve(n_tokens_req)` computes:

```text
prompt bound     = min(n_ctx, n_ubatch)
generation bound = min(prompt bound, max(n_seq_max, n_outputs_max))
```

The server sets `n_outputs_max` to:

```text
min(batch_size, parallel * (1 + resolved_draft_max))
```

For this experiment, the target and MTP generation reservation is seven
tokens. A submitted logical batch above seven selects the full physical
ubatch—512 target tokens or 128 MTP tokens. Shorter trailing prompt chunks do
not shrink it. The first generation-size submission begins shrink.

`llama_decode()` and encoder evaluation publish actual submitted token counts
to `sched_reserve()`. Memory-update re-reservation uses the currently reserved
geometry rather than accidentally restoring the full prompt shape.

### Generic shared backing

`ggml/include/ggml-alloc.h` and `ggml/src/ggml-alloc.c` add a caller-owned
`ggml_gallocr_shared_buffers` group. Every graph allocator remains a separate
member with a private tensor-allocation plan. The group:

- registers and unregisters allocator members;
- keys entries by exact `ggml_backend_buffer_type_t` pointer identity;
- stores every member's per-chunk requirements;
- allocates each physical entry at the per-chunk maximum across active members;
- grows immediately when any member needs more;
- frees the old allocation before replacement, avoiding a temporary old+new
  peak that could exceed device capacity;
- increments a physical generation when addresses change;
- increments a plan generation when a shrink epoch begins;
- waits until every active member publishes once before completing shrink;
- retains backing safely when a member unregisters until a live peer republishes
  or the group owner is destroyed.

`ggml/include/ggml-backend.h` and `ggml/src/ggml-backend.cpp` expose this group
through the scheduler while preserving the existing scheduling interface.

This is generic allocator code. It does not inspect model architecture, Qwen,
CUDA device names, or tensor semantics. Different buffer types are never
mistakenly pooled, so heterogeneous backend layouts fail by ordinary allocation
behavior rather than unsafe reinterpretation.

### Target/MTP hookup and ownership

The target context owns a reference-counted shared group. An integrated MTP
context receives the same group through `ctx_other`. Each context still owns
its scheduler and allocation plan. It observes physical and plan generations,
invalidates cached graph addresses after a peer replacement, and republishes
its current plan during a shrink epoch.

Target and MTP use the backing sequentially. `common/speculative.cpp` adds
ownership fences at actual handoffs:

- after target output before first MTP decode;
- after MTP prompt catch-up before returning to target;
- after draft generation before target verification.

The fences complete asynchronous work before another scheduler reuses the same
addresses. They are conditional on both contexts actually sharing backing.

Backend-private KVarN scratch requirements are recorded separately when graphs
are measured. No KVarN, attention, GDN, or model kernel changed.

### Instrumentation

`llama_workspace_stats` reports enabled state, reserved/decode/prefill token
bounds, reserve/grow/shrink counts, elapsed reserve microseconds, and current
host/device backing bytes. Server slots snapshot cumulative counters at request
launch and return per-request deltas in timing JSON:

```text
workspace_target_reserves
workspace_target_grows
workspace_target_shrinks
workspace_target_reserve_ms
workspace_draft_reserves
workspace_draft_grows
workspace_draft_shrinks
workspace_draft_reserve_ms
```

The same totals are printed with speculative timing statistics.

## Design chronology and failed variants

### 1. Rejected teardown/unload strategy

The first conceptual route was to unload or destroy MTP/prompt resources around
phase boundaries. It was rejected because model/context ownership contains
persistent recurrent state, rollback state, output buffers, graph metadata,
and future-turn state. Mixing all of those lifetimes to reclaim a transient
compute arena would be brittle and would make later prompts difficult to prove
safe.

The retained architecture changes only the compute backing beneath intact
schedulers. In practical terms, MTP remains loaded during prefill; it simply
does not retain a second simultaneous prompt-sized physical arena. After
prefill, prompt-sized backing is released and generation backing remains.

### 2. First shared-backing prototype

The first generic group correctly allocated the maximum target/MTP requirement
instead of their sum. It incorrectly inferred that every smaller reserve plan
meant the workload had entered a smaller phase.

Speculative rollback replay disproved that assumption. Replay creates valid
graph variants while still in generation. Each member's new plan changed the
shared physical generation, causing the peer to republish, which then caused
another replacement. Measured failure:

```text
short, 4 replay cycles: target 6 reserves + draft 6 reserves
5K, 88 replay cycles:   target 90 reserves + draft 90 reserves
5K reserve time:        target 462.110 ms + draft 265.997 ms
total transition churn: 728.107 ms
candidate decode:       51.422 t/s
```

This was a bug in the candidate protocol, not upstream replay. Upstream is
correct to build graph variants for deterministic rollback/replay; allocation
lifetime must distinguish graph-plan variation from a real phase change.

### 3. Intermediate high-water permission

A per-member high-water and shrink-permission revision was tested. The short
artifact `candidate-short-v2b` still reported six plus six reserves, four
replays, 81.399 t/s decode, and 55.247/20.616 ms target/draft reserve time.
Temporary logs identified every extra reserve as `peer_plan=1, local=0`.
Local high-water state could not solve a group coordination problem, so this
variant was replaced.

### 4. Retained explicit group shrink epoch

Only the context observing an actual prompt-to-generation token-geometry
transition requests shrink. That starts one group epoch. Growth is always
immediate; shrink waits for all members to publish once. Replay plan changes
outside that epoch may update private plans but cannot reduce physical backing.

The final short, 5K, 138K, and later-turn runs each perform two reserves per
context: one grow and one shrink. The 5K transition cost fell from 728.107 ms to
41.795 ms.

## Request generation

Create a local artifact root:

```bash
mkdir -p /tmp/phase-bench
```

### Short number-sequence request

```bash
jq -n \
  --arg prompt 'Continue the sequence and explain the rule briefly: 1, 4, 9, 16, 25,' \
  '{prompt:$prompt,n_predict:128,temperature:0,seed:1234,cache_prompt:false,stream:false}' \
  > /tmp/phase-bench/request-short.json
```

Expected SHA-256:

```text
408fc8936598794b092fef75283311a3f0ff9df35318416ebe4f4ffa3cbe011a
```

Use endpoint `/completion`.

### 5,000-token orbital-sandbox coding request

```bash
jq -cn \
  --arg content $'Write a single self-contained HTML file: an interactive orbital mechanics sandbox.\nCanvas-based, 60fps. Users can click-drag to fling new planets into the system;\ngravity is simulated with velocity Verlet integration against a central star.\nInclude trailing orbit paths that fade, collision merging with a mass-conserving\nflash, a mass/velocity readout on hover, and a pause/reset UI. No libraries,\nno assets — one file, pure JS.' \
  '{model:"mtp6-coding-5k",messages:[{role:"user",content:$content}],max_tokens:5000,seed:1234,stream:false,cache_prompt:false}' \
  > /tmp/phase-bench/request-coding-5k.json
```

Expected SHA-256:

```text
6edbe87e0896f13681887883efba7b18a669e29802a161fcfa6a737bd5695995
```

Use endpoint `/v1/chat/completions`. No request `temperature` is present; this
intentionally uses the model/server default on both sides.

### 138,000-token long request

```bash
jq -n \
  '{prompt:[range(138000)|1000+(.%4)],n_predict:64,temperature:0,seed:1234,cache_prompt:false,stream:false}' \
  > /tmp/phase-bench/request-long-138k.json
```

Expected SHA-256:

```text
b60a55d02f8cf01f53b73ab80bf46b22d8d5886ff970c09a4f660475a0e0f546
```

The prompt is an explicit token-ID vector repeating
`[1000, 1001, 1002, 1003]`. Use endpoint `/completion`.

## Plain benchmark harness behavior

The retained harness was `/tmp/phase-bench/run-phase-benchmark.sh`, SHA-256:

```text
64d7e49d020b8323bff5fdea45f111f21cd2b7970da9f691faae63b41f1bcc3d
```

Its interface was:

```text
run-phase-benchmark.sh LABEL SERVER PORT PHASE(on|off) ENDPOINT REQUEST_JSON
```

It performed these steps for every clean process:

1. Copy and hash the request into `/tmp/phase-bench/LABEL`.
2. Write the shell-escaped server command and relevant environment.
3. Launch the server with the five environment variables unset.
4. Poll `/health` for up to 180 seconds, printing startup progress every five
   seconds and failing if the process exits.
5. Record initialized `nvidia-smi` process VRAM and `/proc/PID/status` values.
6. Sample process VRAM, `VmRSS`, `VmHWM`, and `VmLck` every 0.5 seconds.
7. Poll `/slots` every ten seconds, recording request progress.
8. Submit the JSON with `curl --no-buffer`, recording HTTP and wall time.
9. Record the completed memory checkpoint and summarize min/peak values.
10. Hash only generated text, extract timing/spec/workspace counters, stop the
    server cleanly, and preserve relevant allocation/shutdown logs.

The memory sampler used this GPU query:

```bash
nvidia-smi --id=0 \
  --query-compute-apps=pid,used_memory \
  --format=csv,noheader,nounits
```

and these process fields:

```bash
awk '
  /^VmRSS:/ { rss=$2 }
  /^VmHWM:/ { hwm=$2 }
  /^VmLck:/ { lck=$2 }
  END { printf "%s,%s,%s\n", rss+0, hwm+0, lck+0 }
' /proc/PID/status
```

Output hashing used `.content` for `/completion` and concatenated
`reasoning_content` plus visible `content` for chat completion. It did not hash
timing fields or response metadata.

Run the matrix in clean, non-overlapping processes:

```bash
BASE=/home/gencoolpc/beellama-phase-baseline/build-cuda-all/bin/llama-server
CAND=/home/gencoolpc/beellama-prefill-decode/build-cuda-all/bin/llama-server
H=/tmp/phase-bench/run-phase-benchmark.sh

$H baseline-short-140k          "$BASE" 8100 off /completion \
  /tmp/phase-bench/request-short.json
$H candidate-short-final-140k   "$CAND" 8101 on  /completion \
  /tmp/phase-bench/request-short.json
$H candidate-default-off-short-140k "$CAND" 8102 off /completion \
  /tmp/phase-bench/request-short.json

$H baseline-coding-5k-140k        "$BASE" 8103 off /v1/chat/completions \
  /tmp/phase-bench/request-coding-5k.json
$H candidate-coding-final-5k-140k "$CAND" 8104 on  /v1/chat/completions \
  /tmp/phase-bench/request-coding-5k.json

$H baseline-long-138k        "$BASE" 8105 off /completion \
  /tmp/phase-bench/request-long-138k.json
$H candidate-long-final-138k "$CAND" 8106 on  /completion \
  /tmp/phase-bench/request-long-138k.json
```

Port numbers need only be unique and idle. Check the GPU and server table
between processes; do not run baseline and candidate concurrently on the same
GPU.

## Plain benchmark results

### Short control

| Metric | Baseline | Candidate | Default-off candidate |
| --- | ---: | ---: | ---: |
| Prompt tokens / output tokens | 26 / 128 | identical | identical |
| Prefill | 197.651 t/s | 149.254 t/s | 210.641 t/s |
| Decode | 84.099 t/s | 82.783 t/s | 83.508 t/s |
| Wall | 1.6547 s | 1.7216 s | 1.6576 s |
| Init VRAM | 15,768 MiB | 14,660 MiB | 15,768 MiB |
| Complete VRAM | 15,790 MiB | 14,682 MiB | 15,790 MiB |
| Candidate peak | — | 14,874 MiB | — |
| Draft generated/accepted | 103/88 | 103/88 | 103/88 |
| Replay cycles/tokens | 4/26 | 4/26 | 4/26 |
| Output SHA-256 | `e87b691…f223b` | identical | identical |

The prompt is intentionally too short for a stable prefill comparison. Its
purpose is fast correctness, allocation, replay, and default-off control.

Candidate transition counts were target 2 (one grow/one shrink, 35.302 ms) and
draft 2 (one grow/one shrink, 9.429 ms). The default-off candidate reproduced
full baseline VRAM exactly. A one-time 0.256 ms draft setup counter on that path
does not change physical policy; target transitions were zero and full backing
remained resident.

### 5,000-token coding result

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Prompt tokens | 149 | 149 | same |
| Output tokens | 5,000 | 5,000 | same |
| Prefill | 633.102 t/s | 571.953 t/s | cold short prompt; noisy |
| Decode | 52.0979 t/s | 51.1455 t/s | -1.83% |
| Wall | 96.2113 s | 98.0232 s | +1.88% |
| Init VRAM | 15,768 MiB | 14,660 MiB | -1,108 MiB |
| Complete VRAM | 15,800 MiB | 14,692 MiB | -1,108 MiB |
| Peak VRAM | 15,800 MiB | 14,874 MiB | -926 MiB |
| Draft generated/accepted | 2,193/1,851 | identical | zero |
| Replay cycles/batch tokens | 88/430 | identical | zero |
| Target transition | none | 2, 32.482 ms | one grow/one shrink |
| Draft transition | none | 2, 9.313 ms | one grow/one shrink |
| Output SHA-256 | `2f994f6…57f44e` | identical | byte-identical |

The exact output SHA-256 is:

```text
2f994f600563b04a0a8ce172b59b8f04515edae27664f0b15cab003ad257f44e
```

### 138,000-token prompt result

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Prompt tokens / output tokens | 138,000 / 64 | identical | same |
| Prefill | 742.5194 t/s | 741.6613 t/s | -0.12% |
| Decode | 30.6745 t/s | 33.0247 t/s | short tail; no stable claim |
| Wall | 187.9586 s | 188.0301 s | +0.04% |
| Init VRAM | 15,768 MiB | 14,660 MiB | -1,108 MiB |
| Complete VRAM | 15,800 MiB | 14,692 MiB | -1,108 MiB |
| Peak VRAM | 15,800 MiB | 14,898 MiB | -902 MiB |
| Draft generated/accepted | 54/54 | identical | zero |
| Replay cycles/tokens | 0/0 | identical | zero |
| Target transition | none | 2, 36.074 ms | one grow/one shrink |
| Draft transition | none | 2, 9.929 ms | one grow/one shrink |
| Output SHA-256 | `02ba58a…bf54dc` | identical | byte-identical |

The exact output SHA-256 is:

```text
02ba58ab1c1df7dec6df58c435c3fa9eeb5c42fb04c549c82552639aaebf54dc
```

This is the authoritative prefill comparison. The 64-token generation tail is
too short to interpret as a decode improvement.

### Host memory

| Workload and point | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 5K initialized RSS | 7,480,900 KiB | 7,237,808 KiB | -243,092 KiB |
| 5K complete RSS | 7,929,392 KiB | 7,686,608 KiB | -242,784 KiB |
| 5K process `VmHWM` | 14,580,312 KiB | 14,535,664 KiB | -44,648 KiB |
| 138K initialized RSS | 7,480,912 KiB | 7,235,776 KiB | -245,136 KiB |
| 138K complete RSS | 8,551,064 KiB | 8,306,416 KiB | -244,648 KiB |
| 138K process `VmHWM` | 14,580,692 KiB | 14,534,252 KiB | -46,440 KiB |

Model loading dominates `VmHWM`; it is not a clean measure of live allocator
residency. `VmLck` was zero on both sides. CUDA pinned allocations are not
reliably represented by `VmLck`, so do not infer pageable host memory from that
field.

### Named allocator sizes

Baseline full reservations:

```text
target: CUDA0 1054.62 MiB, CUDA_Host 157.03 MiB
draft:  CUDA0  892.05 MiB, CUDA_Host  39.46 MiB
```

Candidate stable shared generation backing:

```text
shared physical: CUDA0 840.82 MiB, CUDA_Host 2.41 MiB
```

Candidate prompt backing grows to the active target maximum:

```text
shared physical: CUDA0 1054.62 MiB, CUDA_Host 157.03 MiB
```

The draft prompt plan is retained logically but does not create a second
simultaneous physical maximum.

## Nsight Systems reproduction

The profiler harness was `/tmp/phase-bench/run-phase-nsys.sh`, SHA-256:

```text
fe48787f6ab25c8d053111133808e3c49d2f557eeae99b8ae93c59e80c0f6d07
```

It uses the same environment clearing and server arguments, wraps the server
with:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=/tmp/phase-bench/LABEL/trace \
  SERVER [arguments]
```

It polls health and `/slots`, submits the 5K chat request, sends `SIGINT` to the
actual child server after completion, waits for report finalization, and runs:

```bash
nsys stats --force-export=true \
  --report cuda_gpu_mem_size_sum --format csv TRACE.nsys-rep
nsys stats \
  --report cuda_gpu_mem_time_sum --format csv TRACE.nsys-rep
```

Run:

```bash
/tmp/phase-bench/run-phase-nsys.sh \
  baseline-nsys-5k-140k \
  /home/gencoolpc/beellama-phase-baseline/build-cuda-all/bin/llama-server \
  8110 off /tmp/phase-bench/request-coding-5k.json

/tmp/phase-bench/run-phase-nsys.sh \
  candidate-nsys-final-5k-140k \
  /home/gencoolpc/beellama-prefill-decode/build-cuda-all/bin/llama-server \
  8111 on /tmp/phase-bench/request-coding-5k.json
```

Results:

| Operation | Baseline | Candidate |
| --- | ---: | ---: |
| H2D total/count | 378,465.886 MB / 200,628 | identical |
| H2D avg/median/max | 1.886 / 1.114 / 874.086 MB | identical |
| D2H total/count | 7,145.663 MB / 143,212 | identical |
| D2H avg/median/max | 0.050 / 0.004 / 6.953 MB | identical |
| Memset total/count | 969.646 MB / 3,159 | identical |
| H2D total time | 8.5195 s | 8.1640 s |
| D2H total time | 0.2743 s | 0.2625 s |
| Decode | 50.4977 t/s | 50.2745 t/s |
| Wall | 99.2590 s | 99.7935 s |

The timing difference inside one trace pair is noise and is not claimed as an
improvement. Exact byte and operation-count equality proves that workspace
replacement adds no H2D/D2H mechanism. Nsight totals are authoritative;
serialized checkpoint payload sizes are not transfer measurements.

Trace identities:

```text
baseline trace, 238 MiB:
54c6b8c1e1c93a576554d38c6775687b76ae96edcfecfb9589d1a5f16387ee87

candidate trace, 239 MiB:
90369d273652c9c353ee5e5f300f3e7471044640f485224c7f5b6f1badf853d5
```

Both profiled outputs have the same 5K generated-text hash as the plain pair.

## Later-turn live-context test

Start one candidate server with the matched command and
`--phase-aware-workspace`, then submit the short request twice without
restarting it:

```bash
for turn in 1 2; do
  curl --fail --show-error \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/phase-bench/request-short.json \
    http://127.0.0.1:PORT/completion \
    -o "/tmp/phase-bench/candidate-two-turn-short-140k/response-$turn.json"
done
```

Both generated outputs have SHA-256:

```text
e87b691143030a0ac2025ab750ed5aeebbca7296734de5c24b884cbdda7f223b
```

Turn 1 grew/shrank target and draft once, with 34.122/9.982 ms transition
time. Turn 2 repeated the full safe lifecycle, with 33.396/9.427 ms. The server
log shows target 7→512→7 and draft 7→128→7 on the second request. Turn 2 took
81.531 t/s versus 84.115 t/s on turn 1. Its replay count was five rather than
four because server prompt checkpoint restoration/invalidated context changed
the internal speculative path; generated output remained byte-identical.

## Validation commands

Cheap development gates:

```bash
cmake --build build-phase-dev \
  --target test-alloc test-arg-parser -j 24
build-phase-dev/bin/test-alloc
build-phase-dev/bin/test-arg-parser
```

Final CUDA allocator test:

```bash
build-cuda-all/bin/test-alloc
```

Focused regression set:

```bash
ctest --test-dir build-cuda-all --output-on-failure \
  -R '^(test-server-loop-guard-checkpoint-static|test-kvarn-rollback-static|test-generate-models|test-recurrent-state-rollback|test-arg-parser|test-server-loop-guard|test-server-prompt-checkpoint|test-backend-sampler|test-alloc)$'
```

Result: 9/9 passed.

Broad set after classifying three independent failures:

```bash
ctest --test-dir build-cuda-all --output-on-failure \
  -E '^(test-upstream-merge-keepers-static|test-tokenizers-ggml-vocabs|test-backend-ops)$'
```

Result: 93/93 passed. The final audit rerun completed in 56.30 seconds.

The excluded tests were investigated rather than silently ignored:

1. `test-upstream-merge-keepers-static` fails identically in the exact detached
   baseline worktree.
2. `test-tokenizers-ggml-vocabs` requires an external fixture that is a Git LFS
   pointer in this checkout.
3. CUDA `test-backend-ops` reaches the existing assertion at
   `ggml/src/ggml-cuda/fattn.cu:380` after approximately 221 seconds. This
   candidate changes no CUDA kernel or attention dispatch source.

Parser coverage checks default false, positive CLI, negative override,
environment enable, environment cleanup, and INI preset parsing. Allocator
coverage checks maximum-not-sum allocation, immediate growth, coalesced shrink,
unchanged-peer publication, member removal, plan/physical generations, and
disjoint buffer-type membership.

## Artifact map and identities

All measurement artifacts remain under `/tmp/phase-bench` on the benchmark
host:

```text
baseline-short-140k/
candidate-short-final-140k/
candidate-default-off-short-140k/
baseline-coding-5k-140k/
candidate-coding-final-5k-140k/
baseline-long-138k/
candidate-long-final-138k/
baseline-nsys-5k-140k/
candidate-nsys-final-5k-140k/
candidate-two-turn-short-140k/
```

Each ordinary harness directory contains the copied request and hash, escaped
server command, relevant environment, health response, server PID/log,
response, curl timing, extracted timing summary, output hash, 0.5-second memory
samples, memory summary, initialized/completed checkpoints, and relevant log
subset. Nsight directories additionally contain the profile command, report,
SQLite export, and CSV summaries.

Response JSON hashes differ across baseline and candidate because timings and
new workspace fields differ. Generated-text hashes are the correctness key.
For provenance, the principal response JSON SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| baseline short | `698cf28240f13c633f33504bea0129252760e2afac4152e7294c01e684e62c18` |
| candidate short | `c3f60f2e80fdde508a0968c1070640746117ec908046debca55caed7d6f19bc6` |
| candidate default-off short | `800cc39f4bf21d1dd42171f86941be8b54916db3e0a6d2a4bad650e38c895d42` |
| baseline coding 5K | `9cf3c341e71125b430d5b2f2ac29f4ddc5b3d7b16e988716523bd7e0a84d08` |
| candidate coding 5K | `b78e6d2f05f265a58298c338229c2bc5f5d8cf627c273c2479fe3f67ab50fe26` |
| baseline long 138K | `aa00ee8c383e4638917ae8996ddc80d6f9654265c879654080826d326b09de24` |
| candidate long 138K | `6f6aba47d61f44891389b574ef790438182feb3e05a605291a1899a9c3a05a2b` |
| baseline Nsight 5K | `801bdc9e1fbb1d65a1e82b16ad11a1c6a3b0ec74aaffa1c7b6430b72f5e9c86b` |
| candidate Nsight 5K | `ad91a2a35321e6904e05d870c30064e525be01f8d8b5232e12d62f56b1b2a0ee` |
| candidate live turn 1 | `aedb2adf50442e28f11f554d3c0c34e0100ba399a74bc674cf8160a33dadb3b8` |
| candidate live turn 2 | `ecd97a6ea4b51844108a1980952bf550b640d722e90e10a3fc66ed322251e883` |

## Integration and review checklist

Before moving this candidate into another branch or a PR:

1. Preserve the public option default as false.
2. Move the entire source diff as one coherent allocator/context/server change;
   do not copy only the context resizing without the group epoch and MTP fences.
3. Resolve upstream scheduler/allocator conflicts against current abstractions;
   do not reintroduce fork-private verifier state.
4. Rebuild a production binary and record its actual commit and SHA-256.
5. Rerun `test-alloc`, parser coverage, the focused regression set, default-off
   control, same-process two-turn control, 5K output equivalence, and 138K
   prefill/peak measurement.
6. Rerun Nsight if backend scheduling, ownership fences, or allocation transfer
   behavior changed during rebase.
7. Keep model, projector, cache formats, MTP depth, plane count, affinity,
   thread count, batch/ubatch sizes, and cleared environment identical on both
   sides.
8. Add a commit ID to Experiment 014 only after the measured code and docs are
   committed together. Do not rewrite the current base commit as if it already
   contained the candidate.

The implementation can be separated conceptually into a generic GGML shared
backing change and a llama/server phase-policy consumer, but they are not
independently useful as benchmark candidates. A clean review series may use:

1. generic GGML shared-backing API plus allocator tests;
2. phase-aware llama-context lifecycle, MTP ownership, and public argument;
3. server instrumentation and documentation.

The series must remain bisectable: intermediate commits should compile and the
public feature should not become selectable before all required ownership and
shrink coordination is present.

## Limits and next VRAM work

This policy removes inactive scheduler high-water and target/draft coexistence.
It does not reduce:

- model weights or the MTP head;
- active recurrent state or recurrent rollback planes;
- CPU target KV or its pinned mapping cost;
- the active target prompt graph's 1,054.62 MiB CUDA and 157.03 MiB CUDA-host
  requirement;
- context-linear Q8 staging, F16 materialization, and explicit causal-mask
  storage inside prompt processing;
- CUDA graph recapture/replacement overhead at phase boundaries.

The result updates the working theory: approximately 1.1 GiB of steady VRAM was
avoidable transient workspace. The next contained optimization is a
fail-closed compact causal-mask descriptor for the supported single-slot
contiguous layout. Direct-Q8 prompt MMA is a larger kernel project. Bounded
fixed-window GPU streaming remains the long-term design when active prompt
workspace must remain approximately flat as maximum context grows.

The separate MTP recurrent-plane cap and sparse GPU replay history is documented
in [`mtp-recurrent-plane-cap-reproduction.md`](mtp-recurrent-plane-cap-reproduction.md).
That feature and this workspace lifecycle are orthogonal: plane count controls
persistent recurrent rollback allocation, while phase-aware workspace controls
transient graph compute backing.
