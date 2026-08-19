# MTP recurrent-plane cap: complete implementation and reproduction record

This is the standalone reconstruction of the configurable MTP recurrent-plane
cap work. It records source lineage, discarded designs, implementation
decisions, build and ccache setup, exact test and benchmark protocols, profiler
queries, artifacts, measurements, failures, and integration. Another developer
should be able to reproduce the work without shell history or conversation
context.

The compact result ledger remains
[`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md). This document
deliberately preserves more process detail, including rejected paths.

## Final outcome

The retained implementation adds `--spec-mtp-rs-planes N` while preserving the
existing full-plane default.

- `N = 0`: allocate `spec-draft-n-max + 1` planes and use ordinary consecutive
  recurrent snapshots.
- `N = spec-draft-n-max + 1`: explicit full allocation; behavior is the same as
  zero/default.
- `2 <= N < spec-draft-n-max + 1`: allocate exactly `N` planes. One holds the
  exact pre-verification input and `N - 1` hold recent verification outputs.
  The direct rejection horizon is therefore `N - 2`.
- Draft depth does not change. MTP-8 with four planes still proposes up to
  eight tokens.
- A rejection beyond the direct horizon reruns the original full target batch
  shape on GPU and writes only the originally accepted recurrent boundary. It
  does not serialize a target checkpoint to the host.
- The cap requires both a model graph and every recurrent-state buffer backend
  to advertise selected sparse-snapshot support. NVIDIA CUDA is the first
  provider. Unsupported combinations fail at startup.

The fixed-seed 5,000-token MTP-8 comparison produced byte-identical reasoning
output. Four planes reduced target recurrent allocation from 1,346.62 MiB to
598.50 MiB and changed profiled decode from 51.383 to 50.979 tokens/s (-0.79%).
The cap performed 38 replay cycles covering 217 actual target batch tokens and
performed zero target checkpoint captures or restores.

Implementation commit:
`d743456922e6005578d5c94e74e99180c0dbe4c7`
(`speculative: cap MTP recurrent rollback planes`).

## Scope and non-goals

The patch changes only MTP recurrent rollback storage and recovery. It does not
change sampling, accepted tokens, draft depth, cache formats, model weights,
quality, visible output, or prompt-cache policy.

Out of scope:

- lower or asymmetric target/draft KV formats;
- partial GPU KV residency in the benchmark comparison;
- different sampling or a quality comparison;
- sparse-snapshot providers for CPU, Vulkan, HIP/ROCm, or MUSA;
- automatic plane selection;
- targeted shorter-batch replay, whose changed CUDA reduction shape was shown
  not to be bitwise equivalent;
- removed BeeLlama verifier, DFlash ring/tape, DDTree, CopySpec, or TurboQuant
  systems.

The implementation is not hard-coded in recurrent memory to a Qwen enum or a
backend name. A graph opts into an abstract operation and a backend exports a
capability. The current graph providers are Qwen3Next, Qwen3.5, and Qwen3.5 MoE
because those delta-net builders use the implemented convolution and gated
delta-net snapshot operations. Other graphs/backends can implement the contract
independently.

## Source lineage and workspace layout

Separate Git worktrees kept the baseline and CPU-KV experiment usable:

| Path | Branch/state during the work | Purpose |
|---|---|---|
| `/home/gencoolpc/beellama.cpp` | `main`, observed at `ba27edad2a84ff045a556df06661e821285c2fab` | Known BeeLlama baseline; not modified by this experiment |
| `/home/gencoolpc/beellama-kv-offload` | `exp/kv-cpu-offload`, initially `c490ebeab626d6002c8fbd28bc2cfdc5df411eda` | Integration target |
| `/home/gencoolpc/beellama-mtp-recurrent-plane-cap` | `exp/mtp-recurrent-plane-cap`, based at `5d6441f5310625911e6e9d2699d711df74af888c` | MTP implementation, tests, and profiles |
| `/home/gencoolpc/llama-pr2-b8f8557` | detached `b8f855761...` | Earlier reference worktree; not an implementation source |

The feature fork point is `5d6441f5310625911e6e9d2699d711df74af888c`.
At integration the CPU-KV branch had seven commits after it:

```text
a074495c4 docs: add CPU KV VRAM reduction roadmap
dbf469277 hybrid-memory: replace CPU-KV env switches with CLI/server options
f31f8b1fb hybrid-memory: fix pinned-KV routing, SWA/hybrid cpu_pinned gaps, ABI break
f727c6226 docs: reject lossless compression of the Q8_0 transfer stream
abb66abb1 kv-cache: add tunable partial GPU KV residency
8221275f9 docs: reconcile partial KV residency integration
c490ebeab kv-cache: align offload policy with memory layout
```

The only local item in the integration worktree was untracked `models.ini`. It
belongs to the user and was neither added nor modified.

A sibling worktree can be created without copying Git objects:

```bash
cd /home/gencoolpc/beellama.cpp
git worktree add \
  /home/gencoolpc/beellama-mtp-recurrent-plane-cap \
  -b exp/mtp-recurrent-plane-cap \
  5d6441f5310625911e6e9d2699d711df74af888c
```

## Hardware, operating system, and model identity

| Component | Recorded value |
|---|---|
| OS | CachyOS Linux, kernel `7.1.8-1-cachyos`, x86-64 |
| CPU | Intel Core Ultra 9 285K, 24 cores/threads, one socket/NUMA node |
| Decode/batch CPU range | CPUs 0-2 with 3 strict workers / CPUs 0-23 with 24 strict workers |
| GPU | RTX 5070 Ti, capability 12.0, PCI `00000000:02:00.0` |
| GPU memory | `nvidia-smi` 16,303 MiB total; server allocator 15,880 MiB usable |
| Driver | 610.57.04 |
| GPU limits | 300 W, maximum SM clock 3,090 MHz |
| System RAM | approximately 63,634 MiB available in the server log |
| CUDA | 13.3, nvcc V13.3.73 |
| GCC/G++ | 16.2.1 |
| CMake / Nsight Systems / ccache | 4.4.2 / 2026.1.3.425 / 4.13.6 |

Capture equivalent host facts before comparison:

```bash
uname -a
lscpu
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version,pci.bus_id,power.limit,clocks.max.sm \
  --format=csv,noheader
nvcc --version
gcc --version
g++ --version
cmake --version
nsys --version
ccache --version
```

Exact files:

```text
/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf
  14,437,471,712 bytes
  ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c

/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf
  927,606,912 bytes
  2f0a90f140322e570130adffe50ae45355f1a79715a641ce0e11ac6b1cdc822c
```

Verify them:

```bash
sha256sum \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf
```

The MTP layer is embedded in the target GGUF, so there is no separate draft
model path.

## ccache and build setup

### Why related worktrees initially compiled cold

The work initially encountered repeated cold compiles. ccache hashed absolute
build/source directories, preventing otherwise identical sibling-worktree
compilations from matching. One build also used
`GGML_CUDA_FA_ALL_QUANTS=ON` while the experimental builds used `OFF`; those
objects are genuinely different and cannot share cache entries.

All running builds were stopped before normalizing
`/home/gencoolpc/.config/ccache/ccache.conf`:

```ini
cache_dir = /home/gencoolpc/.cache/ccache
max_size = 50G
base_dir = /home/gencoolpc
hash_dir = false
absolute_paths_in_stderr = true
```

This makes equivalent paths under `/home/gencoolpc` reusable across worktrees.
It does not override compiler identity, arguments, generated headers, source
content, feature definitions, or CUDA architecture.

All four `build-cuda-all` directories were checked for:

```text
CMAKE_C_COMPILER_LAUNCHER=ccache
CMAKE_CXX_COMPILER_LAUNCHER=ccache
CMAKE_CUDA_COMPILER_LAUNCHER=ccache
GGML_CCACHE=ON
```

Inspect effectiveness with:

```bash
ccache --show-config
ccache --zero-stats
cmake --build build-cuda-all -j24 --target llama-server
ccache --show-stats
```

### Exact MTP build

```bash
cd /home/gencoolpc/beellama-mtp-recurrent-plane-cap
cmake -S . -B build-cuda-all \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_CUDA_KVARN=ON \
  -DGGML_CUDA_NCCL=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-cuda-all -j24 --target \
  llama-server test-backend-ops test-arg-parser \
  test-recurrent-state-rollback test-server-prompt-checkpoint \
  test-server-loop-guard test-backend-sampler
```

CMake recorded architecture `120`; current CMake/CUDA emits the corresponding
`120a` mode. Do not compare to an all-quant build unless both rows use it.

### Avoiding an accidental high-fanout rebuild

A new internal GGML declaration was briefly added to public
`ggml/include/ggml.h`, starting a large dependent rebuild. The build was
stopped, that edit was reverted, and the declaration was kept local to the
model builder with proper export linkage:

```cpp
extern "C" {
GGML_API ggml_tensor * ggml_gated_delta_net_ext(...);
}
```

The first attempt, `extern "C" GGML_API`, was invalid because this platform's
`GGML_API` expansion already includes linkage-related tokens.

After confirming public-header content exactly matched Git, its timestamp was
restored to the committed timestamp so the interrupted build did not rebuild
every dependent source merely because of mtime:

```bash
git diff --exit-code -- ggml/include/ggml.h
header_epoch=$(git log -1 --format=%ct -- ggml/include/ggml.h)
touch -d "@${header_epoch}" ggml/include/ggml.h
```

Only use this repair after the content check succeeds.

## Interface and validation

| Source | Name |
|---|---|
| CLI | `--spec-mtp-rs-planes N` |
| Environment | `LLAMA_ARG_SPEC_MTP_RS_PLANES=N` |
| INI | `spec-mtp-rs-planes = N` |

Validation occurs after all speculative modes are parsed, making CLI, env, and
INI order-independent. It rejects a negative value, one plane, above
`draft_max + 1`, any use without `draft-mtp` (including explicit zero), a
positive cap with draft max below one, or a true cap combined with EAGLE3,
DFlash, or DSpark recurrent rollback. An explicit full value is not a cap and
does not itself add a mixed-mode restriction. The server validates again after
resolving omitted DFlash depth and before allocating contexts.

For MTP-8:

| Value | Planes | Direct rejection horizon | Policy |
|---:|---:|---:|---|
| omitted/0 | 9 | 8 | ordinary snapshots, no cap replay |
| 9 | 9 | 8 | same as default |
| 4 | 4 | 2 | full-shape selected replay beyond horizon |
| 2 | 2 | 0 | every nonzero rejection replays |

A four-plane full allocation at MTP-3 exposes three ordinary rollback
positions, while a four-plane cap at MTP-8 exposes two: the cap reserves one
plane for its exact input state.

## Implementation map

### Arguments and propagation

- `common/common.h`: values, explicitness, true-cap predicate, and `n_rs_seq`.
- `common/arg.cpp` and `common/common.cpp`: CLI/env/INI and validation.
- common parameter copies preserve it through context fitting, model reload,
  and sleep/resume reconstruction.
- `include/llama.h` and `src/llama-context.cpp`: capability probe and mode
  selection for the server without exposing recurrent internals.

### Capability contract

- `llama_model::graph_supports_recurrent_sparse_snapshots()` defaults false.
- implemented delta-net graphs override it true.
- recurrent memory checks every actual buffer device through backend registry
  function `ggml_backend_recurrent_sparse_snapshots_supported`.
- native NVIDIA CUDA registers true; HIP/ROCm and MUSA return false.
- a meta device passes only if every child supports it.
- CPU/unknown backends have no provider and fail closed.

This prevents enablement based only on a model enum or a buffer name containing
`CUDA`.

### Plane bookkeeping and graph work

Recurrent memory records the absolute sequence position in each physical plane.
Ordinary snapshots retain existing relative rollback. Sparse rollback locates
an exact position. Sequence copy, keep, shift, divide, clear, save, and restore
update or invalidate sparse metadata with the corresponding state operation.

Ordinary capped verification writes:

```text
plane 0              newest verification output
plane 1              next-newest output
...
plane N-2            oldest directly retained output
plane N-1            exact pre-verification input
```

Selected replay writes only the accepted boundary to plane zero. Convolution
state and GDN state use the same policy. The CUDA GDN operation receives
trailing-snapshot count, selected token, and input-reservation flag. Its fused
cache matcher accepts the selected one-plane destination. Graph reuse keys
include sparse mode and selected token for recurrent, hybrid, and hybrid-iSWA
wrappers.

### Server transaction

The server preserves the first verification's tokens and sampler. On deep
rejection it:

1. saves accepted IDs and sampler;
2. selects the pre-verification plane;
3. rewinds draft context and speculative state through existing APIs;
4. selects the originally accepted output index;
5. submits the same full verification token batch again;
6. ignores replay logits, restores first-pass sampling, and selects the replayed
   boundary;
7. returns to normal sparse mode.

Replay metrics count cycles and actual target batch tokens. Existing checkpoint
counters remain available to other speculative policies.

## Development chronology and failures

### 1. Rejected host checkpoint/replay

The first design retained consecutive snapshots, captured a target checkpoint
when actual draft exceeded the horizon, and restored/replayed a shorter accepted
prefix only when rejection exceeded it.

Short outputs matched, but four planes reduced MTP-5 decode by 34.1% and MTP-8
by 38.1%. The long capped run captured 132 target checkpoints and restored 18:

```text
cumulative target payload: 20,710,056,048 bytes
capture wall time:          2,888.89 ms
restore wall time:            167.14 ms
extra Nsight H2D:           10,817.267 MB
extra Nsight D2H:           20,635.192 MB
```

The 5K output diverged at byte 1,337. Two clean full-plane controls matched,
excluding ordinary run variation. Retaining the first sampled token did not fix
later divergence. Replaying a shorter prefix changed CUDA batch shape and
floating-point reduction results; later logits crossed a decision boundary.
This was a candidate-design bug, not an upstream defect. The path was rejected
and removed.

### 2. Sparse replay graph-reuse bug

The first GPU selected-replay attempt restored the wrong boundary because the
graph cache considered normal verification and selected replay equivalent. It
reused a graph writing the final state. Adding sparse mode and selected-token
identity to all three graph input `can_reuse()` checks fixed it. A temporary
forced full-nine-plane diagnostic compared 156,894,364 serialized recurrent
bytes and found exact equality; diagnostic code was then removed.

### 3. Fused selected-write miss

The first correct long trace still had 4,888.461 MB D2D because the fused GDN
matcher required a four-plane destination. Selected replay writes one plane.
Accepting the one-plane destination restored fusion; final D2D is zero.

### 4. Explicit full and capability cleanup

Checks initially used `mtp_rs_planes > 0`, incorrectly treating explicit full
as capped. `is_mtp_rs_capped()` now requires
`0 < N < draft_max + 1`. MTP-3 with explicit four logged horizon three and
replay disabled.

The first working backend boundary used Qwen enums and a CUDA buffer-name
check. It was replaced with the graph/backend capability contract. A fresh
short run preserved hash, draft counts, replay, and zero checkpoint counts;
CPU recurrent state failed at startup as intended.

### 5. Build divergence and cleanup

Apparent divergence also came from sibling path-dependent cache keys, different
all-quant configuration, the public-header edit, and interrupted targets older
than source. All builds were stopped, ccache normalized, the header edit
removed, and intended targets rebuilt. No build/server/Nsight process remained
at final feature review.

## Automated validation

Parser tests cover omitted/zero, minimum/full, invalid ranges, non-MTP use,
capped mixed modes, explicit-full mixed mode, environment propagation, and INI.
Prompt-checkpoint tests cover generic capture/no-restore/deep-restore policy and
JSON metrics. The capped path bypasses target host capture; generic policy
coverage remains relevant to other modes and documents the rejected design.

Commands:

```bash
cd /home/gencoolpc/beellama-mtp-recurrent-plane-cap
cmake --build build-cuda-all -j24 --target \
  llama-server test-backend-ops test-arg-parser \
  test-recurrent-state-rollback test-server-prompt-checkpoint \
  test-server-loop-guard test-backend-sampler

ctest --test-dir build-cuda-all --output-on-failure \
  -R 'test-server-loop-guard-checkpoint-static|test-generate-models|test-recurrent-state-rollback|test-arg-parser|test-server-loop-guard|test-server-prompt-checkpoint|test-backend-sampler'

build-cuda-all/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET
git diff --check
```

Observed before commit:

```text
focused CTest suite:              7/7 passed
CUDA GATED_DELTA_NET operations: 36/36 passed
git diff --check:                clean
```

The covered surface includes recurrent rollback, prompt checkpoints, sampler
and loop-guard rollback, server fixtures, parser behavior, and CUDA recurrent
kernels.

## Historical versus merged placement controls

The MTP feature branch predates the CPU-KV branch's supported placement flags.
Exact historical measurements required both:

```bash
GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1
```

Omitting the second leaves recurrent state on CPU under `--no-kv-offload` and
the cap fails closed. Omitting the first changes host KV from pinned to pageable
and invalidates performance comparison.

After merge into `exp/kv-cpu-offload`, use the supported equivalents:

```text
--kv-cpu-pinned --recurrent-state-offload
```

The old experiment variables were intentionally removed there. Do not use only
old variables with the merged build.

## Short fixed-seed functional sweep

Use one clean process per row, changing only `DEPTH`, `PLANES`, `PORT`, and
`ALIAS`:

```bash
cd /home/gencoolpc/beellama-mtp-recurrent-plane-cap

GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 \
build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --ctx-size 4096 --parallel 1 --cont-batching \
  --kv-unified --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max DEPTH \
  --spec-mtp-rs-planes PLANES --spec-draft-ubatch-size 128 \
  --draft-p-min 0.85 --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --threads 3 --threads-batch 24 --cpu-range 0-2 \
  --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --seed 1234 --cache-ram 0 --alias ALIAS \
  --host 127.0.0.1 --port PORT
```

Wait for health and request 128 greedy tokens:

```bash
until curl --fail --silent http://127.0.0.1:PORT/health >/dev/null; do
  sleep 1
done

jq -nc '{
  prompt:"Continue the sequence and explain the rule briefly: 1, 4, 9, 16, 25,",
  n_predict:128,
  temperature:0,
  seed:1234,
  cache_prompt:false,
  stream:false
}' | curl --fail --silent --show-error \
  -H 'Content-Type: application/json' --data-binary @- \
  http://127.0.0.1:PORT/completion > RESPONSE.json

jq -r '.content' RESPONSE.json | sha256sum
jq '.timings | {
  predicted_per_second,draft_n,draft_n_accepted,
  spec_checkpoint_captures,spec_checkpoint_restores,
  spec_replay_cycles,spec_replay_batch_tokens
}' RESPONSE.json
```

`jq -r` adds a newline; preserve that when comparing the short hash. Every row
produced
`64ccba06fd390281d73f4bf6d55e49f21e8cbf42a5a2064693b3a883d2c6e7c3`.

| MTP depth | Planes | Decode | RS allocation | Replay cycles/tokens | Checkpoints |
|---:|---:|---:|---:|---:|---:|
| 3 | default full (4) | 91.1833 t/s | 598.50 MiB | 0/0 | 0/0 |
| 3 | explicit full (4) | 90.9932 t/s | 598.50 MiB | 0/0 | 0/0 |
| 5 | default full (6) | 93.6697 t/s | 897.75 MiB | 0/0 | 0/0 |
| 5 | capped (4) | 87.2313 t/s | 598.50 MiB | 3/18 | 0/0 |
| 8 | default full (9) | 94.5937 t/s | 1,346.62 MiB | 0/0 | 0/0 |
| 8 | capped (4) | 85.2903 t/s | 598.50 MiB | 5/39 | 0/0 |

After fusion, MTP-8/four-plane short decode was 85.2396 t/s. After capability
refactoring it was 85.1355 t/s with draft/accepted 114/89, replay 5/39, and
checkpoints 0/0. Hash/counters were unchanged. The final short artifact is
`/tmp/mtp8-cap4-capability-short.json`, SHA-256
`4c8db94d1dd4850b69aea5955d0101b23460a28a9ddaf990aed03fb1f6fba637`.

For the fail-closed control, omit `GGML_RECURRENT_STATE_OFFLOAD=1` historically
or `--recurrent-state-offload` after merge. Startup must exit nonzero with:

```text
capped MTP recurrent planes require a model graph and recurrent-state backend with selected sparse-snapshot support
```

Recorded negative log:
`/tmp/mtp8-cap4-capability-cpu-reject.log`, SHA-256
`f580ebd3ed84726f36bfaa61ae5cf88f455f6c83f27dd97adc9e6ff37ff5a830`.

## Exact 5,000-token Nsight comparison

### Fixed configuration

Both rows used clean processes, context 32,000, one slot, target batch/ubatch
1,024/512, draft ubatch 128, MTP maximum 8, symmetric target/draft Q8_0 KV,
pinned CPU attention KV, GPU recurrent state, all layers on GPU, FlashAttention,
split none, fit off, three strict decode workers on CPUs 0-2, 24 strict batch
workers on CPUs 0-23, draft p-min 0.85, seed 1234, default chat sampling, prompt
cache off, streaming off, and reasoning loop guard `force-close`.

Nsight traced CUDA, NVTX, and OS runtime with CPU sampling and context-switch
collection disabled. The only experimental difference was:

```text
full: --spec-mtp-rs-planes 0
cap:  --spec-mtp-rs-planes 4
```

The exact prompt was:

```text
Write a single self-contained HTML file: an interactive orbital mechanics sandbox.
Canvas-based, 60fps. Users can click-drag to fling new planets into the system;
gravity is simulated with velocity Verlet integration against a central star.
Include trailing orbit paths that fade, collision merging with a mass-conserving
flash, a mass/velocity readout on hover, and a pause/reset UI. No libraries,
no assets — one file, pure JS.
```

Create the request with the repository editing tool or another JSON-safe editor:

```json
{"model":"mtp8-5k","messages":[{"role":"user","content":"Write a single self-contained HTML file: an interactive orbital mechanics sandbox.\nCanvas-based, 60fps. Users can click-drag to fling new planets into the system;\ngravity is simulated with velocity Verlet integration against a central star.\nInclude trailing orbit paths that fade, collision merging with a mass-conserving\nflash, a mass/velocity readout on hover, and a pause/reset UI. No libraries,\nno assets — one file, pure JS."}],"max_tokens":5000,"seed":1234,"stream":false,"cache_prompt":false}
```

Save that exact single object as `/tmp/mtp8-5k-request.json`. It intentionally
omits `temperature`, using the model/server default. Adding temperature zero is
a different benchmark. The server produced 149 prompt and 5,000 completion
tokens, all completion tokens reported as reasoning.

### Server launch

Set one row at a time:

```bash
export MTP_PLANES=0
export MTP_PORT=8090
export MTP_TAG=full
```

For cap use planes 4, another unused port, and tag `cap4-fused`. Launch:

```bash
cd /home/gencoolpc/beellama-mtp-recurrent-plane-cap

GGML_KV_CPU_PINNED=1 GGML_RECURRENT_STATE_OFFLOAD=1 LLAMA_TRACE=1 \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o "/tmp/mtp8-gpu-replay-5k-${MTP_TAG}-trace" \
  build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --ctx-size 32000 --parallel 1 --cont-batching \
  --kv-unified --batch-size 1024 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 8 \
  --spec-mtp-rs-planes "${MTP_PLANES}" --spec-draft-ubatch-size 128 \
  --draft-p-min 0.85 --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --threads 3 --threads-batch 24 --cpu-range 0-2 \
  --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
  --alias mtp8-5k --host 127.0.0.1 --port "${MTP_PORT}" \
  > "/tmp/mtp8-gpu-replay-5k-${MTP_TAG}.log" 2>&1 &

export MTP_NSYS_PID=$!
```

On the merged CPU-KV branch, remove the two historical environment variables
and add `--kv-cpu-pinned --recurrent-state-offload` after
`--no-kv-offload`.

### Corrected progress and memory monitoring

The original full-run helper monitored the Nsight wrapper PID, not the child
server, and wrote zeros for init/live/peak VRAM. A manual server sample of
15,314 MiB plus native allocation logs was retained. The capped helper correctly
recorded 14,538/14,576 MiB. Use the corrected child-PID procedure:

```bash
until curl --fail --silent "http://127.0.0.1:${MTP_PORT}/health" >/dev/null; do
  printf 'waiting for port %s\n' "${MTP_PORT}"
  sleep 5
done

MTP_SERVER_PID=$(pgrep -n -f "build-cuda-all/bin/llama-server.*--port ${MTP_PORT}")
test -n "${MTP_SERVER_PID}"
printf 'nsys_pid=%s server_pid=%s\n' "${MTP_NSYS_PID}" "${MTP_SERVER_PID}"

MTP_INIT_VRAM=$(nvidia-smi --query-compute-apps=pid,used_memory \
  --format=csv,noheader,nounits | awk -F, -v pid="${MTP_SERVER_PID}" \
  '$1 + 0 == pid { gsub(/ /,"",$2); print $2 }')

(
  while kill -0 "${MTP_SERVER_PID}" 2>/dev/null; do
    date --iso-8601=seconds
    curl --silent "http://127.0.0.1:${MTP_PORT}/slots" | \
      jq 'map({id,state,n_prompt_tokens,n_decoded,n_remaining})'
    nvidia-smi --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits | awk -F, -v pid="${MTP_SERVER_PID}" \
      '$1 + 0 == pid { print "vram_mib=" $2 }'
    sleep 10
  done
) > "/tmp/mtp8-gpu-replay-5k-${MTP_TAG}-progress.log" 2>&1 &
export MTP_MONITOR_PID=$!
```

This exposes progress for the long run. The server log also prints native
prompt/decode progress.

### Request, resource capture, and shutdown

```bash
MTP_REQUEST_START=$(date +%s)
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/mtp8-5k-request.json \
  "http://127.0.0.1:${MTP_PORT}/v1/chat/completions" \
  > "/tmp/mtp8-gpu-replay-5k-${MTP_TAG}-response.json"
MTP_REQUEST_END=$(date +%s)

MTP_LIVE_VRAM=$(nvidia-smi --query-compute-apps=pid,used_memory \
  --format=csv,noheader,nounits | awk -F, -v pid="${MTP_SERVER_PID}" \
  '$1 + 0 == pid { gsub(/ /,"",$2); print $2 }')
MTP_VMHWM=$(awk '/^VmHWM:/ { print $2 }' "/proc/${MTP_SERVER_PID}/status")

printf 'server_pid=%s\ninit_vram_mib=%s\nlive_vram_mib=%s\nvmhwm_kib=%s\nwall_request_s=%s\n' \
  "${MTP_SERVER_PID}" "${MTP_INIT_VRAM}" "${MTP_LIVE_VRAM}" \
  "${MTP_VMHWM}" "$((MTP_REQUEST_END - MTP_REQUEST_START))" \
  > "/tmp/mtp8-gpu-replay-5k-${MTP_TAG}.mem"

kill -INT "${MTP_SERVER_PID}"
wait "${MTP_NSYS_PID}"
kill "${MTP_MONITOR_PID}" 2>/dev/null || true
```

Read `/proc/$pid/status` before shutdown. `nvidia-smi` estimates process device
allocation; it does not account for page-locked system memory.

### Output-equivalence check

Hash the reasoning field without adding a newline:

```bash
jq -j '.choices[0].message.reasoning_content' \
  /tmp/mtp8-gpu-replay-5k-full-response.json | sha256sum
jq -j '.choices[0].message.reasoning_content' \
  /tmp/mtp8-gpu-replay-5k-cap4-fused-response.json | sha256sum

cmp \
  <(jq -j '.choices[0].message.reasoning_content' \
      /tmp/mtp8-gpu-replay-5k-full-response.json) \
  <(jq -j '.choices[0].message.reasoning_content' \
      /tmp/mtp8-gpu-replay-5k-cap4-fused-response.json)
```

Expected for both:

```text
09ab907a3b42bb586f7714eb1327461ff853e37cf4895795ab4328f99382bdb6
```

The full JSON hashes differ because timings and model metadata differ. Compare
reasoning content, token counts, and draft statistics separately.

### Nsight export and transfer query

```bash
nsys export --type sqlite --force-overwrite=true \
  --output=/tmp/mtp8-gpu-replay-5k-full-trace.sqlite \
  /tmp/mtp8-gpu-replay-5k-full-trace.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output=/tmp/mtp8-gpu-replay-5k-cap4-fused-trace.sqlite \
  /tmp/mtp8-gpu-replay-5k-cap4-fused-trace.nsys-rep
```

Query bytes and counts by direction:

```bash
sqlite3 TRACE.sqlite '
SELECT e.name, SUM(m.bytes) AS bytes, COUNT(*) AS copies
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
JOIN ENUM_CUDA_MEMCPY_OPER AS e ON e.id = m.copyKind
GROUP BY e.name
ORDER BY e.name;'
```

Or use:

```bash
nsys stats --report cuda_gpu_mem_size_sum TRACE.nsys-rep
```

The ledger reports decimal MB (`bytes / 1,000,000`). Serialized checkpoint
payload is not a transfer measurement; Nsight is authoritative for H2D/D2H.

### Recorded results

| Measurement | Full, 9 planes | Cap, 4 planes | Difference |
|---|---:|---:|---:|
| Prompt | 663.2628 t/s | 674.4126 t/s | +1.68% |
| Decode under Nsight | 51.383169 t/s | 50.979222 t/s | -0.7861% |
| Prompt tokens | 149 | 149 | identical |
| Completion/reasoning tokens | 5,000/5,000 | 5,000/5,000 | identical |
| Draft generated/accepted | 2,027/1,689 | 2,027/1,689 | identical |
| Replay cycles/batch tokens | 0/0 | 38/217 | +38/+217 |
| Target captures/restores | 0/0 | 0/0 | identical |
| Target checkpoint payload | 0 | 0 | identical |
| Target recurrent allocation | 1,346.62 MiB | 598.50 MiB | -748.12 MiB |
| Init/live/peak process VRAM | helper bug; manual live 15,314 MiB | 14,538/14,576/14,576 MiB | at least -738 MiB live |
| Process `VmHWM` | 15,587,944 KiB (15,222.60 MiB) | 15,553,280 KiB (15,188.75 MiB) | -33.85 MiB |
| Request wall time | 100 s | 100 s | same one-second resolution |
| Nsight H2D | 388,825.670 MB | 392,764.233 MB | +3,938.563 MB |
| Nsight D2H | 6,646.602 MB | 6,896.808 MB | +250.206 MB |
| Nsight D2D | 0 MB | 0 MB | unchanged |

The extra transfers are from 38 full-shape replay batches, not repeated
149.6 MiB state serialization. The accepted trade on this workload is a 0.79%
decode cost for 748.12 MiB less recurrent VRAM.

## Artifact manifest

`/tmp` is ephemeral; copy these reports to durable storage before reboot if
needed.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `/tmp/mtp8-gpu-replay-5k-full-response.json` | 20,247 | `5b16aec785f00e92e819fd564ebcebd5d1556da4b51d74121d4655f2ae90da8d` |
| `/tmp/mtp8-gpu-replay-5k-cap4-fused-response.json` | 20,242 | `fa63e363698763007c53182a3bcd438fd338dc37ff381a87b5167dfd8fdf402a` |
| `/tmp/mtp8-gpu-replay-5k-full-trace.nsys-rep` | 267,374,416 | `74bf4fb434a2af2142b137efe533b1e8623787639aca4a90340d59d000597430` |
| `/tmp/mtp8-gpu-replay-5k-cap4-fused-trace.nsys-rep` | 254,447,383 | `af6b001dd36d322a3a333e263b4dff677a0d9c55d51f4c9f3ac0b7d1c29a0913` |
| `/tmp/mtp8-gpu-replay-5k-full-trace.sqlite` | 782,966,784 | `56649faaa471836cb7f2c032a99dd95cc1bb7b45e09856a83ed19badf2ed35b7` |
| `/tmp/mtp8-gpu-replay-5k-cap4-fused-trace.sqlite` | 732,631,040 | `700b5249f9d6c4c038824a17af723f1f0127e9a4c2291fa40f47298b9c0cf2ea` |
| `/tmp/mtp8-gpu-replay-5k-full.log` | 249,639 | `bcc7ebf66875319e407d7ee44f97a3415ab02d523a04cc71eb6443919eaf6c50` |
| `/tmp/mtp8-gpu-replay-5k-cap4-fused.log` | 250,871 | `d52a742fa1f9ec04fe7c731c8a66714e106db1d091c8be03e1e57adfb671b7dc` |
| `/tmp/mtp8-gpu-replay-5k-full.mem` | 105 | `3ae83eba5a76eafb539f799838699a0eeaa9e07fc843e669cc6d99956dd17279` |
| `/tmp/mtp8-gpu-replay-5k-cap4-fused.mem` | 117 | `d5cdf265fd606d5875355395bfe28121b0399d181824ef6aeb0c536370678663` |
| `/tmp/mtp8-cap4-capability-short.json` | 2,917 | `4c8db94d1dd4850b69aea5955d0101b23460a28a9ddaf990aed03fb1f6fba637` |
| `/tmp/mtp8-cap4-capability-cpu-reject.log` | 916 | `f580ebd3ed84726f36bfaa61ae5cf88f455f6c83f27dd97adc9e6ff37ff5a830` |

The server identified the measured binary as version 11217,
`5d6441f53-dirty`, because the candidate was built from the uncommitted diff on
the recorded base. The implementation commit at this document's top is the
source-of-truth retained diff; the artifact log's dirty identity alone is not.

## Reproduction acceptance gates

A conforming reproduction must satisfy all of these:

1. default/zero and explicit full allocate full depth without selected replay;
2. four-plane MTP-8 reports 598.50 MiB recurrent allocation for this exact
   model, with no hidden full-depth buffer;
3. full and capped fixed-seed reasoning output is byte-identical;
4. paired long-run draft generated/accepted totals match;
5. cap reports replay work but zero target captures, restores, and payload;
6. unsupported graph/backend combinations fail closed;
7. Nsight shows no selected-write D2D regression;
8. binary, models, cache types, placement, affinity, context/batches, prompt,
   request, and sampling match; only the plane value differs;
9. long steps expose progress and no overlapping build/server load contaminates
   the result.

Throughput need not reproduce to the last decimal, but hardware/toolchain
differences and variance must be reported. Output and allocation gates are hard.

## Integration procedure

The feature and CPU-KV branches diverged at `5d6441f...`; use a real merge, not
an overwrite:

```bash
cd /home/gencoolpc/beellama-mtp-recurrent-plane-cap
git status --short
git diff --check
# commit the reviewed implementation and this record

cd /home/gencoolpc/beellama-kv-offload
git status --short
git merge --no-ff exp/mtp-recurrent-plane-cap \
  -m 'Merge MTP recurrent-plane cap into CPU KV offload'
```

Conflict-resolution requirements:

- preserve `--kv-cpu-pinned` and `--recurrent-state-offload`, not removed getenv
  reads;
- preserve the unified per-layer KV buffer plan and partial-GPU-layer policy;
- add the MTP option alongside those parameters;
- retain both branches' experiment history instead of choosing one file;
- do not add or edit untracked `models.ini`;
- rebuild the merged tree instead of treating a pre-merge dirty binary as a
  merge-commit build.

Minimum merged validation: focused CTest suite, all CUDA GDN cases, parser,
`git diff --check`, and a clean-process capped MTP-8 short request using the
supported placement flags.

## Post-merge validation record

Merge commit: `TO_BE_FILLED_AFTER_MERGE`.

Post-merge build/test/runtime results: `TO_BE_FILLED_AFTER_MERGE_VALIDATION`.
