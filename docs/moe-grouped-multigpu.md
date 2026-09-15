# Stage A grouped MoE layer split

This stage adds owner-local grouped MoE caches to the existing layer-split path. It does not add tensor or expert parallelism.

## Architecture

- `build_moe_sources()` records the layer of each ordinary/chunk semantic source group. Snapshot ordering and full V2 broadcast to every backend are unchanged. Buffer ownership is not a filter for candidate publication.
- `build_moe_ffn()` records the exact operations allocated during that call and their direct boundary inputs. Regions carry layer, semantic group, domain, route, down bank and output identity. Collection does not follow pre-existing ancestors or infer identity from tensor names.
- Runtime and reserve graph construction place complete eligible regions on the backend for `model.dev_layer(il)` before scheduler allocation/splitting. Placement requires layer split, cache enabled, no active LoRA map, canonical local `ARGSORT -> VIEW` routing, supported operations and compatible weight sources. Preallocated operations and externally supplied routes are declined without partial assignments.
- Allocation-free fitting builds the same MoE source metadata and retains explicit owner assignments while measuring reserve sizes.
- Non-view operations are explicitly assigned with `ggml_backend_sched_set_tensor_backend()`. Internal views follow their source; boundary views remain on the external source owner. The scheduler transfers boundary activations. Existing attention, KV placement, ordered reduction and replacement policy are unchanged.
- Layer-owner/capability changes synchronize/reset scheduler state, invalidate runtime/reserve graph results, advance the certificate owner generation and request reservation again. Existing graph parameter and candidate generation invalidation remain in place.
- Required-grouped capability is no longer an OR across available endpoints. Early policy checks the cached layers' nominal owners; submission checks every scheduled MMID owner. The latter does not inspect post-split bank buffer types, because scheduler copies can replace those buffers.
- Each backend retains its own grouped context, resources, slots, streams, clocks, leases, graph fingerprints and generations. There is no cross-owner slot access, NCCL operation, expert/tensor-parallel split, or P2P correctness requirement.
- Cross-physical CUDA asynchronous and synchronous buffer copies decline when `cudaDeviceCanAccessPeer()` is false. Existing scheduler synchronization and host-copy fallback then handle the transfer. `GGML_CUDA_NO_PEER_COPY=ON` also forces this path.

Owner selection is not limited to two CUDA devices, but physical validation currently covers two. Grouped cache state and capability negotiation are not transported through the RPC backend; ordinary RPC layer offload remains separate from this feature.

External routes and unsupported regions retain existing certification/fallback behavior; affinity is not permission to bypass the route certificate. A required-grouped refusal remains a failure. Earlier splits may already have submitted work when a later backend fails; callers must not consume failed graph outputs or assume output/KV rollback.

## Proof telemetry

Enable `--experimental-logs`. Use `--verbosity 4` as well for runtime `moe-placement` debug lines.

- `moe-grouped-plan`: context owner, CUDA logical/physical ordinal, generation, semantic group, domain, selected plan outcome and reason. Outcomes: 0 prefill legacy, 1 grouped decode, 2 legacy decode, 3 error. Reasons 0 and 8 mean eligible and route respectively; remaining reasons are defined by `ggml_cuda_moe_graph_plan`.
- `moe-grouped-owner-group`: per-group ready, submitted and completed counts while the semantic table is available, including candidate replacement boundaries.
- `moe-grouped-owner`: owner, physical device, PCI identity, generation, grouped/legacy dispatch counts, direct/capture/replay dispatch counts, ready/submitted/completed counts and preparation/completion errors. Shutdown retains owner totals even after the semantic table is cleared.

Dispatch counters count groups admitted to a dispatch, before reader submission. `submitted` counts final-reader submission; `completed` counts successful completion-event recording, not a host wait for GPU completion. The fixture synchronizes before checking counts and exact output. Capture/replay counters alone are not numerical or completion proof.

The old `fallback` counter had no writer. It now counts decode groups dispatched through legacy execution, including a grouped plan forced into legacy dispatch. The aggregate line format is unchanged. `rollback` remains a reserved zero field, not evidence of graph-output rollback. Statistics still have process-wide reset boundaries and are not strict per-request attribution with concurrent requests.

## Fixtures

`test-moe-cache --grouped-multigpu-only` requires two distinct physical CUDA GPUs. It prints PCI identity and peer capability, rejects duplicate physical identities, and returns 77 with a clear SKIP reason when two distinct exposed devices are unavailable. CTest registers that return code as a skip, not a positive hardware result.

The two-layer fixture uses the production graph builder and real scheduler, shared immutable host sources, and separate CUDA owners. It covers fused Q4_0, separate Q4_K and ungated layouts; direct execution, capture and replay; changing routes; a 512 KiB bounded source/staging budget; candidate replacement on one owner; owner teardown; and injected failure on the second layer. Every positive owner must complete exactly eight grouped executions during the main comparison, with zero legacy dispatches, against the existing exact-output reference helper. The reference uses the same layer placement with an empty candidate snapshot. Separate bidirectional exact-copy checks require both CUDA copy interfaces to decline when peer access is unsupported, then exercise the generic synchronous and asynchronous fallback APIs.

The negative route case deliberately moves the second route producer to CPU. Ordinary decode must report legacy fallback and match the reference; required-grouped execution must fail. Capability tests reject an unsupported MMID owner even while another capable owner is available. Placement tests reject external routes, unsupported operations and preallocated outputs without partial assignments, then verify every operation's owner and view/source consistency.

`--grouped-layer-only` runs the builder/placement cases on one CUDA device and separately exercises two backend contexts sharing source weights, replacement and teardown. It also runs the survivor legacy/grouped transition for every positive layout; only the physical runner destroys the other device's owner first. The survivor check resets prior telemetry and requires three fresh, exact-output grouped completions, zero legacy/fallback/error activity, and capture/replay (or three direct dispatches when graphs are disabled). It explicitly does not claim two-physical-device validation.

## Local validation (2026-09-15)

Local hardware is one RTX 5070 Ti, CUDA 13.3, native SM 120. Build directories and logs are ignored artifacts in this worktree.

| Check | Result / artifact |
| --- | --- |
| CUDA build, forced no peer copy, NCCL disabled | PASS; `build-stage-a/final-build.log`, `build-stage-a/final-incremental-build.log` |
| CUDA build, normal peer capability path | PASS; `build-stage-a/default-build.log`; test, CLI and server targets |
| Complete existing MoE cache suite, both builds | PASS; `build-stage-a/full-suite.log`, `build-stage-a/default-full-suite.log` |
| New layer suite, capture/replay enabled | PASS; `build-stage-a/layer-final.log` |
| New layer suite, CUDA graphs disabled | PASS; `build-stage-a/layer-direct-final.log` |
| New layer suite, normal transfer build | PASS; `build-stage-a/default-layer-final.log` |
| Registry, grouped decode, replay and multirow focused modes | PASS; focused logs in `build-stage-a` |
| CUDA Q4_0 MMID against CPU | 74/74 PASS; `build-stage-a/backend-mmid.log` |
| CTest layer / physical multi-GPU gates | Layer PASS, multi-GPU SKIP (one physical GPU); `build-stage-a/ctest-final.log` |
| Formatting / whitespace | New fixture formatted with clang-format; `git diff --check` clean |

The full-suite run initially exposed an assertion that the previously unwritten fallback counter was zero during legacy execution. The affected assertions now require exactly one legacy dispatch/fallback and retain exact-output, zero-grouped-completion and error checks.

An exploratory virtual-device attempt hit an existing legacy CUDA MMID invalid-device-ordinal error before candidate validation (`build-stage-a/owners.log`). The temporary virtual runner was removed. Virtual devices are not accepted as physical multi-GPU proof.

Unvalidated here: actual two-physical-device layer transfers, PHB/no-P2P behavior, heterogeneous SM 89/86 exact output, two-physical-device teardown/failure, full-model load/decode and runtime/reserve placement, and real-model throughput/RAM/VRAM. The synthetic single-device tests do not close these hardware gates. Builds retain existing compiler warnings about `ggml_can_fuse` indices and CUDA namespace extra semicolons; they are not warning-clean.

### Physical-gate sentinel follow-up

The user reported that Piggi's forced-no-peer-copy/NCCL-disabled build passed bidirectional boundary copies, both owners' eight grouped completions without legacy dispatch, and bounded-budget cases, but failed at `test-moe-cache-fixtures.cpp:988`, `(gate == sentinel) == skipped`, in the survivor check. This report was not independently verified by accessing Piggi or its full log.

The local layer suite now exercises that previously two-device-only survivor transition. Before correcting its expectation, it reproduced the exact line-988 failure (`build-stage-a/sentinel-before-fix.log`, exit 1). Separate Q4_K execution can fuse both MMID readers and GLU, writing the GLU result directly while leaving the intermediate gate/up sentinels untouched. The survivor caller incorrectly used the helper's default `f3_skipped=false`. The ungated case has no gate tensor and cannot reach that assertion.

The caller now derives the expectation from the actual survivor device's MMID capability, separate-bank compatibility, and fusion setting, matching existing F3 fixtures. The sentinel assertions, route checks, and exact-output helper remain unchanged; no runtime/CUDA implementation change was needed. Fresh telemetry and three passes also prevent retired-generation activity from satisfying the survivor completion/replay checks.

Follow-up validation on the local RTX 5070 Ti:

| Check | Result / artifact |
| --- | --- |
| Rebuild `test-moe-cache`, forced no-peer/NCCL off and normal peer/NCCL on | PASS; `build-stage-a/sentinel-fix-build.log`, `build-stage-a/sentinel-fix-default-build.log` |
| Expanded layer suite, all four fusion on/off and graphs on/off combinations | PASS; `build-stage-a/sentinel-after-fix.log`, `sentinel-direct.log`, `sentinel-no-fusion.log`, `sentinel-no-fusion-direct.log` in the same directory |
| Expanded layer suite, normal transfer build | PASS; `build-stage-a/sentinel-default.log` |
| Complete existing MoE-cache suite, forced no-peer build | PASS; `build-stage-a/sentinel-full-suite.log` |
| CTest layer / physical multi-GPU gates | Layer PASS, physical gate SKIP (one GPU); `build-stage-a/sentinel-ctest.log` |

## Piggi physical and real-model validation (2026-09-15)

The corrected fixture was transferred to `/home/gen/Services/llama-moe-grouped-multigpu` at the same baseline and validated on an RTX 4070 (SM 89, PCI `0000:01:00.0`) plus RTX 3060 (SM 86, PCI `0000:06:00.0`). The devices are PHB-connected and report no peer access in either direction.

| Check | Result / artifact |
| --- | --- |
| Forced no-peer-copy, `CUDA_VISIBLE_DEVICES=0,1` | PASS; `build-stage-a-host/multigpu-01-rerun.log` |
| Forced no-peer-copy, `CUDA_VISIBLE_DEVICES=1,0` | PASS; `build-stage-a-host/multigpu-10.log` |
| Forced no-peer-copy, CUDA graphs disabled | PASS; `build-stage-a-host/multigpu-01-direct.log` |
| Complete MoE-cache suite | PASS; `build-stage-a-host/full-suite.log` |
| Normal peer-capability build, both GPU orders | PASS; `build-stage-a-host-peer/multigpu-01.log`, `multigpu-10.log` |
| Gemma-4-26B-A4B real decode, both GPU orders | PASS; `build-stage-a-host/gemma4-a4b-real-01-load-none.log`, `gemma4-a4b-real-10.log` |

Every physical fixture run reported exact bidirectional boundary copies through the host fallback and ended with `two-physical-device grouped layer execution OK`. Positive cases completed eight grouped executions on each owner with zero legacy dispatch. The intentional unsupported-route case retained its expected cached legacy fallback and required-grouped refusal.

GGUF metadata confirmed `gemma4`, 30 blocks, 128 experts and 8 active experts. The 4070-first 64-token run completed 1,008 grouped calls on the 4070 owner and 882 on the 3060 owner at 9.6 generation tokens/s. The reversed 128-token run completed 2,032 grouped calls on the 3060 owner and 1,778 on the 4070 owner at 11.0 generation tokens/s. Both reported zero grouped fallback, unsupported, prepare and finish errors, and produced coherent text. These are correctness runs with different token counts, not a controlled performance comparison.

Bounded pinning with the default mmap load was rejected before inference because read-only auxiliary registration is unsupported. `--load-mode none` supplied writable host allocations and succeeded while retaining the 2 GiB bound. The successful run reported 1.551 GiB source memory, 434 MiB peak staging and 1.975 GiB peak pinned accounting.

Ornith-1.5-35B-A3B (`qwen35moe`, 41 layers, 256 experts, 8 active) was also tested with a 256-token deterministic decode. Both owners completed grouped work with zero fallback, preparation or completion errors. Equal layer split reached 39.6 tokens/s with 256 slots, zero cache evictions and 8.009/7.649 GiB peak VRAM. A 128-slot `4,1` split reached 38.8 tokens/s with 36 cache evictions and 6.876/1.927 GiB peak VRAM. These results show a capacity/performance tradeoff, not parallel speed scaling; the slower GPU remains on the sequential layer path.

Future work is intentionally separate: preserve compatible expert residency across legacy-prefill/grouped-decode handoffs, and evaluate expert or tensor parallel execution for simultaneous multi-GPU compute. RPC grouped-cache transport would require protocol-level capability, state and telemetry support.

## Remote reproduction commands

Run from the remote isolated worktree; avoid other running model workloads during measurements.

```sh
cmake -S . -B build-stage-a-host -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  '-DCMAKE_CUDA_ARCHITECTURES=89-real;86-real' \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_CUDA_NO_PEER_COPY=ON -DGGML_CUDA_NCCL=OFF
cmake --build build-stage-a-host --target test-moe-cache llama-cli llama-server -j 12

env -u GGML_CUDA_DEVICES CUDA_VISIBLE_DEVICES=0,1 \
  ./build-stage-a-host/bin/test-moe-cache --grouped-multigpu-only
env -u GGML_CUDA_DEVICES CUDA_VISIBLE_DEVICES=1,0 \
  ./build-stage-a-host/bin/test-moe-cache --grouped-multigpu-only
env -u GGML_CUDA_DEVICES CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_DISABLE_GRAPHS=1 \
  ./build-stage-a-host/bin/test-moe-cache --grouped-multigpu-only
env -u GGML_CUDA_DEVICES CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_DISABLE_FUSION=1 \
  ./build-stage-a-host/bin/test-moe-cache --grouped-multigpu-only
./build-stage-a-host/bin/test-moe-cache
```

Keep each run's stdout/stderr and exit status. Direct invocation must return 0, not 77, on Piggi. Then configure a second build with `GGML_CUDA_NO_PEER_COPY=OFF` and repeat the physical fixture: Piggi's reported PHB topology and both NS peer matrices should exercise the runtime host-fallback decision even without the forced build option. No NCCL or peer-access enablement is needed for this layer split.

The validated sparse model is Gemma-4-26B-A4B. Its metadata shows nonzero routed expert counts; the Qwen3.8-27B files are dense and are not valid evidence for this feature.

```sh
MOE_MODEL=/home/piggidragon/Services/models/llama-cpp/Gemma4-26b/gemma-4-26B-A4B-heretic-APEX-I-Compact.gguf
env -u GGML_CUDA_DEVICES CUDA_VISIBLE_DEVICES=0,1 \
  ./build-stage-a-host/bin/llama-cli -m "$MOE_MODEL" \
  --load-mode none \
  --gpu-layers all --split-mode layer --tensor-split 1,1 --fit off \
  --cpu-moe --moe-expert-cache-size 16 --moe-expert-cache-host-pinned-mb 2048 \
  --ctx-size 2048 --batch-size 128 --ubatch-size 128 \
  --seed 1 --temp 0 --n-predict 128 --single-turn --simple-io \
  --experimental-logs --verbosity 4 \
  --prompt 'Explain how a CPU cache works in three short paragraphs.'
```

This is a conservative starting configuration, not a measured optimum or a fit guarantee. Cache size must cover the active routed expert count. The 2048 MiB limit bounds MoE pinned source/staging memory, not total RSS; 0 means full pinning, not disabled pinning. Check that the mandatory staging requirement fits and monitor available RAM, swap activity and per-GPU VRAM on the 32 GB host. Cache pools are not fully accounted for by automatic fit. Cache enabled overrides `--cpu-moe` for matched expert tensors. In layer mode, `--tensor-split` supplies layer placement proportions; it does not request ordinary tensor parallelism.

Require distinct PCI owners, eligible decode plans, nonzero synchronized grouped submissions/completions on both owners, coherent generated output, and no unexpected fallback or prepare/finish errors. Prefill legacy activity is expected and is not decode failure. Do not treat `offloaded N/N layers`, model load success, or `rollback=0` as grouped proof.

After correctness, compare against the unchanged baseline binary using identical model, prompt, GPU order, layer proportions, cache slots, context, batch sizes and sampling. Preserve full logs and output; run cold and repeated warm cases separately. Single-token layer split executes successive layers on different GPUs, not the same expert group concurrently. No throughput improvement or optimal split ratio is claimed without measurement.

## Changed-file inventory

- `src/llama-model.h`, `src/llama-model.cpp`: source-group layer identity.
- `src/llama-graph.h`, `src/llama-graph.cpp`: typed exact-region metadata and atomic affinity validation/application.
- `src/llama-context.h`, `src/llama-context.cpp`: runtime/reserve placement, owner invalidation and participating-owner capability.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: no-peer synchronous/asynchronous-copy decline.
- `ggml/src/ggml-cuda/moe-cache.cuh`, `ggml/src/ggml-cuda/moe-cache.cu`: owner-local proof telemetry and fallback accounting.
- `tests/test-moe-cache-multigpu.cpp` (new): physical gate, production-builder fixture, failure and lifetime tests.
- `tests/test-moe-cache.cpp`, `tests/test-moe-cache.h`, `tests/CMakeLists.txt`: runners and CTest registration.
- `tests/test-moe-cache-registry.cpp`: layer identity assertions.
- `tests/test-moe-cache-dispatch.cpp`, `tests/test-moe-cache-graphs.cpp`: exact fallback counter expectations.
- `docs/moe-grouped-multigpu.md` (new), `docs/fork-features.md`: design, evidence and validation handoff.
