# Profile-aware cache capacities

Owner approved implementation, then commit and push after validation. Base: published moe-cache 28d73c87c. This isolated branch excludes the unfinished upstream merge, private specialized port and other workers' prefill changes.

## Behavior

- Without a profile, preserve existing allocation.
- `--moe-cache-allocation uniform` preserves existing allocation even with a profile.
- With a profile and the default `auto` policy, allocate fixed group capacities using actual group costs and profile priorities within the existing device budget.
- Keep a positive execution floor, original capacity checks, one grouped residency owner and existing kernels. Adaptation replaces identities within fixed group capacities.
- Preserve source identity, context/domain separation and device ownership. No runtime resizing or extra replay synchronization.

## Work

- [x] Reuse canonical statistics and validated STRP ordering for a bounded capacity planner.
- [x] Propagate group capacities through transactional candidate publication and resource identity.
- [x] Add public target/draft policy options and retain uniform/no-profile behavior.
- [x] Extend existing planner/registry/argument tests, including overflow and rejection.
- [x] Build under the ordered shared locks, then run focused ownership/graph/state checks.
- [x] Serve Flash Next with real 8194-token prompt and 2048 output, MTP3/.5, plus a distinct supported model check.
- [x] Review budget accounting, model independence and private-code exclusion.
Publication is authorized after validation; its exact commit and remote identity are recorded in the external handoff.

Build lock: /tmp/beellama-cuda-build.lock. GPU lock: /tmp/beellama-single-gpu.lock. Never acquire them in reverse order or queue behind an owned GPU job. Live runs retain finite timeout, whole-tree teardown, GPU ceiling and emergency host headroom stop. Do not publish incomplete validation or unrelated merge changes.

## Qualification on 2026-10-08

Same Release/CUDA120a build, one RTX5070Ti, real 8194-token prompt, 2048 output tokens, MTP3/.5, backend sampling, graph-capacity reuse, decode and boundary overlap. EOS was honored. Flash used the same ranked profile, 64-slot storage budget and 0.17 GPU miss fraction; Ornith used a canonical statistics profile and 144-slot storage budget.

| Case | Native decode tok/s | TTFT seconds | Target slots | Group capacity range | Target payload bytes |
|---|---:|---:|---:|---:|---:|
| Flash uniform | 79.46 | 7.188 | 3072 | 64..64 | 5364121600 |
| Flash profile | 78.84 | 7.187 | 3057 | 35..83 | 5363686400 |
| Ornith profile | 95.31 | 3.063 | 5651 | 105..229 | 8386609152 |

Flash's observed difference is -0.79%, with different generated IDs after token 125 and different MTP work. This single pair establishes neither a gain nor a statistical regression. Different CPU/GPU route ownership can change arithmetic and outputs. Explicit uniform preserves the previous allocation mechanism. Ornith is a serving/support check, not a matched performance comparison; its shared-model MTP group retained 144 slots. All cases completed 2048 output tokens, with zero grouped fallback, rollback, prepare/finish errors or source-core drain failures. Global peak VRAM was 12945/12891/12959 MiB, below the owner ceiling.

Existing planner, argument, registry, routed-bank and static-profile fixtures passed. The graph fixture covers local capacity 3 against uniform 12, exact copied payloads, capture/replay, replacement to uniform 12, stale witnesses and finite drain/retirement. It does not establish physical multi-GPU or Windows partial-pin qualification. Host staging keeps its existing admission limits; no runtime capacity lending or automatic maximum-VRAM search was added.

A broader materialization test stopped in the unchanged file-mmap fixture callback at `tests/test-moe-cache-fixtures.cpp:259`. That callback uses the opaque cached-buffer context as an mmap address. The focused capacity/graph gates passed independently; this broader test is not reported as passing. Its fixture repair is outside this focused slice and coordinated with the prefill worker.
