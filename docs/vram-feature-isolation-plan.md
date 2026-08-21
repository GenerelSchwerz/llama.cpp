# VRAM feature-isolation and integration plan

This plan applies to every VRAM or CPU-KV branch derived from published KV base
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`. It replaces the cumulative
parallel-tree workflow captured in the read-only snapshot
`f52988ee150cd27a94d6897cc049326c1e77c3e2`.

The goal is to answer two separate questions:

1. Does one feature pass correctness, quality, resource, and performance gates
   against the unchanged base with only declared prerequisites?
2. Does the same feature remain acceptable after composition with other
   independently accepted work?

A combined tree can answer the second question only after the first is closed.

## Publication lanes

| Lane | Exact identity observed during consolidation | Ownership | Current status |
|---|---|---|---|
| K: shared KV documentation | PR 9 on `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`; retained journal base `6a20757854395309b32248dd4109d73e99c3e675` | Current protocol, decisions, evidence index, roadmap, isolation plan, feature delta. | Documentation-only draft. |
| N: native standard-quant attention | PR 4, `72ee96bbfcf91c17a7fb5b3b32703aae812af330` | Native quantized FlashAttention source, tests, focused docs, archived composed manifests. | Published draft. |
| W06: PPL capacity | PR 5 evidence head `8d2f8452eb140ba52d8472ecd791cc90212a9307`; merge `50ee5b2d765c91a0d9cd23728ac17a27ac510e3e` | `llama-perplexity` output-capacity fix and evidence. | Merged into the published base. |
| W02: allocation telemetry | PR 6 merged head `3bd7a088199922b1e5e20973cd8cb6d970cde111`; merge/base `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9` | Opt-in physical allocation/VMM telemetry. | Merged into the published base. |
| W03: compact causal mask | PR 7, observed `b3ce3a5c23f5ce3213d0ddb735a7e3bcd5b490e5` | Causal representation and branch-owned evidence. | Moving draft; final identity/evidence pending. |
| W04/W05/W09: live workspace | PR 8, `143cd6aee137e3a9974db64460e33e1de1f7d4bd` | Live reservation, exact plan publication, idle trim, source and user docs. | Published draft; re-fetch before integration. |

An observed head is not a permanent evidence tag. Record the final head and
`updatedAt` immediately before review or merge simulation. PR 7 must not be
treated as stable until its committed fix and deeper-context evidence are
published.

## What belongs in the shared KV line

Every future KV-derived branch should inherit:

- the sole current protocol and clean-process/GPU-lock rules;
- durable development decisions and rejected-path summaries;
- the evidence/identity index;
- the VRAM roadmap;
- this isolation plan; and
- the source-verified KV feature delta.

Branch-specific source, test fixtures, generated arguments, presets, user
documentation, and detailed reproduction records stay with the feature until
its source lands. In particular, PR 8 owns every user-facing spelling of
`--live-context-workspace`, and PR 4 owns native-Q8 composed manifests and its
generated argument surface.

## Source-isolation gate

Before performance testing, compare the candidate with the exact base and
classify every changed path:

- required source;
- focused tests;
- feature-owned documentation;
- generated/index consequence; or
- unrelated contamination, which must be removed.

Runtime default-off is not sufficient isolation. A disabled feature can still
alter template instantiation, kernel ABI, graph keys, allocator plans, public
structure layout, or default execution order. Where possible, use the same
binary for runtime-policy A/B/A. When the source itself is the candidate, use
separate reproducible builds and prove the default/off build path is unchanged
or quantify it explicitly.

No lane may rely on a model-name exception where a capability and layout proof
is possible. Unsupported layouts must preserve the established path or fail
closed.

## Measurement contract

Each independently accepted feature must satisfy:

1. Exact base and candidate commits plus binary/library hashes.
2. Clean GPU preflight and a shared lock held for the full process lifecycle,
   using exactly `flock /tmp/beellama-single-gpu.lock -c 'COMMAND'` with safe
   inner quoting, never flock's direct-command form.
3. Fresh A/B/A processes with identical model, prompt, sampler, affinity,
   batch geometry, cache formats, build options, and runtime settings except
   the named variable.
4. Native llama worker affinity; no whole-process affinity wrapper.
5. Observable progress and retained result/log/progress artifacts.
6. PPL with matching `-b` and `-ub`; the published base includes W06 full-batch
   output capacity for phase-aware contexts.
7. Deterministic output and same-MTP-geometry exactness when scheduling,
   workspace, placement, or recurrent state can affect output.
8. Repeated prefill, 4K decode, and long-context decode.
9. Startup, prompt peak, decode-resident, long-idle, shortened-next-turn, and
   final-idle process VRAM when allocation lifetime changes.
10. Ordinary system RAM, allocator-reported pinned memory, and VMM live/mapped
    high-water where applicable.
11. Profiler evidence only after a stable unprofiled result, with Nsight
    directly targeting the llama binary and a bounded capture.
12. A disposition: retained, revised, neutral, or reverted.

A result from a cumulative tree is interaction evidence, not isolated feature
evidence. A contaminated performance result may motivate a clean experiment,
but it cannot be promoted after the fact by subtracting unrelated changes.

## Feature-specific gates

### Native standard-quant attention

- Preserve default fallback when the build/runtime opt-in is absent.
- Exercise every claimed type pair, head dimension, architecture, and route.
- Compare native and fallback outputs and PPL.
- Verify that later workspace/mask candidates do not silently materialize the
  quantized source or leave the qualified route.

### PPL capacity

- Prove the tool sets full output capacity before context creation.
- Keep the ordinary/default PPL path unchanged.
- Use exact repeated PPL and focused parser/plumbing coverage.

### Allocation telemetry

- Disabled telemetry must remain dormant.
- Enabled counters must reconcile physical allocation classes and return to
  the expected live/mapped state after destruction.
- Do not interpret telemetry as a trimming or memory-policy feature.

### Compact causal masking

- Re-fetch final PR 7 source/evidence.
- Prove representable layouts and fallback layouts independently.
- Test target-only and target-plus-MTP behavior.
- Measure long-context prefill/decode and memory in fresh A/B/A processes.
- Inspect default-path compile and graph effects, not only the runtime flag.
- Do not reuse an intermediate causal attribution from the moving draft.

### Live-context workspace

- Keep source, public arguments, presets, generated docs, and tests atomic in
  PR 8.
- Measure the full grow/full-depth/shrink/post-idle lifecycle.
- Distinguish configured capacity, live physical extent, allocator high-water,
  and mapped residency.
- Require quality/output exactness and quantify reserve-transition cost.

## Comparison and merge simulation

Fetch PR heads into private local refs without checking them out, then verify
their merge bases:

```bash
git fetch --no-tags generel \
  refs/pull/4/head:refs/codex/consolidation-pr4 \
  refs/pull/7/head:refs/codex/consolidation-pr7 \
  refs/pull/8/head:refs/codex/consolidation-pr8

git merge-base 4a7f9b496 refs/codex/consolidation-pr4
git merge-base 4a7f9b496 refs/codex/consolidation-pr7
git merge-base 4a7f9b496 refs/codex/consolidation-pr8
```

Before changing the published base, simulate:

- each PR against the base;
- the shared-doc branch against the base and each PR;
- every remaining PR pair in both likely orders where source overlap exists;
- the proposed full integration order.

Use `git merge-tree` or disposable temporary worktrees. Do not merge,
fast-forward, reset, stash, or normalize a user-owned dirty worktree for the
simulation. Record conflicts by path and distinguish textual conflicts from
semantic interactions. A conflict-free merge tree does not prove runtime
composition.

For each simulated combined tree, inspect:

- changed production paths and public structures;
- option/default ownership and generated docs;
- graph keys, buffer sizing, scheduler lifetime, CUDA template/dispatch
  selection, and fallback behavior;
- tests/manifests that become invalid or redundant; and
- documentation links and source identities.

The integration proposal remains provisional until PR 7 and PR 8 final heads
are published and all comparisons are repeated. No current instruction
authorizes merging or fast-forwarding `beellama-kv-cpu-offload`.

### Consolidation-time comparison matrix

`git merge-tree --write-tree` was rerun on 2026-08-21 with published base
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`, the reconciled PR 9 tree, and the
exact PR 4/7/8 heads above. Every comparison conflicts at the independent
append point in `docs/cpu-kv-offload-experiments.md`. Resolve that documentation
conflict by preserving Experiments 001-020 and W06 once and adding the accepted
branch evidence to the shared post-KV index; it is not by itself evidence of
source incompatibility.

| Comparison | Additional textual conflicts beyond the experiment record |
|---|---|
| Base + PR 4 | None; W02 CUDA-backend and benchmark overlap auto-merged and still requires semantic review. |
| Base + PR 7 | None; W02 CUDA-backend overlap auto-merged and still requires semantic review. |
| Base + PR 8 | `docs/cpu-kv-offload-development.md` and `ggml/src/ggml-cuda/ggml-cuda.cu`; current-testing, benchmark, and test overlap auto-merged. |
| PR 9 docs + PR 4 | None; W02 CUDA-backend and benchmark overlap auto-merged. |
| PR 9 docs + PR 7 | None; W02 CUDA-backend overlap auto-merged. |
| PR 9 docs + PR 8 | `docs/cpu-kv-offload-current-testing.md`, `docs/cpu-kv-offload-development.md`, and `ggml/src/ggml-cuda/ggml-cuda.cu`; benchmark and test overlap auto-merged. |
| PR 4 + PR 7 | `ggml/src/ggml-cuda/CMakeLists.txt`, `fattn-mma-f16.cuh`, and `tests/test-backend-ops.cpp`. |
| PR 4 + PR 8 | `tools/llama-bench/llama-bench.cpp`. |
| PR 7 + PR 8 | None; multiple KV/context/CUDA files auto-merged and require interaction testing. |

This matrix is diagnostic only. PR 7 is moving, PR 8 may receive review
updates, and an automatic textual merge does not establish allocator, graph,
or kernel compatibility. Re-fetch and repeat the matrix before choosing a
merge order.

PR 8 overlaps the reconciled current protocol and development journal in
addition to its source conflict with merged W02. Any later resolution must keep
the consolidated protocol and complete Experiments 001-020 plus W06, then
retain PR-owned detail until its source lands. Do not resolve conflicts by
dropping either historical KV evidence or a PR's source-coupled documentation.

## Post-composition acceptance

After individually accepted changes compose, rerun at minimum:

- default/off equivalence for every feature;
- native/fallback quantized attention routes;
- target-only PPL and deterministic output;
- same-MTP-geometry exactness;
- 4K and long-context A/B/A performance;
- maximum-depth then shortened-next-turn lifecycle;
- pinned/ordinary host memory and VMM reconciliation; and
- focused unit/static tests from every participating lane.

If composition regresses a gate that each branch passed independently, stop.
Create a named interaction experiment and assign the fix to the smallest
responsible lane. Do not hide the interaction in a merge commit or relabel a
combined result as one feature's evidence.
