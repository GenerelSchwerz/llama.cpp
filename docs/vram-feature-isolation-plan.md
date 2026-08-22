# VRAM feature-isolation and integration plan

This plan applies to every VRAM or CPU-KV branch derived from exact
source-bearing KV baseline
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9` or its documentation-only
descendants. It replaces the cumulative parallel-tree workflow captured in the
read-only snapshot
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
| K: shared KV documentation | PR 9 merge `8e858fcec39049fa028ce6fcb144a0c08b03abd3` on source-bearing base `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`; retained journal base `6a20757854395309b32248dd4109d73e99c3e675` | Current protocol, decisions, evidence index, roadmap, isolation plan, feature delta. | Merged documentation-only; production source is unchanged. |
| N: native standard-quant attention | PR 4, `72ee96bbfcf91c17a7fb5b3b32703aae812af330` | Native quantized FlashAttention source, tests, focused docs, archived composed manifests. | Published draft. |
| W06: PPL capacity | PR 5 evidence head `8d2f8452eb140ba52d8472ecd791cc90212a9307`; merge `50ee5b2d765c91a0d9cd23728ac17a27ac510e3e` | `llama-perplexity` output-capacity fix and evidence. | Merged into the published base. |
| W02: allocation telemetry | PR 6 merged head `3bd7a088199922b1e5e20973cd8cb6d970cde111`; merge/base `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9` | Opt-in physical allocation/VMM telemetry. | Merged into the published base. |
| W03: compact causal mask | PR 7 published head `565233f79faebb5bace9e41f0e2d0ba9c70930cf`; preserved evidence head `d4183adb8b4902a125b9339cd39032a095fca013`; this investigation composes locally onto `f6341a15779eb58fe6ad9e1b890e331c32b676c7` | Causal representation and branch-owned isolated/composed evidence. | New test-only screening phase is uncommitted; draft remains unmerged. |
| W04/W05/W09: live workspace | PR 8 enabled evidence head `0c8df007a504f16aa35fc5982303e3e1b9883331`; disabled-gate runtime head `4cdd2d74e7acc432fcdde4a9d1e5e832fe80e148`; merged through `35e179272` and present in current KV base `f6341a15779eb58fe6ad9e1b890e331c32b676c7` | Live reservation, exact plan publication, idle trim, source and user docs. | Merged; it remains an unrelated opt-in and is omitted from compact-mask causal commands. |

An observed head is not a permanent evidence tag. PR 7 and PR 8 final
identities and comparisons were refreshed on 2026-08-21. Verify each head and
`updatedAt` again immediately before review or integration.

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

- Use only final PR 7 head `d4183adb8b4902a125b9339cd39032a095fca013`.
- Preserve the evidence boundary: the isolated c9 dense/Candidate-1/dense A/B/A
  owns resource/performance claims; composed source checkpoint `ae60c7321` owns
  compatibility checks only.
- Retain the proved capability/fallback boundary for representable standard-KV
  layouts; there is no runtime control or context threshold.
- Keep exact output/PPL, deep serving memory, focused decode, and direct-target
  allocation evidence together; do not import invalid probes or rejected noisy
  timing.
- Treat target-plus-MTP and composition with PR 4/final PR 8 as integration
  gates rather than claims already established by isolated target serving.

### Live-context workspace

- Keep source, public arguments, presets, generated docs, and tests atomic in
  PR 8.
- Preserve enabled evidence at `0c8df007a504f16aa35fc5982303e3e1b9883331`
  and disabled-source evidence at runtime head
  `4cdd2d74e7acc432fcdde4a9d1e5e832fe80e148`; verify the published PR head
  contains only source-equivalent implementation plus evidence/harness updates.
- Measure the full grow/full-depth/shrink/post-idle lifecycle.
- Distinguish configured capacity, live physical extent, allocator high-water,
  mapped residency, synchronized CUDA device-used high-water, and sampled
  process VRAM.
- Preserve Experiment 021's exact output/PPL, 32K startup/post-shrink savings,
  neutral repeated throughput, and explicit +8 MiB 4K/+56 MiB 30K prefill
  device-used costs as separate measurement classes.
- Accept only explicit Bash-wrapped process substitution for its recorded
  exactness provenance; do not import invalid setup or probe launches.
- Require quality/output exactness and quantify reserve-transition cost.
- Require exact-base A/B/A at both omission and explicit off. The 2026-08-21
  gate passed upstream upfront reservation, exact output/PPL, identical W02
  allocation/VMM fields, and neutral repeated 4K/30K throughput.

## Comparison and merge simulation

Fetch PR heads into private local refs without checking them out, then verify
their merge bases:

```bash
git fetch --no-tags generel \
  refs/pull/4/head:refs/codex/consolidation-pr4 \
  refs/pull/7/head:refs/codex/consolidation-pr7-final \
  refs/pull/8/head:refs/codex/consolidation-pr8-current

git merge-base 4a7f9b496 refs/codex/consolidation-pr4
git merge-base 4a7f9b496 refs/codex/consolidation-pr7-final
git merge-base 4a7f9b496 refs/codex/consolidation-pr8-current
```

Before integrating any feature PR into the published line, simulate:

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

The 2026-08-21 readiness audit validated and authorized PR 9's
documentation-only consolidation for coordinator merge into
`beellama-kv-cpu-offload`. That authorization does not extend to PR 7 or PR 8
feature source. Their integration and ordering remain provisional until the
combined post-composition gates pass.

### Consolidation-time comparison matrix

`git merge-tree --write-tree` was rerun in both directions on 2026-08-21 with
source baseline `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`, the final PR 9 refresh tree,
final PR 7 `d4183adb8`, and final PR 8 `0c8df007a`. Baseline plus either feature
PR is clean/fast-forwardable because both are directly based on it. Independently
appended evidence comparisons conflict in `docs/cpu-kv-offload-experiments.md`;
resolve those by preserving Experiments 001-020 and W06 once and retaining the
shared post-KV index plus source-owned Experiment 021 where its source lands.
That append conflict is not by itself evidence of source incompatibility.

| Comparison | Additional textual conflicts beyond the experiment record |
|---|---|
| Base + PR 4 | None; W02 CUDA-backend and benchmark overlap auto-merged and still requires semantic review. |
| Base + final PR 7 | Clean/fast-forwardable; the final branch already includes W06/W02 and its composed validation. |
| Base + final PR 8 | Clean/fast-forwardable; PR 8 is directly based on the exact source baseline. |
| PR 9 docs + PR 4 | None; W02 CUDA-backend and benchmark overlap auto-merged. |
| PR 9 docs + final PR 7 | None beyond the experiment record; CUDA-backend changes are already composed with W02. |
| PR 9 docs + final PR 8 | `docs/cpu-kv-offload-current-testing.md` plus the experiment record; development auto-merges. |
| PR 4 + final PR 7 | `ggml/src/ggml-cuda/CMakeLists.txt`, `fattn-mma-f16.cuh`, and `tests/test-backend-ops.cpp`. |
| PR 4 + final PR 8 | `tools/llama-bench/llama-bench.cpp`. |
| Final PR 7 + final PR 8 | None beyond the experiment record; CUDA VMM and KV/context source overlap auto-merges and still requires interaction testing. |

This matrix is diagnostic only. The final PR 7/PR 8/PR 9 simulations produced
the same conflict paths in both orders. An automatic textual merge does not
establish allocator, graph, or kernel compatibility; no PR 7/PR 8 feature
integration order is selected until the shared append conflicts and
post-composition gates are resolved deliberately.

PR 8 overlaps the reconciled current protocol and experiment append; its
development journal auto-merges in the final PR 9 comparison. Any later
resolution must keep the consolidated protocol and complete Experiments
001-020 plus W06, then place PR 8's source-owned Experiment 021 after those
sections while retaining the shared post-KV index. Do not resolve conflicts by
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
