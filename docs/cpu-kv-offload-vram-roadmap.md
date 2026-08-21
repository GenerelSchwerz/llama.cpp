# CPU KV-offload VRAM roadmap

This roadmap ranks shared VRAM work for every future branch derived from exact
source-bearing KV baseline
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9` or its documentation-only
descendants. Status refers to production source actually present at that
baseline; merging shared PR 9 documentation does not make an independent
feature PR part of the source line.

Use this together with the
[`feature-isolation plan`](vram-feature-isolation-plan.md) and the
[`evidence index`](cpu-kv-offload-experiments.md). Numerical claims belong in
the evidence owner, not duplicated here.

## Capacity model

CPU attention-KV placement removes persistent target KV from device memory but
does not make process VRAM independent of context. Device demand still
includes:

- model and projector placement;
- recurrent R/S state and MTP rollback planes;
- partially resident target or draft KV layers;
- prompt/generation scheduler workspace;
- explicit masks or compact descriptors;
- canonical store staging and host-to-device attention staging;
- CUDA graphs, VMM pool retention, loaded modules, and driver context; and
- temporary server/checkpoint state that may instead appear in system RAM.

Pinned host KV consumes real unswappable system memory and CUDA mapping
resources. `nvidia-smi` process VRAM, allocator counters, ordinary RSS, and
page-locked allocation are separate measurements.

## Available in the source-bearing KV baseline

These are current controls and should be exhausted before adding source:

| Lever | Benefit | Cost / constraint |
|---|---|---|
| Keep the multimodal projector on CPU | Avoids a large optional device allocation. | CPU projector compute and ordinary RAM; only appropriate when the workload permits it. |
| Configure only required context/parallel capacity | Avoids unused persistent KV and context-scaled workspace. | Reduces maximum request or concurrency. |
| Use qualified homogeneous standard KV widths | Reduces host KV and repeated transfer volume. | Requires matched quality and backend qualification. |
| `--kv-gpu-layers N` | Trades device memory for less target KV transfer. | Context-linear VRAM; selected by owned layer, not recency. |
| `--spec-draft-kv-gpu-layers N` | Removes transfer for independently owned draft KV. | Smaller context-linear device cost; benefit is workload-specific. |
| `--spec-mtp-rs-planes N` | Reduces fixed recurrent rollback storage without lowering draft maximum. | Rejection-dependent replay and capability requirements. |
| `--phase-aware-workspace` | Removes inactive target/MTP prompt high-water during generation. | Synchronization, reserve transitions, and graph recapture; opt-in. |

The base also keeps supported hybrid recurrent state independently on the GPU
and canonicalizes host-resident standard-Q8 stores. Those are correctness and
performance foundations, not optional VRAM-only experiments.

Merged W06 support lets `llama-perplexity` declare full-batch output capacity
before context creation, including phase-aware runs. Merged W02 support extends
opt-in `llama-bench --kv-memory` with physical allocation classes and CUDA VMM
live/mapped/high-water telemetry. W02 reports allocation behavior; it does not
trim a pool or change allocation policy.

## Near-term integration lanes

### 1. Use the merged protocol support as the measurement foundation

[PR 5](https://github.com/GenerelSchwerz/llama.cpp/pull/5) merged the
`llama-perplexity` output-capacity correctness fix as
`50ee5b2d765c91a0d9cd23728ac17a27ac510e3e`. It is a measurement-tool
prerequisite, not a VRAM feature.

[PR 6](https://github.com/GenerelSchwerz/llama.cpp/pull/6) merged the W02
telemetry as current base `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`.
Use it for allocator-lifetime acceptance that depends on live/mapped
high-water, without treating it as a trimming policy.

### 2. Integrate native standard-quant attention as its own capability

[PR 4](https://github.com/GenerelSchwerz/llama.cpp/pull/4) owns native
same-type standard-quant FlashAttention. Its qualified compiled matrix,
fallback behavior, exactness gates, and source identity must remain intact.
Do not use the historical mega-tree composed manifests as a current KV
protocol. Recreate only the combinations needed after the source is merged.

Native Q8 is an enabling route for later CPU-KV VRAM work: it can remove the
full-source F16 materialization path where the exact PR capability applies.
It is not evidence that any separate workspace or mask feature is correct.

### 3. Integrate compact causal masking only from final PR 7

[PR 7](https://github.com/GenerelSchwerz/llama.cpp/pull/7) has preserved
evidence head `d4183adb8b4902a125b9339cd39032a095fca013`. Its evidence publication
contained six feature commits over exact source baseline `4a7f9b496`; the
current draft inherits merged PR 9 documentation base
`8e858fcec39049fa028ce6fcb144a0c08b03abd3` without changing the causal source
delta or evidence identity. It automatically uses an I64 causal-prefix
descriptor only for proved
single-stream contiguous standard-KV layouts and otherwise retains the dense
mask. Per-consumer views remove the measured allocator fragmentation without
changing allocator policy, public controls, or attention-kernel ownership.

Keep its two evidence lanes distinct:

- isolated c9 dense/Candidate-1/dense A/B/A owns the resource and performance
  claims: exact output and PPL, serving savings of 20/30/34 MiB at 4K/30K/34K
  and 64/96/128 MiB at 64K/98K/128K, plus a local 98 MiB 49K allocator-bin
  result that must not be extrapolated; and
- `ae60c7321d950937a36af096112525db777ae13f` on the merged base owns only the
  composed build, correctness, PPL/output, and W02 telemetry coexistence gates.
  It is not a replacement A/B/A performance or memory comparison.

Do not copy invalid setup/probe results or rejected noisy timing. Target-plus-
MTP and composition with PR 4 and final PR 8 remain integration gates.

### 4. Keep live-context workspace source and docs atomic in PR 8

[PR 8](https://github.com/GenerelSchwerz/llama.cpp/pull/8) owns the default-off
live physical-KV reservation policy and all-idle trim. Its source, presets,
generated arguments, and user-facing documentation must merge together. It is
final at `0c8df007a504f16aa35fc5982303e3e1b9883331`, directly based on
`4a7f9b496` with six feature commits. Reconcile inherited PR 9 documentation
and refresh merge metadata before feature integration. Until its source lands,
`--live-context-workspace` is absent from current runnable commands and from
current preset/generated argument documentation.

Final Experiment 021 passed exact 1K and full 32K lifecycle output, identical
PPL, W02 lifecycle reconciliation, and focused coverage on the merged W02/W06
base. It measured meaningful startup/post-shrink process-VRAM savings and
neutral repeated 4K/30K throughput, while separately exposing synchronized
device-used prefill high-water costs. Treat those W02 device-used values as
allocator evidence, not `nvidia-smi` process VRAM. Exact values and accepted
Bash-wrapped provenance commands are indexed in the experiment record.

## Later research, in priority order

### A. Close cross-feature integration without losing isolation

PR 7 and PR 8 are now final. Their pairwise comparisons and comparisons with
the shared documentation tree must be simulated on the merged W06/W02 base
before integrating either feature source. PR 9's validated documentation-only
consolidation was cleared for independent coordinator merge by the 2026-08-21
readiness audit. Re-run default/off controls after feature composition; a
runtime-disabled feature can still change template instantiation, graph
signatures, or allocator geometry. The isolation plan is the gate.

### B. Bound attention staging without changing the native reduction

Independently normalized fixed windows are rejected: positive partitions
changed arithmetic and deterministic output. Any renewed design must carry the
native kernel's exact accumulator/reduction ownership across partitions or
fail closed. The earlier vector-partition branch is unfinished research, not
accepted source.

Required gates include:

- chunk-zero source/build equivalence;
- positive-partition PPL with matching batch geometry;
- deterministic long-context output;
- process VRAM and VMM live/mapped high-water;
- pinned and ordinary host memory;
- repeated prefill/decode; and
- target-plus-MTP interaction where the full-source path may still dominate.

Do not double-buffer large K/V windows unless the extra high-water is explicitly
worth the overlap; capacity is the objective.

### C. Reassess direct quantized prompt MMA after structural savings

Only after native quant attention, compact mask, and live-workspace candidates
have been integrated and remeasured should direct prompt MMA or broader tuning
be reconsidered. The work has a large template/kernel test matrix and should
be justified by the remaining measured prompt peak, not an old cumulative-tree
profile.

### D. Explore lower homogeneous KV widths under quality gates

Lower standard widths may reduce pinned RAM and context-linear transfer. Each
candidate requires matching `llama-perplexity` `-b`/`-ub`, exact route
validation, and performance/resource measurements. Asymmetric K/V, KVarN
CPU-offload, multi-GPU, Vulkan, and HIP are separate qualification lanes.

### E. Revisit selective pinned allocation only with a resource policy

Pinning the full configured host KV is fast but consumes unswappable RAM and
driver mappings. A later policy could distinguish active transfer windows or
layers, but it must preserve fallback behavior, avoid repeated registration
churn, and report pinned bytes independently from device VRAM.

## Closed or deferred directions

| Direction | Status |
|---|---|
| Forced CPU attention | Rejected: severe decode loss without the required memory outcome. |
| Broad zero-copy CUDA reads from mapped host KV | Rejected: prompt and decode kernel slowdown. |
| Lossless Q8 transfer compression | Rejected: insufficient measured redundancy for complexity. |
| Smaller MTP draft ubatch | Rejected for MTP exactness; retained only as a generic non-MTP control. |
| Host recurrent checkpoints | Rejected; selected full-geometry GPU replay replaced them. |
| F16 persistent recurrent S state | Rejected by deterministic PPL increase. |
| Independently normalized positive staging chunks | Rejected for exact serving; chunk zero only. |
| Staging merge barrier removal | Neutral and reverted. |
| Cold CUDA graph eviction as a Bee feature | Deferred: only a small shape-specific saving with mixed performance; upstream control already exists. |
| Cross-context VMM pool sharing | Deferred pending explicit concurrency and lifetime ownership. |
| Token-recency GPU KV | Research only; requires exact segmented attention reduction. |

## Integration order rule

The 2026-08-21 readiness audit cleared PR 9 as a documentation-only inheritance
update for immediate coordinator merge. That integration does not select or
validate a feature-source order. The likely feature dependency shape is support
fixes/telemetry already merged, then native quant attention and independently
accepted mask and workspace features. PR 7 and PR 8 integration/order remains
held until their combined post-composition correctness, resource, and
performance gates pass.
