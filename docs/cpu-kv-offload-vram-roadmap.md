# CPU KV-offload VRAM roadmap

This roadmap ranks shared VRAM work for every future branch derived from the
published KV base `4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`. Status refers to
source actually present in that base; remaining independent PRs are candidates,
not current features.

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

## Available in the published KV base

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

### 3. Review compact causal masking only from final PR 7 evidence

[PR 7](https://github.com/GenerelSchwerz/llama.cpp/pull/7) is actively receiving
a committed fix and deeper-context evidence. Its currently observed head is
`b3ce3a5c23f5ce3213d0ddb735a7e3bcd5b490e5`; keep it pending and do not copy
intermediate speed, allocation, or causal conclusions into the KV base. Review
the final published head for:

- a source-only comparison against unchanged KV base plus declared prerequisites;
- explicit fallback for unrepresentable layouts;
- target-only and target-plus-MTP exactness;
- 4K and long-context memory, prefill, and decode A/B/A;
- direct-target profiler evidence only when it explains a stable unprofiled result; and
- no default-path compile contamination when the feature is absent or inactive.

### 4. Keep live-context workspace source and docs atomic in PR 8

[PR 8](https://github.com/GenerelSchwerz/llama.cpp/pull/8) owns the default-off
live physical-KV reservation policy and all-idle trim. Its source, presets,
generated arguments, and user-facing documentation must merge together. Until
then, `--live-context-workspace` is absent from current commands and docs.

The acceptance question is lifecycle residency, not merely allocation at
context creation. A valid comparison must include startup, full-depth growth,
decode, a shorter following request, post-idle mapped residency, system/pinned
memory, exact output, PPL, and repeated prompt/decode performance.

## Later research, in priority order

### A. Close cross-feature integration without losing isolation

After the remaining PR 4, PR 7, and PR 8 heads stabilize, simulate all pairwise
comparisons and likely merge orders on top of the merged W06/W02 base before
touching the published branch again. Re-run default/off controls after
composition; a runtime-disabled feature can still change template
instantiation, graph signatures, or allocator geometry. The isolation plan is
the gate.

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

There is no authorized merge order yet. The likely dependency shape is support
fixes/telemetry already merged, then native quant attention and independently
accepted workspace and mask features. Final ordering must be chosen from
simulated final PR heads and source overlap. Do not merge or fast-forward the
published KV base while PR 7 and PR 8 are changing.
