# CPU KV-offload VRAM reduction roadmap

This document ranks VRAM-reduction options for the CPU-KV-offload branch by
implementation complexity. It distinguishes real peak reductions from changes
that only improve startup or steady-state decode residency.

Read this with the development journal and experiment ledger. MTP allocation
experiments may use separate branches, but their memory categories remain here
because target prefill and speculative decode must ultimately coexist.

## Reference memory model

Measure four phases separately:

1. **Startup:** contexts allocated before a request.
2. **Prefill peak:** target prompt workspace and speculative synchronization.
3. **Decode resident:** memory retained during token generation.
4. **Next-turn peak:** an established context ingests another prompt.

At 240,128 context cells, the measured non-MTP target uses 12,879.47 MiB for
CUDA model weights, 149.62 MiB for active CUDA recurrent state, and 1,751.09
MiB for CUDA compute. The Q8 KV itself is 7,973.00 MiB of CUDA-host memory.

The context-scaled CUDA-compute slope is exactly 7,296 bytes per cell:

- 4,096 bytes: prompt FlashAttention F16 K/V materialization.
- 1,088 bytes: Q8 K device staging.
- 1,088 bytes: Q8 V device staging.
- 1,024 bytes: GPU F16 attention mask at target ubatch 512.

The source mask adds another 1,024 bytes per cell in pinned host memory.

## Complexity 0: existing configuration choices

### Keep the multimodal projector off GPU

Use `--no-mmproj-offload`, or omit the projector for text-only service. The
Qwen projector is approximately 885 MiB. Text decode is unaffected, while image
processing becomes CPU-bound.

### Configure only the required context capacity

Reducing `--ctx-size` immediately reduces masks, staging, prompt workspace, and
host KV. This is not a solution when a genuinely full 240K context is required.

### Use lower homogeneous KV precision

Q6/Q6 or Q4/Q4 reduces CPU KV, PCIe traffic, and device staging with existing
code. It changes cache quality and requires matched KLD/perplexity validation.
Asymmetric quantization is outside the current experiment scope.

### Selective model-layer CPU placement

Existing placement controls can save hundreds of MiB, but repeating layers are
used every token and are expected to damage decode. Treat this as a capacity
fallback, not a preferred optimization.

## Complexity 1: small allocation controls

### Independent draft ubatch (implemented)

`--spec-draft-ubatch-size 128` leaves target ubatch at 512. At MTP depth 5 and
32K it saved 88 MiB of live process VRAM with decode unchanged within
single-run noise and a 0.9% prompt-throughput cost. Draft ubatch 32 saved 110
MiB but reduced prompt throughput by 4.3%.

### Cap speculative recurrent rollback planes

Each MTP rollback plane costs approximately 149.625 MiB for this model. A cap
can save multiples of that amount if the scheduler defines safe behavior when
draft depth exceeds available planes. Checkpoint, rejection, prompt-cache, and
sequence-removal correctness are mandatory. This work belongs on its dedicated
MTP experiment branch.

### Remove padding and duplicate reservations

Audit alignment, output buffers, scheduler copy slots, and rounded tensor
extents. Expected savings are modest, but changes can be low risk when tensor
lifetimes and graph topology remain unchanged.

## Complexity 2: phase-aware allocation policy

### Shrink to a decode-only scheduler after prefill (implemented)

The retained `--phase-aware-workspace` implementation rebuilds the scheduler
reservation for the active geometry. It starts at a speculation-aware
generation bound, grows to the full physical ubatch for prompt work, and
shrinks when generation resumes. A second prompt on the same live server was
verified to regrow and shrink both target and MTP contexts safely.

At 140K with MTP-6, three recurrent planes, target ubatch 512, and draft ubatch
128, it reduced initialized/steady VRAM by 1,108 MiB and measured peak VRAM by
902-926 MiB. The 138K prompt changed prefill by -0.12%; the plain 5K run changed
decode by -1.83%. Target plus draft transition work was 41.8 ms in the 5K run
and 46.0 ms in the 138K run.

- Reduces startup and steady generation residency.
- Reduces coexistence peak by sharing sequential target/draft backing, but does
  not shrink the active target prompt geometry itself.
- Requires allocation, graph-address invalidation, and recapture at each phase
  boundary.
- Retains model weights, KV, recurrent state, rollback planes, samplers, and
  checkpoints unchanged.

### Grow workspace with live context

Reserve for current live KV and grow in bounded steps instead of allocating for
maximum `--ctx-size` at startup.

- Reduces startup and short-context residency.
- Does not reduce a genuinely full-context peak.
- Requires safe buffer relocation and graph invalidation at growth boundaries.

### Phase MTP allocation against target prefill (implemented)

Target and integrated MTP schedulers now retain private allocation plans while
using one physical backing group keyed by exact backend buffer type. The active
allocation is the maximum requirement, not the sum, because execution is
sequential. Explicit synchronization protects each target/MTP ownership
handoff, and a coalesced shrink epoch waits for every active member to publish
its current plan.

- The 140K generation allocation was 840.82 MiB CUDA plus 2.41 MiB CUDA-host
  backing, instead of retaining the full target 1,054.62/157.03 MiB and draft
  892.05/39.46 MiB reservations concurrently.
- Later-turn prompt ingestion passed with identical fixed-seed output.
- Active recurrent state and checkpoints remain persistent; this policy only
  phases transient graph workspace.
- The allocator protocol is backend-type based and contains no Qwen or CUDA
  architecture check.

### Evict cold CUDA graphs

Retain common decode graphs and evict uncommon prompt shapes. The 90K/240K
startup trace showed that primary initial growth is the scheduler buffer rather
than additional graph count, so this targets post-execution growth only.

## Complexity 3: contained graph or kernel-interface changes

### Compact implicit causal mask

At 240K and target ubatch 512, the explicit F16 mask occupies approximately
234.5 MiB on GPU and another 234.5 MiB in pinned host memory. Pass compact query
positions, sequence metadata, and KV bounds so CUDA derives validity instead of
reading `[n_kv, 512]` values.

- Potential saving: 234.5 MiB GPU and 234.5 MiB pinned host at 240K.
- Also removes mask PCIe transfer and device mask reads.
- Must support or fall back for multiple sequences, padding, unified KV,
  sequence removal, non-contiguous positions, and sliding-window layouts.
- Existing no-mask attention is insufficient because it loses causal/padding
  semantics.

Use a fail-closed descriptor only when the memory context proves its layout is
representable. Retain the explicit mask as the general fallback.

### Reduced-precision active recurrent state

The non-speculative active F32 recurrent state is 149.62 MiB. BF16 or F16 could
roughly halve it. This changes recurrent numerics and requires long-generation,
restore, output-equivalence, quality, and conversion-overhead testing.

### Compact or delta recurrent checkpoints

If checkpoints duplicate unchanged recurrent regions, share immutable backing
or store only modified data. First measure actual changed bytes; ownership and
rollback semantics make speculative designs premature without that evidence.

## Complexity 4: direct Q8 prompt MMA

Prompt ubatch 512 selects the F16 MMA kernel. It materializes complete Q8 K and
V as contiguous F16 so hundreds of query rows can efficiently reuse them. The
direct quantized vector kernel is intended for one or two decode queries and
would repeatedly read/dequantize KV if forced onto prefill.

A real replacement must load Q8 blocks and scales, dequantize once into shared
F16 tiles, and reuse each tile across prompt queries and GQA heads.

- Potential saving at 240K: approximately 950 MiB of F16 K/V.
- Remaining context-scaled allocation: approximately 498 MiB Q8 K/V staging.
- Risks: tensor-core utilization, shared-memory pressure, supported layouts,
  numerical differences, and architecture-specific performance.
- Validate prefill, decode, peak VRAM, output tolerance, KLD, compiled cache
  pairs, and fail-closed fallback behavior.

## Complexity 5: fixed-window online-softmax streaming

Stream bounded Q8 K/V windows from CPU through reusable device buffers and
merge attention with numerically stable online-softmax state. Double buffering
may overlap PCIe transfer with attention computation.

- Caps Q8 staging, F16 tile storage, and mask storage independently of context.
- Reduces full-context prefill and decode peaks, not merely post-prefill
  residency.
- Does not reduce the total PCIe bytes required to consume CPU KV.
- Requires new scheduling, additional launches, online-softmax validation,
  GQA/multi-sequence support, graph integration, and short-context tuning.

This has not been implemented. Earlier rejected experiments were full CPU
attention and CUDA zero-copy mapped-host attention; neither used bounded
explicit staging with GPU online-softmax accumulation.

## Recommended order

1. Retained: capped MTP recurrent planes and phase-aware target/draft workspace
   have separate controls and matched correctness/resource measurements.
2. Implement a fail-closed compact mask for single-slot contiguous causality
   and verify the predicted 1,024-byte-per-cell GPU and host reductions.
3. Reassess peak pressure. Pursue direct-Q8 prompt MMA only if its approximately
   950 MiB saving justifies the kernel-development and tuning cost.
4. Treat fixed-window streaming as the long-term solution when VRAM must remain
   approximately flat at genuinely full 90K-to-240K contexts.

Every experiment must record model files, command, prompt, sampling settings,
hardware, commit, target/draft batch geometry, prefill, decode, process and peak
VRAM, pinned host memory, and correctness or quality results.
