# CPU KV-offload feature delta from BeeLlama v0.4.3

## Scope and identities

This is a source-backed capability inventory, not a performance report. It
compares BeeLlama v0.4.3 at
`ba27edad2a84ff045a556df06661e821285c2fab` with published CPU-KV source at
`4a7f9b496b58a5c782b4d4c97597cd076fe0b2e9`. The retained journal and
Experiments 001-019 derive from local tip
`6a20757854395309b32248dd4109d73e99c3e675`; the published base also includes
the merged W06 perplexity-capacity fix and W02 allocation telemetry. The
canonical integration line additionally contains the PR 8 live-context
workspace source and documentation since merge
`35e179272d510a0d5aefbf8dccb9dcf30fb31556`.

The evidence and limitations for each feature are indexed in
[`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md). This document
deliberately avoids copying hardware-specific throughput numbers into a source
inventory.

Existing BeeLlama features such as KVarN, standard low-bit KV formats, DFlash,
the adaptive DFlash controller, reasoning-loop guard, and the realtime control
endpoint are outside this delta.

## User-visible controls added by the KV line

| Control | Source-backed behavior | Default |
|---|---|---|
| `--kv-cpu-pinned` / `--no-kv-cpu-pinned` | Routes supported host-resident attention KV through the accelerator's host buffer type, with backend fallback where supported. | Off |
| `--recurrent-state-offload` / `--no-recurrent-state-offload` | Keeps supported hybrid recurrent R/S state on the accelerator independently of host attention KV. | Off |
| `--kv-gpu-layers N` | Places the first `N` independently owned target attention-KV layers on the accelerator under host-KV policy. | `0` |
| `--spec-draft-kv-gpu-layers N` | Overrides target placement for independently owned draft KV. Omission inherits target policy; zero explicitly selects host placement. | Inherit |
| `--spec-draft-ubatch-size N` | Gives a separate model-backed draft context its own physical ubatch. For MTP it must be omitted/inherited or equal to target ubatch. | Inherit |
| `--spec-mtp-rs-planes N` | Caps total target recurrent planes, including the current plane; zero resolves to full `draft_max + 1`. | Full |
| `--phase-aware-workspace` / `--no-phase-aware-workspace` | Enables explicit prompt/generation reservation transitions and shared backing for supported sequential target/MTP schedulers. | Off |
| `--live-context-workspace` / `--no-live-context-workspace` | Bounds supported standard-attention graph reservations by padded live physical KV extent; merged after the core source checkpoint through PR 8. | Off |

These policies flow through the normal common-argument configuration and the
applicable CLI, environment, INI, context, server, and benchmark surfaces.
Current environment names use `LLAMA_ARG_*`. The old experimental
`GGML_KV_CPU_PINNED` and `GGML_RECURRENT_STATE_OFFLOAD` variables are removed.

## Architectural changes

### Accelerator-visible pinned host KV

Base BeeLlama placed host KV in ordinary CPU buffers. The KV line can instead
select the accelerator-associated host buffer for each supported layer. On
CUDA this is page-locked accelerator-visible system memory. It remains host
storage: pinning improves transfer behavior but consumes unswappable RAM and
does not eliminate context-linear history transfer.

Pinned routing is propagated through the supported standard, KVarN, SWA,
hybrid, and hybrid-iSWA constructors. That propagation does not by itself
qualify every format/backend combination; unsupported placement must retain an
explicit fallback or fail closed.

### Independent recurrent-state placement

Base hybrid memory coupled recurrent R/S placement to attention-KV placement.
The KV line passes separate attention and recurrent decisions through hybrid
memory construction. Host attention history and fixed accelerator recurrent
state can therefore coexist. Recurrent state remains a fixed device cost and
MTP rollback planes multiply the applicable state storage.

### Owned-layer target and draft placement plans

Standard KV constructs one per-layer placement plan reused by capability
probing, allocation, canonical store planning, and precision-tail handling.
Shared KV follows its real owner and does not consume a target or draft layer
budget.

Target and draft policies are independent only for independently owned layers.
They are layer-residency controls, not token-recency windows; each chosen layer
keeps its configured cache on one buffer type.

### Storage and attention execution are separate policies

The graph parameters add an internal `offload_attn_compute` decision distinct
from `offload_kqv`. Supported pinned-host or partially resident KV can remain
eligible for CUDA attention while persistent storage stays on the host.
`--no-op-offload` still selects the CPU execution route when requested.

This separation does not make host KV free: normal CUDA attention may still
stage growing history to the device. Storage capability and execution
capability are validated independently.

### Canonical accelerator quantization for host standard KV

CPU and accelerator quantizers can produce different standard-Q8 bytes even
when both are valid implementations. For supported quantized host storage whose
layer executes on an accelerator, the KV line:

1. converts new F32 rows to the cache type on the accelerator;
2. transfers only the newly quantized rows to the host; and
3. copies the same-type bytes into persistent host KV.

This gives CPU- and GPU-resident standard KV one numerical owner. The bounded
new-row store stage and D2H traffic are separate from the much larger repeated
history transfer used by CUDA attention. Unsupported store/execution routes
fail during route construction rather than silently changing quantization.

### Independent draft ubatch with an MTP restriction

Separate model-backed speculative contexts may use an independent physical
draft ubatch. Integrated MTP cannot: long-output testing showed that changing
MTP prompt/recurrent chunk geometry can diverge after short screens have
passed. Complete speculative-mode validation therefore rejects an MTP draft
ubatch that differs from target ubatch, independent of option parsing order.

### Configurable MTP recurrent planes

Base MTP keeps `draft_max + 1` target recurrent planes. The KV line permits a
smaller total in `2..draft_max+1` while retaining the draft maximum. One plane
holds the pre-verification state; the remaining planes hold recent verification
outputs. A rejection outside the direct retained horizon triggers selected
full-geometry replay.

The cap is a VRAM/replay trade. Full planes remain the default and broadest
compatibility path.

### Selected sparse recurrent snapshots and deterministic replay

The model graph and recurrent backend expose capability for selected snapshots.
During deep rejection the server restores the retained input state, replays the
original full verification shape, and writes only the accepted boundary.
Convolution and gated-delta-net state use the same selected boundary, and graph
reuse keys include sparse mode/selection.

This contract is capability-based and fails closed on unsupported graphs or
backends. It is not a general claim that all architectures or devices support
capped MTP.

### Phase-aware shared scheduler backing

Each scheduler retains an independent graph allocation plan. A shared backing
group for schedulers proven sequential allocates the maximum current
requirement for each exact backend buffer type rather than the sum of members.
Generation tracking invalidates stale captured addresses after replacement.

Growth is immediate. Shrink waits for an explicit phase epoch in which every
active member has published its plan. The server derives prompt/generation
bounds, synchronizes ownership handoffs, and regrows for later prompts. The
feature changes transient compute lifetime, not persistent KV or recurrent
state.

### Telemetry, benchmark identity, and exactness infrastructure

The KV line adds server counters for checkpoint/replay and workspace
transitions, benchmark fields for retained placement controls, a manifest-driven
fresh-server exactness runner, and an Nsight Systems workflow with progress and
artifact provenance.

The exactness runner compares prompt tokens, request semantics, output token
IDs and bytes, acceptance/replay work, commands, and identities. The maintained
MTP oracle compares the same MTP geometry across placements; target-only output
is not an MTP correctness reference.

The published base's opt-in `llama-bench --kv-memory` reports component and
device checkpoints, physical device/accelerator-host/ordinary-host allocation
classes, and CUDA VMM live/mapped/high-water state. The counters remain dormant
without opt-in and do not change allocation or trimming policy.

`llama-perplexity` declares `n_batch` as its maximum output-row requirement
before context creation. This preserves the tool's full-logits contract when a
phase-aware context would otherwise default to a serving-sized output reserve.
It adds no new argument or performance policy.

## Behavior intentionally absent from the KV base

The following were found in parallel or historical work but are not source
features of `4a7f9b496`:

| Absent behavior | Owner or disposition |
|---|---|
| Native standard-quantized FlashAttention and `--flash-attn-native-quants` | PR 4. |
| Automatic compact causal-prefix masking | Final draft PR 7 at `d4183adb8b4902a125b9339cd39032a095fca013`; no user control, not yet in the base. |
| `--live-context-workspace`, preset spelling, generated/user docs, and idle trim | Absent from core checkpoint `4a7f9b496`; final PR 8 source at `0c8df007a504f16aa35fc5982303e3e1b9883331` landed on the canonical integration line through merge `35e179272d510a0d5aefbf8dccb9dcf30fb31556`. |
| Positive bounded host-attention staging and `--kv-attn-staging-chunk` | Rejected for exact serving; unfinished research is not accepted source. |
| F16 persistent recurrent S state | Rejected by the quality gate. |
| A Bee-specific CUDA graph-disable option | Not implemented; the audited upstream environment control is not a KV feature. |
| Token-recency GPU KV | Not implemented. |
| Cross-context VMM pool sharing | Not implemented. |

Historical native-Q8-composed manifests remain archived with their original
branch. They must not be copied into current protocol because the KV base does
not contain their native source or option.

## Practical tradeoffs

The KV line turns one coarse placement decision into a memory hierarchy:

```text
target attention history       host, optionally pinned
target recurrent state         independently host or accelerator
selected target KV layers      optional accelerator residency
independently owned draft KV   inherited or independently placed
attention execution            independently eligible for accelerator
MTP rollback capacity          full or capped with selected replay
target/MTP compute backing     upfront or phase-aware shared lifetime
```

The advantages are explicit capacity/performance controls, residency-independent
standard-Q8 bytes, exactness infrastructure, and failure at unsupported
capability boundaries. The costs are a larger configuration/test surface,
substantial pinned system memory, context-linear transfer that still dominates
deep decode, fixed recurrent/rollback VRAM, allocator/graph lifecycle
complexity, and qualification that is deepest for CUDA homogeneous Q8.

Use the evidence index for measured outcomes. Do not add gains from different
hardware, contexts, models, or experiment branches as if they were one
composable performance result.
