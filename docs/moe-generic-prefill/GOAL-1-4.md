# Active goal: fully generic implementation of features 1-4

Saved 2026-10-08 for thread 01a0fe7a-0aaa-70b1-bda0-96f6acbc3a61. The owner selected features 1-4 from the difficulty ranking, with complete implementation and multi-model testing. This is the active acceptance contract; older research remains evidence rather than additional scope.

## Feature contracts

| Feature | Required implementation | Completion evidence |
| --- | --- | --- |
| 1. Learned profile save/reload | Persist learned full-model ranks/statistics independently of runtime slot count or selected layers. Define snapshot consistency, provenance, normalization, validation, versioning and publication. Reload through the existing profile/ranking owner. | Round-trip, malformed/incompatible input, multiple-capacity placement and continued adaptation checks; independent equal-byte held-out quality. Offline corpus collection or saving only current occupants does not qualify. |
| 2. CPU-assisted prefill | Partition prompt work through shared graph/backend semantics and the existing CPU workers. Preserve original routing, activation, precision, auxiliary readers, reduction and per-request ownership. Derive capacity/split from descriptors and hardware policy. | Numerical/state checks, actual CPU/GPU work, pinned/pageable transport, failure/cancellation, matched prefill plus subsequent decode/MTP across distinct real models. |
| 3. Cross-layer streaming/lookahead | Extend existing prefetch/event scheduling to copy bounded future-layer expert data before consumption. Use available layer/source metadata, never future trace routing labels. Preserve one aggregate resource budget and canonical ownership. | Ready/late/unused bytes, overlap and wait time, original IDs, source/reader lifetimes, cold/warm and concurrent/cancellation cases, matched prefill/decode. Existing same-layer two-lane overlap does not qualify cross-layer streaming. |
| 4. Temporary cache lending/restoration | Borrow reclaimable canonical cache storage for staging; fence readers, copies and adaptation; restore current learned occupants before decode. Account actual per-device backend ranges, with no independent residency map. | Invalidation/publication, bounded storage accounting, held-reader/adaptation/failure/cancellation tests, repeated prefill-to-decode/MTP restoration and matched resource/performance results. |

## Generic architecture

All four are first-class behavior in the same hybrid pipeline. Model support follows llama.cpp graphs, tensor/source descriptors and shared GGML/backend contracts, including future graph/model updates. No per-model helpers, fixed specialized geometry, permanent hybrid operation whitelist, native/generic tiers, mandatory requantization, second residency authority or whole-model ordinary fallback substituted for hybrid support.

Reuse arithmetic kernels, CPU workers and the canonical owner. Preserve original route tensors for auxiliary readers; retain graph, source, copy and output lifetimes through actual completion. Bound aggregate resources across requests/devices; cancellation must not invalidate unrelated work. Identity tags or serialized smoke tests alone do not qualify asynchronous requests or multi-GPU. Any shared lifetime/control extension must be a concrete prerequisite for these four features, not an unrelated executor redesign.

Keep one-time full-model statistics, hardware budget/split tuning and online residency adaptation distinct. Persist learned information with explicit provenance and snapshot semantics, not only runtime placement. Full profile coverage is independent of runtime capacity. Future held-out counts cannot be used as a training profile.

## Model and acceptance matrix

Inventory available model files/metadata without broad model hashing. Test distinct MoE graphs and scalar/quantized formats, including available Nemotron, Mixtral, Flash Next/RCO and DeepSeek V4, plus Qwen, GPT-OSS, LFM and Ornith where available. They are validation samples, never a model support list. Record unavailable samples; preserve an open gate or justify equivalent structural coverage. Synthetic graph/format tests supplement unavailable cases and do not replace real-model demonstrations.

Require component and matched real-model evidence for every feature, individually and in combination. Exercise repeated prefill/decode, static/adaptive profiles, MTP including MTP3, parallel/staggered requests and per-device isolation. Windows partial pinning and physical multi-GPU remain required. Linux/source and single-device checks cannot qualify native platforms; unavailable required gates remain explicitly incomplete.

Compare the exact combined candidate against matched frozen generic controls; retain updated specialized comparisons with configuration differences exposed. Preserve actual input/output IDs and work, numerical/state bounds, route/residency counts, hits/missed bytes, MTP acceptance, phase timings, source/binary hashes, peak resources and verified teardown. CPU/GPU arithmetic may differ within justified numerical bounds; report resulting acceptance/output differences before attributing speed changes. Short or unmatched runs are observational.

No accepted performance regressions against matched generic controls. Held-out saved-profile quality must be equal or better than Strata at equal actual resident bytes, including representative code generation and prose/context workloads independent of training/tuning. Preserve existing losing held-out comparisons. Loading a file, one short win or component speed alone does not close serving/quality gates.

## Execution boundaries

Use /home/gencoolpc/llama-moe-generic-prefill-20261007 and own builds. Preserve checkpoint a69f285ddca71c9446354b687a6d8c44b0c5c9cd and incoming release 28d73c87cb90a9b78d4357ff1b456f456bfa1164. The owner requested committing the combined incoming merge and feature checkpoint while implementation is paused. Complete CPU tools and CUDA backend/test linkage,8 sequential private-prefill checks,48 owner checks and14 existing decode checks pass; full combined serving remains unqualified. Preserve parent, specialized/frozen sources and other sessions' edit ownership.

Recheck interrupted processes, builds, tests and artifacts before relying on them. Hold /tmp/beellama-cuda-build.lock then /tmp/beellama-single-gpu.lock for whole build/GPU jobs. Follow CRASH-RECOVERY.md: verified hard RAM budgets, zero swap, finite timeout, fatal-CUDA whole-process-tree teardown, shared 12GiB GPU cap and preserved incident evidence. Full model-server/original Flash replay remains blocked under the saved recovery protocol. Containment/tiny fixtures do not prove the original root defect fixed. No blind retry, unrelated kills, global changes or ccache clearing. Extended drains remain skipped; focused completion/cancellation/teardown checks remain required.

No extra agents, additional commit/push/PR or default promotion without explicit owner authorization. Keep durable MISSION/PLAN/TODO/STATUS/DESIGN and feature-by-feature results in the established evidence directory.

## Deferred scope and completion

Kernel fusion, elastic cache resizing (#5), RAM-complement exchange (#6) and complete-expert wave reorganization (#7) are deferred. Optional paired fusion stays disabled after measured regressions. Cross-layer streaming first reuses existing arithmetic/scheduling rather than silently becoming #7. Document source evidence and the smallest shared prerequisite if a boundary must change.

The goal is complete only when all four full generic implementations, source review and required correctness, profile-quality, performance, multi-model, concurrency and platform gates pass. Historical frozen runs, fallback-only support, partial implementations and explicitly open platform gates do not count as completion.
