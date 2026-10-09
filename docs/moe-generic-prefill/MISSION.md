# Fully generic Strata feature parity: features 1-4

Owner request, 2026-10-08: implement all four features from the difficulty ranking fully generically and test across multiple models. This replaces the prior broader active research objective. The complete acceptance contract is [GOAL-1-4.md](GOAL-1-4.md).

1. Save and reload learned full-model profiles.
2. CPU-assisted prefill.
3. Bounded cross-layer expert prefill streaming/lookahead.
4. Temporary expert-cache lending and restoration.

Support must follow llama.cpp graphs, tensor/source descriptors and shared GGML/backend contracts, including future model updates. Validation models are samples, not a support list. No model-specific admission helpers, fixed specialized dimensions, permanent hybrid operation whitelist, native/generic support tiers, mandatory requantization or whole-model fallback counted as hybrid support. Reuse existing arithmetic kernels and CPU workers, with one canonical residency owner.

Model statistics, hardware capacity/split tuning and online adaptation remain separate. Profiles cover the full model independently of runtime capacity. Completion requires numerical/state checks, held-out profile quality at least Strata's at equal actual byte budgets, and no accepted matched performance regressions. Preserve MTP, parallel/staggered requests, multiple devices and Windows partial pinning. Unavailable required physical platform gates remain incomplete.

Owner clarification: hybrid mode must use the generalized CPU/multi-token prefill pipeline, including graph-derived complete-expert wave scheduling. CPU share zero remains inside hybrid; unavailable expert admission fails explicitly instead of selecting ordinary prefill. Kernel fusion, elastic cache resizing and RAM-complement exchange remain deferred. Historical TODO entries do not expand the active scope.

Use the established own worktree /home/gencoolpc/llama-moe-generic-hybrid-features-20261007 and own builds. Preserve checkpoint a69f285ddca71c9446354b687a6d8c44b0c5c9cd and incoming release 28d73c87cb90a9b78d4357ff1b456f456bfa1164. The pending combined tree passes current CPU-tool/CUDA-backend/test linkage and scoped synchronous owner tests; full serving qualification remains open. Preserve parent, specialized/frozen and other sessions' source and edit ownership. No extra agents, additional commit/push/PR or default promotion without explicit owner authorization.

Hold /tmp/beellama-cuda-build.lock then /tmp/beellama-single-gpu.lock for whole build/GPU jobs. Read [CRASH-RECOVERY.md](CRASH-RECOVERY.md); preserve crash evidence, verified hard RAM budgets, zero swap, finite timeouts, fatal-CUDA whole-process-tree teardown and the shared 12GiB GPU cap. Verify interrupted processes/artifacts. Full model-server/original Flash replay remains blocked under the saved recovery protocol. Containment does not prove the root defect fixed. No blind replay, unrelated kills, global changes or ccache clearing. Extended drains remain skipped per owner.

Maintain PLAN, TODO, STATUS, DESIGN and feature-by-feature evidence. The active goal remains incomplete until all four full implementations and required acceptance gates pass.
