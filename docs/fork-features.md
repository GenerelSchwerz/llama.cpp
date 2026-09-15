# GenerelSchwerz fork feature guide

This guide documents the user-visible features and internal backend integration carried by this maintained fork relative to its `llama/dev` baseline.

## CUDA MoE expert cache

The expert cache keeps a configurable number of routed MoE expert slabs in GPU memory and pages cold slabs from host memory. It is opt-in and CUDA-only. Grouped decode uses frequency-aware eviction with LRU tie-breaking by default; prefill and the legacy cache retain their existing policies.

Frequency history ages by one halving every 16 grouped planning steps, so previously hot experts do not stay protected indefinitely. This changes cache replacement, not expert routing, tensor values, or compute kernels. It uses small per-expert counters rather than extra expert slots. Set `GGML_CUDA_MOE_FREQUENCY=0` before starting the process to use LRU replacement in grouped decode; omit it or set `1` to restore the default. The setting is fixed when the CUDA grouped context is created, including for captured graphs.

| Public control | Default and disable behavior | Effect |
| --- | --- | --- |
| `--moe-expert-cache-size N` or `LLAMA_ARG_MOE_EXPERT_CACHE_SIZE=N` | `0` by default; set `0` or omit it to disable; negative values are rejected | Keeps `N` expert slabs per cached expert tensor on each owning CUDA backend. This is not a process-wide total. |
| `--spec-draft-moe-expert-cache-size N` or `LLAMA_ARG_SPEC_DRAFT_MOE_EXPERT_CACHE_SIZE=N` | Inherits `--moe-expert-cache-size` by default; set `0` to disable caching only for a separately loaded draft model | Overrides the expert cache size for a speculative draft model without changing target-model placement. |
| `llama_model_params::moe_expert_cache_slots` | `0` from `llama_model_default_params()` | Library equivalent of `--moe-expert-cache-size`; set it before loading the model. |
| `--moe-expert-cache-host-pinned-mb N` or `LLAMA_ARG_MOE_EXPERT_CACHE_HOST_PINNED_MB=N` | `0` by default preserves full pinning; negative values are rejected | Bounds model-wide MoE source registration and pinned staging in MiB, for allocated and memory-mapped sources. The main expert cache must also be enabled. |
| `--experimental-logs` | Off by default; omit it to disable | Enables detailed experimental CUDA MoE cache, matrix-dispatch, grouped-execution, and page-residency logs. It has no environment alias. Basic hit, miss, eviction, and hit-rate summaries remain available without it. |

When enabled, the loader prepends a cache placement override for routed `ffn_up`, `ffn_down`, `ffn_gate`, and fused `ffn_gate_up` expert weights, including chunk-expert names. This override wins over matching `--cpu-moe`, `--n-cpu-moe`, and user tensor buffer overrides. Shared experts, dense FFNs, and unrelated tensors keep their normal placement. On a build without CUDA, the model parameter has no effect and normal placement is retained.

Without a host budget, cold weights use pinned host memory when possible and memory-mapped models page directly from their mapped source. Pinned-allocation failure or `GGML_CUDA_NO_PINNED` uses pageable host backing while retaining CUDA cache ownership. Unexpected pinned-allocation failure logs a warning. With a host budget, mandatory staging and auxiliary sources are reserved first, then complete expert groups are registered in model order. Remaining experts use bounded pinned staging. A budget below the mandatory working space is rejected. Embedders set `llama_model_params::moe_expert_cache_host_pinned_size` in bytes before loading the model. This replaces the former mmap-only L2 setting; there is no independent L2 budget.

GPU memory use scales approximately with `N` times the expert-slab stride for every cached expert tensor assigned to a device, plus metadata, staging, and eligible auxiliary storage. Automatic `--fit` accounting does not include these pools and can overestimate available VRAM; use `-fit off` and size the cache explicitly when fit decisions are tight.

Cache misses, eviction protection, sibling prefetch, bounded overflow staging, cached prefill MMQ/MMVQ dispatch, and CUDA graph capture/replay are automatic implementation details. Eligible small contiguous F32 expert biases can remain GPU-resident for prefill under a fixed 32 MiB candidate-table budget. Grouped decode is also automatic: it accepts certified target-model decode graphs with independent rows, including `-np 1` and parallel one-token-per-sequence batches, when the complete expert group and CUDA kernel capabilities match and the total routed experts fit in `N` slots. Unsupported layouts, types, graph shapes, mixed prompt/decode batches, incomplete manifests, active LoRA, tensor overrides, insufficient slots, and unavailable optimized kernels fail closed to the existing cached `mul_mat_id` path. Setting the cache size to `0` restores baseline model placement and execution.

The cache setting is inherited by a separately loaded speculative draft model unless `--spec-draft-moe-expert-cache-size` overrides it. Setting the draft value to `0` restores normal draft placement, so `--spec-draft-ngl all` can keep an eligible draft model entirely on the selected GPU. MTP without a separate draft file uses the target model's weights. Draft and MTP contexts have distinct execution namespaces, and the grouped decode fast path is restricted to the main target context. Draft/MTP work and unsupported speculative shapes continue through the legacy cached path. The cache does not change sampling controls: target `-bs`/`--backend-sampling` remains opt-in and off by default, request-local `backend_sampling: false` and its CPU fallbacks remain intact, and draft/MTP sampling defaults are unchanged.

All CUDA devices can consume the cache buffer type and cache resources are owned by the CUDA backend context for the device that executes a layer, so normal layer-split placement can use device-local caches. The cache override is not a CUDA split buffer: row- and tensor-split sharding of matched expert weights is not implemented or covered by this fork's cache tests. Tensor split with a nonzero expert cache is rejected before weight allocation because the tensor-mode meta backend cannot consume the cached buffer. Use layer split or set the expert cache size to zero for tensor split.

Stage A explicitly places complete eligible grouped regions on their layer owner before runtime/reserve scheduling. See [grouped MoE multi-GPU validation](moe-grouped-multigpu.md) for the physical two-device fixture, no-P2P transfer path, owner-local diagnostics and hardware validation status.

The server prints and resets aggregate cache statistics at request timing boundaries and model unload. With parallel requests, this is a process-wide reset boundary rather than strict per-request attribution.

### Internal backend integration

The fork resolves MoE buffer operations, cache statistics and typed expert manifests through `ggml_backend_reg_get_proc_address()`. The internal procedure declarations and snapshot layouts live in [ggml-backend-moe.h](../ggml/src/ggml-backend-moe.h), not the installed public backend headers. Deploy matching fork-built executables and llama/ggml/CUDA libraries; these private procedures are not a stable API for mixing builds.

The llama runtime publishes complete V2 candidate snapshots and refreshes them after LoRA changes. Rejected, incomplete or unknown snapshots are ineligible for grouped execution. Snapshot arrays are borrowed for the call; tensor pointers in an accepted snapshot remain valid until replacement. Buffers and active cache resources retain the bounded host-budget owner.

## Generic fork helpers

- Graph execution certificates: `ggml_graph_execution_certificate` describes MAIN, DRAFT, or MTP ownership and independent, sequential, or speculative row semantics. Pass it with `ggml_backend_sched_graph_compute_ext()` or `ggml_backend_sched_graph_compute_async_ext()`. Invalid or absent certificates run as ordinary uncertified graphs. The scheduler stamps source and split graph IDs only for each backend compute callback; a backend must validate and copy needed fields synchronously and must not retain the certificate address. Existing scheduler compute APIs are unchanged. The llama runtime automatically certifies eligible independent decode batches.

- Draft context namespace: `LLAMA_CONTEXT_TYPE_DRAFT` distinguishes a separate speculative draft context from the default target and `LLAMA_CONTEXT_TYPE_MTP` contexts. Common speculative setup assigns it automatically; the default context type and MTP behavior are unchanged.

- Dense penalty counts: the penalties sampler automatically uses a vocabulary-sized dense count table for valid token IDs, with a map fallback for out-of-range IDs. This has no flag or API change and preserves repeat, frequency, and presence penalty semantics, including duplicate candidate IDs, reset, and clone behavior.
