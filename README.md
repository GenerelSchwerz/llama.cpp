# llama.cpp

## GenerelSchwerz fork feature guide

This section documents the user-visible and public API additions carried by this maintained fork relative to its `llama/dev` baseline. The upstream README is unchanged below this guide.

### CUDA MoE expert cache

The expert cache keeps a configurable number of routed MoE expert slabs in GPU memory and pages cold slabs from host memory with LRU eviction. It is opt-in and CUDA-only.

| Public control | Default and disable behavior | Effect |
| --- | --- | --- |
| `--moe-expert-cache-size N` or `LLAMA_ARG_MOE_EXPERT_CACHE_SIZE=N` | `0` by default; set `0` or omit it to disable; negative values are rejected | Keeps `N` expert slabs per cached expert tensor on each owning CUDA backend. This is not a process-wide total. |
| `llama_model_params::moe_expert_cache_slots` | `0` from `llama_model_default_params()` | Library equivalent of `--moe-expert-cache-size`; set it before loading the model. |
| `--moe-expert-cache-l2-pinned-mb N` or `LLAMA_ARG_MOE_EXPERT_CACHE_L2_PINNED_MB=N` | `0` by default; set `0` or omit it to disable; negative values are rejected | Sets a total MiB budget for a second, pinned-host LRU used only by memory-mapped expert sources. The main expert cache must also be enabled. |
| `--experimental-logs` | Off by default; omit it to disable | Enables detailed experimental CUDA MoE cache, matrix-dispatch, grouped-execution, L2, and page-residency logs. It has no environment alias. Basic hit, miss, eviction, and hit-rate summaries remain available without it. |

When enabled, the loader prepends a cache placement override for routed `ffn_up`, `ffn_down`, `ffn_gate`, and fused `ffn_gate_up` expert weights, including chunk-expert names. This override wins over matching `--cpu-moe`, `--n-cpu-moe`, and user tensor buffer overrides. Shared experts, dense FFNs, and unrelated tensors keep their normal placement. On a build without CUDA, the model parameter has no effect and normal placement is retained.

Cold weights use pinned host memory when possible. If pinned allocation fails or `GGML_CUDA_NO_PINNED` is set, allocation falls back to regular CPU memory, which preserves loading and correctness but can make misses slower. Memory-mapped models page directly from their mapped source. The optional L2 retains recently read mmap expert slabs in pinned host memory before the GPU copy; its total budget is divided among registered mmap banks and it disables itself on pinned-allocation failure. The L2 and debug process settings are latched by an existing CUDA backend context when it first acquires cache resources, so embedders should set them before first inference.

GPU memory use scales approximately with `N` times the expert-slab stride for every cached expert tensor assigned to a device, plus metadata, staging, and eligible auxiliary storage. Automatic `--fit` accounting does not include these pools and can overestimate available VRAM; use `-fit off` and size the cache explicitly when fit decisions are tight.

Cache misses, eviction protection, sibling prefetch, bounded overflow staging, cached prefill MMQ/MMVQ dispatch, and CUDA graph capture/replay are automatic implementation details. Eligible small contiguous F32 expert biases can remain GPU-resident for prefill under a fixed 32 MiB candidate-table budget. Grouped decode is also automatic: it accepts certified target-model decode graphs with independent rows, including `-np 1` and parallel one-token-per-sequence batches, when the complete expert group and CUDA kernel capabilities match and the total routed experts fit in `N` slots. Unsupported layouts, types, graph shapes, mixed prompt/decode batches, incomplete manifests, active LoRA, tensor overrides, insufficient slots, and unavailable optimized kernels fail closed to the existing cached `mul_mat_id` path. Setting the cache size to `0` restores baseline model placement and execution.

The cache setting is inherited by a separately loaded speculative draft model; MTP uses the target model's weights. Draft and MTP contexts have distinct execution namespaces, and the grouped decode fast path is restricted to the main target context. Draft/MTP work and unsupported speculative shapes continue through the legacy cached path. The cache does not change sampling controls: target `-bs`/`--backend-sampling` remains opt-in and off by default, request-local `backend_sampling: false` and its CPU fallbacks remain intact, and draft/MTP sampling defaults are unchanged.

All CUDA devices can consume the cache buffer type and cache resources are owned by the CUDA backend context for the device that executes a layer, so normal layer-split placement can use device-local caches. The cache override is not a CUDA split buffer: row- and tensor-split sharding of matched expert weights is not implemented or covered by this fork's cache tests. Use layer split or a single GPU when that distinction matters.

The server prints and resets aggregate cache statistics at request timing boundaries and model unload. With parallel requests, this is a process-wide reset boundary rather than strict per-request attribution.

#### Library and backend integration

The primary low-level placement APIs are `ggml_backend_cuda_moe_cached_buffer_type()`, `ggml_backend_buft_is_cuda_moe_cached()`, and `ggml_backend_cuda_moe_cached_buffer_from_host_ptr()`. The last function registers a borrowed host range as an mmap cache source. Functional process settings are `ggml_backend_cuda_moe_set_l2_pinned_cache_size()` / `ggml_backend_cuda_moe_get_l2_pinned_cache_size()` and `ggml_backend_cuda_moe_set_debug_mm()` / `ggml_backend_cuda_moe_get_debug_mm()`. `ggml_backend_cuda_moe_log_and_reset_stats()` reports and clears aggregate counters.

Advanced backends can publish complete typed expert manifests through `ggml_backend_reg_get_proc_address()` using `GGML_BACKEND_MOE_CANDIDATE_REPLACE_V1_PROC_NAME` or `GGML_BACKEND_MOE_CANDIDATE_REPLACE_V2_PROC_NAME` and the corresponding snapshot ABI in `ggml-backend.h`. The llama runtime publishes V2 snapshots, refreshes them after LoRA changes, and treats rejected, incomplete, or unknown snapshots as ineligible for grouped execution. Snapshot arrays are borrowed for the call; tensor pointers in an accepted snapshot must remain valid until replacement.

The exported `ggml_backend_cuda_moe_set_cache_slots()` / `ggml_backend_cuda_moe_get_cache_slots()` pair is only a legacy route-publication hint; model-owned resources use candidate snapshots. `ggml_cuda_moe_cache_free_all()`, `ggml_backend_cuda_moe_observe_expert_tensor()`, `ggml_backend_cuda_moe_reset_expert_size_observation()`, `ggml_backend_cuda_moe_preallocate_pool()`, `ggml_backend_cuda_moe_preallocate_pools()`, and `ggml_backend_cuda_moe_prefetch_experts()` are resource-free compatibility shims and do not control the current cache.

### Generic fork helpers

- Graph execution certificates: `ggml_graph_execution_certificate` describes MAIN, DRAFT, or MTP ownership and independent, sequential, or speculative row semantics. Pass it with `ggml_backend_sched_graph_compute_ext()` or `ggml_backend_sched_graph_compute_async_ext()`. Invalid or absent certificates run as ordinary uncertified graphs. The scheduler stamps source and split graph IDs only for each backend compute callback; a backend must validate and copy needed fields synchronously and must not retain the certificate address. Existing scheduler compute APIs are unchanged. The llama runtime automatically certifies eligible independent decode batches.

- Draft context namespace: `LLAMA_CONTEXT_TYPE_DRAFT` distinguishes a separate speculative draft context from the default target and `LLAMA_CONTEXT_TYPE_MTP` contexts. Common speculative setup assigns it automatically; the default context type and MTP behavior are unchanged.

- Dense penalty counts: the penalties sampler automatically uses a vocabulary-sized dense count table for valid token IDs, with a map fallback for out-of-range IDs. This has no flag or API change and preserves repeat, frequency, and presence penalty semantics, including duplicate candidate IDs, reset, and clone behavior.

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
