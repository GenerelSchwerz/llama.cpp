<h1 align="center">llama.cpp / MoE Cache</h1>

<p align="center"><strong>Run MoE models with an expert cache sized for your NVIDIA GPU.</strong><br>
A maintained <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> fork for models whose expert weights exceed VRAM.</p>

---

## What does this fork do?

A mixture-of-experts (MoE) model has many expert networks, but calls on only a few at a time. This fork keeps a configurable set of experts on the GPU and brings in the rest from host memory when needed.

- **Use the VRAM you have.** Set a GPU cache budget instead of trying to fit every expert in VRAM.
- **Keep the llama.cpp workflow.** Run `llama-server` for a local chat page and OpenAI-compatible API.
- **Opt in when you need it.** The expert cache is CUDA-only and off by default; other llama.cpp models and backends remain available.

> Jump to: [Is my PC enough?](#is-my-pc-enough) | [Get started](#get-started) | [How it works](#how-it-works) | [Cache controls](#cache-controls) | [Full docs](#full-documentation)

---

## Is my PC enough?

| You need | What to check |
| --- | --- |
| **Graphics card** | NVIDIA GPU with a working CUDA driver and enough VRAM for your chosen model placement, context, and at least one expert cache slot. |
| **System memory** | Enough RAM for the model's host-backed weights and your workload. Quantization makes a large difference. |
| **Storage** | Space for a MoE model in GGUF format. An SSD helps when model data must be read from disk. |
| **Build tools** | CMake, a C++ compiler, and the CUDA toolkit. |

There is no single minimum RAM or VRAM figure for every MoE model. Choose a GGUF that fits your machine, then size the cache from the VRAM left after other model and context allocations.

## Get started

**1. Build the `moe-cache` branch with CUDA.**

```sh
git clone --branch moe-cache https://github.com/GenerelSchwerz/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

**2. Start the server with your GGUF model.**

```sh
./build/bin/llama-server -m /path/to/model.gguf --moe-expert-cache-mib 4096
```

**3. Open the local address printed by the server** to chat, or connect an OpenAI-compatible client to its API.

The `4096` MiB cache budget is an example, not a recommended setting for every GPU. If the model does not fit, lower the budget or adjust model placement and context size. See the [CUDA build guide](docs/build.md#cuda) and [server guide](tools/server/README.md) for more options.

## How it works

![Diagram of the GGUF file, host-backed expert weights, and a budgeted GPU expert cache](media/moe-cache-memory.svg)

The model selects experts as it runs. A cache hit reuses weights already on the GPU; a miss moves the selected weights from their host backing into the cache. The cache does not change which experts the model selects. Eligible decode workloads can also use the fork's grouped CUDA path automatically.

Cache speed depends on the model, quantization, available VRAM and RAM, storage, and request pattern. The [feature guide](docs/fork-features.md) covers grouped-decode eligibility and fallback behavior.

## Cache controls

| Option | What it does |
| --- | --- |
| `--moe-expert-cache-mib MiB` | Set a per-device VRAM budget for cached experts. |
| `--moe-expert-cache-size N` | Set expert slots per cached tensor instead of a MiB budget. |
| `--moe-expert-cache-layers N[,N-M,...]` | Cache only selected MoE layers. |
| `--moe-expert-cache-host-pinned-mb N` | Bound host memory registration and staging. |

Use **either** a MiB budget **or** a nonzero slot count. Both are off by default. After a request, check the server log for `moe-cache` hit and miss statistics to confirm that caching ran. A startup flag alone does not prove it was active.

## Full documentation

- [Fork feature guide](docs/fork-features.md) - defaults, host memory, speculative drafts, and limitations.
- [Build guide](docs/build.md) and [server guide](tools/server/README.md) - installation and API usage.
- [Multi-GPU notes](docs/moe-grouped-multigpu.md) - layer-split cache behavior and validation. Tensor split with the expert cache enabled is unsupported.

## Work with me or support the project

I'm looking for a job in GPU inference, systems engineering, or local AI based on the work in this fork. If that matches your team, find me on [GitHub](https://github.com/GenerelSchwerz).

If this work helps you, you can [support me on Ko-fi](https://ko-fi.com/generel).

---

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). The code is under the repository's [MIT license](LICENSE); model files have their own licenses.
