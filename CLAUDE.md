# BeeLlama.cpp development guide

BeeLlama.cpp is Anbeeld's llama.cpp fork. The fork tracks upstream closely and
keeps a small set of maintained extensions:

- KVarN target-context KV-cache compression (`kvarn2` through `kvarn8`).
- Low-bit standard KV cache types (`q2_0`, `q2_1`, `q3_0`, `q3_1`, `q6_0`, and
  `q6_1`).
- Quantized-native CUDA MMA FlashAttention (`--flash-attn-native-quants`).
- Profit-only adaptive draft depth for upstream `draft-dflash` speculation.
- Server-side reasoning-loop protection.
- KLD save/load support in `llama-perplexity` for KVarN validation.

The speculative implementation itself is upstream llama.cpp. Use
`--spec-type draft-dflash`; the old `--spec-type dflash` alias was removed in
v0.4.0 and now errors.
Draft GGUFs must use upstream's `dflash` architecture, metadata, and tensor
names.

## Build

```powershell
# Windows, RTX 3090 example
cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_NATIVE=ON `
  -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 16
```

For Windows hosts matching CUDA 13.1 and compute capability 8.6, use:

```powershell
powershell -File scripts/build-win-cuda-13.1-sm_86.ps1 -AllTests
powershell -File scripts/build-win-cuda-13.1-sm_86-default.ps1 -AllTests
powershell -File scripts/build-win-vulkan.ps1 -AllTests
```

The first CUDA script compiles the expanded quant matrix; the `-default`
variant compiles the default pair matrix. The Vulkan script requires a Vulkan
SDK. Adapt the architecture, toolkit, and build-name parameters for other
hardware rather than reusing the `sm_86` artifact names.

CUDA FlashAttention vector cache coverage has two build modes:

- Default: 50 of the 169 ordered pairs over `f16`, `bf16`, `q8_0`, `q6_1`,
  `q6_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `q3_1`, `q3_0`, `q2_1`, and the
  fork's internal q2 fallback type: the two homogeneous float pairs plus the
  48 quant pairs admitted by `ggml_cuda_get_fattn_vec_default_pairs` in
  `ggml/CMakeLists.txt`, which is also asserted there. On the bit ladder
  8-6-5-4-3-2, V sits at K's position or up to two below it, never above, and
  at equal position a `_1` K may pair with a `_0` V but not the reverse.
- `-DGGML_CUDA_FA_ALL_QUANTS=ON`: all 169 ordered vector pairs.

`--flash-attn-native-quants` lets the CUDA MMA FlashAttention kernel read a
quantized K/V cache in place instead of casting it to F16 first, which removes
the transient F16 copy of the attention window. It is off by default, output is
unchanged, and it applies only where a native loader exists for the cache types
and head dimension; every other case keeps the F16-casting path. K and V may be
different types: the native route compiles the same ordered pair set as the
vector path above, so `--cache-type-k q8_0 --cache-type-v q6_0` is native while
`q8_0`/`q4_0` and any pair whose V outranks its K fall back. The kernels are
compiled only when `GGML_CUDA_FATTN_Q8_NATIVE=ON`, so on a build without them
the option keeps the standard path and reports a warning. The cache type still
comes exclusively from `--cache-type-k` and `--cache-type-v`; the native option
does not select a second layout. See
[`docs/quantized-native-flash-attention.md`](docs/quantized-native-flash-attention.md)
for the exact kernel inventory, architecture, tradeoffs, and validation gates.

There is no `GGML_CUDA_FA_HALF_QUANTS` tier. KVarN has 15 balanced fast-decode
pairs by default and all 36 with `GGML_CUDA_FA_ALL_QUANTS=ON`; every valid KVarN
bit pair remains available through descriptor-native MMA when it is outside the
fast matrix. `GGML_CUDA_KVARN=OFF` omits all dedicated CUDA KVarN kernels and
templates.

## Layout

- `src/llama-kvarn.{h,cpp}`: KVarN descriptors, layouts, type parsing, and
  runtime validation.
- `src/llama-kv-cache-kvarn.{h,cpp}`: KVarN cache metadata, stream handling,
  and state serialization.
- `ggml/src/ggml-cuda/kvarn.*`: CUDA KVarN store support.
- `ggml/src/ggml-cuda/fattn-kvarn-dispatch.*`: explicit KVarN FlashAttention
  dispatch and descriptor-native fallback.
- `tools/server/server-adaptive-dm.h`: profit controller.
- `tools/server/server-loop-guard.*`: reasoning-loop protection.
- `common/speculative.*` and `src/models/dflash.cpp`: upstream DFlash support.

## Validation

Run the narrowest relevant checks first:

```powershell
ctest --test-dir build --output-on-failure -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"
build/bin/llama-bench -m model.gguf -p 0 -n 64 -t 1
build/bin/llama-perplexity -m model.gguf -f test.txt -c 4096
```

For KVarN changes, validate both numerical quality and routing: `test-kvarn`
exercises all 36 K/V bit combinations when CUDA is available, and the local KLD
harnesses under `tmp/` provide model-level measurements. Do not compare benchmark
numbers across different models, prompts, cache settings, hardware, or commits.

## Compatibility notes

TurboQuant/TCQ cache types and their GGUF formats are removed. The command-line
cache spellings `turbo2`, `turbo3`, `turbo4`, and their `_tcq` variants are warned
and redirected to the matching KVarN type for target caches or a standard low-bit
type for draft caches. Re-quantize removed TQ weight GGUFs from source.

DDTree and CopySpec are not supported in v0.4.0. Use upstream speculative modes
such as `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, or the n-gram
modes documented by `llama-server --help`.
