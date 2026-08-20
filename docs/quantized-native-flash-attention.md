# Quantized-native CUDA FlashAttention

BeeLlama can optionally let the CUDA MMA FlashAttention kernel consume a
supported quantized K/V cache directly. This removes the temporary F16 K/V
materialization used by the standard MMA path. It does not change the
persistent cache format, cache placement, or quantization policy.

The feature is deliberately explicit at both build time and run time. Its
current kernel inventory is `Q8_0`, `Q4_0`, `Q5_0`, and `Q6_0` by default, each
with K and V of the same type and equal head dimensions 64, 128, or 256, on
NVIDIA GPUs that use the Ampere MMA implementation. `GGML_CUDA_FA_ALL_QUANTS`
adds `Q4_1`, `Q5_1`, `Q6_1`, `Q3_0`, `Q3_1`, `Q2_0`, and `Q2_1` to the same
head-dimension set. Unsupported cache pairs, dimensions, devices, and builds
keep the standard materializing path.

## Selection model

`--cache-type-k` and `--cache-type-v` select the actual cache layouts.
`--flash-attn-native-quants` only permits a backend to consume those selected
layouts directly when it has a registered kernel. The native option neither
selects nor converts the cache type. For example:

| K / V cache | Native option | Result |
|---|---|---|
| `f16 / f16` | off or on | Existing F16 attention path |
| `q8_0 / q8_0` | off | Standard Q8-to-F16 materialization, then MMA |
| `q8_0 / q8_0` | on | Native Q8 MMA when the build, device, and head size support it |
| `q4_0 / q4_0` | on | Native Q4 MMA when the build, device, and head size support it |
| `q6_0 / q6_0` | on | Native Q6 MMA when the build, device, and head size support it |
| `q8_0 / q4_0` | on | Standard materializing fallback; the route requires K and V to agree |
| `q2_1 / q2_1` | on | Native MMA only in a `GGML_CUDA_FA_ALL_QUANTS` build; otherwise standard materializing fallback |

This avoids a second K/V type selector that could disagree with the tensors in
the graph. The graph records a boolean permission; CUDA derives the concrete
pair from the K and V tensors themselves.

## Build and use

The native MMA family (`Q8_0`, `Q4_0`, `Q5_0`, `Q6_0`; plus `Q4_1`, `Q5_1`,
`Q6_1`, `Q3_0`, `Q3_1`, `Q2_0`, `Q2_1` under `GGML_CUDA_FA_ALL_QUANTS`) is not
compiled by default:

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FATTN_Q8_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Enable it at run time only for workloads where it has been measured:

```bash
build/bin/llama-server -m model.gguf --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn-native-quants
```

The run-time option is exposed by the server, CLI, perplexity, batched,
parallel, and benchmark tools. A build without native kernels accepts the
option, retains the standard route, and reports a once-only warning. A build
with the native family reports the same kind of warning when an opted-in
quantized pair or head dimension has no registered kernel. F16/BF16 attention
is not treated as an unsupported quantized request.

`GGML_CUDA_FA_ALL_QUANTS` extends this family's own registration list (see
above) in addition to controlling the existing vector FlashAttention pair
matrix; it does not implicitly instantiate every K/V pair for the native MMA
route.

## Implementation structure

The implementation follows the existing upstream CUDA template-instance
structure:

- The normal MMA case and `ncols1`/`ncols2` selectors are templated on the K and
  V storage types. Their default remains F16, so existing call sites and
  generated F16 instances do not change behavior.
- The native route specializes that shared case with one type for both K and V; it does not carry
  a copied launcher or a copied selector tree.
- The per-type tile loaders are `fattn_quant_type_traits<GGML_TYPE_*>`
  specializations in `fattn-mma-quant.cuh`. Each dequantizes into the same
  shared-memory `half2` tiles consumed by the existing MMA math.
- Most types share one of two packed extraction shapes rather than carrying
  their own loader. `fattn_quant_nibble_traits` covers `Q4_1`, `Q5_0` and
  `Q5_1`, whose qs byte holds one element per nibble;
  `fattn_quant_2bit_traits` covers `Q2_0S`, `Q2_1`, `Q3_0` and `Q3_1`, whose
  qs byte holds the two-bit fields of elements j, j+8, j+16 and j+24. Each
  takes a flag for the optional one-bit plane in qh. `Q6_0` and `Q6_1` keep
  their own extraction for the two-bit plane, and all of them finish through
  the shared `fattn_quant_store4` conversion. A lane unpacks four codes at a
  time with 32-bit word operations instead of walking the block per element.
- `fattn_quant_store4` subtracts a zero-point type's bias in float rather than
  in the packed word. `(code - bias)*d` is the cast path's own expression and
  is exact for these small integers, and it avoids `__vsubss4`, which is
  emulated on every NVIDIA architecture since Kepler.
- Per-type dequantization must reproduce the F16-casting route bit for bit,
  and which helper achieves that differs per type. `Q8_0` reuses the established
  `dequantize_V_q8_0` helper from
  `fattn-common.cuh`. The helper's historical name is V-specific, but its
  operation is a generic Q8_0 row dequantization and is also valid for K.
- `template-instances/generate_cu_files.py` owns the supported head-size and
  tile inventory. It emits one translation unit per `(ncols1, ncols2)` shape,
  matching the existing F16 and KVarN convention and preserving incremental
  parallel compilation.
- CMake includes those generated sources only when
  `GGML_CUDA_FATTN_Q8_NATIVE=ON`.

There are 16 tile shapes and three supported head dimensions. The default
four-type registration (`Q8_0`, `Q4_0`, `Q5_0`, `Q6_0`) yields 192 explicit
template cases; `GGML_CUDA_FA_ALL_QUANTS` adds the remaining seven types for
528 total. A future native pair is not inferred from the standard quant
matrix. It requires an intentional loader/registration, a bounded generated
inventory, route and fallback tests, output validation, and performance and
memory evidence.

## Expected effects

The standard MMA path materializes quantized K and V as F16 for the attention
window. The native path removes that conversion and the corresponding
temporary device buffer. For equal K/V head dimension `D`, the avoided F16
storage per visible token is:

```text
2 tensors * n_kv_heads * D * sizeof(F16)
```

The graph allocator can reuse that temporary storage between layers, so peak
device-memory savings normally correspond to one layer's materialized window,
not the sum of all attention layers. The exact observed saving can also be
hidden by a larger reusable graph allocation, such as a logits buffer.

This is primarily a prefill and multi-token attention optimization. CUDA's
existing vector FlashAttention route already consumes Q8_0 directly for the
small decode shapes it selects, so single-token decode often never reaches the
native MMA family. Speculative decoding can reach it when a verification batch
is large enough for MMA routing.

Performance is shape- and device-dependent. Removing conversion traffic can
reduce memory use without guaranteeing a meaningful speedup, and an in-kernel
dequantization schedule can lose to the materialized path on an untested GPU or
geometry. This is why both the build and run-time defaults remain off.

## Validation protocol

For a source change, validate both build modes and both execution outcomes:

1. Generate instances from `ggml/src/ggml-cuda/template-instances` and confirm
   a second generator run is byte-idempotent.
2. Build with `GGML_CUDA_FATTN_Q8_NATIVE=OFF`; verify no native quantized MMA
   instance is compiled or linked and an opted-in graph uses the fallback with
   a warning.
3. Build with it `ON`; run the focused `FLASH_ATTN_EXT` cases filtered by
   `native_quants=1`. They cover D=64/128/256, padded and unpadded KV lengths,
   GQA-selected column geometries, query-batch tile choices, and unsupported
   fallbacks against the CPU reference.
4. Run the full FlashAttention backend suite to protect the unchanged default
   F16 and quantized materializing routes.
5. Compare deterministic end-to-end output with and without
   `--flash-attn-native-quants` in the same opt-in build.
6. Measure representative short, medium, and very-long contexts with identical
   model, prompt, batch, cache, affinity, and repetition settings. Record
   prefill throughput, live decode or verification throughput, peak process
   VRAM, and the reserved CUDA compute buffer.

For route auditing only,
`GGML_CUDA_FATTN_Q8_NATIVE_VERBOSE=1` logs each launch that actually selects
the native quantized MMA family, for any of its registered types. It does not
enable or disable the route and is not part of a serving configuration.

## Recorded validation

The structural revision at source commit `afaf37c31` was tested on an NVIDIA
GeForce RTX 5070 Ti (compute capability 12.0, driver 610.57.04) with the
Qwen3.8 27B model and homogeneous Q8_0 target and MTP cache types. The detailed
commands, hashes, and artifacts are recorded in Experiment 020 of
[`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md).

- Both CMake modes built from the same commit. The disabled build contained no
  Q8 MMA instances and passed all 98 opted-in cases through the once-warned
  standard fallback. The enabled build passed the same 98 cases, selecting all
  registered D=64/128/256 geometries and retaining the mixed-pair and D=72
  fallbacks.
- A maintained 1,000-token MTP exactness comparison changed only the native
  permission. Prompt tokens, request semantics, all output token IDs, and
  response bytes matched exactly.
- Matched 512-token prefill screens at depths 4,096, 32,768, and 245,760 found
  changes of -1.67%, +3.45%, and +22.70%, respectively. The very-long case
  reduced the synchronized peak process allocation and CUDA compute arena by
  974 MiB. These are three-repetition screens, not cross-device performance
  guarantees.
- A one-run live MTP screen at 30,565 prompt tokens and 64 generated tokens
  improved measured generation throughput by 3.32% and reduced sampled peak
  process VRAM by 140 MiB. Acceptance, generated drafts, and replay work were
  identical. Its prefill result was 1.32% lower, within the variability of a
  single ordered pair.

Enabling the family increased `libggml-cuda.so` by 7,469,120 bytes (7.12 MiB,
4.02%) in these otherwise matched builds. This build-size cost, the mixed
short-depth performance, and the limited hardware coverage are why the build
and run-time defaults remain off.

## Limitations

- The registered pairs are same-type: `Q8_0/Q8_0`, `Q4_0/Q4_0`, `Q5_0/Q5_0`,
  and `Q6_0/Q6_0` by default, plus `Q4_1/Q4_1`, `Q5_1/Q5_1`, `Q6_1/Q6_1`,
  `Q3_0/Q3_0`, `Q3_1/Q3_1`, `Q2_0/Q2_0`, and `Q2_1/Q2_1` under
  `GGML_CUDA_FA_ALL_QUANTS`; mixed and other quantized pairs use fallback.
- The only registered equal head dimensions are 64, 128, and 256.
- Routing requires the NVIDIA Ampere MMA implementation. AMD, MUSA, older
  NVIDIA paths, and non-CUDA backends are unchanged.
- The native loader disables the existing asynchronous F16 copy pipeline and
  performs synchronous dequantization into shared memory. That is a
  deliberate first implementation, not a claim that every device's optimal
  schedule has been found.
- Compile time and CUDA-library size increase when the family is built because
  all explicit cases are emitted: 192 by default, 528 with
  `GGML_CUDA_FA_ALL_QUANTS`. Default builds pay neither cost.
- Existing Q8 quality characteristics are unchanged because the cache format
  is unchanged. This option is not a quality or memory-compression setting.

## Adding a cache type

Bit-identity is against the F16-casting route the option replaces, so the
reference is that type's cast kernel in `convert.cu`, not the vector path's
helper. The two do not always agree:

- `Q8_0` casts via `dequantize_block_q8_0_f16`, which multiplies in `half2`.
  `dequantize_V_q8_0` does the same, so the shared helper is exact here.
- `Q4_0` has no dedicated cast kernel and goes through the generic
  `dequantize_block_q4_0<half>`, which computes in float and rounds once
  (`dm = -8*d`, `y = cast<half>(d*q + dm)`). `dequantize_V_q4_0` instead biases
  the quant by `-8` as an integer and multiplies in `half2`. Both are correct
  and they differ in the last bit, so `Q4_0` carries its own dequantization.
  That helper is also limited to `ne` of 2 or 4 and cannot serve the
  16-element run the tile loader issues.

A new type therefore needs a `fattn_quant_type_traits` specialization whose
`dequant()` matches its cast kernel, an entry in
`ggml_cuda_fattn_mma_quant_type`, one line in `DECL_FATTN_MMA_QUANT_CASE_TYPES`
and in the generator's `FATTN_MMA_QUANT_TYPES`, a case in the routing switch,
and a pass in the `test-backend-ops` sweep. Tolerance-based tests are not
sufficient evidence on their own: with random inputs they would accept a loader
that permuted elements within a block. A byte-identical generation comparison
against the same build with the option off is what actually gates a
reconstructed dequantization.
