# Quantized-native CUDA FlashAttention

BeeLlama can optionally let the CUDA MMA FlashAttention kernel consume a
supported quantized K/V cache directly. This removes the temporary F16 K/V
materialization used by the standard MMA path. It does not change the
persistent cache format, cache placement, or quantization policy.

The feature is deliberately explicit at both build time and run time. Its
current type inventory is `Q8_0`, `Q4_0`, `Q5_0`, and `Q6_0` by default, at
equal head dimensions 64, 128, or 256, on NVIDIA GPUs that use the Ampere MMA
implementation. `GGML_CUDA_FA_ALL_QUANTS` adds `Q4_1`, `Q5_1`, `Q6_1`, `Q3_0`,
`Q3_1`, `Q2_0`, and `Q2_1` to the same head-dimension set. K and V need not be
the same type; which mixed pairs are covered is the pair policy below.
Unsupported cache pairs, dimensions, devices, and builds keep the standard
materializing path.

## K/V pairing

K and V are independent template parameters of the kernel and reach the tile
loader through separate calls, so any ordered pair of native types is
expressible. **The route covers all of them.** Whatever the F16-casting path
accepts for K and V, the native path accepts too, so no cache configuration has
to reason about which pairings are eligible: 121 ordered pairs over the eleven
types with `GGML_CUDA_FA_ALL_QUANTS`, 16 over the four of the default tier.

Coverage does not cost one kernel per pair. The symmetric pair is instantiated
with V as a template argument; every mixed pair shares a single kernel per K
type that selects its V loader at runtime, from a switch that runs once per K/V
tile rather than per element. So n types cost 2n kernels rather than n^2.

Nothing about a pair's direction is special-cased. `q8_0 / q2_0` and
`q2_0 / q8_0` are both native, as are all the `_0`/`_1` crossings. Whether a
given pairing is a *sensible* cache configuration is a separate question from
whether it has a kernel, and the route does not answer the first one.

One predicate drives all three consumers.
`ggml_cuda_fattn_mma_quant_pair()` in `fattn-mma-quant.cuh` is asked by the
host-side route decision and by the device-side kernel selection, and
`generate_cu_files.py` emits the matching cross product. A disagreement is a
missing-symbol link error naming the exact pair, not a silent gap.

Note that the *vector* FlashAttention path is tiered separately and is not
widened by this. `ggml_cuda_fattn_quant_pair_policy()` still describes its
band — on the bit ladder 8-6-5-4-3-2 the V type sits at K's position or up to
two below it, never above — and `ggml_cuda_get_fattn_vec_default_pairs()` in
`ggml/CMakeLists.txt` is its build-side twin.

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
| `q8_0 / q6_0` | on | Native mixed-pair MMA; K and V load different quant layouts into the same tile |
| `q4_1 / q2_0` | on | Native; K carries a min and unpacks nibbles while V has none and unpacks two-bit fields |
| `q4_0 / q8_0` | on | Native; a V more precise than its K is unusual but has a kernel like any other pair |
| `q8_0 / q8_0`, `logit_softcap != 0` | on | Standard materializing fallback; the native route requires a zero softcap |
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
- The native route specializes that shared case with an independent type for K
  and for V; it does not carry a copied launcher or a copied selector tree. The
  dispatcher switches on the two types in sequence and guards the inner switch
  with the pair predicate, so the compiler only reaches kernels that were
  actually instantiated.
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
  tile inventory. It emits one translation unit per K type and `(ncols1,
  ncols2)` shape. The extra K dimension relative to the F16 and KVarN convention
  is what keeps each translation unit near its former size now that the pair
  matrix is 4.4x the cases the symmetric matrix had, so the build still
  parallelises instead of serialising behind a few very large files.
- The explicit instantiation declarations live in `fattn-mma-quant-decl.cuh`
  rather than in `fattn-mma-f16.cuh`, and only `fattn.cu` includes them. They are
  the full unfiltered cross product; declaring the pairs the policy rejects costs
  nothing because they are never odr-used, and it removes any chance of the
  declaration list and the generator disagreeing.
- CMake includes those generated sources only when
  `GGML_CUDA_FATTN_Q8_NATIVE=ON`.

There are 16 tile shapes and three supported head dimensions, and two kernels
per type: one specializing V for the symmetric pair, one selecting V at runtime
for every mixed pair. The default four-type registration (`Q8_0`, `Q4_0`,
`Q5_0`, `Q6_0`) is 384 explicit template cases and covers its 16 ordered pairs;
`GGML_CUDA_FA_ALL_QUANTS` adds the remaining seven types for 1,056 cases
covering all 121 pairs. Adding an nth type costs two more kernels, not `2n-1`
more pairs, so the inventory is linear in the type list.

The split exists for a measured reason. Making V runtime for the symmetric pair
as well cuts the inventory further, to 528 cases, but pushes the `D = 256`
kernel from 16 to 48 bytes of register spill and costs `-0.73%` (Welch
`t = -2.88`) on `q8_0/q8_0` prefill — enough to put the native path behind the
materializing path it replaces. With the symmetric pair specialized, the same
measurement is `+1.09%` (`t = +3.87`). A future native pair is not inferred from the standard quant
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

- Every ordered pair of compiled types is registered, so no quantized K/V
  combination that the F16-casting path accepts falls back for lack of a pair.
  What still falls back is an unsupported *type*, head dimension, device, or
  build.
- A mixed pair buys allocation, not throughput. Measured for `q8_0 / q6_0` at
  depth 32,768, prefill is `+0.22%` with `t = 0.877`, which is neutral; the
  reserve-time `CUDA0 compute` buffer falls 46.6% at that depth and 54.6% at
  131,072. The removed transient is an F16 copy of the attention window, whose
  size does not depend on the source quantization.
- The only registered equal head dimensions are 64, 128, and 256.
- Routing requires the NVIDIA Ampere MMA implementation. AMD, MUSA, older
  NVIDIA paths, and non-CUDA backends are unchanged.
- The native loader disables the existing asynchronous F16 copy pipeline and
  performs synchronous dequantization into shared memory. That is a
  deliberate first implementation, not a claim that every device's optimal
  schedule has been found.
- Compile time and CUDA-library size increase when the family is built: 384
  explicit cases by default, 1,056 with `GGML_CUDA_FA_ALL_QUANTS`. Coverage of
  every ordered pair costs two kernels per type rather than one per pair,
  because only the symmetric pair specializes V at compile time. Each case is
  one device kernel rather than two, because the route requires a zero logit
  softcap and so never names the softcap specialization. Default llama.cpp
  builds pay neither cost.
- Existing Q8 quality characteristics are unchanged because the cache format
  is unchanged. This option is not a quality or memory-compression setting.

## Known limitation: quantized K/V with a logit softcap

A non-zero `logit_softcap` combined with any quantized K/V cache aborts the
CUDA FlashAttention backend:

```
CUDA error: unspecified launch failure
ggml/src/ggml-cuda/ggml-cuda.cu:109: CUDA error
```

Since the pruning of the unreachable softcap kernels, the native route also
declines a non-zero softcap outright and leaves it on the standard path, so the
abort below is reached only through that path.

**This is not caused by the native route and is not fixed by avoiding it.** It
reproduces identically with `--flash-attn-native-quants` withheld, i.e. through
the ordinary F16-materializing path, and F16 K/V with the same softcap passes
120/120. It affects models that apply attention logit softcapping, such as the
Gemma-2 family, whenever their K/V cache is quantized.

It went unnoticed because nothing exercised the combination: before the test
additions described below, no `FLASH_ATTN_EXT` case in the suite paired a
quantized K/V type with a non-zero softcap, with or without the native
permission.

`test-backend-ops` deliberately does not cover it. The failure is an abort
rather than a wrong result, so a case for it would take down the entire
`FLASH_ATTN_EXT` sweep instead of reporting one failure. Reproduce it by
adding a case such as

```cpp
test_cases.emplace_back(new test_flash_attn_ext(
            64, 64, 4, {4, 1}, 512, 16, true, false, 0.0f, 10.0f, GGML_PREC_F32,
            GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, /* native_quants = */ false));
```

Fixing it is out of scope for the quantized-native work, which neither
introduces nor worsens it.

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
`dequant()` matches its cast kernel, an entry in `FATTN_MMA_QUANT_TYPES` in
`fattn-mma-quant.cuh`, one line in each of `DECL_FATTN_MMA_QUANT_CASE_V` and
`DECL_FATTN_MMA_QUANT_CASE_TYPES` in `fattn-mma-quant-decl.cuh`, an entry in the
generator's `FATTN_MMA_QUANT_TYPES`, and an entry in the `native_types` array in
the `test-backend-ops` sweep, which pairs it against every existing type
automatically. It also wants a rung in `ggml_cuda_fattn_quant_pair_rank` if the
vector path should carry it. The routing switch needs no change: it expands from
the same type list. Tolerance-based tests are not
sufficient evidence on their own: with random inputs they would accept a loader
that permuted elements within a block. A byte-identical generation comparison
against the same build with the option off is what actually gates a
reconstructed dequantization.
