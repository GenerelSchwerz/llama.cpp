# Quantized-native CUDA FlashAttention

`--flash-attn-native-quants` lets the CUDA MMA FlashAttention kernel read a
quantized K/V cache in place instead of casting the visible attention window to
F16 first. Results are unchanged; what goes away is the transient F16 copy.

The option is off by default and only affects the CUDA backend.

## Why

The standard quantized-K/V MMA route casts the visible K and V window to F16
before the kernel runs. That temporary lives in the compute buffer and scales
with the window:

```
2 * n_kv_heads * head_dim * sizeof(F16) * visible_tokens
```

For a four-KV-head, D=256 model that is 4 KiB per visible token. The native
loaders instead dequantize the current tile straight into the shared-memory
`half2` tiles that the existing MMA body already consumes, so nothing is
materialized.

## Using it

```
llama-cli   --flash-attn-native-quants -fa on -ctk q8_0 -ctv q8_0 ...
llama-bench -fanq 1                    -fa 1  -ctk q8_0 -ctv q8_0 ...
```

The environment variable `LLAMA_ARG_FLASH_ATTN_NATIVE_QUANTS` sets the same
option. `GGML_CUDA_FATTN_NATIVE_VERBOSE=1` logs one line per launch that
actually takes the native route, which is how a measurement can state that it
exercised the kernel rather than inferring it from throughput.

Requesting the option where no native kernel exists is not an error: the request
is declined, the established F16-casting path runs, and the backend logs one
warning naming the K/V types and head dimensions it could not serve.

## Where it applies

`ggml_cuda_fattn_native_applies()` in `ggml/src/ggml-cuda/fattn.cu` requires all
of:

- the graph opted in for this node (the runtime option);
- an NVIDIA device using the Ampere MMA implementation;
- a compiled native K/V pair (see the tiers below);
- equal head dimensions of 64, 128 or 256, or symmetric Q4_0 or Q8_0 D=512 with GQA enabled, a GQA ratio above 4, and a query batch above 4;
- `logit_softcap == 0`.

Anything else keeps the standard path. In particular the dispatcher checks the
vector conditions first, so single-token quantized decode still takes the
existing vector kernel rather than being displaced onto the new route.

## Compiled type tiers

The cache-type inventory lives in exactly one place,
`ggml/src/ggml-cuda/fattn-mma-quant-types.h`:

| Type | Tier | Compiled by |
|---|---|---|
| `q8_0` | DEFAULT | every CUDA FlashAttention build |
| `q4_0` | DEFAULT | every CUDA FlashAttention build |
| `q4_1` | EXTRA | `GGML_CUDA_FA_ALL_QUANTS=ON` |
| `q5_0` | EXTRA | `GGML_CUDA_FA_ALL_QUANTS=ON` |
| `q5_1` | EXTRA | `GGML_CUDA_FA_ALL_QUANTS=ON` |

The tiers mirror `ggml_cuda_fattn_kv_type_supported()`: a default build only
ever sees `q4_0` and `q8_0` caches, so native kernels for the other types would
be dead code there.

Mixed K/V pairs follow the same rule. `ggml_cuda_get_best_fattn_kernel()`
declines `K->type != V->type` unless `GGML_CUDA_FA_ALL_QUANTS` is set, so a
default build compiles only the symmetric kernel per type. The all-quants build
adds one runtime-V kernel per K type, which covers every mixed pair of that K.
That is what keeps coverage linear: five types cover all 25 ordered pairs with
ten kernels per tile shape, not 25.

The D=512 Q4_0 and Q8_0 specializations are symmetric-only. Mixed and other-type D=512 requests stay on the materializing path until those kernels have separate performance evidence.

`fattn-mma-quant-types.h` is read by the route predicate, the extern
declarations, the host K dispatch, the device runtime-V selection, the instance
generator, CMake's extra-tier source filter, and the `test-backend-ops`
coverage. The generator and CMake parse it textually and fail closed: an
unparsable manifest fails the configure rather than silently filtering nothing,
and the generator rejects a file-name stem that does not match its enum.

Adding a type is that one manifest line plus its `fattn_quant_type_traits`
specialization.

## Implementation

The patch changes the storage loaders and reuses the existing F16 MMA
attention/reduction body rather than copying a native-specific attention kernel.
Concretely:

- `fattn-mma-quant.cuh` holds one `fattn_quant_type_traits<T>` per type. Each
  `dequant()` reproduces that type's F16 cast path bit for bit, because the
  route it replaces is the reference. Which helper achieves that differs per
  type and is documented at each specialization.
- `flash_attn_ext_f16` gains `type_K` / `type_V` template parameters, defaulting
  to `GGML_TYPE_F16`, so the F16 instantiations are unchanged.
- A sentinel `type_V` selects the V loader from the trailing `type_V_rt` kernel
  argument instead of instantiating it. An `if constexpr` removes that
  argument's only use for every static-V instantiation.
- `launch_fattn` gained a trailing argument pack, so the tile and vector kernels
  keep their existing signature.
- Multi-stage cp.async loading is disabled for native tiles: the loader writes
  the tile itself, so there is no pipeline to stage.

## Build cost

Measured on this tree, `sm_86;sm_89`, CUDA 13.3, Release:

| Build | `libggml-cuda.so` | Delta |
|---|---:|---:|
| base, default | 99,614,224 B | |
| this branch, default | 121,767,024 B | +22,152,800 B (+21.1 MiB, +22.24%) |
| base, `GGML_CUDA_FA_ALL_QUANTS=ON` | 135,156,856 B | |
| this branch, `GGML_CUDA_FA_ALL_QUANTS=ON` | 276,719,560 B | +141,562,704 B (+135.0 MiB, +104.74%) |

The default tier has 49 Q4_0 kernels and 49 Q8_0 kernels (16 tile shapes x 3 base head dimensions, plus one D=512 8x8 kernel per type).
The all-quants tier keeps both D=512 specializations, adds 48 runtime-V kernels per default type, and carries 48 symmetric plus 48 runtime-V kernels per extra type. These
counts are read back out of the built library by
`scripts/fattn-native-inventory.py`, which checks them against the manifest.

The library-size figures are a build and distribution tradeoff. They are not a
process-VRAM cost, and they are architecture dependent; the runtime result below
is the one that should drive the decision. The default tier's +22% is paid by
every CUDA FlashAttention build, which is what makes `--flash-attn-native-quants`
usable without a rebuild. The extra tier roughly doubles the library again,
which is why it stays behind `GGML_CUDA_FA_ALL_QUANTS`.

## Validation

RTX 4070 (sm_89) and RTX 3060 (sm_86), driver 610.57.04, CUDA 13.3.

Correctness, `test-backend-ops`:

| Build | Selection | CUDA0 | CUDA1 |
|---|---|---|---|
| default | `-o FLASH_ATTN_EXT` | 3146/3146 | not rerun |
| default | `-o FLASH_ATTN_EXT -p native_quants=1` | 210/210 | not rerun |
| all-quants | `-o FLASH_ATTN_EXT` | 4572/4572 | not rerun |
| all-quants | `-o FLASH_ATTN_EXT -p native_quants=1` | 623/623 | not rerun |

With `GGML_CUDA_FATTN_NATIVE_VERBOSE=1` the all-quants focused run takes the
native route for all 25 ordered pairs of the five compiled types; the default
run takes it for `q8_0/q8_0` and `q4_0/q4_0` and correctly declines the rest.

Kernel inventory (`scripts/fattn-native-inventory.py`): the default library carries 49 Q8_0 and 49 Q4_0 symmetric kernels, and none for the extra tier; the all-quants library adds 48 runtime-V kernels per default type and carries 48 + 48 for each extra type.
Regenerating the instance files reproduces the committed ones byte for byte.

Runtime, Qwen3.8-27B-UD-IQ2_M (D=256, 24 heads, 4 KV heads), `q8_0/q8_0` cache:

| Configuration | Off | On | Delta |
|---|---:|---:|---:|
| 1 GPU, reserve compute buffer @ 16K ctx | 160.28 MiB | 138.28 MiB | -13.7% |
| 2 GPU, reserve compute buffer @ 64K ctx, per device | 633.13 MiB | 419.13 MiB | -33.8% |
| 1 GPU, pp2048 @ depth 4096 | 1194.31 t/s | 1195.39 t/s | +0.09% |
| 1 GPU, pp2048 @ depth 16384 | 1036.99 t/s | 1056.22 t/s | +1.85% |
| 2 GPU, pp2048 @ depth 0 | 986.67 t/s | 977.58 t/s | -0.92% |
| 2 GPU, tg64 @ depth 16384 | 23.46 t/s | 23.13 t/s | -1.41% |

The two-GPU arm is a heterogeneous pair (4070 + 3060) with layer split, so its
throughput numbers are dominated by the slower card and should be read as
"neutral", not as a measured regression. The memory result is the one that is
large and structural; throughput is neutral at short depth and mildly positive
at depth on a single device.

Not covered: tensor-parallel split and `-nkvo`, which abort on this tree
independently of this change.

## Scope boundary

The route is deliberately narrow, and the conditions in
`ggml_cuda_fattn_native_applies()` are the boundary. Widening it costs evidence:

- **A new cache type** owes a tile loader that is bit-identical to that type's
  F16 cast path, its manifest line, all ordered-pair coverage, and matched
  runtime allocation and performance evidence. Widening the pair matrix inside
  an already-compiled tier is not an expansion; adding to the type set is.
- **A new head geometry** (unequal `DKQ`/`DV`, or a size outside 64/128/256/512) is
  a separate measured change. The tile loaders assert alignment against the
  quant block size, and those assertions are what currently confine the route.
- **A new device family** is a separate measured change. The predicate names
  the Ampere MMA implementation because that is what was measured.
- **A non-zero `logit_softcap`** stays on the standard path on purpose:
  compiling the softcap specialization would double the generated kernels for a
  dispatch that cannot reach them.

The generic option name does not imply any of these.
