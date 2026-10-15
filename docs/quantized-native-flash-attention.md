# Quantized-native CUDA FlashAttention

For a set of geometries, the CUDA MMA FlashAttention kernel reads a quantized K/V cache in place instead of casting the visible attention window to F16 first. The transient F16 copy goes away. On Ampere and Ada the route is faster than the F16-casting path for prompt processing, and several times faster for decode and for the small query batches of speculative decoding.

It is opt-in. `--flash-attn-native-quants` turns it on (`llama-bench -fanq 0,1` to compare); without it every FlashAttention node keeps the established path. When it is on, the route is taken wherever a kernel is compiled for the geometry and nothing else is consulted, at every query batch size.

The K/V values the kernel computes with are the ones the F16 cast produces: for `q4_0`, `q5_0` and `q8_0` bit for bit, for `q4_1` and `q5_1` up to the last bit of a converted value (see **Conversion**). The attention output is not bit identical to the F16-casting path. Some query batches use another tile shape, which changes the summation order, and on `sm_86`, `sm_89` and `sm_120` the KQ products are partly summed in half precision (see **KQ accumulation**). Measured against an F16 cache, the output stays within the run-to-run noise of the F16-casting path (see **Validation**).

## Why

The standard quantized-K/V MMA route casts the visible K and V window to F16 before the kernel runs and reads that copy back:

```
2 * n_kv_heads * head_dim * sizeof(F16) * visible_tokens
```

For a model with four KV heads and D=256 that is 4 KiB per visible token, allocated in the compute buffer, written once and read once. The native route instead copies the raw quantized rows into shared memory and converts them there, so nothing is materialized in device memory and K/V is read at 4.5 to 8.5 bits per value instead of 16.

For decode and small query batches the difference is larger. These steps are bound by reading K/V. Up to 2 query tokens on Ada and 1 on Turing and Ampere, the vector kernel reads the quantized cache directly, but once per Q head, so at a GQA ratio of 6 every K/V row is requested six times. Above that, the cast reads the cache, writes the F16 copy and the kernel reads the copy again. The native kernels read every K/V row once per KV head in both cases.

## Where it applies

The graph opts in per node via `ggml_flash_attn_ext_set_native_quants()`, which `llm_graph_context::build_attn_mha()` sets from the context parameter. `ggml_cuda_fattn_native_supported()` in `ggml/src/ggml-cuda/fattn.cu` decides whether a kernel exists and returns the tile shape it uses; the dispatcher asks it before the vector kernel. Common to every row:

- an NVIDIA device with Turing MMA or newer (`sm_75`+);
- `logit_softcap == 0`;
- the same native cache type for K and V;
- the GQA optimizations apply (mask present, no ALiBi, padded K/V, aligned strides);
- K/V data and strides aligned to the copy size of a staged row (16, 8 or 4 bytes, see **Conversion**), which a KV cache always is;
- no sparse K/V set (`n_kv_max == 0`): the sparse gather exists only in the F16 kernel.

| Head dim | GQA ratio | Query batch | Tile (sm_80+) | Tile (Turing) |
|---|---|---|---|---|
| 64, 128, 256 | 3 or more | 1 | 1x8 | 1x8 |
| 64, 128, 256 | 3 or more | 2 | 2x8 | 2x8 |
| 64, 128, 256 | 3 or more | 3-4 | 4x8 | 4x8 |
| 64, 128, 256 | 3, 4 | 5-8 | 8x8 | 4x8 |
| 64, 128, 256 | even above 4, not a multiple of 8 | 5-16 | 8x8 | 4x8 |
| 64, 128, 256 | odd above 4, or a multiple of 8 | > 4 | 8x8 | 4x8 |
| 64, 128, 256 | 2 | 1-4 | 4x2 | 4x2 |
| 64, 128, 256 | 2, 3, 4 | 5-16 (GQA 2), 9-16 (GQA 3, 4) | 16x2 | 16x2 |
| 64, 128, 256 | 3 and up, not a multiple of 8, not odd above 4 | > 16 | 32x2 | 16x2 |
| 64, 128, 256 | 2 | > 16 | 32x2 | 16x2 |
| 512 | > 4 | 1, 2, 3-4 | 1x8, 2x8, 4x8 | 4x8 |
| 512 | > 4 | > 4 | 8x8 | 4x8 |

D=64, D=128 and D=256 have kernels for `q4_0`, `q4_1`, `q5_0`, `q5_1` and `q8_0`, D=512 for `q4_0` and `q8_0`. A tile `AxB` covers A query tokens of B Q heads that share a KV head. Small query batches are bound by reading K/V, and a tile of 8 heads reads every K/V row once per KV head. Below 8 heads per KV head the unused ones cost arithmetic only, which these batches can afford, so up to 4 query tokens take 8 heads per tile at any GQA ratio from 3, and up to 8 tokens at GQA 3 and 4. GQA 2 takes 2 heads per tile, which also reads K/V once. Larger query batches of a GQA ratio that is not a multiple of 8 take 2 heads per tile, because 8 heads per tile would compute zeros for the missing ones (25% of the tile at GQA 6, half of it at GQA 4); an odd ratio above 4 keeps 8 heads per tile, as the F16 path does. A GQA ratio of 1 keeps the F16 path: there the vector kernel already reads K/V once per KV head for decode.

`fattn-mma-quant-decl.cuh` declares exactly these shapes and nothing else, so a disagreement with `ggml_cuda_fattn_native_supported()` is a link error.

## Compiled type tiers

The cache-type inventory lives in `ggml/src/ggml-cuda/fattn-mma-quant-types.h`:

| Type | Tier | Compiled by |
|---|---|---|
| `q8_0` | DEFAULT | every CUDA FlashAttention build |
| `q4_0` | DEFAULT | every CUDA FlashAttention build |
| `q4_1` | EXTRA | `GGML_CUDA_FA_QUANTS` with `q4_1-q4_1`, or `all` |
| `q5_0` | EXTRA | `GGML_CUDA_FA_QUANTS` with `q5_0-q5_0`, or `all` |
| `q5_1` | EXTRA | `GGML_CUDA_FA_QUANTS` with `q5_1-q5_1`, or `all` |

An extra type follows its own `type-type` pair, the same selection that compiles its vector kernel. Without its pair a cache of that type still runs, on the F16-casting path. That gives 50 kernels in a default build, 21 more per selected extra pair and 113 with `all`. With all five types and `86-real;89-real`, `libggml-cuda.so` grows from 71.0 to 73.3 MB against the previous revision with D=256 and D=512 only. `scripts/fattn-native-inventory.py --fa-quants <value>` reads the built library back and fails on a missing, unexpected or duplicated kernel. The route is CUDA only; HIP and MUSA do not build the instances.

## Implementation

`flash_attn_ext_f16` gains `type_K` / `type_V` template parameters, defaulting to `GGML_TYPE_F16`, so the F16 instantiations are unchanged. Only the way the half2 K/V tiles in shared memory are filled differs, plus the KQ accumulation below; the softmax, the VKQ MMA and the combine code are shared. How the tiles are filled depends on the architecture and the configuration (`ggml_cuda_fattn_mma_nstages_quant()`):

- **Two-stage (Ampere and newer, D=64 to D=256).** The F16 path loads K/V with a two-stage cp.async pipeline here. The native route keeps it: the raw quantized rows of the next tile are copied to shared memory with cp.async while the current tile computes. Rows are copied as they are, with consecutive threads on consecutive chunks of 16, 8 or 4 bytes, the largest size that divides both the row and its pitch in shared memory: 16 bytes at D=256 and D=512, 8 bytes for the 72-, 88- and 136-byte rows of `q4_0`, `q5_0` and `q8_0` at D=128, 4 bytes for their rows at D=64. K and V get separate half2 tiles where shared memory allows two blocks per SM: V of the current tile is converted while the KQ MMA reads K, and K of the next tile while the VKQ MMA reads V, so the conversion runs in the shadow of the tensor cores. For `q8_0` the mask tile does not fit next to the four K/V tiles, so each thread loads its mask values into registers instead.
- **Chunked (Ampere and newer, D=512).** K and V are processed in two chunks of 256 values each. Two raw chunk buffers rotate through K hi, K lo, V lo, V hi, each chunk copied two chunks ahead, so no global memory latency is left on the critical path. The F16 path waits for each chunk here.
- **Synchronous (Turing).** Without cp.async, each thread reads its quant blocks from global memory and writes the converted values to the tile directly.

### Conversion

A staged block is read from shared memory as 4-byte words. Blocks of 18, 22 and 34 bytes (`q4_0`, `q5_0`, `q8_0`) alternate between aligned and 2-byte misaligned; even and odd blocks are converted in separate passes, so the misalignment is a compile-time constant and costs at most one funnel shift per word. The 8 lanes of a quarter warp take 8 consecutive rows of the same block: through the tile swizzle their 16-byte stores hit 8 distinct bank groups. With 4 or more blocks per pass the four quarter warps take four blocks of the same rows; with fewer (D=128 and D=64) they take 2 or 4 groups of rows 8 apart. A staged row of (4/q)*odd words, where q is the number of row groups, keeps the reads of all 32 lanes on distinct banks: rows of 2 mod 4 byte blocks have that length as they are, rows of 4-byte aligned blocks are padded to it. A value becomes half2 by placing quant bytes into `0x64XX` (half(1024 + q)) with `__byte_perm`, subtracting the bias and scaling, which reproduces the F16 cast exactly. `q4_1` and `q5_1` compute `q*d + m` in half2 with one rounding where the cast rounds to float first; the two can differ in the last bit.

### KQ accumulation

GeForce cards run MMAs with float accumulators at half the rate of half accumulators. On `sm_86`, `sm_89` and `sm_120` the native kernels therefore sum KQ in half over 32 values of D (two m16n8k16 MMAs) and add those partial sums in float. This saves up to 11% of kernel time at D=256 and 10-15% at D=512 on the 3060 and 4070. A partial sum over 32 values stays far below the half range for realistic logits, and its rounding error is small against the error the quantized cache itself introduces. Longer half runs were measured and rejected: summing all 256 values in half saves another 7-9% of kernel time, but one sensitive token in the wikitext sample moved from a KL divergence of at most 2.3 to between 5 and 18 against the F16 cache. `sm_120` (RTX 50) is included on the same grounds but has not been measured here. Other architectures keep float: `sm_80` runs both accumulators at the same rate. Professional cards, which run float accumulators at full rate, gain nothing from it. Defining `GGML_FATTN_QUANT_KQ_FLOAT` at build time (`-DCMAKE_CUDA_FLAGS=-DGGML_FATTN_QUANT_KQ_FLOAT`) keeps float accumulation on every architecture, which is the comparison to run on a card not measured here. The VKQ product is summed in half, as in the F16 path.

### Shared MMA path

Two changes to the code both paths share make the F16 path faster as well, with identical results:

- `load_ldmatrix_swizzled` and `load_ldmatrix_trans_swizzled` pass a 32-bit shared address to `ldmatrix`. With a generic pointer the compiler added and subtracted the shared window base for every load.
- Where K is the B operand of the KQ mma (wide tiles), it is loaded with the 8x8 matrices in the register order the mma takes them, which removes the register moves that formed the operand pairs.

## Validation

RTX 4070 (`sm_89`, Ada) and RTX 3060 (`sm_86`, Ampere), CUDA 13.4, `CMAKE_CUDA_ARCHITECTURES=86-real;89-real`, all five types selected.

- `test-backend-ops -o FLASH_ATTN_EXT`: 3993/3993 on both GPUs; `-o GATED_DELTA_NET` (shares the ldmatrix helpers): 43/43.
- `test-backend-ops -o NATIVE_QUANT_EQUIVALENCE`: 96/96 on both GPUs. Each case computes the attention over the quantized cache and over an F16 copy of it in one graph, requires the two CUDA results to agree within an NMSE of 1e-4, and checks which path the dispatcher took through the backend's native-launch counter. The largest NMSE between the two halves is 2.9e-5 (n_kv=16384), a tenth of the difference between the CPU's own two halves (3.5e-4, since the CPU takes quantized dot products for a quantized cache), and the same as with float KQ accumulation. The cases cover every tile shape at D=64, 128, 256 and 512, decode and small query batches, GQA 2, 3, 4, 6, 8 and 16, attention sinks, two sequences and query batches large enough for the mask scan; seven cases assert the F16-casting path where no kernel exists.
- `compute-sanitizer` racecheck, synccheck and memcheck: no errors or warnings on the native kernels, on both GPUs, for the two-stage, shadow and chunked modes, the 8-, 16- and 32-column tiles at decode and small batches, the 8- and 4-byte staging at D=128 and D=64, and the half KQ accumulation at every head size. The one racecheck warning is in upstream's F16 vector kernel, which computes the reference half of a decode case on Ada; `master` reports the same warning there.
- `scripts/fattn-native-inventory.py`: 113 declared kernels with all types selected, all compiled, nothing else.

KL divergence of Qwen3.8-27B IQ3_XXS (D=256, GQA 6) against the same model with an F16 cache, wikitext-2 test, `-c 4096 -b 1024`. Sample 1 is the first 6 chunks, sample 2 4 chunks from the middle of the file. "native, float KQ" is the route before the half accumulation; the column the route ships with is "native".

| Cache | Sample | Path | Mean KLD | 99% KLD | Max KLD | Same top token | PPL ratio |
|---|---|---|--:|--:|--:|--:|--:|
| `q8_0` | 1 | F16-casting | 0.000953 | 0.00609 | 1.62 | 98.82% | 1.0029 |
| `q8_0` | 1 | native, float KQ | 0.000735 | 0.00597 | 0.52 | 98.80% | 1.0021 |
| `q8_0` | 1 | native | 0.000818 | 0.00632 | 0.91 | 98.75% | 1.0024 |
| `q8_0` | 2 | F16-casting | 0.00213 | 0.00757 | 4.02 | 98.82% | 1.0015 |
| `q8_0` | 2 | native, float KQ | 0.00310 | 0.00764 | 6.35 | 98.88% | 1.0026 |
| `q8_0` | 2 | native | 0.00195 | 0.00679 | 6.31 | 98.85% | 1.0014 |
| `q4_0` | 1 | F16-casting | 0.00346 | 0.0260 | 2.28 | 97.57% | 1.0025 |
| `q4_0` | 1 | native, float KQ | 0.00340 | 0.0270 | 2.17 | 97.46% | 1.0016 |
| `q4_0` | 1 | native | 0.00334 | 0.0263 | 3.75 | 97.68% | 1.0018 |

The mean and the maximum are dominated by a few sensitive tokens whose divergence changes with any change in summation order, as the two float paths show.

D=128 models against an F16 cache, wikitext-2 test, first 4 chunks, `-c 4096 -b 1024`, RTX 4070 (half KQ accumulation):

| Model | Cache | Path | Mean KLD | 99% KLD | Max KLD | Median KLD | Same top token |
|---|---|---|--:|--:|--:|--:|--:|
| Llama-3.2-3B | `q8_0` | F16-casting | 0.00176 | 0.0138 | 0.62 | 0.00096 | 98.131% |
| Llama-3.2-3B | `q8_0` | native | 0.00173 | 0.0136 | 0.10 | 0.00099 | 98.119% |
| Llama-3.2-3B | `q4_0` | F16-casting | 0.01888 | 0.1568 | 3.39 | 0.00949 | 93.942% |
| Llama-3.2-3B | `q4_0` | native | 0.01795 | 0.1412 | 2.59 | 0.00907 | 93.649% |
| Qwen3-4B | `q8_0` | F16-casting | 0.00360 | 0.0405 | 0.80 | 0.00094 | 97.643% |
| Qwen3-4B | `q8_0` | native | 0.00378 | 0.0430 | 2.01 | 0.00091 | 97.912% |
| Qwen3-4B | `q4_0` | F16-casting | 0.04261 | 0.5097 | 11.63 | 0.01121 | 92.758% |
| Qwen3-4B | `q4_0` | native | 0.04368 | 0.5508 | 10.54 | 0.01160 | 92.452% |

The native route stays within the run-to-run noise of the F16-casting path here too: the means differ by less than their standard errors (0.00176 +- 0.00009 against 0.00173 +- 0.00004 for Llama `q8_0`, 0.00360 +- 0.00018 against 0.00378 +- 0.00029 for Qwen3 `q8_0`), and the medians are the same. The median KLD is the same for all three paths: 0.00028 and 0.00030 for `q8_0` on the two samples, 0.0012 for `q4_0`.

Greedy decoding with Qwen3.8-27B UD-IQ4_XS on 4070 + 3060, 1442 prompt tokens and 256 generated tokens: without the flag the output is byte identical to upstream `master` for `q8_0` and `q4_0` caches. With it the text stays coherent and departs from upstream after 25 to 65 words, as any change in summation order does.

Turing: the kernels compile for `sm_75` and use the synchronous loaders. A build with `CMAKE_CUDA_ARCHITECTURES=75-virtual` runs the Turing code on the RTX 4070 (the host takes the Turing tiles since sm_75 is the highest compiled architecture, the device code is the sm_75 PTX): `NATIVE_QUANT_EQUIVALENCE` 96/96. That checks the results, not the speed on a Turing card. At D=512 Turing batches K in 96 half2 below 32 columns, which the synchronous loader cannot split into quant blocks, so Turing takes the 4x8 tile for up to 4 query tokens there.

## Throughput

`test-backend-ops perf -o FLASH_ATTN_EXT -p expected_native`, us/run, lower is better, on an otherwise idle machine. "F16 path" is this branch without the flag, cast kernel included; the shared MMA changes make it 1-5% faster than upstream `master`. Rows where a type has no kernel are left out.

### Prompt processing


**RTX 3060 (Ampere, sm_86)**

| type | D | GQA | n_kv | n_q | F16 path | native | change |
|---|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 1024 | 512 | 608 | 398 | -34.6% |
| `q4_0` | 256 | 6 | 1024 | 2048 | 2157 | 1562 | -27.6% |
| `q4_0` | 256 | 6 | 4096 | 512 | 2280 | 1460 | -36.0% |
| `q4_0` | 256 | 6 | 4096 | 2048 | 8204 | 5824 | -29.0% |
| `q4_0` | 256 | 6 | 16384 | 512 | 9030 | 5723 | -36.6% |
| `q4_0` | 256 | 6 | 16384 | 2048 | 32458 | 22838 | -29.6% |
| `q4_0` | 256 | 2 | 1024 | 512 | 368 | 287 | -22.0% |
| `q4_0` | 256 | 2 | 1024 | 2048 | 1219 | 1067 | -12.5% |
| `q4_0` | 256 | 2 | 4096 | 512 | 1367 | 1046 | -23.5% |
| `q4_0` | 256 | 2 | 4096 | 2048 | 4565 | 3951 | -13.4% |
| `q4_0` | 256 | 2 | 16384 | 512 | 5374 | 4082 | -24.0% |
| `q4_0` | 256 | 2 | 16384 | 2048 | 17977 | 15499 | -13.8% |
| `q4_0` | 512 | 16 | 1024 | 512 | 996 | 756 | -24.1% |
| `q4_0` | 512 | 16 | 1024 | 2048 | 3779 | 2890 | -23.5% |
| `q4_0` | 512 | 16 | 4096 | 512 | 3641 | 2674 | -26.5% |
| `q4_0` | 512 | 16 | 4096 | 2048 | 13773 | 10191 | -26.0% |
| `q4_0` | 512 | 16 | 16384 | 512 | 14265 | 10380 | -27.2% |
| `q4_0` | 512 | 16 | 16384 | 2048 | 53893 | 39494 | -26.7% |
| `q4_0` | 128 | 8 | 1024 | 512 | 340 | 283 | -16.8% |
| `q4_0` | 128 | 8 | 1024 | 2048 | 1186 | 1047 | -11.7% |
| `q4_0` | 128 | 8 | 4096 | 512 | 1254 | 1021 | -18.6% |
| `q4_0` | 128 | 8 | 4096 | 2048 | 4407 | 3801 | -13.8% |
| `q4_0` | 128 | 8 | 16384 | 512 | 4910 | 3963 | -19.3% |
| `q4_0` | 128 | 8 | 16384 | 2048 | 17228 | 14795 | -14.1% |
| `q4_0` | 128 | 4 | 1024 | 512 | 365 | 295 | -19.0% |
| `q4_0` | 128 | 4 | 1024 | 2048 | 1222 | 1107 | -9.4% |
| `q4_0` | 128 | 4 | 4096 | 512 | 1358 | 1092 | -19.6% |
| `q4_0` | 128 | 4 | 4096 | 2048 | 4559 | 4114 | -9.8% |
| `q4_0` | 128 | 4 | 16384 | 512 | 5334 | 4267 | -20.0% |
| `q4_0` | 128 | 4 | 16384 | 2048 | 17887 | 16128 | -9.8% |
| `q4_0` | 64 | 8 | 1024 | 512 | 355 | 328 | -7.7% |
| `q4_0` | 64 | 8 | 1024 | 2048 | 1234 | 1269 | +2.8% |
| `q4_0` | 64 | 8 | 4096 | 512 | 1309 | 1189 | -9.2% |
| `q4_0` | 64 | 8 | 4096 | 2048 | 4593 | 4620 | +0.6% |
| `q4_0` | 64 | 8 | 16384 | 512 | 5127 | 4642 | -9.5% |
| `q4_0` | 64 | 8 | 16384 | 2048 | 17999 | 18009 | +0.1% |
| `q8_0` | 256 | 6 | 1024 | 512 | 616 | 448 | -27.3% |
| `q8_0` | 256 | 6 | 1024 | 2048 | 2196 | 1763 | -19.7% |
| `q8_0` | 256 | 6 | 4096 | 512 | 2324 | 1632 | -29.8% |
| `q8_0` | 256 | 6 | 4096 | 2048 | 8305 | 6475 | -22.0% |
| `q8_0` | 256 | 6 | 16384 | 512 | 9145 | 6386 | -30.2% |
| `q8_0` | 256 | 6 | 16384 | 2048 | 32799 | 25426 | -22.5% |
| `q8_0` | 256 | 2 | 1024 | 512 | 373 | 323 | -13.4% |
| `q8_0` | 256 | 2 | 1024 | 2048 | 1235 | 1203 | -2.6% |
| `q8_0` | 256 | 2 | 4096 | 512 | 1390 | 1176 | -15.4% |
| `q8_0` | 256 | 2 | 4096 | 2048 | 4628 | 4402 | -4.9% |
| `q8_0` | 256 | 2 | 16384 | 512 | 5462 | 4605 | -15.7% |
| `q8_0` | 256 | 2 | 16384 | 2048 | 18184 | 17219 | -5.3% |
| `q8_0` | 512 | 16 | 1024 | 512 | 999 | 786 | -21.3% |
| `q8_0` | 512 | 16 | 1024 | 2048 | 3792 | 3024 | -20.2% |
| `q8_0` | 512 | 16 | 4096 | 512 | 3659 | 2812 | -23.1% |
| `q8_0` | 512 | 16 | 4096 | 2048 | 13825 | 10762 | -22.2% |
| `q8_0` | 512 | 16 | 16384 | 512 | 14281 | 10909 | -23.6% |
| `q8_0` | 512 | 16 | 16384 | 2048 | 53913 | 41681 | -22.7% |
| `q8_0` | 128 | 8 | 1024 | 512 | 342 | 330 | -3.4% |
| `q8_0` | 128 | 8 | 1024 | 2048 | 1194 | 1223 | +2.5% |
| `q8_0` | 128 | 8 | 4096 | 512 | 1261 | 1198 | -5.1% |
| `q8_0` | 128 | 8 | 4096 | 2048 | 4439 | 4452 | +0.3% |
| `q8_0` | 128 | 8 | 16384 | 512 | 4961 | 4654 | -6.2% |
| `q8_0` | 128 | 8 | 16384 | 2048 | 17376 | 17320 | -0.3% |
| `q8_0` | 128 | 4 | 1024 | 512 | 369 | 333 | -9.6% |
| `q8_0` | 128 | 4 | 1024 | 2048 | 1233 | 1265 | +2.6% |
| `q8_0` | 128 | 4 | 4096 | 512 | 1371 | 1211 | -11.7% |
| `q8_0` | 128 | 4 | 4096 | 2048 | 4599 | 4583 | -0.3% |
| `q8_0` | 128 | 4 | 16384 | 512 | 5383 | 4728 | -12.2% |
| `q8_0` | 128 | 4 | 16384 | 2048 | 18058 | 17911 | -0.8% |
| `q8_0` | 64 | 8 | 1024 | 512 | 357 | 347 | -3.0% |
| `q8_0` | 64 | 8 | 1024 | 2048 | 1245 | 1344 | +7.9% |
| `q8_0` | 64 | 8 | 4096 | 512 | 1319 | 1264 | -4.2% |
| `q8_0` | 64 | 8 | 4096 | 2048 | 4634 | 4909 | +5.9% |
| `q8_0` | 64 | 8 | 16384 | 512 | 5166 | 4926 | -4.6% |
| `q8_0` | 64 | 8 | 16384 | 2048 | 18146 | 19139 | +5.5% |
| `q4_1` | 256 | 6 | 1024 | 512 | 618 | 419 | -32.2% |
| `q4_1` | 256 | 6 | 1024 | 2048 | 2200 | 1640 | -25.5% |
| `q4_1` | 256 | 6 | 4096 | 512 | 2334 | 1526 | -34.6% |
| `q4_1` | 256 | 6 | 4096 | 2048 | 8356 | 6037 | -27.8% |
| `q4_1` | 256 | 6 | 16384 | 512 | 9156 | 5935 | -35.2% |
| `q4_1` | 256 | 6 | 16384 | 2048 | 32756 | 23504 | -28.2% |
| `q4_1` | 256 | 2 | 1024 | 512 | 374 | 300 | -19.8% |
| `q4_1` | 256 | 2 | 1024 | 2048 | 1231 | 1110 | -9.9% |
| `q4_1` | 256 | 2 | 4096 | 512 | 1383 | 1088 | -21.3% |
| `q4_1` | 256 | 2 | 4096 | 2048 | 4621 | 4073 | -11.9% |
| `q4_1` | 256 | 2 | 16384 | 512 | 5469 | 4247 | -22.3% |
| `q4_1` | 256 | 2 | 16384 | 2048 | 18159 | 15846 | -12.7% |
| `q4_1` | 128 | 8 | 1024 | 512 | 341 | 284 | -16.8% |
| `q4_1` | 128 | 8 | 1024 | 2048 | 1192 | 1049 | -12.0% |
| `q4_1` | 128 | 8 | 4096 | 512 | 1257 | 1025 | -18.5% |
| `q4_1` | 128 | 8 | 4096 | 2048 | 4412 | 3817 | -13.5% |
| `q4_1` | 128 | 8 | 16384 | 512 | 4926 | 3973 | -19.3% |
| `q4_1` | 128 | 8 | 16384 | 2048 | 17300 | 14825 | -14.3% |
| `q4_1` | 128 | 4 | 1024 | 512 | 370 | 289 | -21.8% |
| `q4_1` | 128 | 4 | 1024 | 2048 | 1224 | 1086 | -11.2% |
| `q4_1` | 128 | 4 | 4096 | 512 | 1362 | 1046 | -23.2% |
| `q4_1` | 128 | 4 | 4096 | 2048 | 4574 | 3942 | -13.8% |
| `q4_1` | 128 | 4 | 16384 | 512 | 5343 | 4064 | -23.9% |
| `q4_1` | 128 | 4 | 16384 | 2048 | 17911 | 15355 | -14.3% |
| `q4_1` | 64 | 8 | 1024 | 512 | 355 | 334 | -5.9% |
| `q4_1` | 64 | 8 | 1024 | 2048 | 1235 | 1283 | +4.0% |
| `q4_1` | 64 | 8 | 4096 | 512 | 1313 | 1166 | -11.2% |
| `q4_1` | 64 | 8 | 4096 | 2048 | 4608 | 4532 | -1.6% |
| `q4_1` | 64 | 8 | 16384 | 512 | 5134 | 4511 | -12.1% |
| `q4_1` | 64 | 8 | 16384 | 2048 | 18015 | 17549 | -2.6% |
| `q5_0` | 256 | 6 | 1024 | 512 | 620 | 426 | -31.3% |
| `q5_0` | 256 | 6 | 1024 | 2048 | 2202 | 1669 | -24.2% |
| `q5_0` | 256 | 6 | 4096 | 512 | 2344 | 1556 | -33.6% |
| `q5_0` | 256 | 6 | 4096 | 2048 | 8348 | 6166 | -26.1% |
| `q5_0` | 256 | 6 | 16384 | 512 | 9225 | 6084 | -34.1% |
| `q5_0` | 256 | 6 | 16384 | 2048 | 32989 | 24164 | -26.8% |
| `q5_0` | 256 | 2 | 1024 | 512 | 378 | 306 | -19.2% |
| `q5_0` | 256 | 2 | 1024 | 2048 | 1240 | 1134 | -8.6% |
| `q5_0` | 256 | 2 | 4096 | 512 | 1409 | 1113 | -21.0% |
| `q5_0` | 256 | 2 | 4096 | 2048 | 4650 | 4183 | -10.0% |
| `q5_0` | 256 | 2 | 16384 | 512 | 5534 | 4331 | -21.7% |
| `q5_0` | 256 | 2 | 16384 | 2048 | 18213 | 16286 | -10.6% |
| `q5_0` | 128 | 8 | 1024 | 512 | 343 | 306 | -10.7% |
| `q5_0` | 128 | 8 | 1024 | 2048 | 1192 | 1133 | -5.0% |
| `q5_0` | 128 | 8 | 4096 | 512 | 1262 | 1122 | -11.1% |
| `q5_0` | 128 | 8 | 4096 | 2048 | 4426 | 4179 | -5.6% |
| `q5_0` | 128 | 8 | 16384 | 512 | 4949 | 4389 | -11.3% |
| `q5_0` | 128 | 8 | 16384 | 2048 | 17322 | 16340 | -5.7% |
| `q5_0` | 128 | 4 | 1024 | 512 | 369 | 306 | -17.2% |
| `q5_0` | 128 | 4 | 1024 | 2048 | 1232 | 1153 | -6.4% |
| `q5_0` | 128 | 4 | 4096 | 512 | 1380 | 1122 | -18.7% |
| `q5_0` | 128 | 4 | 4096 | 2048 | 4597 | 4236 | -7.9% |
| `q5_0` | 128 | 4 | 16384 | 512 | 5417 | 4359 | -19.5% |
| `q5_0` | 128 | 4 | 16384 | 2048 | 17987 | 16540 | -8.0% |
| `q5_0` | 64 | 8 | 1024 | 512 | 357 | 341 | -4.7% |
| `q5_0` | 64 | 8 | 1024 | 2048 | 1244 | 1328 | +6.7% |
| `q5_0` | 64 | 8 | 4096 | 512 | 1324 | 1246 | -5.9% |
| `q5_0` | 64 | 8 | 4096 | 2048 | 4632 | 4858 | +4.9% |
| `q5_0` | 64 | 8 | 16384 | 512 | 5188 | 4871 | -6.1% |
| `q5_0` | 64 | 8 | 16384 | 2048 | 18138 | 18964 | +4.6% |
| `q5_1` | 256 | 6 | 1024 | 512 | 620 | 427 | -31.0% |
| `q5_1` | 256 | 6 | 1024 | 2048 | 2198 | 1674 | -23.9% |
| `q5_1` | 256 | 6 | 4096 | 512 | 2340 | 1567 | -33.1% |
| `q5_1` | 256 | 6 | 4096 | 2048 | 8322 | 6203 | -25.5% |
| `q5_1` | 256 | 6 | 16384 | 512 | 9176 | 6116 | -33.4% |
| `q5_1` | 256 | 6 | 16384 | 2048 | 32872 | 24309 | -26.0% |
| `q5_1` | 256 | 2 | 1024 | 512 | 376 | 306 | -18.7% |
| `q5_1` | 256 | 2 | 1024 | 2048 | 1237 | 1141 | -7.7% |
| `q5_1` | 256 | 2 | 4096 | 512 | 1403 | 1117 | -20.4% |
| `q5_1` | 256 | 2 | 4096 | 2048 | 4637 | 4196 | -9.5% |
| `q5_1` | 256 | 2 | 16384 | 512 | 5538 | 4367 | -21.1% |
| `q5_1` | 256 | 2 | 16384 | 2048 | 18219 | 16380 | -10.1% |
| `q5_1` | 128 | 8 | 1024 | 512 | 344 | 301 | -12.3% |
| `q5_1` | 128 | 8 | 1024 | 2048 | 1190 | 1118 | -6.1% |
| `q5_1` | 128 | 8 | 4096 | 512 | 1262 | 1102 | -12.7% |
| `q5_1` | 128 | 8 | 4096 | 2048 | 4418 | 4104 | -7.1% |
| `q5_1` | 128 | 8 | 16384 | 512 | 4947 | 4298 | -13.1% |
| `q5_1` | 128 | 8 | 16384 | 2048 | 17307 | 16041 | -7.3% |
| `q5_1` | 128 | 4 | 1024 | 512 | 369 | 327 | -11.4% |
| `q5_1` | 128 | 4 | 1024 | 2048 | 1228 | 1237 | +0.7% |
| `q5_1` | 128 | 4 | 4096 | 512 | 1373 | 1179 | -14.1% |
| `q5_1` | 128 | 4 | 4096 | 2048 | 4583 | 4464 | -2.6% |
| `q5_1` | 128 | 4 | 16384 | 512 | 5392 | 4591 | -14.8% |
| `q5_1` | 128 | 4 | 16384 | 2048 | 18010 | 17449 | -3.1% |
| `q5_1` | 64 | 8 | 1024 | 512 | 357 | 347 | -2.8% |
| `q5_1` | 64 | 8 | 1024 | 2048 | 1239 | 1339 | +8.1% |
| `q5_1` | 64 | 8 | 4096 | 512 | 1317 | 1218 | -7.5% |
| `q5_1` | 64 | 8 | 4096 | 2048 | 4617 | 4732 | +2.5% |
| `q5_1` | 64 | 8 | 16384 | 512 | 5159 | 4702 | -8.8% |
| `q5_1` | 64 | 8 | 16384 | 2048 | 18077 | 18331 | +1.4% |

**RTX 4070 (Ada, sm_89)**

| type | D | GQA | n_kv | n_q | F16 path | native | change |
|---|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 1024 | 512 | 291 | 204 | -29.9% |
| `q4_0` | 256 | 6 | 1024 | 2048 | 1040 | 743 | -28.6% |
| `q4_0` | 256 | 6 | 4096 | 512 | 1027 | 675 | -34.2% |
| `q4_0` | 256 | 6 | 4096 | 2048 | 3808 | 2641 | -30.7% |
| `q4_0` | 256 | 6 | 16384 | 512 | 4723 | 2584 | -45.3% |
| `q4_0` | 256 | 6 | 16384 | 2048 | 16692 | 10237 | -38.7% |
| `q4_0` | 256 | 2 | 1024 | 512 | 188 | 146 | -22.3% |
| `q4_0` | 256 | 2 | 1024 | 2048 | 603 | 514 | -14.8% |
| `q4_0` | 256 | 2 | 4096 | 512 | 656 | 464 | -29.2% |
| `q4_0` | 256 | 2 | 4096 | 2048 | 2127 | 1777 | -16.5% |
| `q4_0` | 256 | 2 | 16384 | 512 | 4462 | 1789 | -59.9% |
| `q4_0` | 256 | 2 | 16384 | 2048 | 11143 | 7086 | -36.4% |
| `q4_0` | 512 | 16 | 1024 | 512 | 388 | 330 | -14.9% |
| `q4_0` | 512 | 16 | 1024 | 2048 | 1430 | 1215 | -15.0% |
| `q4_0` | 512 | 16 | 4096 | 512 | 1346 | 1122 | -16.7% |
| `q4_0` | 512 | 16 | 4096 | 2048 | 5212 | 4340 | -16.7% |
| `q4_0` | 512 | 16 | 16384 | 512 | 5287 | 4284 | -19.0% |
| `q4_0` | 512 | 16 | 16384 | 2048 | 21800 | 16796 | -23.0% |
| `q4_0` | 128 | 8 | 1024 | 512 | 158 | 133 | -15.4% |
| `q4_0` | 128 | 8 | 1024 | 2048 | 583 | 515 | -11.7% |
| `q4_0` | 128 | 8 | 4096 | 512 | 535 | 448 | -16.3% |
| `q4_0` | 128 | 8 | 4096 | 2048 | 2041 | 1776 | -12.9% |
| `q4_0` | 128 | 8 | 16384 | 512 | 2101 | 1733 | -17.5% |
| `q4_0` | 128 | 8 | 16384 | 2048 | 7879 | 6847 | -13.1% |
| `q4_0` | 128 | 4 | 1024 | 512 | 167 | 137 | -18.1% |
| `q4_0` | 128 | 4 | 1024 | 2048 | 598 | 529 | -11.4% |
| `q4_0` | 128 | 4 | 4096 | 512 | 592 | 462 | -21.9% |
| `q4_0` | 128 | 4 | 4096 | 2048 | 2097 | 1835 | -12.5% |
| `q4_0` | 128 | 4 | 16384 | 512 | 2887 | 1787 | -38.1% |
| `q4_0` | 128 | 4 | 16384 | 2048 | 9338 | 7214 | -22.7% |
| `q4_0` | 64 | 8 | 1024 | 512 | 159 | 144 | -9.2% |
| `q4_0` | 64 | 8 | 1024 | 2048 | 592 | 575 | -2.8% |
| `q4_0` | 64 | 8 | 4096 | 512 | 560 | 510 | -8.9% |
| `q4_0` | 64 | 8 | 4096 | 2048 | 2100 | 2057 | -2.1% |
| `q4_0` | 64 | 8 | 16384 | 512 | 2251 | 1995 | -11.4% |
| `q4_0` | 64 | 8 | 16384 | 2048 | 8314 | 8000 | -3.8% |
| `q8_0` | 256 | 6 | 1024 | 512 | 294 | 220 | -25.1% |
| `q8_0` | 256 | 6 | 1024 | 2048 | 1053 | 809 | -23.2% |
| `q8_0` | 256 | 6 | 4096 | 512 | 1042 | 739 | -29.1% |
| `q8_0` | 256 | 6 | 4096 | 2048 | 3840 | 2906 | -24.3% |
| `q8_0` | 256 | 6 | 16384 | 512 | 4765 | 3008 | -36.9% |
| `q8_0` | 256 | 6 | 16384 | 2048 | 16748 | 11915 | -28.9% |
| `q8_0` | 256 | 2 | 1024 | 512 | 190 | 156 | -17.9% |
| `q8_0` | 256 | 2 | 1024 | 2048 | 614 | 552 | -10.2% |
| `q8_0` | 256 | 2 | 4096 | 512 | 674 | 512 | -24.1% |
| `q8_0` | 256 | 2 | 4096 | 2048 | 2163 | 1965 | -9.2% |
| `q8_0` | 256 | 2 | 16384 | 512 | 4536 | 2234 | -50.8% |
| `q8_0` | 256 | 2 | 16384 | 2048 | 11160 | 9151 | -18.0% |
| `q8_0` | 512 | 16 | 1024 | 512 | 391 | 347 | -11.2% |
| `q8_0` | 512 | 16 | 1024 | 2048 | 1437 | 1280 | -10.9% |
| `q8_0` | 512 | 16 | 4096 | 512 | 1355 | 1184 | -12.6% |
| `q8_0` | 512 | 16 | 4096 | 2048 | 5219 | 4575 | -12.3% |
| `q8_0` | 512 | 16 | 16384 | 512 | 5307 | 4538 | -14.5% |
| `q8_0` | 512 | 16 | 16384 | 2048 | 21822 | 17807 | -18.4% |
| `q8_0` | 128 | 8 | 1024 | 512 | 158 | 150 | -5.0% |
| `q8_0` | 128 | 8 | 1024 | 2048 | 590 | 583 | -1.3% |
| `q8_0` | 128 | 8 | 4096 | 512 | 543 | 510 | -6.1% |
| `q8_0` | 128 | 8 | 4096 | 2048 | 2056 | 2033 | -1.2% |
| `q8_0` | 128 | 8 | 16384 | 512 | 2125 | 1972 | -7.2% |
| `q8_0` | 128 | 8 | 16384 | 2048 | 7946 | 7937 | -0.1% |
| `q8_0` | 128 | 4 | 1024 | 512 | 167 | 150 | -10.0% |
| `q8_0` | 128 | 4 | 1024 | 2048 | 600 | 578 | -3.6% |
| `q8_0` | 128 | 4 | 4096 | 512 | 599 | 510 | -14.9% |
| `q8_0` | 128 | 4 | 4096 | 2048 | 2116 | 2051 | -3.1% |
| `q8_0` | 128 | 4 | 16384 | 512 | 2906 | 2132 | -26.6% |
| `q8_0` | 128 | 4 | 16384 | 2048 | 9274 | 8470 | -8.7% |
| `q8_0` | 64 | 8 | 1024 | 512 | 161 | 154 | -4.0% |
| `q8_0` | 64 | 8 | 1024 | 2048 | 594 | 614 | +3.2% |
| `q8_0` | 64 | 8 | 4096 | 512 | 564 | 549 | -2.8% |
| `q8_0` | 64 | 8 | 4096 | 2048 | 2109 | 2200 | +4.3% |
| `q8_0` | 64 | 8 | 16384 | 512 | 2270 | 2143 | -5.6% |
| `q8_0` | 64 | 8 | 16384 | 2048 | 8388 | 8572 | +2.2% |
| `q4_1` | 256 | 6 | 1024 | 512 | 293 | 209 | -28.7% |
| `q4_1` | 256 | 6 | 1024 | 2048 | 1053 | 769 | -26.9% |
| `q4_1` | 256 | 6 | 4096 | 512 | 1034 | 694 | -32.8% |
| `q4_1` | 256 | 6 | 4096 | 2048 | 3831 | 2704 | -29.4% |
| `q4_1` | 256 | 6 | 16384 | 512 | 4743 | 2646 | -44.2% |
| `q4_1` | 256 | 6 | 16384 | 2048 | 16817 | 10502 | -37.6% |
| `q4_1` | 256 | 2 | 1024 | 512 | 185 | 149 | -19.7% |
| `q4_1` | 256 | 2 | 1024 | 2048 | 608 | 523 | -14.0% |
| `q4_1` | 256 | 2 | 4096 | 512 | 662 | 472 | -28.7% |
| `q4_1` | 256 | 2 | 4096 | 2048 | 2140 | 1812 | -15.3% |
| `q4_1` | 256 | 2 | 16384 | 512 | 4552 | 1858 | -59.2% |
| `q4_1` | 256 | 2 | 16384 | 2048 | 11242 | 7432 | -33.9% |
| `q4_1` | 128 | 8 | 1024 | 512 | 158 | 134 | -15.1% |
| `q4_1` | 128 | 8 | 1024 | 2048 | 587 | 519 | -11.6% |
| `q4_1` | 128 | 8 | 4096 | 512 | 540 | 450 | -16.7% |
| `q4_1` | 128 | 8 | 4096 | 2048 | 2054 | 1779 | -13.4% |
| `q4_1` | 128 | 8 | 16384 | 512 | 2117 | 1726 | -18.5% |
| `q4_1` | 128 | 8 | 16384 | 2048 | 7914 | 6832 | -13.7% |
| `q4_1` | 128 | 4 | 1024 | 512 | 168 | 136 | -18.5% |
| `q4_1` | 128 | 4 | 1024 | 2048 | 600 | 526 | -12.4% |
| `q4_1` | 128 | 4 | 4096 | 512 | 596 | 459 | -23.0% |
| `q4_1` | 128 | 4 | 4096 | 2048 | 2111 | 1819 | -13.8% |
| `q4_1` | 128 | 4 | 16384 | 512 | 2919 | 1780 | -39.0% |
| `q4_1` | 128 | 4 | 16384 | 2048 | 9354 | 7152 | -23.5% |
| `q4_1` | 64 | 8 | 1024 | 512 | 160 | 146 | -8.6% |
| `q4_1` | 64 | 8 | 1024 | 2048 | 594 | 610 | +2.6% |
| `q4_1` | 64 | 8 | 4096 | 512 | 562 | 508 | -9.7% |
| `q4_1` | 64 | 8 | 4096 | 2048 | 2106 | 2058 | -2.3% |
| `q4_1` | 64 | 8 | 16384 | 512 | 2261 | 1961 | -13.3% |
| `q4_1` | 64 | 8 | 16384 | 2048 | 8338 | 7906 | -5.2% |
| `q5_0` | 256 | 6 | 1024 | 512 | 293 | 215 | -26.8% |
| `q5_0` | 256 | 6 | 1024 | 2048 | 1052 | 794 | -24.6% |
| `q5_0` | 256 | 6 | 4096 | 512 | 1038 | 719 | -30.7% |
| `q5_0` | 256 | 6 | 4096 | 2048 | 3829 | 2800 | -26.9% |
| `q5_0` | 256 | 6 | 16384 | 512 | 4769 | 2755 | -42.2% |
| `q5_0` | 256 | 6 | 16384 | 2048 | 16787 | 10906 | -35.0% |
| `q5_0` | 256 | 2 | 1024 | 512 | 188 | 154 | -18.1% |
| `q5_0` | 256 | 2 | 1024 | 2048 | 607 | 543 | -10.5% |
| `q5_0` | 256 | 2 | 4096 | 512 | 668 | 490 | -26.7% |
| `q5_0` | 256 | 2 | 4096 | 2048 | 2150 | 1885 | -12.3% |
| `q5_0` | 256 | 2 | 16384 | 512 | 4549 | 1930 | -57.6% |
| `q5_0` | 256 | 2 | 16384 | 2048 | 11107 | 7655 | -31.1% |
| `q5_0` | 128 | 8 | 1024 | 512 | 159 | 143 | -10.0% |
| `q5_0` | 128 | 8 | 1024 | 2048 | 590 | 556 | -5.9% |
| `q5_0` | 128 | 8 | 4096 | 512 | 542 | 487 | -10.1% |
| `q5_0` | 128 | 8 | 4096 | 2048 | 2052 | 1937 | -5.6% |
| `q5_0` | 128 | 8 | 16384 | 512 | 2123 | 1885 | -11.2% |
| `q5_0` | 128 | 8 | 16384 | 2048 | 7922 | 7476 | -5.6% |
| `q5_0` | 128 | 4 | 1024 | 512 | 169 | 144 | -14.5% |
| `q5_0` | 128 | 4 | 1024 | 2048 | 600 | 557 | -7.1% |
| `q5_0` | 128 | 4 | 4096 | 512 | 598 | 487 | -18.5% |
| `q5_0` | 128 | 4 | 4096 | 2048 | 2114 | 1931 | -8.7% |
| `q5_0` | 128 | 4 | 16384 | 512 | 2927 | 1902 | -35.0% |
| `q5_0` | 128 | 4 | 16384 | 2048 | 9378 | 7589 | -19.1% |
| `q5_0` | 64 | 8 | 1024 | 512 | 161 | 152 | -5.1% |
| `q5_0` | 64 | 8 | 1024 | 2048 | 595 | 606 | +1.9% |
| `q5_0` | 64 | 8 | 4096 | 512 | 564 | 540 | -4.2% |
| `q5_0` | 64 | 8 | 4096 | 2048 | 2105 | 2174 | +3.3% |
| `q5_0` | 64 | 8 | 16384 | 512 | 2262 | 2117 | -6.4% |
| `q5_0` | 64 | 8 | 16384 | 2048 | 8334 | 8474 | +1.7% |
| `q5_1` | 256 | 6 | 1024 | 512 | 292 | 216 | -26.1% |
| `q5_1` | 256 | 6 | 1024 | 2048 | 1053 | 799 | -24.2% |
| `q5_1` | 256 | 6 | 4096 | 512 | 1042 | 727 | -30.3% |
| `q5_1` | 256 | 6 | 4096 | 2048 | 3834 | 2835 | -26.1% |
| `q5_1` | 256 | 6 | 16384 | 512 | 4753 | 2783 | -41.5% |
| `q5_1` | 256 | 6 | 16384 | 2048 | 16816 | 11080 | -34.1% |
| `q5_1` | 256 | 2 | 1024 | 512 | 189 | 154 | -18.2% |
| `q5_1` | 256 | 2 | 1024 | 2048 | 609 | 545 | -10.5% |
| `q5_1` | 256 | 2 | 4096 | 512 | 671 | 495 | -26.2% |
| `q5_1` | 256 | 2 | 4096 | 2048 | 2150 | 1908 | -11.2% |
| `q5_1` | 256 | 2 | 16384 | 512 | 4517 | 1973 | -56.3% |
| `q5_1` | 256 | 2 | 16384 | 2048 | 11284 | 8050 | -28.7% |
| `q5_1` | 128 | 8 | 1024 | 512 | 159 | 142 | -10.2% |
| `q5_1` | 128 | 8 | 1024 | 2048 | 591 | 552 | -6.5% |
| `q5_1` | 128 | 8 | 4096 | 512 | 544 | 481 | -11.5% |
| `q5_1` | 128 | 8 | 4096 | 2048 | 2057 | 1915 | -6.9% |
| `q5_1` | 128 | 8 | 16384 | 512 | 2121 | 1857 | -12.4% |
| `q5_1` | 128 | 8 | 16384 | 2048 | 7953 | 7384 | -7.2% |
| `q5_1` | 128 | 4 | 1024 | 512 | 168 | 147 | -12.4% |
| `q5_1` | 128 | 4 | 1024 | 2048 | 602 | 572 | -4.9% |
| `q5_1` | 128 | 4 | 4096 | 512 | 598 | 500 | -16.4% |
| `q5_1` | 128 | 4 | 4096 | 2048 | 2118 | 2020 | -4.6% |
| `q5_1` | 128 | 4 | 16384 | 512 | 2927 | 2005 | -31.5% |
| `q5_1` | 128 | 4 | 16384 | 2048 | 9348 | 8094 | -13.4% |
| `q5_1` | 64 | 8 | 1024 | 512 | 160 | 154 | -4.0% |
| `q5_1` | 64 | 8 | 1024 | 2048 | 597 | 639 | +7.0% |
| `q5_1` | 64 | 8 | 4096 | 512 | 564 | 535 | -5.1% |
| `q5_1` | 64 | 8 | 4096 | 2048 | 2106 | 2162 | +2.7% |
| `q5_1` | 64 | 8 | 16384 | 512 | 2272 | 2072 | -8.8% |
| `q5_1` | 64 | 8 | 16384 | 2048 | 8348 | 8323 | -0.3% |

At D=256 the route is 20-45% faster at GQA 6 and 3-60% at GQA 2, and 11-27% faster at D=512. At D=128 it is 3-39% faster at GQA 4 and up to 19% at GQA 8, where `q8_0` is about even. At D=64 it is 3-13% faster at 512-token batches. At 2048-token batches `q8_0`, `q5_0` and `q5_1` run up to 8% slower than the F16 path there: every query tile converts the K/V rows it reads again, while the cast converts them once for all tiles, and D=64 has the least arithmetic per converted value to hide that behind. `llama.cpp` takes 512-token batches unless `-ub` says otherwise. Each value is the faster of two runs, for both paths alike.

### Decode and small query batches

Each cell is "without the flag / with it (change)", us/run, the faster of two runs. Without the flag a single query token (two on Ada) takes the vector kernel, larger batches the cast and the F16 MMA kernel. "n/a": the F16-casting path did not fit next to another model on the 4070.


**RTX 3060 (Ampere, sm_86)**

| type | D | GQA | n_kv | n_q=1 | n_q=2 | n_q=3 | n_q=4 | n_q=8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 4096 | 115 / 23 (-80%) | 175 / 25 (-85%) | 174 / 31 (-82%) | 174 / 32 (-82%) | 187 / 54 (-71%) |
| `q4_0` | 256 | 6 | 16384 | 406 / 64 (-84%) | 634 / 76 (-88%) | 638 / 86 (-87%) | 639 / 87 (-86%) | 651 / 143 (-78%) |
| `q4_0` | 256 | 6 | 65536 | 1559 / 232 (-85%) | 2499 / 281 (-89%) | 2495 / 304 (-88%) | 2497 / 307 (-88%) | 2510 / 509 (-80%) |
| `q4_0` | 256 | 2 | 4096 | 82 / 41 (-50%) | 328 / 41 (-87%) | 327 / 42 (-87%) | 327 / 44 (-87%) | 331 / 49 (-85%) |
| `q4_0` | 256 | 2 | 16384 | 286 / 123 (-57%) | 1254 / 124 (-90%) | 1255 / 127 (-90%) | 1255 / 127 (-90%) | 1259 / 159 (-87%) |
| `q4_0` | 256 | 2 | 65536 | 1082 / 459 (-58%) | 5000 / 460 (-91%) | 5001 / 462 (-91%) | 5028 / 463 (-91%) | 4992 / 599 (-88%) |
| `q4_0` | 512 | 16 | 4096 | 90 / 45 (-50%) | 81 / 30 (-63%) | 93 / 43 (-53%) | 93 / 44 (-52%) | 119 / 71 (-40%) |
| `q4_0` | 512 | 16 | 16384 | 262 / 93 (-64%) | 266 / 82 (-69%) | 312 / 125 (-60%) | 312 / 127 (-59%) | 373 / 188 (-50%) |
| `q4_0` | 512 | 16 | 65536 | 967 / 305 (-68%) | 1002 / 288 (-71%) | 1190 / 462 (-61%) | 1193 / 467 (-61%) | 1398 / 668 (-52%) |
| `q4_0` | 128 | 8 | 4096 | 87 / 14 (-84%) | 134 / 17 (-87%) | 137 / 20 (-86%) | 136 / 20 (-85%) | 143 / 31 (-78%) |
| `q4_0` | 128 | 8 | 16384 | 322 / 37 (-88%) | 492 / 41 (-92%) | 494 / 49 (-90%) | 497 / 49 (-90%) | 502 / 80 (-84%) |
| `q4_0` | 128 | 8 | 65536 | 1246 / 123 (-90%) | 1921 / 126 (-93%) | 1925 / 164 (-91%) | 1923 / 166 (-91%) | 1932 / 267 (-86%) |
| `q4_0` | 128 | 4 | 4096 | 92 / 24 (-73%) | 254 / 25 (-90%) | 254 / 28 (-89%) | 254 / 29 (-89%) | 258 / 43 (-83%) |
| `q4_0` | 128 | 4 | 16384 | 328 / 69 (-79%) | 969 / 68 (-93%) | 967 / 85 (-91%) | 968 / 86 (-91%) | 970 / 135 (-86%) |
| `q4_0` | 128 | 4 | 65536 | 1269 / 239 (-81%) | 3841 / 237 (-94%) | 3813 / 313 (-92%) | 3814 / 317 (-92%) | 3819 / 502 (-87%) |
| `q4_0` | 64 | 8 | 4096 | 80 / 13 (-84%) | 220 / 17 (-92%) | 220 / 19 (-91%) | 219 / 20 (-91%) | 223 / 27 (-88%) |
| `q4_0` | 64 | 8 | 16384 | 297 / 37 (-87%) | 833 / 46 (-95%) | 827 / 54 (-93%) | 831 / 54 (-93%) | 835 / 80 (-90%) |
| `q4_0` | 64 | 8 | 65536 | 1189 / 124 (-90%) | 3263 / 158 (-95%) | 3262 / 196 (-94%) | 3262 / 197 (-94%) | 3262 / 295 (-91%) |
| `q8_0` | 256 | 6 | 4096 | 107 / 35 (-67%) | 171 / 38 (-78%) | 174 / 43 (-75%) | 175 / 44 (-75%) | 188 / 60 (-68%) |
| `q8_0` | 256 | 6 | 16384 | 374 / 113 (-70%) | 639 / 116 (-82%) | 643 / 123 (-81%) | 642 / 124 (-81%) | 657 / 160 (-76%) |
| `q8_0` | 256 | 6 | 65536 | 1527 / 426 (-72%) | 2509 / 430 (-83%) | 2510 / 440 (-82%) | 2516 / 440 (-82%) | 2533 / 573 (-77%) |
| `q8_0` | 256 | 2 | 4096 | 80 / 65 (-19%) | 329 / 65 (-80%) | 328 / 66 (-80%) | 329 / 67 (-80%) | 333 / 69 (-79%) |
| `q8_0` | 256 | 2 | 16384 | 274 / 220 (-20%) | 1264 / 220 (-83%) | 1262 / 221 (-82%) | 1263 / 222 (-82%) | 1268 / 231 (-82%) |
| `q8_0` | 256 | 2 | 65536 | 1053 / 845 (-20%) | 6126 / 846 (-86%) | 5000 / 846 (-83%) | 4994 / 851 (-83%) | 5001 / 868 (-83%) |
| `q8_0` | 512 | 16 | 4096 | 97 / 49 (-49%) | 83 / 36 (-57%) | 96 / 46 (-52%) | 95 / 47 (-50%) | 121 / 74 (-39%) |
| `q8_0` | 512 | 16 | 16384 | 281 / 106 (-62%) | 283 / 95 (-66%) | 328 / 137 (-58%) | 328 / 139 (-58%) | 388 / 200 (-49%) |
| `q8_0` | 512 | 16 | 65536 | 1037 / 344 (-67%) | 1072 / 335 (-69%) | 1265 / 510 (-60%) | 1264 / 514 (-59%) | 1469 / 712 (-52%) |
| `q8_0` | 128 | 8 | 4096 | 87 / 20 (-77%) | 134 / 26 (-81%) | 135 / 26 (-80%) | 140 / 27 (-81%) | 139 / 37 (-74%) |
| `q8_0` | 128 | 8 | 16384 | 316 / 60 (-81%) | 489 / 64 (-87%) | 493 / 70 (-86%) | 491 / 70 (-86%) | 498 / 92 (-82%) |
| `q8_0` | 128 | 8 | 65536 | 1226 / 217 (-82%) | 1910 / 220 (-88%) | 1912 / 233 (-88%) | 1911 / 233 (-88%) | 1919 / 313 (-84%) |
| `q8_0` | 128 | 4 | 4096 | 90 / 38 (-58%) | 252 / 37 (-85%) | 253 / 41 (-84%) | 252 / 41 (-84%) | 255 / 49 (-81%) |
| `q8_0` | 128 | 4 | 16384 | 324 / 117 (-64%) | 962 / 115 (-88%) | 965 / 121 (-87%) | 964 / 121 (-87%) | 965 / 158 (-84%) |
| `q8_0` | 128 | 4 | 65536 | 1262 / 428 (-66%) | 3802 / 429 (-89%) | 3811 / 444 (-88%) | 3800 / 444 (-88%) | 3806 / 592 (-84%) |
| `q8_0` | 64 | 8 | 4096 | 79 / 20 (-75%) | 218 / 22 (-90%) | 217 / 23 (-89%) | 218 / 24 (-89%) | 222 / 28 (-87%) |
| `q8_0` | 64 | 8 | 16384 | 296 / 60 (-80%) | 824 / 62 (-92%) | 821 / 65 (-92%) | 822 / 65 (-92%) | 830 / 87 (-89%) |
| `q8_0` | 64 | 8 | 65536 | 1186 / 216 (-82%) | 3245 / 223 (-93%) | 3243 / 233 (-93%) | 3244 / 232 (-93%) | 3253 / 323 (-90%) |

**RTX 4070 (Ada, sm_89)**

| type | D | GQA | n_kv | n_q=1 | n_q=2 | n_q=3 | n_q=4 | n_q=8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 4096 | 48 / 11 (-78%) | 99 / 13 (-87%) | 58 / 15 (-74%) | 59 / 16 (-73%) | 68 / 25 (-63%) |
| `q4_0` | 256 | 6 | 16384 | 175 / 28 (-84%) | 359 / 36 (-90%) | 386 / 38 (-90%) | 386 / 39 (-90%) | 395 / 64 (-84%) |
| `q4_0` | 256 | 6 | 65536 | 741 / 167 (-77%) | 1760 / 170 (-90%) | 1523 / 179 (-88%) | 1524 / 180 (-88%) | 1535 / 241 (-84%) |
| `q4_0` | 256 | 2 | 4096 | 33 / 18 (-45%) | 68 / 19 (-73%) | 144 / 19 (-87%) | 145 / 20 (-87%) | 149 / 23 (-84%) |
| `q4_0` | 256 | 2 | 16384 | 123 / 54 (-56%) | 308 / 55 (-82%) | 773 / 57 (-93%) | 771 / 58 (-92%) | 774 / 78 (-90%) |
| `q4_0` | 256 | 2 | 65536 | 553 / 327 (-41%) | 1373 / 328 (-76%) | n/a / 329 | n/a / 331 | n/a / 339 |
| `q4_0` | 512 | 16 | 4096 | 47 / 19 (-60%) | 46 / 16 (-65%) | 49 / 21 (-56%) | 49 / 22 (-55%) | 60 / 33 (-45%) |
| `q4_0` | 512 | 16 | 16384 | 134 / 38 (-72%) | 140 / 39 (-72%) | 161 / 57 (-65%) | 163 / 59 (-64%) | 189 / 85 (-55%) |
| `q4_0` | 512 | 16 | 65536 | 696 / 132 (-81%) | 691 / 131 (-81%) | 723 / 206 (-71%) | 723 / 210 (-71%) | 787 / 301 (-62%) |
| `q4_0` | 128 | 8 | 4096 | 41 / 7 (-82%) | 57 / 9 (-84%) | 44 / 11 (-76%) | 44 / 11 (-75%) | 49 / 16 (-68%) |
| `q4_0` | 128 | 8 | 16384 | 140 / 17 (-88%) | 200 / 18 (-91%) | 183 / 23 (-87%) | 183 / 24 (-87%) | 195 / 36 (-81%) |
| `q4_0` | 128 | 8 | 65536 | 533 / 55 (-90%) | 758 / 58 (-92%) | 915 / 79 (-91%) | 916 / 80 (-91%) | 924 / 127 (-86%) |
| `q4_0` | 128 | 4 | 4096 | 41 / 13 (-68%) | 57 / 11 (-80%) | 77 / 14 (-82%) | 77 / 14 (-82%) | 79 / 20 (-74%) |
| `q4_0` | 128 | 4 | 16384 | 140 / 30 (-78%) | 200 / 28 (-86%) | 454 / 39 (-91%) | 455 / 39 (-91%) | 458 / 61 (-87%) |
| `q4_0` | 128 | 4 | 65536 | 574 / 167 (-71%) | 776 / 169 (-78%) | 1820 / 174 (-90%) | 1803 / 175 (-90%) | 1816 / 244 (-87%) |
| `q4_0` | 64 | 8 | 4096 | 40 / 7 (-82%) | 56 / 9 (-85%) | 63 / 10 (-84%) | 63 / 10 (-84%) | 66 / 13 (-80%) |
| `q4_0` | 64 | 8 | 16384 | 129 / 16 (-88%) | 192 / 20 (-90%) | 271 / 25 (-91%) | 272 / 25 (-91%) | 279 / 37 (-87%) |
| `q4_0` | 64 | 8 | 65536 | 491 / 57 (-88%) | 742 / 75 (-90%) | 1262 / 94 (-93%) | 1263 / 94 (-93%) | 1267 / 139 (-89%) |
| `q8_0` | 256 | 6 | 4096 | 46 / 12 (-74%) | 97 / 14 (-85%) | 58 / 17 (-71%) | 60 / 18 (-70%) | 77 / 27 (-65%) |
| `q8_0` | 256 | 6 | 16384 | 165 / 34 (-80%) | 359 / 41 (-89%) | 401 / 48 (-88%) | 404 / 49 (-88%) | 412 / 92 (-78%) |
| `q8_0` | 256 | 6 | 65536 | 747 / 306 (-59%) | 1909 / 309 (-84%) | 1586 / 319 (-80%) | 1588 / 320 (-80%) | 1599 / 339 (-79%) |
| `q8_0` | 256 | 2 | 4096 | 32 / 21 (-36%) | 66 / 21 (-68%) | 162 / 21 (-87%) | 168 / 22 (-87%) | 173 / 25 (-85%) |
| `q8_0` | 256 | 2 | 16384 | 170 / 159 (-7%) | 353 / 159 (-55%) | 802 / 160 (-80%) | 802 / 160 (-80%) | 805 / 167 (-79%) |
| `q8_0` | 256 | 2 | 65536 | 672 / 607 (-10%) | 1377 / 609 (-56%) | n/a / 615 | n/a / 612 | n/a / 616 |
| `q8_0` | 512 | 16 | 4096 | 32 / 19 (-40%) | 30 / 18 (-41%) | 33 / 23 (-30%) | 34 / 24 (-31%) | 45 / 34 (-24%) |
| `q8_0` | 512 | 16 | 16384 | 149 / 40 (-73%) | 154 / 45 (-71%) | 180 / 62 (-65%) | 179 / 64 (-64%) | 205 / 89 (-56%) |
| `q8_0` | 512 | 16 | 65536 | 744 / 214 (-71%) | 736 / 176 (-76%) | 770 / 236 (-69%) | 772 / 239 (-69%) | 839 / 324 (-61%) |
| `q8_0` | 128 | 8 | 4096 | 39 / 8 (-78%) | 55 / 10 (-81%) | 43 / 12 (-72%) | 43 / 12 (-72%) | 48 / 17 (-65%) |
| `q8_0` | 128 | 8 | 16384 | 136 / 19 (-86%) | 194 / 21 (-89%) | 200 / 27 (-87%) | 202 / 27 (-86%) | 207 / 40 (-81%) |
| `q8_0` | 128 | 8 | 65536 | 554 / 157 (-72%) | 761 / 160 (-79%) | 924 / 165 (-82%) | 925 / 166 (-82%) | 933 / 178 (-81%) |
| `q8_0` | 128 | 4 | 4096 | 39 / 15 (-63%) | 55 / 13 (-77%) | 74 / 16 (-79%) | 75 / 16 (-79%) | 76 / 22 (-71%) |
| `q8_0` | 128 | 4 | 16384 | 138 / 36 (-74%) | 197 / 37 (-81%) | 462 / 49 (-89%) | 463 / 50 (-89%) | 466 / 74 (-84%) |
| `q8_0` | 128 | 4 | 65536 | 600 / 307 (-49%) | 788 / 309 (-61%) | 1823 / 314 (-83%) | 1820 / 315 (-83%) | 1825 / 322 (-82%) |
| `q8_0` | 64 | 8 | 4096 | 39 / 8 (-78%) | 55 / 9 (-83%) | 62 / 11 (-83%) | 62 / 11 (-83%) | 65 / 14 (-78%) |
| `q8_0` | 64 | 8 | 16384 | 127 / 20 (-84%) | 188 / 23 (-88%) | 277 / 28 (-90%) | 277 / 28 (-90%) | 283 / 40 (-86%) |
| `q8_0` | 64 | 8 | 65536 | 508 / 156 (-69%) | 761 / 159 (-79%) | 1253 / 161 (-87%) | 1253 / 162 (-87%) | 1257 / 174 (-86%) |

The native kernels read every K/V row once per KV head and nothing else, so long-context steps run close to the DRAM bandwidth: `q8_0` decode at GQA 6 and n_kv=65536 reads K/V at about 90% of it. One or two query tokens take the 1x8 and 2x8 tiles at any GQA ratio from 3, up to 4 the 4x8 tile, and GQA 2 the 4x2 tile. Up to 8 query tokens at GQA 3 and 4 take 8 heads per tile as well, which reads K/V half as often as the 2-head tiles. At D=512 the step is not bandwidth bound: the 1x8 kernel there keeps too many values in registers and spills. Without the flag, two to eight query tokens pay for the cast, which reads the cache, writes an F16 copy two to four times its size and has the kernel read that again.

### End to end

`llama-bench`, Qwen3.8-27B IQ3_XXS (16 of 65 layers are attention), one GPU each, `-fa 1 -p 512,4 -n 128 -d 0,16384 -r 2`, t/s, best of two passes. `pp4` stands in for a speculative-decoding step that verifies 4 tokens. "native off/on" is this branch with `-fanq 0` and `-fanq 1`; differences within about 2% are noise.

| GPU | cache | test | upstream | native off | native on | on vs upstream |
|---|---|---|--:|--:|--:|--:|
| RTX 3060 | `q8_0` | pp512 | 571.7 | 564.7 | 563.2 | -1.5% |
| RTX 3060 | `q8_0` | pp4 | 53.8 | 53.5 | 53.5 | -0.6% |
| RTX 3060 | `q8_0` | tg128 | 21.1 | 21.0 | 20.9 | -1.1% |
| RTX 3060 | `q8_0` | pp512 @ d16384 | 480.5 | 481.3 | 500.3 | +4.1% |
| RTX 3060 | `q8_0` | pp4 @ d16384 | 48.7 | 48.4 | 52.4 | +7.4% |
| RTX 3060 | `q8_0` | tg128 @ d16384 | 18.6 | 18.5 | 20.2 | +8.5% |
| RTX 3060 | `q4_0` | pp512 | 565.8 | 572.3 | 564.3 | -0.3% |
| RTX 3060 | `q4_0` | pp4 | 53.4 | 53.3 | 53.2 | -0.4% |
| RTX 3060 | `q4_0` | tg128 | 21.0 | 21.1 | 20.9 | -0.5% |
| RTX 3060 | `q4_0` | pp512 @ d16384 | 476.0 | 480.8 | 505.0 | +6.1% |
| RTX 3060 | `q4_0` | pp4 @ d16384 | 47.4 | 46.8 | 51.0 | +7.6% |
| RTX 3060 | `q4_0` | tg128 @ d16384 | 18.3 | 18.3 | 20.4 | +11.2% |
| RTX 4070 | `q8_0` | pp512 | 1225.1 | 1225.3 | 1231.6 | +0.5% |
| RTX 4070 | `q8_0` | pp4 | 104.9 | 107.5 | 108.5 | +3.4% |
| RTX 4070 | `q8_0` | tg128 | 38.2 | 38.2 | 38.2 | -0.1% |
| RTX 4070 | `q8_0` | pp512 @ d16384 | 1028.5 | 1034.7 | 1098.0 | +6.8% |
| RTX 4070 | `q8_0` | pp4 @ d16384 | 91.5 | 92.1 | 101.6 | +11.0% |
| RTX 4070 | `q8_0` | tg128 @ d16384 | 34.4 | 34.4 | 36.5 | +6.2% |
| RTX 4070 | `q4_0` | pp512 | 1226.9 | 1226.8 | 1233.1 | +0.5% |
| RTX 4070 | `q4_0` | pp4 | 106.9 | 106.9 | 106.2 | -0.7% |
| RTX 4070 | `q4_0` | tg128 | 38.2 | 38.2 | 38.2 | +0.1% |
| RTX 4070 | `q4_0` | pp512 @ d16384 | 1032.4 | 1039.6 | 1120.0 | +8.5% |
| RTX 4070 | `q4_0` | pp4 @ d16384 | 90.3 | 90.4 | 103.7 | +14.8% |
| RTX 4070 | `q4_0` | tg128 @ d16384 | 34.2 | 34.2 | 37.3 | +8.8% |

Two-token steps (speculative decoding with one draft token), one GPU each, `-p 2 -n 0 -d 16384 -r 3`:

| GPU | cache | test | upstream | native off | native on | on vs upstream |
|---|---|---|--:|--:|--:|--:|
| RTX 3060 | `q8_0` | pp2 @ d16384 | 31.1 | 30.9 | 35.4 | +13.7% |
| RTX 3060 | `q4_0` | pp2 @ d16384 | 30.8 | 30.6 | 35.7 | +16.1% |
| RTX 4070 | `q8_0` | pp2 @ d16384 | 49.2 | 49.3 | 63.0 | +27.9% |
| RTX 4070 | `q4_0` | pp2 @ d16384 | 49.0 | 48.9 | 64.0 | +30.6% |

Both GPUs (`-sm layer`), 64k context, `-p 512,4 -n 32 -d 65536 -r 1`, best of up to two passes:

| GPU | cache | test | upstream | native off | native on | on vs upstream |
|---|---|---|--:|--:|--:|--:|
| 4070 + 3060 | `q8_0` | pp512 @ d65536 | 445.7 | 459.6 | 519.9 | +16.6% |
| 4070 + 3060 | `q8_0` | pp4 @ d65536 | 49.2 | 49.1 | 67.3 | +36.9% |
| 4070 + 3060 | `q8_0` | tg32 @ d65536 | 18.3 | 18.3 | 23.9 | +30.7% |
| 4070 + 3060 | `q4_0` | pp512 @ d65536 | 451.0 | 458.6 | 555.2 | +23.1% |
| 4070 + 3060 | `q4_0` | pp4 @ d65536 | 50.3 | 50.3 | 70.0 | +39.1% |
| 4070 + 3060 | `q4_0` | tg32 @ d65536 | 18.2 | 18.2 | 25.4 | +39.1% |

The gain grows with the context, since attention is a growing share of each step. Without the flag the branch matches upstream.

D=128 models, RTX 4070, `-fa 1 -p 512 -n 128 -d 0,16384 -r 2` and `-p 2,4 -n 0 -d 16384 -r 3`, t/s. The pp512 results at depth 0 vary by up to 7% between repetitions (run time of 50 ms).

| Model | cache | test | native off | native on | change |
|---|---|---|--:|--:|--:|
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q8_0` | pp512 | 9528.0 | 10051.2 | +5.5% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q8_0` | tg128 | 188.2 | 185.8 | -1.3% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q8_0` | pp512 @ d16384 | 3715.4 | 4411.7 | +18.7% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q8_0` | tg128 @ d16384 | 113.9 | 130.1 | +14.2% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q8_0` | pp2 @ d16384 | 177.2 | 225.9 | +27.5% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q8_0` | pp4 @ d16384 | 238.3 | 429.1 | +80.0% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q4_0` | pp512 | 10104.5 | 9598.7 | -5.0% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q4_0` | tg128 | 188.2 | 187.3 | -0.5% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q4_0` | pp512 @ d16384 | 3790.4 | 4944.7 | +30.5% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q4_0` | tg128 @ d16384 | 116.9 | 151.4 | +29.5% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q4_0` | pp2 @ d16384 | 180.0 | 258.9 | +43.8% |
| Llama-3.2-3B Q4_K_M (D=128, GQA 3) | `q4_0` | pp4 @ d16384 | 238.8 | 467.5 | +95.7% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q8_0` | pp512 | 7166.4 | 7557.8 | +5.5% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q8_0` | tg128 | 146.5 | 144.8 | -1.2% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q8_0` | pp512 @ d16384 | 2832.8 | 3464.6 | +22.3% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q8_0` | tg128 @ d16384 | 80.6 | 101.5 | +25.9% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q8_0` | pp2 @ d16384 | 124.9 | 180.0 | +44.1% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q8_0` | pp4 @ d16384 | 187.0 | 332.0 | +77.5% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q4_0` | pp512 | 7511.9 | 7645.5 | +1.8% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q4_0` | tg128 | 146.4 | 145.1 | -0.9% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q4_0` | pp512 @ d16384 | 2903.0 | 3867.7 | +33.2% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q4_0` | tg128 @ d16384 | 81.8 | 117.9 | +44.2% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q4_0` | pp2 @ d16384 | 125.5 | 206.7 | +64.7% |
| Qwen3-4B Q4_K_M (D=128, GQA 4) | `q4_0` | pp4 @ d16384 | 189.7 | 378.8 | +99.8% |

At depth 0 a token generation step attends to at most 128 tokens, one or two K/V tiles. There the native kernel costs about 1% more than the vector kernel: its fixed work per launch (Q load, staging, conversion of a whole tile) is larger. From a few thousand tokens of context on the native route is faster.

## Memory

The removed F16 copy is part of the compute buffer, which is reserved for the full context. Compute buffer size reported at context creation, Qwen3.8-27B IQ3_XXS split over RTX 4070 + RTX 3060, `-ub 512`, the same for `q4_0` and `q8_0` caches:

| Context | 4070 off | 4070 on | 3060 off | 3060 on | Saved |
|--:|--:|--:|--:|--:|--:|
| 32768 | 467 MiB | 381 MiB | 704 MiB | 704 MiB | 86 MiB |
| 131072 | 1235 MiB | 765 MiB | 1185 MiB | 1088 MiB | 567 MiB |
| 262144 | 2259 MiB | 1277 MiB | 2209 MiB | 1600 MiB | 1591 MiB |

The copy is `2 * n_kv_heads * head_dim * sizeof(F16)` per token of context, 4 KiB for this model. Where the FlashAttention node is not the largest consumer of the compute buffer, as on the second GPU at short context, less of it shows.
