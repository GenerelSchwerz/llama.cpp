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
- K/V data and strides 16-byte aligned, which a KV cache always is;
- no sparse K/V set (`n_kv_max == 0`): the sparse gather exists only in the F16 kernel.

| Head dim | GQA ratio | Query batch | Tile (sm_80+) | Tile (Turing) |
|---|---|---|---|---|
| 256 | 2, 4 | 1-4 | 4x2 | 4x2 |
| 256 | 2, 4 | 5-16 | 16x2 | 16x2 |
| 256 | even, not a multiple of 8 | > 16 | 32x2 | 16x2 |
| 256 | even above 4, not a multiple of 8 | 5-16 | 8x8 | 4x8 |
| 256 | odd above 4, or a multiple of 8 | > 4 | 8x8 | 4x8 |
| 256 | > 4 | 2-4 | 4x8 | 4x8 |
| 256 | > 4 | 1 | 1x8 | 1x8 |
| 512 | > 4 | > 4 | 8x8 | 4x8 |
| 512 | > 4 | 1-4 | 4x8 | 4x8 |

D=256 has kernels for `q4_0`, `q4_1`, `q5_0`, `q5_1` and `q8_0`, D=512 for `q4_0` and `q8_0`. A tile `AxB` covers A query tokens of B Q heads that share a KV head. An even GQA ratio that is not a multiple of 8 takes 2 heads per tile when the query batch is large, because 8 heads per tile would compute zeros for the missing ones (25% of the tile at GQA 6). Small query batches take narrower shapes: they are bound by reading K/V, and at D=256 a single token (or up to 4 at 2 heads per tile) gets an 8-column tile, which loads 64 K/V rows per stage instead of 32 and does a quarter of the arithmetic. With 2 or more tokens at 8 heads per tile, one 4x8 tile beats several 1x8 tiles, which would read K/V once each.

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

An extra type follows its own `type-type` pair, the same selection that compiles its vector kernel. Without its pair a cache of that type still runs, on the F16-casting path. That gives 16 kernels in a default build, six more per selected extra pair and 34 with `all`. `scripts/fattn-native-inventory.py --fa-quants <value>` reads the built library back and fails on a missing, unexpected or duplicated kernel. The route is CUDA only; HIP and MUSA do not build the instances.

## Implementation

`flash_attn_ext_f16` gains `type_K` / `type_V` template parameters, defaulting to `GGML_TYPE_F16`, so the F16 instantiations are unchanged. Only the way the half2 K/V tiles in shared memory are filled differs, plus the KQ accumulation below; the softmax, the VKQ MMA and the combine code are shared. How the tiles are filled depends on the architecture and the configuration (`ggml_cuda_fattn_mma_nstages_quant()`):

- **Two-stage (Ampere and newer, D=256).** The F16 path loads K/V with a two-stage cp.async pipeline here. The native route keeps it: the raw quantized rows of the next tile are copied to shared memory with cp.async while the current tile computes. A row of 256 or 512 values is a whole number of 16-byte chunks for every type, so rows are copied as they are, with consecutive threads on consecutive chunks. K and V get separate half2 tiles where shared memory allows two blocks per SM: V of the current tile is converted while the KQ MMA reads K, and K of the next tile while the VKQ MMA reads V, so the conversion runs in the shadow of the tensor cores. For `q8_0` the mask tile does not fit next to the four K/V tiles, so each thread loads its mask values into registers instead.
- **Chunked (Ampere and newer, D=512).** K and V are processed in two chunks of 256 values each. Two raw chunk buffers rotate through K hi, K lo, V lo, V hi, each chunk copied two chunks ahead, so no global memory latency is left on the critical path. The F16 path waits for each chunk here.
- **Synchronous (Turing).** Without cp.async, each thread reads its quant blocks from global memory and writes the converted values to the tile directly.

### Conversion

A staged block is read from shared memory as 4-byte words. Blocks of 18, 22 and 34 bytes (`q4_0`, `q5_0`, `q8_0`) alternate between aligned and 2-byte misaligned; even and odd blocks are converted in separate passes, so the misalignment is a compile-time constant and costs at most one funnel shift per word. The 8 lanes of a quarter warp take 8 consecutive rows of the same block: through the tile swizzle their 16-byte stores hit 8 distinct bank groups, and their reads of the packed rows do not conflict either. A value becomes half2 by placing quant bytes into `0x64XX` (half(1024 + q)) with `__byte_perm`, subtracting the bias and scaling, which reproduces the F16 cast exactly. `q4_1` and `q5_1` compute `q*d + m` in half2 with one rounding where the cast rounds to float first; the two can differ in the last bit.

### KQ accumulation

GeForce cards run MMAs with float accumulators at half the rate of half accumulators. On `sm_86`, `sm_89` and `sm_120` the native kernels therefore sum KQ in half over 32 values of D (two m16n8k16 MMAs) and add those partial sums in float. This saves up to 11% of kernel time at D=256 and 10-15% at D=512 on the 3060 and 4070. A partial sum over 32 values stays far below the half range for realistic logits, and its rounding error is small against the error the quantized cache itself introduces. Longer half runs were measured and rejected: summing all 256 values in half saves another 7-9% of kernel time, but one sensitive token in the wikitext sample moved from a KL divergence of at most 2.3 to between 5 and 18 against the F16 cache. `sm_120` (RTX 50) is included on the same grounds but has not been measured here. Other architectures keep float: `sm_80` runs both accumulators at the same rate. Professional cards, which run float accumulators at full rate, gain nothing from it. Defining `GGML_FATTN_QUANT_KQ_FLOAT` at build time (`-DCMAKE_CUDA_FLAGS=-DGGML_FATTN_QUANT_KQ_FLOAT`) keeps float accumulation on every architecture, which is the comparison to run on a card not measured here. The VKQ product is summed in half, as in the F16 path.

### Shared MMA path

Two changes to the code both paths share make the F16 path faster as well, with identical results:

- `load_ldmatrix_swizzled` and `load_ldmatrix_trans_swizzled` pass a 32-bit shared address to `ldmatrix`. With a generic pointer the compiler added and subtracted the shared window base for every load.
- Where K is the B operand of the KQ mma (wide tiles), it is loaded with the 8x8 matrices in the register order the mma takes them, which removes the register moves that formed the operand pairs.

## Validation

RTX 4070 (`sm_89`, Ada) and RTX 3060 (`sm_86`, Ampere), CUDA 13.4, `CMAKE_CUDA_ARCHITECTURES=86-real;89-real`, all five types selected.

- `test-backend-ops -o FLASH_ATTN_EXT`: 3993/3993 on both GPUs; `-o GATED_DELTA_NET` (shares the ldmatrix helpers): 43/43.
- `test-backend-ops -o NATIVE_QUANT_EQUIVALENCE`: 37/37 on both GPUs. Each case computes the attention over the quantized cache and over an F16 copy of it in one graph, requires the two CUDA results to agree within an NMSE of 1e-4, and checks which path the dispatcher took through the backend's native-launch counter. The largest NMSE between the two halves is 2.9e-5 (n_kv=16384), a tenth of the difference between the CPU's own two halves (3.5e-4, since the CPU takes quantized dot products for a quantized cache), and the same as with float KQ accumulation. The cases cover every tile shape, decode and small query batches, GQA 4 and 8, two sequences and query batches large enough for the mask scan; five cases assert the F16-casting path where no kernel exists.
- `compute-sanitizer` racecheck, synccheck and memcheck: no errors or warnings on the native kernels, on both GPUs, for the two-stage, shadow and chunked modes, the 8- and 32-column tiles at decode and small batches, and the half KQ accumulation at D=256 and D=512. The one racecheck warning is in upstream's F16 vector kernel, which computes the reference half of a decode case on Ada; `master` reports the same warning there.
- `scripts/fattn-native-inventory.py`: 34 declared kernels with all types selected, all compiled, nothing else.

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

The mean and the maximum are dominated by a few sensitive tokens whose divergence changes with any change in summation order, as the two float paths show. The median KLD is the same for all three paths: 0.00028 and 0.00030 for `q8_0` on the two samples, 0.0012 for `q4_0`.

Greedy decoding with Qwen3.8-27B UD-IQ4_XS on 4070 + 3060, 1442 prompt tokens and 256 generated tokens: without the flag the output is byte identical to upstream `master` for `q8_0` and `q4_0` caches. With it the text stays coherent and departs from upstream after 25 to 65 words, as any change in summation order does.

Turing: the kernels compile for `sm_75` and use the unchanged synchronous loaders. They were verified on a Quadro RTX 8000 before the rebase; this revision, including decode and small batches on the native route, has not been measured there.

## Throughput

`test-backend-ops perf -o FLASH_ATTN_EXT -p expected_native`, us/run, lower is better, on an otherwise idle machine. "F16 path" is this branch without the flag, cast kernel included; the shared MMA changes make it 1-5% faster than upstream `master`. Rows where a type has no kernel are left out.

### Prompt processing


**RTX 3060 (Ampere, sm_86)**

| type | D | GQA | n_kv | n_q | F16 path | native | change |
|---|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 1024 | 512 | 610 | 393 | -35.6% |
| `q4_0` | 256 | 6 | 1024 | 2048 | 2167 | 1547 | -28.6% |
| `q4_0` | 256 | 6 | 4096 | 512 | 2296 | 1445 | -37.1% |
| `q4_0` | 256 | 6 | 4096 | 2048 | 8225 | 5744 | -30.2% |
| `q4_0` | 256 | 6 | 16384 | 512 | 9064 | 5656 | -37.6% |
| `q4_0` | 256 | 6 | 16384 | 2048 | 32487 | 22556 | -30.6% |
| `q4_0` | 256 | 2 | 1024 | 512 | 369 | 284 | -23.1% |
| `q4_0` | 256 | 2 | 1024 | 2048 | 1220 | 1054 | -13.6% |
| `q4_0` | 256 | 2 | 4096 | 512 | 1374 | 1034 | -24.7% |
| `q4_0` | 256 | 2 | 4096 | 2048 | 4576 | 3903 | -14.7% |
| `q4_0` | 256 | 2 | 16384 | 512 | 5401 | 4020 | -25.6% |
| `q4_0` | 256 | 2 | 16384 | 2048 | 17964 | 15281 | -14.9% |
| `q4_0` | 512 | 16 | 1024 | 512 | 997 | 755 | -24.2% |
| `q4_0` | 512 | 16 | 1024 | 2048 | 3781 | 2886 | -23.7% |
| `q4_0` | 512 | 16 | 4096 | 512 | 3647 | 2671 | -26.8% |
| `q4_0` | 512 | 16 | 4096 | 2048 | 13784 | 10196 | -26.0% |
| `q4_0` | 512 | 16 | 16384 | 512 | 14219 | 10315 | -27.5% |
| `q4_0` | 512 | 16 | 16384 | 2048 | 53744 | 39420 | -26.7% |
| `q8_0` | 256 | 6 | 1024 | 512 | 614 | 447 | -27.2% |
| `q8_0` | 256 | 6 | 1024 | 2048 | 2186 | 1763 | -19.4% |
| `q8_0` | 256 | 6 | 4096 | 512 | 2319 | 1634 | -29.5% |
| `q8_0` | 256 | 6 | 4096 | 2048 | 8292 | 6463 | -22.1% |
| `q8_0` | 256 | 6 | 16384 | 512 | 9125 | 6383 | -30.1% |
| `q8_0` | 256 | 6 | 16384 | 2048 | 32660 | 25363 | -22.3% |
| `q8_0` | 256 | 2 | 1024 | 512 | 373 | 322 | -13.5% |
| `q8_0` | 256 | 2 | 1024 | 2048 | 1231 | 1199 | -2.6% |
| `q8_0` | 256 | 2 | 4096 | 512 | 1389 | 1173 | -15.5% |
| `q8_0` | 256 | 2 | 4096 | 2048 | 4616 | 4396 | -4.8% |
| `q8_0` | 256 | 2 | 16384 | 512 | 5455 | 4589 | -15.9% |
| `q8_0` | 256 | 2 | 16384 | 2048 | 18129 | 17182 | -5.2% |
| `q8_0` | 512 | 16 | 1024 | 512 | 1002 | 797 | -20.4% |
| `q8_0` | 512 | 16 | 1024 | 2048 | 3801 | 3067 | -19.3% |
| `q8_0` | 512 | 16 | 4096 | 512 | 3669 | 2853 | -22.2% |
| `q8_0` | 512 | 16 | 4096 | 2048 | 13861 | 10909 | -21.3% |
| `q8_0` | 512 | 16 | 16384 | 512 | 14325 | 11079 | -22.7% |
| `q8_0` | 512 | 16 | 16384 | 2048 | 54079 | 42336 | -21.7% |
| `q4_1` | 256 | 6 | 1024 | 512 | 612 | 418 | -31.8% |
| `q4_1` | 256 | 6 | 1024 | 2048 | 2182 | 1630 | -25.3% |
| `q4_1` | 256 | 6 | 4096 | 512 | 2320 | 1519 | -34.5% |
| `q4_1` | 256 | 6 | 4096 | 2048 | 8299 | 6002 | -27.7% |
| `q4_1` | 256 | 6 | 16384 | 512 | 9143 | 5939 | -35.0% |
| `q4_1` | 256 | 6 | 16384 | 2048 | 32735 | 23458 | -28.3% |
| `q4_1` | 256 | 2 | 1024 | 512 | 371 | 299 | -19.3% |
| `q4_1` | 256 | 2 | 1024 | 2048 | 1230 | 1111 | -9.7% |
| `q4_1` | 256 | 2 | 4096 | 512 | 1383 | 1087 | -21.4% |
| `q4_1` | 256 | 2 | 4096 | 2048 | 4631 | 4072 | -12.1% |
| `q4_1` | 256 | 2 | 16384 | 512 | 5447 | 4232 | -22.3% |
| `q4_1` | 256 | 2 | 16384 | 2048 | 18068 | 15836 | -12.4% |
| `q5_0` | 256 | 6 | 1024 | 512 | 618 | 424 | -31.4% |
| `q5_0` | 256 | 6 | 1024 | 2048 | 2193 | 1661 | -24.3% |
| `q5_0` | 256 | 6 | 4096 | 512 | 2331 | 1549 | -33.5% |
| `q5_0` | 256 | 6 | 4096 | 2048 | 8292 | 6133 | -26.0% |
| `q5_0` | 256 | 6 | 16384 | 512 | 9175 | 6060 | -33.9% |
| `q5_0` | 256 | 6 | 16384 | 2048 | 32786 | 24010 | -26.8% |
| `q5_0` | 256 | 2 | 1024 | 512 | 377 | 304 | -19.2% |
| `q5_0` | 256 | 2 | 1024 | 2048 | 1235 | 1129 | -8.5% |
| `q5_0` | 256 | 2 | 4096 | 512 | 1403 | 1107 | -21.1% |
| `q5_0` | 256 | 2 | 4096 | 2048 | 4634 | 4161 | -10.2% |
| `q5_0` | 256 | 2 | 16384 | 512 | 5509 | 4324 | -21.5% |
| `q5_0` | 256 | 2 | 16384 | 2048 | 18211 | 16291 | -10.5% |
| `q5_1` | 256 | 6 | 1024 | 512 | 619 | 425 | -31.3% |
| `q5_1` | 256 | 6 | 1024 | 2048 | 2181 | 1668 | -23.5% |
| `q5_1` | 256 | 6 | 4096 | 512 | 2321 | 1558 | -32.9% |
| `q5_1` | 256 | 6 | 4096 | 2048 | 8274 | 6170 | -25.4% |
| `q5_1` | 256 | 6 | 16384 | 512 | 9149 | 6082 | -33.5% |
| `q5_1` | 256 | 6 | 16384 | 2048 | 32620 | 24216 | -25.8% |
| `q5_1` | 256 | 2 | 1024 | 512 | 375 | 305 | -18.8% |
| `q5_1` | 256 | 2 | 1024 | 2048 | 1232 | 1132 | -8.1% |
| `q5_1` | 256 | 2 | 4096 | 512 | 1399 | 1115 | -20.3% |
| `q5_1` | 256 | 2 | 4096 | 2048 | 4607 | 4173 | -9.4% |
| `q5_1` | 256 | 2 | 16384 | 512 | 5497 | 4352 | -20.8% |
| `q5_1` | 256 | 2 | 16384 | 2048 | 18090 | 16372 | -9.5% |

**RTX 4070 (Ada, sm_89)**

| type | D | GQA | n_kv | n_q | F16 path | native | change |
|---|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 1024 | 512 | 294 | 201 | -31.5% |
| `q4_0` | 256 | 6 | 1024 | 2048 | 1045 | 744 | -28.7% |
| `q4_0` | 256 | 6 | 4096 | 512 | 1035 | 671 | -35.2% |
| `q4_0` | 256 | 6 | 4096 | 2048 | 3820 | 2618 | -31.5% |
| `q4_0` | 256 | 6 | 16384 | 512 | 4760 | 2576 | -45.9% |
| `q4_0` | 256 | 6 | 16384 | 2048 | 16738 | 10180 | -39.2% |
| `q4_0` | 256 | 2 | 1024 | 512 | 183 | 146 | -20.3% |
| `q4_0` | 256 | 2 | 1024 | 2048 | 609 | 512 | -15.9% |
| `q4_0` | 256 | 2 | 4096 | 512 | 659 | 460 | -30.3% |
| `q4_0` | 256 | 2 | 4096 | 2048 | 2133 | 1763 | -17.3% |
| `q4_0` | 256 | 2 | 16384 | 512 | 4523 | 1785 | -60.5% |
| `q4_0` | 256 | 2 | 16384 | 2048 | 11213 | 7017 | -37.4% |
| `q4_0` | 512 | 16 | 1024 | 512 | 388 | 330 | -14.9% |
| `q4_0` | 512 | 16 | 1024 | 2048 | 1434 | 1220 | -14.9% |
| `q4_0` | 512 | 16 | 4096 | 512 | 1350 | 1121 | -17.0% |
| `q4_0` | 512 | 16 | 4096 | 2048 | 5212 | 4324 | -17.0% |
| `q4_0` | 512 | 16 | 16384 | 512 | 5313 | 4288 | -19.3% |
| `q4_0` | 512 | 16 | 16384 | 2048 | 21808 | 16816 | -22.9% |
| `q8_0` | 256 | 6 | 1024 | 512 | 292 | 222 | -24.0% |
| `q8_0` | 256 | 6 | 1024 | 2048 | 1059 | 810 | -23.5% |
| `q8_0` | 256 | 6 | 4096 | 512 | 1043 | 744 | -28.7% |
| `q8_0` | 256 | 6 | 4096 | 2048 | 3843 | 2912 | -24.2% |
| `q8_0` | 256 | 6 | 16384 | 512 | 4783 | 3029 | -36.7% |
| `q8_0` | 256 | 6 | 16384 | 2048 | 16844 | 11954 | -29.0% |
| `q8_0` | 256 | 2 | 1024 | 512 | 188 | 156 | -16.9% |
| `q8_0` | 256 | 2 | 1024 | 2048 | 615 | 552 | -10.3% |
| `q8_0` | 256 | 2 | 4096 | 512 | 677 | 511 | -24.5% |
| `q8_0` | 256 | 2 | 4096 | 2048 | 2158 | 1977 | -8.4% |
| `q8_0` | 256 | 2 | 16384 | 512 | 4546 | 2257 | -50.3% |
| `q8_0` | 256 | 2 | 16384 | 2048 | 11129 | 9274 | -16.7% |
| `q8_0` | 512 | 16 | 1024 | 512 | 391 | 351 | -10.3% |
| `q8_0` | 512 | 16 | 1024 | 2048 | 1436 | 1296 | -9.7% |
| `q8_0` | 512 | 16 | 4096 | 512 | 1358 | 1198 | -11.7% |
| `q8_0` | 512 | 16 | 4096 | 2048 | 5225 | 4638 | -11.2% |
| `q8_0` | 512 | 16 | 16384 | 512 | 5336 | 4599 | -13.8% |
| `q8_0` | 512 | 16 | 16384 | 2048 | 21840 | 18052 | -17.3% |
| `q4_1` | 256 | 6 | 1024 | 512 | 291 | 210 | -27.8% |
| `q4_1` | 256 | 6 | 1024 | 2048 | 1054 | 773 | -26.7% |
| `q4_1` | 256 | 6 | 4096 | 512 | 1038 | 696 | -33.0% |
| `q4_1` | 256 | 6 | 4096 | 2048 | 3835 | 2714 | -29.2% |
| `q4_1` | 256 | 6 | 16384 | 512 | 4790 | 2664 | -44.4% |
| `q4_1` | 256 | 6 | 16384 | 2048 | 16767 | 10537 | -37.2% |
| `q4_1` | 256 | 2 | 1024 | 512 | 185 | 149 | -19.3% |
| `q4_1` | 256 | 2 | 1024 | 2048 | 609 | 528 | -13.4% |
| `q4_1` | 256 | 2 | 4096 | 512 | 664 | 475 | -28.5% |
| `q4_1` | 256 | 2 | 4096 | 2048 | 2147 | 1825 | -15.0% |
| `q4_1` | 256 | 2 | 16384 | 512 | 4490 | 1866 | -58.4% |
| `q4_1` | 256 | 2 | 16384 | 2048 | 11231 | 7506 | -33.2% |
| `q5_0` | 256 | 6 | 1024 | 512 | 292 | 215 | -26.4% |
| `q5_0` | 256 | 6 | 1024 | 2048 | 1053 | 796 | -24.4% |
| `q5_0` | 256 | 6 | 4096 | 512 | 1042 | 723 | -30.6% |
| `q5_0` | 256 | 6 | 4096 | 2048 | 3834 | 2800 | -27.0% |
| `q5_0` | 256 | 6 | 16384 | 512 | 4760 | 2753 | -42.2% |
| `q5_0` | 256 | 6 | 16384 | 2048 | 16738 | 10933 | -34.7% |
| `q5_0` | 256 | 2 | 1024 | 512 | 187 | 154 | -17.9% |
| `q5_0` | 256 | 2 | 1024 | 2048 | 609 | 543 | -10.8% |
| `q5_0` | 256 | 2 | 4096 | 512 | 670 | 490 | -26.8% |
| `q5_0` | 256 | 2 | 4096 | 2048 | 2157 | 1892 | -12.3% |
| `q5_0` | 256 | 2 | 16384 | 512 | 4504 | 1935 | -57.0% |
| `q5_0` | 256 | 2 | 16384 | 2048 | 11231 | 7698 | -31.5% |
| `q5_1` | 256 | 6 | 1024 | 512 | 295 | 218 | -26.3% |
| `q5_1` | 256 | 6 | 1024 | 2048 | 1057 | 800 | -24.3% |
| `q5_1` | 256 | 6 | 4096 | 512 | 1046 | 729 | -30.3% |
| `q5_1` | 256 | 6 | 4096 | 2048 | 3837 | 2839 | -26.0% |
| `q5_1` | 256 | 6 | 16384 | 512 | 4818 | 2797 | -41.9% |
| `q5_1` | 256 | 6 | 16384 | 2048 | 16840 | 11062 | -34.3% |
| `q5_1` | 256 | 2 | 1024 | 512 | 190 | 155 | -18.1% |
| `q5_1` | 256 | 2 | 1024 | 2048 | 615 | 550 | -10.7% |
| `q5_1` | 256 | 2 | 4096 | 512 | 675 | 496 | -26.4% |
| `q5_1` | 256 | 2 | 4096 | 2048 | 2161 | 1909 | -11.7% |
| `q5_1` | 256 | 2 | 16384 | 512 | 4538 | 1997 | -56.0% |
| `q5_1` | 256 | 2 | 16384 | 2048 | 11158 | 8104 | -27.4% |

At D=256 the route is 19-46% faster at GQA 6 and 3-61% at GQA 2, and 19-27% (3060) and 10-23% (4070) at D=512. Each value is the faster of two runs, for both paths alike.

### Decode and small query batches

Each cell is "without the flag / with it (change)", us/run, the faster of two or three runs. Without the flag a single query token (two on Ada) takes the vector kernel, larger batches the cast and the F16 MMA kernel.


**RTX 3060 (Ampere, sm_86)**

| type | D | GQA | n_kv | n_q=1 | n_q=2 | n_q=3 | n_q=4 | n_q=8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 4096 | 112 / 23 (-80%) | 170 / 30 (-83%) | 174 / 32 (-82%) | 174 / 32 (-82%) | 187 / 54 (-71%) |
| `q4_0` | 256 | 6 | 16384 | 402 / 64 (-84%) | 632 / 85 (-87%) | 637 / 87 (-86%) | 636 / 88 (-86%) | 650 / 145 (-78%) |
| `q4_0` | 256 | 6 | 65536 | 1551 / 231 (-85%) | 2489 / 309 (-88%) | 2493 / 312 (-87%) | 2493 / 314 (-87%) | 2515 / 514 (-80%) |
| `q4_0` | 256 | 2 | 4096 | 81 / 41 (-50%) | 326 / 42 (-87%) | 326 / 42 (-87%) | 327 / 43 (-87%) | 330 / 49 (-85%) |
| `q4_0` | 256 | 2 | 16384 | 284 / 123 (-57%) | 1250 / 125 (-90%) | 1253 / 126 (-90%) | 1253 / 127 (-90%) | 1255 / 160 (-87%) |
| `q4_0` | 256 | 2 | 65536 | 1077 / 457 (-58%) | 4987 / 459 (-91%) | 4981 / 461 (-91%) | 4984 / 463 (-91%) | 5024 / 643 (-87%) |
| `q4_0` | 512 | 16 | 4096 | 113 / 57 (-49%) | 85 / 44 (-48%) | 94 / 45 (-52%) | 93 / 44 (-53%) | 122 / 71 (-42%) |
| `q4_0` | 512 | 16 | 16384 | 264 / 122 (-54%) | 266 / 123 (-54%) | 311 / 124 (-60%) | 312 / 126 (-60%) | 373 / 188 (-50%) |
| `q4_0` | 512 | 16 | 65536 | 969 / 455 (-53%) | 1002 / 456 (-55%) | 1189 / 458 (-62%) | 1188 / 462 (-61%) | 1398 / 664 (-53%) |
| `q8_0` | 256 | 6 | 4096 | 106 / 35 (-67%) | 171 / 42 (-75%) | 174 / 43 (-75%) | 175 / 44 (-75%) | 188 / 60 (-68%) |
| `q8_0` | 256 | 6 | 16384 | 377 / 112 (-70%) | 637 / 121 (-81%) | 640 / 123 (-81%) | 641 / 124 (-81%) | 656 / 160 (-76%) |
| `q8_0` | 256 | 6 | 65536 | 1515 / 426 (-72%) | 2503 / 437 (-83%) | 2507 / 438 (-83%) | 2509 / 440 (-82%) | 2528 / 571 (-77%) |
| `q8_0` | 256 | 2 | 4096 | 80 / 64 (-19%) | 327 / 64 (-80%) | 328 / 66 (-80%) | 328 / 66 (-80%) | 334 / 69 (-79%) |
| `q8_0` | 256 | 2 | 16384 | 274 / 219 (-20%) | 1260 / 219 (-83%) | 1265 / 220 (-83%) | 1261 / 221 (-82%) | 1266 / 229 (-82%) |
| `q8_0` | 256 | 2 | 65536 | 1049 / 845 (-20%) | 4994 / 846 (-83%) | 4995 / 848 (-83%) | 4997 / 849 (-83%) | 5005 / 866 (-83%) |
| `q8_0` | 512 | 16 | 4096 | 89 / 45 (-50%) | 83 / 45 (-45%) | 95 / 46 (-51%) | 95 / 47 (-50%) | 120 / 73 (-39%) |
| `q8_0` | 512 | 16 | 16384 | 282 / 134 (-53%) | 282 / 135 (-52%) | 328 / 137 (-58%) | 328 / 138 (-58%) | 389 / 198 (-49%) |
| `q8_0` | 512 | 16 | 65536 | 1039 / 503 (-52%) | 1071 / 505 (-53%) | 1260 / 507 (-60%) | 1260 / 512 (-59%) | 1468 / 710 (-52%) |

**RTX 4070 (Ada, sm_89)**

| type | D | GQA | n_kv | n_q=1 | n_q=2 | n_q=3 | n_q=4 | n_q=8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `q4_0` | 256 | 6 | 4096 | 49 / 11 (-79%) | 99 / 15 (-85%) | 58 / 15 (-73%) | 59 / 16 (-72%) | 68 / 25 (-63%) |
| `q4_0` | 256 | 6 | 16384 | 176 / 28 (-84%) | 358 / 38 (-89%) | 384 / 39 (-90%) | 386 / 40 (-90%) | 394 / 65 (-83%) |
| `q4_0` | 256 | 6 | 65536 | 742 / 167 (-77%) | 1772 / 179 (-90%) | 1537 / 179 (-88%) | 1537 / 181 (-88%) | 1548 / 244 (-84%) |
| `q4_0` | 256 | 2 | 4096 | 33 / 18 (-45%) | 68 / 19 (-72%) | 144 / 19 (-87%) | 147 / 20 (-87%) | 150 / 23 (-84%) |
| `q4_0` | 256 | 2 | 16384 | 129 / 57 (-56%) | 300 / 56 (-81%) | 771 / 57 (-93%) | 770 / 57 (-93%) | 775 / 80 (-90%) |
| `q4_0` | 256 | 2 | 65536 | 554 / 327 (-41%) | 1374 / 328 (-76%) | 3029 / 330 (-89%) | 3035 / 331 (-89%) | 3033 / 341 (-89%) |
| `q4_0` | 512 | 16 | 4096 | 52 / 26 (-50%) | 47 / 22 (-53%) | 49 / 22 (-56%) | 50 / 23 (-55%) | 61 / 33 (-46%) |
| `q4_0` | 512 | 16 | 16384 | 137 / 55 (-60%) | 139 / 56 (-60%) | 161 / 58 (-64%) | 162 / 59 (-63%) | 189 / 85 (-55%) |
| `q4_0` | 512 | 16 | 65536 | 698 / 202 (-71%) | 692 / 206 (-70%) | 723 / 210 (-71%) | 724 / 212 (-71%) | 787 / 301 (-62%) |
| `q8_0` | 256 | 6 | 4096 | 47 / 12 (-75%) | 96 / 16 (-83%) | 58 / 17 (-70%) | 59 / 18 (-69%) | 71 / 27 (-62%) |
| `q8_0` | 256 | 6 | 16384 | 167 / 34 (-80%) | 353 / 51 (-85%) | 402 / 49 (-88%) | 404 / 50 (-88%) | 413 / 86 (-79%) |
| `q8_0` | 256 | 6 | 65536 | 748 / 307 (-59%) | 1910 / 317 (-83%) | 1587 / 319 (-80%) | 1587 / 320 (-80%) | 1600 / 339 (-79%) |
| `q8_0` | 256 | 2 | 4096 | 32 / 21 (-36%) | 65 / 21 (-68%) | 163 / 21 (-87%) | 163 / 22 (-87%) | 164 / 25 (-85%) |
| `q8_0` | 256 | 2 | 16384 | 171 / 159 (-7%) | 352 / 159 (-55%) | 800 / 159 (-80%) | 800 / 159 (-80%) | 805 / 167 (-79%) |
| `q8_0` | 256 | 2 | 65536 | 667 / 607 (-9%) | 1354 / 608 (-55%) | 3148 / 611 (-81%) | 3149 / 611 (-81%) | 3165 / 615 (-81%) |
| `q8_0` | 512 | 16 | 4096 | 32 / 21 (-32%) | 30 / 23 (-24%) | 34 / 23 (-32%) | 34 / 24 (-31%) | 45 / 35 (-24%) |
| `q8_0` | 512 | 16 | 16384 | 147 / 60 (-60%) | 155 / 61 (-61%) | 179 / 62 (-65%) | 179 / 63 (-65%) | 207 / 90 (-57%) |
| `q8_0` | 512 | 16 | 65536 | 738 / 229 (-69%) | 734 / 232 (-68%) | 771 / 236 (-69%) | 771 / 238 (-69%) | 840 / 325 (-61%) |

The native kernels read every K/V row once per KV head and nothing else, so these steps run close to the DRAM bandwidth: `q8_0` decode at GQA 6 and n_kv=65536 reads K/V at about 90% of it on both cards. A single query token at 8 heads per tile, or up to 4 at 2 heads per tile, takes the 8-column tiles. Without the flag, two to eight query tokens pay for the cast, which reads the cache, writes an F16 copy two to four times its size and has the kernel read that again.

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

## Memory

The removed F16 copy is part of the compute buffer, which is reserved for the full context. Compute buffer size reported at context creation, Qwen3.8-27B IQ3_XXS split over RTX 4070 + RTX 3060, `-ub 512`, the same for `q4_0` and `q8_0` caches:

| Context | 4070 off | 4070 on | 3060 off | 3060 on | Saved |
|--:|--:|--:|--:|--:|--:|
| 32768 | 467 MiB | 381 MiB | 704 MiB | 704 MiB | 86 MiB |
| 131072 | 1235 MiB | 765 MiB | 1185 MiB | 1088 MiB | 567 MiB |
| 262144 | 2259 MiB | 1277 MiB | 2209 MiB | 1600 MiB | 1591 MiB |

The copy is `2 * n_kv_heads * head_dim * sizeof(F16)` per token of context, 4 KiB for this model. Where the FlashAttention node is not the largest consumer of the compute buffer, as on the second GPU at short context, less of it shows.
