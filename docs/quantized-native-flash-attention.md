# Quantized-native CUDA FlashAttention

For a small set of measured geometries, the CUDA MMA FlashAttention kernel reads
a quantized K/V cache in place instead of casting the visible attention window to
F16 first. Results are unchanged; what goes away is the transient F16 copy.

It is opt-in. `--flash-attn-native-quants` turns it on; without it every
FlashAttention node keeps the established path. When it is on, the route is
taken wherever a kernel is compiled for the geometry, and nothing else is
consulted: if it does not suit a machine, turn the flag off.

## Why

The standard quantized-K/V MMA route casts the visible K and V window to F16
before the kernel runs, then reads that copy back:

```
2 * n_kv_heads * head_dim * sizeof(F16) * visible_tokens
```

For a four-KV-head, D=256 model that is 4 KiB per visible token, written once
and read once. The native loaders instead dequantize the current tile straight
into the shared-memory `half2` tiles that the existing MMA body already consumes,
so nothing is materialized.

The gain is the traffic, not the allocation, and it is not uniform: at long
context on Ada this is up to 40% faster, at short context it is 8-23% slower,
and Ampere pays on every D=256 row. See **Throughput** for the full matrix, and
**Where the Ampere cost comes from** for why.

## Where it applies

The graph opts in per node via `ggml_flash_attn_ext_set_native_quants()`, which
`llm_graph_context::build_attn_mha()` sets from the context parameter.
`ggml_cuda_fattn_native_supported()` in `ggml/src/ggml-cuda/fattn.cu` then
decides whether a kernel exists and returns the tile shape it uses. Common to
every row:

- an NVIDIA device with Turing MMA or newer (`sm_75`+). Ampere and Ada are
  measured for throughput; Turing is verified correct but has no throughput
  comparison yet: see **Turing** below;
- `logit_softcap == 0`;
- the same native cache type for K and V;
- the GQA optimizations apply (mask present, no ALiBi, padded K/V, aligned strides);
- for `q5_1`, a K/V base pointer and row stride that are 8-byte aligned.

The rows themselves:

| Head dim | GQA ratio | Query batch | Cache types | Tile (sm_80+) | Tile (Turing) |
|---|---|---|---|---|---|
| 256 | 2 | > 16 | `q4_0`, `q8_0` | 32x2 | 16x2 |
| 256 | > 4 | > 4 | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0` | 8x8 | 4x8 |
| 512 | > 4 | > 4 | `q4_0`, `q8_0` | 8x8 | 4x8 |

Each tile shape is the one the generic `switch_ncols1`/`switch_ncols2` would pick
inside those bounds, written out so that the compiled kernel set is exactly the
selectable set. `fattn-mma-quant-decl.cuh` declares the same rows and nothing
else, so a disagreement between the two is a link error.

The two tile columns are the same rows at different widths: `switch_ncols1` caps
`ncols1 * ncols2` at 32 on Turing, so each row loses half its columns there. A
build carries both shapes and picks between them at dispatch, because one build
serves whichever card it runs on.

Nothing narrows those rows further. The caller asked for this route, so the KV
length and where the cache lives are not second-guessed. Two consequences worth
knowing before turning the flag on:

- on Ampere every D=256 row measured slower than the F16 path, by 11% to 42%;
  the D=512 rows gain for `q4_0` and lose for `q8_0`. See **Throughput**;
- D=256 with a GQA ratio of 8 is reachable. PR 55 recorded one measured `q8_0`
  case at that geometry with an open correctness and memory-safety question
  under graph and workspace reuse, which was never diagnosed.

Anything else keeps the standard path. In particular the dispatcher checks the
vector conditions first, so single-token quantized decode still takes the
existing vector kernel rather than being displaced onto this route.

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
be dead code there. The extra tier only adds the D=256 GQA-wide row, because that
is the only row those types can reach.

`q4_0`, `q5_0` and `q8_0` have hand-tuned loaders. `q4_1` and `q5_1` share the
generic nibble loader in `fattn-mma-quant-packed.cuh`, which is correct but was
not tuned per type.

That gives 12 kernels in a default build and 18 with `GGML_CUDA_FA_ALL_QUANTS`:
six and nine type-and-geometry combinations, each at both tile widths.
`scripts/fattn-native-inventory.py` reads the built library back and fails on a
missing, unexpected or duplicated one, and on any mixed K/V or logit-softcap
kernel, neither of which the route can select.

The route is CUDA only. HIP and MUSA exclude the generated instances from their
source globs and `FATTN_MMA_QUANT_AVAILABLE` keeps them from naming the kernels.

## Implementation

The patch changes the storage loaders and reuses the existing F16 MMA
attention/reduction body rather than copying a native-specific attention kernel.
Concretely:

- one `fattn_quant_type_traits<T>` per type, in `fattn-mma-quant-<type>.cuh`.
  Each `dequant()` reproduces that type's F16 cast path bit for bit, because the
  route it replaces is the reference. Which helper achieves that differs per
  type and is documented at each specialization.
- `flash_attn_ext_f16` gains `type_K` / `type_V` template parameters, defaulting
  to `GGML_TYPE_F16`, so the F16 instantiations are unchanged.
- Multi-stage cp.async loading is disabled for native tiles: the loader writes
  the tile itself, so there is no pipeline to stage.
- The tile is XOR swizzled with the same map `fattn-swizzle.cuh` applies to the
  F16 loads, so the MMA body reads it unchanged. The swizzle permutes whole
  16-byte units, which is the store width the loader already uses.

## Scope boundary

The route is deliberately narrow, and the rows above are the boundary. Widening
it costs evidence:

- **A new cache type** owes a tile loader that is bit-identical to that type's
  F16 cast path, its manifest line, equivalence coverage, and matched runtime
  allocation and performance evidence.
- **A new row** (head geometry, GQA ratio, KV-length range) owes its own
  measurement plus a `test-backend-ops` case asserting the route it takes. The
  tile loaders assert alignment against the quant block size, and those
  assertions are what currently confine the head geometry.
- **A new device family** is a separate measured change. Ampere and Ada are
  measured for throughput. Turing is verified correct on hardware but has no
  route-on/route-off comparison, so nothing here claims it is faster there.
- **A non-zero `logit_softcap`** stays on the standard path on purpose:
  compiling the softcap specialization would double the generated kernels for a
  dispatch that cannot reach them.

## Build cost

`CMAKE_CUDA_ARCHITECTURES=86-real;89-real`, CUDA 13.3, Release, one machine,
both tile widths compiled. The base is `llama/dev` at `11f8c737`.

| Build | `libggml-cuda.so` | Delta vs base | Native kernels |
|---|---:|---:|---:|
| base, default | 73,815,568 B | | 0 |
| this, default | 76,261,080 B | +2,445,512 B (+3.31%) | 12 |
| base, all-quants | 107,814,192 B | | 0 |
| this, all-quants | 110,727,992 B | +2,913,800 B (+2.70%) | 14 |

The route table is what keeps both numbers small. PR 55 compiled every tile
shape at D=64, 128 and 256, every mixed K/V pair, and D=512 for all five cache
types it carried: 98 kernels in a default build and 485 with all-quants, for
+29.04% and +130.23% over the base it was measured on. All but a dozen of them
were unreachable.

## Validation

RTX 4070 (`sm_89`, Ada) and RTX 3060 (`sm_86`, Ampere), CUDA 13.3, Release,
`CMAKE_CUDA_ARCHITECTURES=86-real;89-real`.

Correctness, `test-backend-ops -o FLASH_ATTN_EXT`:

| Build | CUDA0 (4070) | CUDA1 (3060) |
|---|---|---|
| default | 2959/2959 | 2959/2959 |
| all-quants | 3972/3972 | 3972/3972 |

Route, `test-backend-ops -o NATIVE_QUANT_EQUIVALENCE`: each case compares the
native result against the same attention over an F16 copy of the same cache, and
asserts which path the dispatcher took by reading the backend's native-launch
counter. A default build runs 6 native and 3 fallback cases, an all-quants build
7 and 5; the difference is the three `q5_0` cases, which a default build skips
because it reports a `q5_0` K/V cache as unsupported. Both counts pass on both
GPUs, and the CI job asserts them, so a run that selects nothing fails instead
of passing vacuously.

Kernel inventory (`scripts/fattn-native-inventory.py`): 12 cases in a default
build, 18 with `GGML_CUDA_FA_ALL_QUANTS`, exactly the declared set and nothing
else. Regenerating the instance files reproduces the committed ones.

### Turing

Correct, not yet compared. Quadro RTX 8000 (TU102, `sm_75`), default build.
These are the one set of numbers here not taken on this base: they come from the
pre-rebase revision, before the tile swizzle and the widened Q4_0 load run.

`test-backend-ops -o NATIVE_QUANT_EQUIVALENCE` passed every case of a default
build, each asserted against the backend's native-launch counter. So the narrow
tile widths produce the same results as the path they replace, and the
dispatcher picks them on real Turing hardware.

End to end, Qwen3.8-27B (D=256, GQA 6) with a `q4_0` cache takes the route and
generates coherent text. Throughput measured 491-588 t/s at `pp512` and about
28 t/s at `tg64`.

Those are absolute numbers with no route-off build beside them, so they say the
route works, not that it is faster. Turing keeps the Ada thresholds in
`ggml_cuda_fattn_native_profitable()` for that reason.

What the Ampere result does suggest: the D=256 regression there comes from the
native loaders forcing `nstages = 0`, which costs a two-stage cp.async pipeline
the F16 path would have used. Turing has no cp.async, so its F16 path already
runs at `nstages = 0` and the native route gives up nothing. That predicts no
regression rather than a gain, and it is reasoning, not measurement.

`scripts/fattn-turing-model-test.sh --ab` produces the missing comparison.

### Throughput

`test-backend-ops perf -o FLASH_ATTN_EXT`, us/run, lower is better. Every route
row against every cache type it admits, on an otherwise idle machine, native
against the F16-casting path with the cast included in both timings. Taken on
this base, with the widened Q4_0 load run.

| D | GQA | type | n_kv | n_q | 4070 F16 | 4070 native | 4070 | 3060 F16 | 3060 native | 3060 |
|--:|--:|--|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 6 | `q4_0` | 1024 | 512 | 292 | 316 | +8.1% | 613 | 729 | +19.0% |
| 256 | 6 | `q4_0` | 1024 | 2048 | 1065 | 1191 | +11.8% | 2184 | 2711 | +24.1% |
| 256 | 6 | `q4_0` | 16384 | 512 | 4743 | 4387 | -7.5% | 9140 | 10716 | +17.2% |
| 256 | 6 | `q4_0` | 16384 | 2048 | 16878 | 16980 | +0.6% | 32821 | 39668 | +20.9% |
| 256 | 6 | `q4_1` | 1024 | 512 | 294 | 333 | +13.3% | 617 | 770 | +24.8% |
| 256 | 6 | `q4_1` | 1024 | 2048 | 1070 | 1288 | +20.3% | 2202 | 2816 | +27.9% |
| 256 | 6 | `q4_1` | 16384 | 512 | 4795 | 4635 | -3.3% | 9192 | 11283 | +22.8% |
| 256 | 6 | `q4_1` | 16384 | 2048 | 16933 | 18669 | +10.3% | 32979 | 41265 | +25.1% |
| 256 | 6 | `q5_0` | 1024 | 512 | 295 | 334 | +13.2% | 620 | 777 | +25.4% |
| 256 | 6 | `q5_0` | 1024 | 2048 | 1071 | 1297 | +21.2% | 2207 | 2859 | +29.5% |
| 256 | 6 | `q5_0` | 16384 | 512 | 4769 | 4725 | -0.9% | 9261 | 11167 | +20.6% |
| 256 | 6 | `q5_0` | 16384 | 2048 | 17008 | 18890 | +11.1% | 33090 | 41804 | +26.3% |
| 256 | 6 | `q5_1` | 1024 | 512 | 295 | 346 | +17.2% | 621 | 808 | +30.1% |
| 256 | 6 | `q5_1` | 1024 | 2048 | 1072 | 1319 | +23.1% | 2208 | 2992 | +35.5% |
| 256 | 6 | `q5_1` | 16384 | 512 | 4795 | 4816 | +0.4% | 9272 | 11877 | +28.1% |
| 256 | 6 | `q5_1` | 16384 | 2048 | 17005 | 19592 | +15.2% | 33151 | 43544 | +31.4% |
| 256 | 6 | `q8_0` | 1024 | 512 | 295 | 330 | +12.1% | 619 | 755 | +21.8% |
| 256 | 6 | `q8_0` | 1024 | 2048 | 1072 | 1254 | +17.0% | 2207 | 2789 | +26.4% |
| 256 | 6 | `q8_0` | 16384 | 512 | 4739 | 4793 | +1.1% | 9245 | 11267 | +21.9% |
| 256 | 6 | `q8_0` | 16384 | 2048 | 17016 | 18420 | +8.3% | 33167 | 41261 | +24.4% |
| 256 | 2 | `q4_0` | 1024 | 512 | 187 | 185 | -1.2% | 372 | 507 | +36.0% |
| 256 | 2 | `q4_0` | 1024 | 2048 | 607 | 672 | +10.6% | 1233 | 1749 | +41.8% |
| 256 | 2 | `q4_0` | 16384 | 512 | 4466 | 2660 | -40.4% | 5446 | 7090 | +30.2% |
| 256 | 2 | `q4_0` | 16384 | 2048 | 10894 | 10595 | -2.7% | 18202 | 24301 | +33.5% |
| 256 | 2 | `q8_0` | 1024 | 512 | 188 | 186 | -0.8% | 373 | 424 | +13.5% |
| 256 | 2 | `q8_0` | 1024 | 2048 | 613 | 676 | +10.3% | 1237 | 1505 | +21.6% |
| 256 | 2 | `q8_0` | 16384 | 512 | 4519 | 2796 | -38.1% | 5476 | 6090 | +11.2% |
| 256 | 2 | `q8_0` | 16384 | 2048 | 11276 | 10217 | -9.4% | 18265 | 21754 | +19.1% |
| 512 | 16 | `q4_0` | 4096 | 512 | 1359 | 1330 | -2.1% | 3682 | 3341 | -9.2% |
| 512 | 16 | `q4_0` | 4096 | 2048 | 5253 | 5240 | -0.2% | 13928 | 12709 | -8.8% |
| 512 | 16 | `q8_0` | 4096 | 512 | 1364 | 1452 | +6.4% | 3684 | 3994 | +8.4% |
| 512 | 16 | `q8_0` | 4096 | 2048 | 5262 | 5719 | +8.7% | 13926 | 15200 | +9.1% |

Read by row rather than as an average. Long context on Ada is where the route
pays: at D=256 GQA 2 and n_kv 16384 it is 38-40% faster, because it reads 4.5 or
8.5 bit weights where the other path reads 16. Short context is where it does
not: at n_kv 1024 every type costs 8-23% there, because the dequant is on the
critical path and there is not enough memory traffic to hide it behind.

Ampere loses on every D=256 row, by 11% to 42%. D=512 is the mirror image: Q4_0
gains 9% on Ampere and Q8_0 loses 8-9% on both cards. **Where the Ampere cost
comes from** below takes that apart.

The two generic loaders are consistently the slowest of the five. Against their
hand tuned counterparts at the same geometry on Ampere, Q4_1 costs about 4
points more than Q4_0 and Q5_1 about 6 more than Q5_0, which is what the hand
tuning is worth.

#### Where the Ampere cost comes from

The native loaders write the shared-memory tile themselves, so they force
`nstages = 0`. Every D=256 entry in the MMA config table sets
`nstages_target = 2` and every D=512 entry sets `1`, so at D=256 the route gives
up a two-stage cp.async pipeline and at D=512 there is none to give up. That is
one cause, and for a long time it was the only one recorded here.

Measuring it needs a third build: the F16-casting path with `nstages` forced to
0, which removes the pipeline as a variable and leaves only the loaders. On the
3060, `test-backend-ops perf`, us/run:

| Row | n_q | F16 `ns=2` | native `ns=0` | F16 `ns=0` | native vs F16 | native vs F16 at equal `ns` |
|---|---:|---:|---:|---:|---:|---:|
| D=256, GQA 6, `q4_0`, n_kv 1024 | 2048 | 2190 | 2761 | 2561 | +26.0% | +7.8% |
| D=256, GQA 6, `q8_0`, n_kv 512 | 2048 | 1162 | 1522 | 1401 | +30.9% | +8.6% |
| D=256, GQA 6, `q8_0`, n_kv 512 | 512 | 329 | 409 | 389 | +24.2% | +4.9% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 2048 | 1224 | 2061 | 1461 | +68.4% | +41.1% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 512 | 371 | 586 | 444 | +58.1% | +31.9% |
| D=256, GQA 2, `q8_0`, n_kv 1024 | 2048 | 1232 | 1498 | 1466 | +21.6% | +2.2% |
| D=256, GQA 2, `q8_0`, n_kv 1024 | 512 | 373 | 425 | 446 | +13.9% | -4.6% |

The last column is the loaders with the pipeline held equal. For `q8_0` it is
between -4.6% and +8.6%: the loader is close to free, and at one row it beats
the path it replaces. For `q4_0` at GQA 2 it is +31.9% and +41.1%. Same tile,
same thread count, same pipeline; the difference is the dequant. `q8_0` converts
bytes to half, `q4_0` extracts nibbles with a mask, a shift, `__byte_perm` and a
bias subtract, and that integer work is what costs on GA106.

Part of it was the per-thread load run rather than the arithmetic.
`fattn_quant_load_width<GGML_TYPE_Q4_0>` used to be 8 for every 128-thread
config, which is every D=256 row, where `q8_0` uses 16. Widening it to 16, which
is what the type now uses, on the 3060:

| Row | n_q | width 8 | width 16 | vs width 8 | width 16 vs F16 at equal `ns` |
|---|---:|---:|---:|---:|---:|
| D=256, GQA 6, `q4_0`, n_kv 1024 | 2048 | 2761 | 2701 | -2.2% | +5.5% |
| D=256, GQA 6, `q4_0`, n_kv 16384 | 512 | 11018 | 10639 | -3.4% | -0.0% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 2048 | 2061 | 1740 | -15.6% | +19.1% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 512 | 586 | 504 | -14.0% | +13.5% |

The 4070 agrees, so the narrower run was not a tradeoff between the two
architectures, just worse:

| Row | n_q | width 8 | width 16 | vs width 8 |
|---|---:|---:|---:|---:|
| D=256, GQA 6, `q4_0`, n_kv 1024 | 2048 | 1234 | 1192 | -3.5% |
| D=256, GQA 6, `q4_0`, n_kv 1024 | 512 | 328 | 316 | -3.8% |
| D=256, GQA 6, `q4_0`, n_kv 16384 | 2048 | 18300 | 16980 | -7.2% |
| D=256, GQA 6, `q4_0`, n_kv 16384 | 512 | 4523 | 4384 | -3.1% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 2048 | 695 | 673 | -3.1% |
| D=512, GQA 16, `q4_0`, n_kv 4096 | 512 | 1325 | 1330 | +0.4% |

The D=512 row already ran at 16, and moves by noise. So `q4_0` now takes the
16-wide default like every other type, and the specialization is gone.

So the D=256 Ampere cost is the cp.async pipeline, plus a load width tuned for
Ada, plus a residual nibble-unpack cost that is real and specific to the nibble
types. At GQA 6 and n_kv 16384, the long-context row, width 16 brings the native
path level with the F16 path at equal `nstages`, and the whole remaining
regression there is the pipeline.

One incidental result from the same runs: at D=512 on Ampere, forcing
`nstages = 0` made the F16-casting path itself faster, 3675 to 2939 us/run. That
is `nstages_target = 1` being a pessimization on GA106 in code this route does
not touch.

#### Staging the quantized tiles, and why it does not work yet

cp.async cannot dequantize; it copies 16 bytes global to shared and nothing
else. So the half2 tile cannot be staged, but the raw quantized bytes can: copy
them into shared while the previous tile computes, then convert out of shared.
That was prototyped and measured, and it is not a win as written.

Against the same native path without it, over all three route rows and all five
types on both cards, it lands between -17.8% and +5.8%. Sorted by cache type the
result is not noise:

| Type | Block size | 4-byte aligned | Staged vs unstaged |
|---|---:|:-:|---|
| `q4_1` | 20 B | yes | -3.3% to -8.7% |
| `q5_1` | 24 B | yes | -5.0% to -9.3% |
| `q4_0` | 18 B | no | +1.3% to +4.9% at D=256 GQA 6 |
| `q5_0` | 22 B | no | +1.7% to +5.8% |
| `q8_0` | 34 B | no | +0.8% to +5.4% |

The two types whose block is a multiple of 4 bytes win everywhere; the three
that are not lose. `block_q4_0` is 18 bytes, so block n starts at byte 18n and
is never 4-byte aligned, and the dequant reads it with a 2-byte aligned access.
Out of global memory that costs little. Out of shared memory it becomes several
sub-word loads, and that costs more than the overlap gains.

Per-block padding would fix the alignment but cp.async copies a contiguous run,
so it cannot scatter blocks to padded slots. Staging would have to hold a
reformatted layout, with the scales separated from the quants so both land
aligned, which changes the loaders rather than only what feeds them.

One row does not fit that account: `q4_0` at D=256 GQA 2 gains 14.0% to 17.8% on
Ampere despite the misalignment, at the same tile shape and thread count as the
GQA 6 row that loses. That is unexplained.

#### Memory

The transient F16 copy is sized by the visible attention window, so what it
costs depends on how much context is actually in use. With the KV cache on the
host, where the GPU holds only the model and the compute buffers, that shows up
directly.

Qwen3.8-27B-UD-IQ2_M on one 4070, `-nkvo`, `q8_0` cache, peak device memory
sampled at 200 ms through a `pp2048` run at each depth:

| Context in use | Route off | Route on | Saved | `pp2048` off | `pp2048` on |
|---:|---:|---:|---:|---:|---:|
| 16384 | 9985 MiB | 9985 MiB | 0 | 800.75 | 804.46 |
| 65536 | 10019 MiB | 10005 MiB | 14 MiB | 469.71 | 482.59 |
| 131072 | 10357 MiB | 10005 MiB | 352 MiB | 305.19 | 321.26 |
| 262144 | 11141 MiB | 10097 MiB | **1044 MiB** | 179.28 | 191.66 |

t/s for the throughput columns, higher is better. The route was asserted on
every row by the backend's native-launch counter, 640 to 8320 launches with it
on and 0 with it off.

The saving is the copy itself: `2 * n_kv_heads * head_dim * n_kv * sizeof(F16)`,
which for this model is 4 KiB per cached token, or 1024 MiB at full context. The
route-on curve is nearly flat, 9985 to 10097 MiB across a 16x increase in
context, while route-off climbs 1156 MiB over the same range.

Nothing shows at 16K because the copy still fits inside pool capacity the
allocator already holds. It appears once the window grows past that, which is
also the regime the route is for.

The throughput columns run the other way from the **Throughput** matrix above,
where D=256 GQA 6 loses on Ampere. That matrix uses a device-resident cache. With
the cache on the host the traffic the route saves outweighs the dequant it adds,
so it is between 0.5% and 5% faster at every depth here. Which side wins depends
on where the cache lives.
