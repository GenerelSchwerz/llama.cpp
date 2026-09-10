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

The measured gain is the traffic, not the allocation: prefill at depth measured
up to 21.5% faster on Ada, while no measured allocation changed. See
**Validation**, and **Older measurements** for which revision those came from.

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

- on Ampere the D=256 rows measured slower than the F16 path, by up to 14% at
  `n_kv 512`; the D=512 rows were the largest win on either card. See
  **Older measurements**;
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

Correct, not yet compared. Quadro RTX 8000 (TU102, `sm_75`), default build,
taken on the pre-rebase revision described under **Older measurements** below.

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

### Older measurements

The throughput, end-to-end and memory numbers below were taken on the revision
this work carried before it was rebased onto the current `llama/dev`: base
commit `01b141fc`, and before the shared-memory K/V tile gained the XOR swizzle
that `fattn-swizzle.cuh` now applies. The route table and the loaders are
unchanged since, so the shape of the result should hold, but the numbers have
not been re-taken on this base and nothing here claims they were.

#### Throughput

Kernel-level, `test-backend-ops perf -o FLASH_ATTN_EXT`, native against the
F16-casting path with the cast kernel included in both timings. Rows that stay on
the F16 path in both builds move by at most 0.4% on the 4070 and 1.7% on the
3060, which is the noise floor for these numbers.

| Route row | n_q | 4070 (Ada) | 3060 (Ampere) |
|---|---:|---:|---:|
| D=256, GQA 6, `q4_0`, n_kv 16384 | 512 | -21.5% | +0.3% |
| D=256, GQA 6, `q4_0`, n_kv 16384 | 2048 | -15.8% | +2.4% |
| D=256, GQA 6, `q4_0`, n_kv 1024 | 512 | -6.8% | +6.7% |
| D=256, GQA 6, `q4_0`, n_kv 1024 | 2048 | -3.7% | +10.8% |
| D=256, GQA 6, `q8_0`, n_kv 512 | 512 | -3.5% | +9.3% |
| D=256, GQA 6, `q8_0`, n_kv 512 | 2048 | +0.9% | +13.9% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 512 | -17.2% | +1.9% |
| D=256, GQA 2, `q4_0`, n_kv 1024 | 2048 | -9.5% | +7.5% |
| D=256, GQA 2, `q8_0`, n_kv 1024 | 512 | -17.0% | +1.0% |
| D=256, GQA 2, `q8_0`, n_kv 1024 | 2048 | -7.0% | +9.2% |
| D=512, GQA 16, `q4_0`, n_kv 4096 | 512 | -12.2% | -15.6% |
| D=512, GQA 16, `q8_0`, n_kv 4096 | 512 | -6.4% | -5.1% |

Every row is faster on Ada. On Ampere the D=512 rows are the largest win of any
row on either card, and the D=256 rows are slower.

There are three causes, separated below under **Where the Ampere cost comes
from**. An earlier revision of this document blamed the loading pipeline alone;
that is wrong, and the measurement that shows it is there.

End to end, Qwen3.8-27B-UD-IQ2_M (D=256, 24 heads, 4 KV heads, GQA 6) with a
`q4_0` cache on one GPU, route asserted by the native-launch counter:

| Test | 4070 off | 4070 on | 3060 off | 3060 on |
|---|---:|---:|---:|---:|
| `pp512` | 1179.46 | 1178.58 | 534.05 | 533.54 |
| `pp2048 @ d16384` | 982.61 | 1022.77 | 453.35 | 450.35 |
| `tg64 @ d16384` | 33.27 | 33.34 | 17.53 | 17.62 |

t/s, higher is better. Decode is unaffected because a single-token query stays on
the vector kernel.

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

#### Memory

The transient F16 copy that this route removes did not change any measured
allocation on this base.

Reserve compute buffer, Qwen3.8-27B-UD-IQ2_M at 16K context on one 4070, is
505.28 MiB with a `q4_0` cache, 505.28 MiB with `q8_0`, and 505.02 MiB with
`f16`, which has no copy to remove at all. Peak device memory sampled during a
`pp2048 @ d16384` run is 10419 MiB with the route on and with it off.

So on this base another node sets the high-water mark and the copy never reaches
it. The route is worth taking for the throughput above, not for the memory. A
model with more KV heads, or a tree where attention dominates the compute
buffer, may still show the saving; nothing here measures that.
