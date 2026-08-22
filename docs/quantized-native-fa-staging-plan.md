# Plan: staged loading for quantized-native CUDA FlashAttention

Status: **plan only, nothing implemented.** This records what an `nstages > 0`
prototype for the quantized-native MMA path would have to do, what it costs in
shared memory, and what would have to be true for it to be accepted. It exists
so the experiment can start from measured constraints instead of from scratch.

Context: the packed-loader work on this branch is finished. Every natively
loadable cache type now beats the materializing path at 32K prefill, from
`+0.52%` (`q6_0`) to `+7.32%` (`q2_1`). The loader itself is synchronous, and
`nstages` is pinned to `0` for quantized K/V.

## Revisions after the pair-matrix and configuration work

This plan was written against a kernel that specialized V at compile time and an
untested assumption that the configuration table was worth retuning. Both
changed, and both change this plan.

### The configuration is fixed, and the tables below still hold

The quantized-specific configuration sweep was run and **rejected**: every
deviation from the inherited F16 table lost, so `nthreads = 128`,
`occupancy = 2`, `nbatch_fa = 32`, `nbatch_K2 = nbatch_V2 = 128` and
`Q_in_reg = true` are retained unchanged for the production `D = 256` 8x8 shape.
The resource tables in this document were computed against exactly those values,
so they need no recalculation. The prerequisite "do not start pipeline
implementation while the winning configuration is unresolved" is satisfied: the
winner is the incumbent.

### Occupancy 2 -> 1 now has a measured price, and it is disqualifying

Two separate sweep arms dropped the production shape to one block per SM, one
explicitly via `occupancy = 1` and one implicitly by setting `Q_in_reg = false`
so that shared memory became `Q + KV + mask` instead of `max(Q, KV + mask)`.
Both landed within 0.07% of each other at roughly **-9.3%** against
materialization.

So the note below that `q8_0` double-buffered "must be measured rather than
assumed" is superseded: it drops to one block per SM, and one block per SM is
worth about -9% on this shape. A staged `q8_0` would have to find more than 9%
of overlap to break even. Treat it as expected-dead and measure only to confirm,
not as an open question.

### `Q_in_reg = false` is not available as a shared-memory escape hatch

If a staged variant runs short of shared memory, the tempting move is to spill Q
out of registers and reclaim its 33,792 B floor. That was measured directly:
**-9.35%, Welch t = -41.45**, because keeping Q out of registers forces a
shared-memory read per MMA. It is not a lever this plan may use.

For the same reason, the register budget is not the thing to optimise for. That
arm *succeeded* at relieving pressure -- `REG:255` down to `REG:219/222` with
spill unchanged -- and still lost 9.35%. Spill did not predict throughput in
either direction across the sweep.

### Runtime-V kernels change what a raw V region must be sized for

Mixed K/V pairs no longer instantiate per pair. There are now two kernel shapes:

- the **symmetric** kernel, with V as a template argument, used when `K == V`;
- the **runtime-V** kernel, one per K type, which selects its V loader from a
  switch at the tile-load site and therefore does not know V's row stride at
  compile time.

A raw staging region for V in the runtime-V kernel must be sized for the
**largest** V row, not the actual one: 272 B per row (`q8_0`) at `D = 256`,
against 80 B for `q2_0s`. The staging budget before crossing the `Q` floor is
`33,792 - (16,896 + 640) = 16,256 B`.

| K type | Row B | Symmetric, single | Fits | Runtime-V, single | Fits | Runtime-V, double | Occupancy |
|---|---:|---:|---|---:|---|---:|---|
| `q8_0` | 272 | 17,408 | no | 17,408 | no | 34,816 | **1/SM** |
| `q6_1` | 224 | 14,336 | yes | 15,872 | yes | 31,744 | 2/SM |
| `q6_0` | 208 | 13,312 | yes | 15,360 | yes | 30,720 | 2/SM |
| `q5_0` | 176 | 11,264 | yes | 14,336 | yes | 28,672 | 2/SM |
| `q4_0` | 144 | 9,216 | yes | 13,312 | yes | 26,624 | 2/SM |
| `q3_0` | 112 | 7,168 | yes | 12,288 | yes | 24,576 | 2/SM |
| `q2_1` | 96 | 6,144 | yes | 11,776 | yes | 23,552 | 2/SM |
| `q2_0s` | 80 | 5,120 | yes | 11,264 | yes | 22,528 | 2/SM |

The result is better than feared. Because `nbytes_shared_Q` sets a 33,792 B
floor that the K/V tile does not approach, single-buffered raw staging stays
free even in the runtime-V kernel for every type except `q8_0`, and double
buffering still holds two blocks per SM for every type except `q8_0`. Only
`q8_0` is squeezed, and it is squeezed identically in both kernel shapes.

### Consequences for the prototype

- Start on the **symmetric** kernel, not the runtime-V one. It is the common
  configuration, it avoids the worst-case V sizing question entirely, and it is
  the shape whose throughput the retained hybrid depends on.
- `q3_0/q3_0` and `q2_1/q2_1` remain the right first types, now for a stronger
  reason: staging is free in shared memory for them under either kernel shape.
- Do not touch `nbatch_fa`, `nbatch_K2` or `nbatch_V2` to make room. Those were
  swept: `nbatch_fa` 32 -> 64 cost -2.32% and `nbatch_K2`/`V2` 128 -> 64 cost
  -2.66%, both on the production shape.
- Extending staging to the runtime-V kernel is a second step with its own
  measurement, because its raw region is worst-case sized and its loader
  selection is a runtime branch.

## What `nstages` means today

`fattn-mma-f16.cuh` carries three loading modes, selected per `(DKQ, DV, ncols)`
by the config table and then filtered by hardware:

| Value | Behavior |
|---|---|
| `0` | Ordinary synchronous loads, global -> register -> shared. |
| `1` | Synchronous, but issued as `cp.async` followed by `cp_async_wait_all()`. |
| `2` | Double-buffered: `tile_K` and `tile_V` are separate regions and V for the next iteration is prefetched with `cp.async` while K is consumed. |

Two lines force quantized and KVarN caches to `0`:

```
constexpr int nstages = is_kvarn_kv || is_quant_kv ? 0 : ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2);
```

and the host-side twin in `ggml_cuda_flash_attn_ext_mma_f16_case`. For F16 the
same shape would pick `nstages_target = 2` on Ampere and Ada.

## Why `cp.async` cannot simply be switched on

`cp.async` is a byte-for-byte DMA from global to shared memory. It performs no
arithmetic. For an F16 cache that is exactly what is needed, because the bytes
in the cache *are* the contents of the `half2` tile the MMA math reads. For a
quantized cache the bytes must pass through a `dequantize_q*` step, which the
DMA cannot do. That is the whole reason the quantized path is synchronous.

So a staged design has to choose *what* it stages, and there are only two
options.

## Design A: stage raw quant bytes, dequantize shared -> shared

`cp.async` the raw quantized rows into a scratch shared region, then run the
existing packed loaders reading from shared memory instead of global, writing
the `half2` tile as they do now.

What it buys: the global-memory latency of the loader moves off the critical
path and can overlap with the previous iteration's MMA math.

What it costs:

- A second shared-memory region for the raw bytes, sized by cache type.
- An extra `__syncthreads()` between the DMA and the dequantize pass, and a
  second one after the tile write. The dequantize work itself is not removed;
  it only happens later.
- The loaders must be re-expressed against a shared-memory source. Their
  `ggml_cuda_memcpy_1<n, 2>` fetches assume a `const char *` row in global
  memory; shared-memory addressing and the 2-byte alignment workaround both
  change.

### Alignment is the first hard constraint

`cp_async_cg_16` is 16-byte only. Whether a row can be DMA'd depends on the row
stride, which is `(D/32) * sizeof(block)`:

| D | Row strides across the eleven types | 16-byte aligned? |
|---|---|---|
| 256 | 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 272 | yes, all |
| 128 | 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 136 | no (40, 56, 72, 88, 104, 136) |
| 64 | 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 68 | no (most) |

Individual quant *blocks* are never 16-byte aligned for any type, so the DMA
has to move whole rows, not blocks. A first prototype should therefore be
restricted to `D = 256`, which is also the production geometry for the target
model. Covering `D = 64` and `D = 128` needs 4- and 8-byte `cp.async` variants
that do not exist in `cp-async.cuh` yet.

### Shared-memory budget

Measured against the production case: Ada, `DKQ = DV = 256`, `ncols1 = ncols2 =
8`, so `nthreads = 128`, `nbatch_fa = 32`, `nbatch_K2 = nbatch_V2 = 128`,
`Q_in_reg = true`, occupancy target 2.

Current totals: the `half2` K/V tile is 16,896 B at one stage and 33,792 B at
two, but `nbytes_shared_Q` is *also* 33,792 B and `Q_in_reg` makes the total
`max(combine, max(Q, KV + mask))`. The one-stage total is therefore 33,792 B
and the F16 two-stage total is only 34,432 B — the second stage is nearly free
here because Q already sets the floor.

Adding a raw staging region on top, per type:

| Type | Raw K+V, single | total | blocks/SM | Raw K+V, double | total | blocks/SM |
|---|---:|---:|---:|---:|---:|---:|
| `q8_0` | 17,408 B | 34,944 B | 2 | 34,816 B | 52,352 B | **1** |
| `q6_0` | 13,312 B | 33,792 B | 2 | 26,624 B | 44,160 B | 2 |
| `q4_0` | 9,216 B | 33,792 B | 2 | 18,432 B | 35,968 B | 2 |
| `q3_0` | 7,168 B | 33,792 B | 2 | 14,336 B | 33,792 B | 2 |
| `q2_0` | 5,120 B | 33,792 B | 2 | 10,240 B | 33,792 B | 2 |

Two readings of this table matter:

1. For the low-bit types the raw region is **free**. It fits underneath the
   33,792 B floor that `nbytes_shared_Q` already sets, even double-buffered.
   That is the opposite of what one would assume, and it makes the low-bit
   types the right place to start rather than `q8_0`.
2. `q8_0` double-buffered is the one configuration that breaks occupancy:
   2 x 52,352 B = 104,704 B against Ada's 102,400 B per SM, so it drops to one
   block per SM. Any prototype must report occupancy per type, not once.

## Design B: double-buffer the dequantized tile only

Keep the synchronous dequantizing loader, but give it two `half2` tiles so the
loader for iteration `i+1` runs while the math for `i` proceeds.

This is what `nstages = 2` does for F16, minus the part that makes it work.
Without `cp.async` there is no asynchrony: the same warps perform both the
dequantize and the math, so this is software pipelining, not overlap. The only
gain is instruction-level overlap of the loader's global loads with math, which
the compiler already achieves in part inside the existing unrolled loop. It
costs the full second tile (16,896 B here) and buys strictly less than
Design A.

Design B is recorded for completeness and should not be the first experiment.

## Constraints that must be relaxed or worked around

The existing two-stage path carries three static asserts that the quantized
path violates:

- `static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline")`
  — the quantized loader *does* use `oob_check` for the tail batch. Either the
  prototype keeps a synchronous fallback for the final partial batch, or the
  staging path grows a bounds-checked DMA. The former is simpler and costs
  nothing on the interior batches.
- `static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading")`
  — satisfied for `D = 256` (`nbatch_K2 = 128 = 256/2`), so this is not a
  blocker for the first prototype.
- `static_assert(!V_is_K_view, ...)` — `V_is_K_view` is `DKQ == 576` only, so
  irrelevant here.

Also note `ggml_cuda_fattn_mma_get_nstages` returns `0` unless `ncols2 >= 2`.
The production 8x8 case satisfies it; the `ncols2 == 1` shapes would stay
synchronous, which is acceptable for a first cut but means the prototype cannot
be evaluated on those geometries.

## Why the upside is real but the risk is high

The loader is not a rounding error in this kernel. The paired-lane experiment
removed roughly a quarter of the loader's load instructions and *lost* 4.0% to
7.7% across every type. A change that only perturbs the loader's latency
structure moved total kernel throughput by that much, which says the loader sits
on the critical path and that its scheduling dominates its instruction count.

That cuts both ways. It means staging has something real to hide, and it means
a staged prototype can easily lose by the same margin through an extra
`__syncthreads()` or a lost block per SM. The prior from this branch is not
encouraging: of the loader experiments run so far, packing won, the float-domain
zero point won, and paired-lane loading lost decisively.

## Proposed sequence

1. **Smallest useful prototype**: Design A, `D = 256` only, single-buffered raw
   staging, synchronous fallback for the `oob_check` tail batch, `q3_0` and
   `q2_1` only. These are the types where the raw region is free in shared
   memory, so the experiment measures overlap alone with occupancy held fixed.
2. Gate it exactly as the paired-lane variant was gated: `test-backend-ops
   FLASH_ATTN_EXT` for the affected types, no local-memory spills, reported
   occupancy unchanged, then two reverse-order `llama-bench` runs of `r=5`
   against the current library with native attention on in both arms.
3. Only if step 1 wins, extend to double buffering, and only then to `q8_0`,
   whose occupancy cliff is now measured rather than open: one block per SM
   costs about 9% on this shape, so staged `q8_0` must find more overlap than
   that to break even.
4. If step 1 loses, record it in `docs/cpu-kv-offload-experiments.md` the way
   paired-lane loading was recorded and stop. Do not proceed to `D = 64` and
   `D = 128`, which additionally require new `cp.async` widths.

## Effort

Design A step 1 is roughly 2-4 hours of implementation, one 23-minute build, a
25-minute correctness sweep and a 40-minute benchmark per iteration, with two
or three iterations likely. Steps 3 and 4 roughly double that. The honest
expected value is a coin flip on a low-single-digit gain, which is why this is
written down rather than built.
