# KV offload: roadmap

This branch contains **no code change**. It is a measurement ground and a plan.

Everything below is either a defect confirmed on `beellama/dev` at `946c1e5b6`
with a fresh build and a reproduction script, or a number measured on the same
base with a patch that is included here. Nothing is carried forward on trust,
and where something could not be confirmed this document says so instead of
repeating it.

- Defects, each with an exact reproduction: [`kv-offload-defects.md`](kv-offload-defects.md)
- Numbers, each with the patch that produced them: [`kv-offload-measurements.md`](kv-offload-measurements.md)
- Scripts: [`repro/`](repro/) · Patches: [`probes/`](probes/)

## Ownership

Three layers. Getting this wrong wastes work, because a fix aimed at the wrong
layer either cannot be merged or has to be maintained forever as a fork patch.

| Layer | Owns | Items |
|---|---|---|
| **upstream llama.cpp** | `ggml`, the CUDA backend, FlashAttention kernels, the scheduler | R1 |
| **BeeLlama** | KVarN in all its forms, its tests, its GGUF formats, vector-kernel routing | R2, R6, R7 |
| **this fork** | the KV-offload line: `--no-kv-offload`, `--kv-cpu-pinned`, `--kv-gpu-layers`, `offload_attn_compute`, `--recurrent-state-offload` | R3, R4, R5 |

A usable test for "is this ours": BeeLlama v0.4.3 carries KVarN across 89 files
and contains no occurrence of `kv_cpu_pinned`, `offload_attn_compute` or
`kv_gpu_layers`. This tree has each in 9-11 files. Touching the second set is
ours; touching KVarN internals is BeeLlama's.

## Summary

| # | Item | Owner | Size | Blast radius | Blocked by |
|---|---|---|---|---|---|
| **R1** | Fix the divergent `__syncthreads()` | upstream | M-L | **very high** | — |
| **R2** | Fix D320 vector routing | BeeLlama | S-M | medium | — |
| **R3** | Interim guard for MTP + `-nkvo` | this fork | S-M | medium | — |
| **R4** | Correct copy/compute overlap | this fork | **XL** | high | design decision |
| **R5** | Re-measure the vector/MMA crossover | this fork | S | low | R1 |
| **R6** | KVarN in the KV-offload line | BeeLlama + us | **XL** | high | ownership decision |
| **R7** | `test-kvarn` on default builds | BeeLlama | S | none | — |
| **R8** | Widen quantized × softcap test coverage | BeeLlama | S | none | — |

---

## R1. Fix the divergent `__syncthreads()`

**Owner: upstream llama.cpp.** Size **M-L**. Blast radius **very high** -- every
CUDA FlashAttention user, not only this configuration.

Defect [D2](kv-offload-defects.md#d2-divergent-syncthreads-in-upstreams-mma-flashattention-kernel).
Localized to `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1732`, guarded by
`if (np > 1 && threadIdx.y % np == 0)` at `:1687` and balanced at `:1768`.
`git blame` attributes both lines to upstream's CUDA maintainer.

**Why it is first.** It gates the two largest numbers in the record:

- MTP with a host-resident cache is **+17.7%** at `-c 8192` and **+75.9%** on a
  24,000-token prompt where it completes, and it aborts (defect
  [D3](kv-offload-defects.md#d3-mtp-with-a-host-resident-kv-cache-aborts-during-sustained-generation)).
- The vector/MMA crossover is **+11.59%** on decode at depth 32,768 (PR #4), and
  cannot be taken because lowering the threshold routes *more* work onto exactly
  the diverging shapes.

**What to do.** File upstream with the reproduction, the `synccheck` output and
the blame. The fix is theirs to choose; the obvious shapes are hoisting the
barrier so every warp reaches the same one, or replacing the block-wide barrier
with a `cooperative_groups` partition over the participating warps. **Do not
patch this quietly in the fork** -- the blast radius is every FA user, and a
silent fork divergence in a barrier is the worst possible thing to maintain.

**Acceptance.** `repro/d2-barrier.sh` reports zero barrier errors, and
`repro/d3-mtp-abort.sh` completes 10 of 10 runs.

## R2. Fix D320 vector routing

**Owner: BeeLlama.** Size **S-M**. Blast radius medium -- changes kernel
selection for one head-dimension family.

Defect [D1](kv-offload-defects.md#d1-head-dimension-320-routed-to-a-vector-kernel-that-does-not-exist).
`320 % 64 == 0` passes `can_use_vector_kernel`, Ada's single-query preference
selects the vector kernel before MMA is considered, and no D320 vector instance
exists, so `ggml_cuda_flash_attn_ext_vec` reaches its terminal
`GGML_ABORT` at `fattn.cu:412`.

**Update to the record.** PR #4 described the confirmed reproducer as GQA 32.
On `946c1e5b6` the suite aborts at `nr23=[4,1], kv=512, nb=75`, so the defect is
reachable at a lower ratio than documented. Whoever fixes it should not assume
the recorded shape is the only one.

**What to do.** Either exclude D320/DV256 from `can_use_vector_kernel` so the
shape falls to MMA, or add the missing vector instance. The first is smaller and
is what the abort implies was intended.

**Acceptance.** `repro/d1-vector-d320.sh` runs the unfiltered
`test-backend-ops -o FLASH_ATTN_EXT` to completion. That command is currently
impossible to pass, which is why PR #4 had to attach an exception to its
validation claim.

## R3. Interim guard for MTP + `-nkvo`

**Owner: this fork.** Size **S-M**. Blast radius medium.

`ncols1 = 4` is the shape speculative verification produces at small draft
batch sizes, which is why MTP reaches R1's defect and plain decode does not. A
guard that routes that shape away from the diverging kernel makes the
configuration usable at some throughput cost, which beats aborting one run in
three.

**Prerequisites, in order.** Confirm the vector kernel is eligible for this head
dimension and cache pair. Measure what the detour costs. Only then write it.

**Delete this when R1 lands.** It is a workaround for someone else's defect and
should not outlive it.

**Acceptance.** `repro/d3-mtp-abort.sh` completes 10 of 10, and the cost of the
detour is recorded next to the guard.

## R4. Correct copy/compute overlap

**Owner: this fork.** Size **XL**. Blast radius high -- scheduler execution loop
plus a new device memory pool.

**Payoff, measured:** +21.3% at 4,096, +77.0% at 16,384, +60.0% at 32,768,
reaching 96-99% of the `max(copy, compute)` ceiling. See
[measurements §3](kv-offload-measurements.md#3-the-overlap-ceiling).

**Why it is available.** The token is `copy + compute` in series and the copy
engine demonstrably retires the transfer underneath the kernels once the
ordering constraint is removed. The bandwidth is already at the practical PCIe
4.0 x16 rate; the ordering is the whole cost.

**Why the two previous attempts failed.** One issued the copy on the compute
stream -- correct, removes the host block, and stream ordering still serializes
it (+0.13%). The other used a copy stream with a device double buffer and got
+1.38% on a build that was **not exact**. Both are in
[measurements §5](kv-offload-measurements.md#5-attempts-that-were-built-and-rejected).

**The structural obstacle.** `ggml-alloc` frees an `input_cpy` after its last
consumer and reuses the block. Writing split `k+1`'s delivery ahead into the
scheduler's own input copies corrupts the split still reading them. That is the
defect class the second attempt hit.

**The shape that works.** A backend-owned ring of staging slots, outside the
graph allocator's reach, one event per slot, lookahead strictly smaller than the
ring depth.

Code to read first, all at `946c1e5b6`:

| What | Where |
|---|---|
| Split execution loop, where input copies are issued | `ggml/src/ggml-backend.cpp:1811` |
| Copy elision -- `supports_buft` only, **no op context** | `ggml/src/ggml-backend.cpp:1052-1071`, used at `:1494` |
| Placement -- `supports_buft` **and** `supports_op` | `ggml/src/ggml-backend.cpp:903-922` |
| Existing fork-local split-input predicate, the pattern to copy | `ggml/src/ggml-backend.cpp:766`, used at `:1370` and `:1453` |
| A second instance of that pattern | `ggml_backend_sched_allows_mapped_host_kv_src`, `fork/exp/kv-gpu-window` @ `9dc737a01` |
| CUDA async copy hook, currently refuses non-CUDA sources | `ggml/src/ggml-cuda/ggml-cuda.cu:2785` |
| The predicate upstream leaves commented out | `ggml/src/ggml-cuda/ggml-cuda.cu:1653` |

**Decision needed before any code is written.** Does the ring live fork-locally
in the CUDA backend, or do we generalize upstream's existing
`n_copies`/`cur_copy`/`events[][]` machinery? Fork-local is faster and
lower-risk; generalizing is upstreamable but is a much larger negotiation.
Recommendation: **build fork-local, prove the gain, then discuss upstreaming.**
That costs a port later and is still the right order.

**Acceptance gates, non-negotiable.** The failed attempt is what skipping them
looks like.

1. Byte-identical greedy server output on all four fixed tasks against the
   control.
2. A/B/A/B at 4,096 / 16,384 / 32,768 with reversed arm order.
3. Device allocation high-water reported -- the ring competes with the model.
4. `GGML_KV_TRANSPORT_STATS=1` showing the deliveries actually converted.

## R5. Re-measure the vector/MMA crossover

**Owner: this fork.** Size **S**. Blast radius low. **Blocked by R1.**

PR #4 measured MMA beating the vector kernel at every batch size -- `+0.99%` at
`n_q = 1` rising to `+18.81%` at 32, and **`+11.59%` on real decode at depth
32,768** -- and did not change the Ada rule that sends quantized `n_q <= 2` to
the vector kernel, because the gain shapes are exactly the diverging shapes.

Once R1 lands: re-measure, then decide the threshold with a low-GQA model as PR
#4 recommends. Cheap, and it is free performance that is currently sitting
behind a defect.

## R6. KVarN in the KV-offload line

**Owner: BeeLlama for KVarN internals, this fork for the offload line.
Unresolved.** Size **XL**. Blast radius high.

Defect [D5](kv-offload-defects.md#d5-kvarn-and-host-resident-kv-silent-cpu-fallback-with-unbounded-workspace):
host-resident KVarN reserves about **0.099 MiB of host workspace per context
token** -- 25,933.90 MiB at 262,144 -- against 20.28 MiB for a standard
quantized cache, and attention falls to the CPU as well. At 262,144 the
workspace is roughly 4.7x the cache it compresses.

**No guard is proposed.** A fail-closed guard was written and discarded: it
converts a slow configuration into an error message, which is not a fix and gets
in the way of the real one.

**Why it happens.** A standard quantized cache reaches attention as copyable
split inputs the scheduler stages to the device. KVarN's records reach attention
through `KVARN_VIEW` chains, so there is nothing to stage and
`ggml_backend_cuda_device_supports_buft` -- which accepts host buffers only on an
integrated GPU -- pushes the whole subgraph to the CPU.

**The work, if it is taken.**

1. Give the KVarN body the same staged-input treatment standard KV gets, so
   attention stays on the accelerator while storage stays on the host. That is
   what `offload_attn_compute` already expresses for standard caches.
2. Decide whether `KVARN_VIEW` operands can be staged at all, or whether the
   records need a device-side mirror with its own residency policy.
3. Extend `--kv-gpu-layers` to KVarN layers so partial residency is expressible.
4. Qualify it the way the standard types were: exactness, KLD, allocation
   high-water, throughput at depth.

**Decision needed first.** Does KVarN host-residency get implemented here, or
does it go to BeeLlama as a requirement? Step 1 changes how `KVARN_VIEW` chains
are built, which is BeeLlama's area, driven by a requirement that exists only in
our line. **This is a conversation, not a technical question**, and it should
happen before anyone estimates the work.

## R7. `test-kvarn` on default builds

**Owner: BeeLlama.** Size **S**. Blast radius none.

Defect [D4](kv-offload-defects.md#d4-test-kvarn-aborts-on-any-default-build-with-a-cuda-device-present).
The test exercises CUDA KVarN paths a default build does not compile, so
`ctest -R "test-kvarn|test-adaptive-dm|test-server-loop-guard"` -- the command
`CLAUDE.md` documents for KVarN changes -- cannot pass as written on a stock
CUDA build. It should skip them when `GGML_CUDA_KVARN` is off, the way the four
static KVarN tests already do.

## R8. Widen quantized × softcap test coverage

**Owner: BeeLlama.** Size **S**. Blast radius none.

Defect [D6](kv-offload-defects.md#d6-quantized-kv-with-non-zero-logit_softcap)
could not be confirmed or closed, because `test-backend-ops` enumerates
`logit_softcap != 0` only at `hsk == 128`
(`tests/test-backend-ops.cpp:10446`) and the quantized cross-product with
softcap is exactly **one case wide**, which passes.

Widening it either reproduces the defect PR #4 recorded or retires the claim.
Either outcome is worth more than the current state, which is that nobody can
tell.

---

## Order

1. **R1** -- largest measured payoff, gates R3 and R5, and it is a defect fix
   rather than a feature. Localization is done; the next step is an upstream
   report.
2. **R7, R8, R2** -- small, independent, and R2 unblocks the full test suite.
3. **R4** -- largest engineering payoff. Settle the ring question first.
4. **R6** -- settle ownership before estimating.

R3 is optional and only worth writing if R1 will take a while.

## What this branch deliberately does not do

No implementation. The five patches under [`probes/`](probes/) are measurement
instruments, three of which produce incorrect output by construction; they are
included so the numbers can be re-derived, not as candidates. The transport
telemetry in patches 01 and 02 is the one piece that could reasonably be landed
on its own, and it is left out of this branch so that the branch stays a plan.
