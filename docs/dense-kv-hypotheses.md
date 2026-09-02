# Dense-model optimization hypotheses: host-resident KV, VRAM, prefill/decode

Working backlog for dense and hybrid-dense models on a VRAM-constrained accelerator.
Reference config while writing: Qwen3.8-27B on an RTX 4070 (12 GiB, gen4 x16). Nothing here
is keyed to an architecture name; every item is expressed in terms of tensor geometry,
placement and link bandwidth.

## Ground rules for this list

Every item below:

- works against `llama/dev` as merged today, with no open pull request as a prerequisite,
- is not a re-measurement of something an open pull request already measures,
- names the mechanism, the measurement that decides it, and what would kill it.

Out of scope on purpose, because it belongs to work already in review:

- pipelining or overlapping the host-to-device delivery of the KV window,
- reading quantized K/V in place inside the attention kernel, and the kernel dispatch
  tables that go with it,
- splitting a host-resident cache by head under `--split-mode tensor`, `--attn-split`,
  and the ordering of `--kv-gpu-layers` across devices.

Two items (H14, H15) address the same cost as the delivery-pipelining work but by a
different mechanism with a higher ceiling. They are kept because they are independent
implementations, not follow-ups. Sequence them after that work lands so the remaining
headroom is known.

## The cost model

A decode graph delivers the whole live window on every host-resident attention layer:

```
bytes_per_decode = n_kv * n_layer_host * (row_size(type_k, n_embd_k_gqa) +
                                          row_size(type_v, n_embd_v_gqa))
```

`n_kv` is `get_n_kv()`, the used extent padded up to `max(n_pad, 256)`. It grows with the
context, and the graph re-delivers all of it for every single token.

For the reference config (16 owned attention layers, `n_embd_k_gqa = n_embd_v_gqa = 1024`,
`q8_0` both sides) that is about 2176 B per token of context per layer, so roughly 35 KiB
per token of context per decode step. At a 19k-token prompt the graph moves about 640 MiB
per generated token, which at gen4 x16 speeds is around 29 ms and caps generation near
34 t/s before any compute is counted.

A fully dense model, where every layer is an attention layer, is roughly 4x worse than the
hybrid reference at the same hidden size.

**Correction from measurement.** Density is the wrong axis; the sliding window dominates it.
Muse-Glimmer-30B is fully dense at 52 attention layers, but 39 of them are SWA-2048 and its
`n_embd_k_gqa` is 256, so beyond the window only 13 layers grow: 7072 B per context token
against the hybrid reference's 34816 B. The fully dense model is 4.9x *cheaper* per context
token here, not 4x worse. Read the geometry, not the architecture class.

Three consequences shape the list:

1. The cost is linear in context and re-paid every token. Only H17 and H18 change that.
2. The cost is independent of how many tokens the graph produces. H2 and H3 exploit this.
3. Host DRAM bandwidth is several times the PCIe bandwidth, so the cache is cheaper to read
   where it already lives than to ship. H5, H10 and H14 exploit this.

Measure host DRAM bandwidth once on the target machine before relying on point 3. The
whole case for H14 rests on that ratio.

**Measured on the reference machine**, RTX 4070 on gen4 x16, 16 host threads:

| path | GB/s |
|---|---|
| host to device, pinned, 256 MiB   | 24.61 |
| host to device, pageable, 256 MiB | 12.63 |
| host DRAM read, 1 thread          | 16.31 |
| host DRAM read, 8 threads         | 42.73 |
| host DRAM read, 16 threads        | 42.83 |

Point 3 is much weaker than assumed: host DRAM is 1.74x the pinned link, not several times
it. H5 measures what the two paths actually deliver rather than what they could.

## Tooling prerequisite

`llama-bench` accepts `-nkvo` and `-ub` and none of the other placement flags, so several
items below cannot be swept from it. `--kv-gpu-layers` is the one with the largest known
effect and is missing everywhere. Add `-kvgl` before starting H4.

**Done.** `llama-bench` now takes `-kvgl`, `-kvcp` and `-rso`, mirroring `--kv-gpu-layers`,
`--kv-cpu-pinned` and `--recurrent-state-offload`. All three sweep and appear as result
columns like every other placement flag. `-rso` turned out to be needed before H5, not just
before H4; see the note below.

## Hybrid models: `-nkvo` also moves the recurrent state

`llama-model.cpp` builds the recurrent memory with `offload_kqv || recurrent_state_offload`,
so on a hybrid model `-nkvo` moves the gated-delta-net state to host memory as well as the
attention KV. The recurrent state does not grow with context, so this buys a fixed and small
amount of VRAM, but it puts every recurrent layer's state ops on the CPU backend.

Measured on the reference config, decode at 2k of context, `-nkvo --kv-cpu-pinned`:

| recurrent state | tg t/s | ms/token |
|---|---|---|
| host (`-rso 0`, the default) | 15.99 | 62.54 |
| device (`-rso 1`)            | 32.93 | 30.37 |

That is 32.2 ms per token, at every context length, and `llama_memory_recurrent` reports the
state it buys back as 149.62 MiB for one sequence, independent of `-c`. Attention KV on the
same run is 8.5 MiB per 256 cells, so 149.62 MiB is about 4500 cells of cache: the trade
loses at any context worth using `-nkvo` for.

Any `-nkvo` measurement on a hybrid model is meaningless without
`--recurrent-state-offload`, and every measurement below sets it. Whether the default should
follow `offload_kqv` at all is a separate question, and it is worth asking: `-nkvo` alone
costs 2.06x of generation for 149.62 MiB.

---

## Quantized KV with `-nkvo` aborted on many models

Found while setting up H13, unrelated to `--split-mode tensor` and not what #57 fixes; it
reproduces on a single device with no split mode at all.

`stage_store_rows()` casts the current rows to the cache type before storing them into a
host-resident cache. That cast is a `GGML_OP_CPY` whose output is a quantized tensor in the
compute buffer, so the backend's `get_alloc_size` sizes it. CUDA pads a quantized tensor
whose row is not a multiple of `MATRIX_ROW_PADDING`, 512, and `GGML_OP_CPY` was missing from
`ggml_backend_op_alloc_size_may_expand()`, so the allocation assert fired at context
creation.

The condition is `n_embd_k_gqa % 512 != 0` with a quantized cache type under `-nkvo`:

| model | `n_embd_k_gqa` | `-nkvo` with `q8_0` |
|---|---|---|
| Qwen3.8-27B      | 1024 | worked, 1024 is a multiple of 512 |
| Muse-Glimmer-30B |  256 | aborted |

`f16` was unaffected; every quantized pair aborted. This is why the whole reference set for
this file is a hybrid model: the fully dense one could not run the configuration at all.

With it fixed, Muse-Glimmer-30B across a 4070 and a 3060, tg64 at depth:

| depth | device KV | host KV | penalty |
|---|---|---|---|
|  2048 | 19.67 | 15.73 | 20% |
|  8192 | 19.23 | 13.69 | 29% |
| 32768 | 17.71 |  9.18 | 48% |

Read the host column as a slow-link case: the second card is gen3 x4, about 3.5 GB/s against
the 4070's 24.6, and it holds half the layers. The fitted slope is 4.8 GB/s, which is what a
mix of those two links predicts.

---

## Tier 0: configuration sweeps, no code

These need only merged code and a benchmark run. Do them first; two of them may close
their own hypotheses without any implementation.

### H1. Prefill transport falls as 1/ubatch, and large ubatches just became affordable

**Claim.** Under `-nkvo`, raising `-ub` reduces total prompt transport close to
proportionally, and the merged compact causal mask removed the reason not to.

**Mechanism.** Prefill delivers the whole live window once per ubatch, so the total prompt
transport is about `N^2 / (2*ub)` bytes per host-resident layer. Doubling `-ub` halves it.
This used to be a bad trade because the dense F16 mask cost `n_kv * ub * 2` bytes, which is
128 MiB at `n_kv = 32768, ub = 2048`. That term is gone for the single-sequence causal case.
What still scales with `ub` is the FFN intermediate, `n_ff * ub * 4` bytes, so the trade is
now compute-buffer VRAM against prompt transport rather than against both.

**Measure.**

```
llama-bench -nkvo 1 -ub 512,2048,4096,8192 -p 4096,16384,32768 -ctk q8_0 -ctv q8_0
```

Record pp t/s and the reported compute buffer size at each `ub`. Repeat without `-nkvo` to
separate the transport effect from the ordinary batching effect.

**Kills it.** If the compute buffer growth forces `--kv-gpu-layers` down by more than the
prefill gain is worth, or if prompt processing is already compute-bound at `ub = 512`.

**Measured: rejected.** Both kill conditions hold on the reference config. Same machine and
model as H5, `-b 8192`, `-rso 1`, 2 reps, pp t/s:

| ub | prompt | host KV | device KV | CUDA0 compute buffer |
|---|---|---|---|---|
|  512 |  4096 | 1154.78 | 1163.10 |  505 MiB |
|  512 | 16384 | 1063.68 | 1088.98 |  505 MiB |
|  512 | 32768 |  953.30 |  991.27 |  505 MiB |
| 2048 |  4096 | 1139.49 | 1148.83 | 2054 MiB |
| 2048 | 16384 | 1071.72 | out of memory | 2054 MiB |
| 2048 | 32768 |  980.97 | out of memory | 2054 MiB |
| 4096 | any   | out of memory | out of memory | - |
| 8192 | any   | out of memory | out of memory | - |

First, the prize is small. The whole prompt transport, measured as host KV against device KV
at `ub = 512`, is 0.7% of prompt time at 4k, 2.3% at 16k and 3.8% at 32k. Prompt processing
is compute-bound at `ub = 512`, so `1/ubatch` is dividing a term that is already noise.

Second, `ub 512 -> 2048` does behave as the mechanism says, recovering most of that: 953.30
to 980.97 t/s at 32k, +2.9%. It costs 1549 MiB of compute buffer. At 32k one owned attention
layer of q8_0 KV is 71 MiB, so that buffer is worth more than the entire 16-layer cache
(1136 MiB). Spending it on `-kvgl` instead buys 12.71 -> 30.69 t/s of generation, +142%,
against +2.9% of prompt. The exchange rate is not close.

Third, the compute buffer grows about 1.01 MiB per ubatch token here, roughly 15x the
`n_ff * ub * 4` term the mechanism above accounts for, so `ub` 4096 and 8192 do not fit at
all on a 12 GiB card with a 9.2 GiB model. Whatever dominates that growth, and not the mask,
is what caps `ub` now.

H19 was gated on this item concluding that large ubatches are the prefill answer. They are
not, on this config.

### H2. A unified cache amortizes one delivery over every slot in the batch

**Claim.** Under `--kv-unified`, aggregate throughput scales close to linearly with the
number of concurrently decoding slots, because the per-token PCIe cost divides by the slot
count.

**Mechanism.** With `kv_unified`, `n_stream == 1` and `get_k()` returns rows `[0, n_kv)` of
one tensor that covers all sequences. The scheduler copies that tensor once per layer per
graph regardless of how many slots contribute tokens to the batch; the mask separates the
sequences afterwards. The delivery is therefore a fixed cost per graph, not per sequence.

If this holds, host-resident KV is a much better fit for a multi-slot server than for a
single interactive stream, which is the opposite of the usual intuition and is not
reflected anywhere in the current defaults or documentation.

**Measure.** `llama-server -nkvo --kv-cpu-pinned --kv-unified -np 8` with a long shared
context, then `scripts/server-bench.py` at 1, 2, 4 and 8 concurrent streams. Report
aggregate and per-stream generation rates separately. The signature to look for is
aggregate t/s rising while per-stream t/s stays roughly flat.

**Kills it.** Compute becoming the floor early, or the non-unified path being forced for
other reasons in the target deployment.

**Measured: mechanism confirmed, claim overstated.** `llama-batched-bench -kvu -pps`, shared
8192-token prompt, 64 generated tokens per slot, same machine and model as H5. Aggregate tg:

| slots | host KV | device KV | host / device | ms per decode graph, host - device |
|---|---|---|---|---|
| 1 |  24.98 |  35.28 | 0.708 | 11.69 |
| 2 |  44.21 |  59.28 | 0.746 | 11.50 |
| 4 |  72.31 |  89.82 | 0.805 | 10.79 |
| 8 |  99.01 | 113.90 | 0.869 | 10.56 |

The last column is the whole hypothesis, and it holds exactly: the delivery costs about 11 ms
per graph no matter how many slots contribute tokens to it. At `n_kv = 8256` the cost model
predicts 11.68 ms, and one slot measures 11.69.

What does not hold is "close to linearly". Aggregate tg rises 3.96x for 8x the slots, not 8x,
because compute grows with the batch. Per-stream tg falls from 24.98 to 12.38.

The useful form of the result is the third column. The host-KV penalty falls from 29% at one
slot to 13% at eight, and host-resident KV scales better with slots than a device-resident
cache does (3.96x against 3.23x). It never wins, it just stops mattering much. A multi-slot
server is the configuration where host-resident KV is least bad, and the single-stream
numbers everyone quotes are its worst case.

### H3. Speculative decoding is worth several times more with a host-resident cache

**Claim.** The speedup from a given draft length is substantially larger under `-nkvo` than
with a device-resident cache, because verification amortizes one delivery over the whole
accepted run.

**Mechanism.** Verifying `d` draft tokens costs exactly one window delivery, the same as
verifying one. With a device-resident cache, speculation buys the difference between a GEMV
and a small GEMM. With a host-resident cache it divides the dominant cost outright. The
fork already ships MTP and an independent draft residency control, so nothing needs writing.

**Measure.** Two arms, same model and prompt: device-resident KV, and
`-nkvo --kv-cpu-pinned`. Sweep `--spec-draft-n-max 0,2,4,8,16` in each (`--draft-max` has
been removed). The result is the ratio of
the two speedup curves, not either curve on its own. Also record accepted-tokens-per-draft
so the two arms are compared at equal acceptance.

**Kills it.** Draft-model KV also being host-resident and eating the gain; check
`--spec-draft-kv-gpu-layers` keeps the draft cache on device.

**Measured: confirmed.** Same model, a 27312-token prompt, 256 generated tokens, greedy at
`--temp 0` with a fixed seed, DFlash2 sidecar as the draft, `--spec-draft-kv-gpu-layers 99`
in the host arm. tg t/s and the speedup over that arm's own `n_max = 0`:

| `--spec-draft-n-max` | device KV | speedup | host KV | speedup | ratio |
|---|---|---|---|---|---|
| 0 | 21.9 | 1.00 |  4.8 | 1.00 | -    |
| 2 | 32.0 | 1.46 |  9.6 | 2.00 | 1.37 |
| 4 | 32.4 | 1.48 | 11.0 | 2.29 | 1.55 |
| 8 | 27.9 | 1.27 | 11.2 | 2.33 | 1.83 |

The last column is the result. Speculation is worth 1.37x to 1.83x more under a host-resident
cache than under a device-resident one, and the gap widens with draft length. The shapes
differ for the reason the mechanism gives: the device arm peaks at `n_max = 4` and regresses
at 8, because past that the draft costs more than the verification saves, while the host arm
keeps climbing because the one delivery it amortizes dominates everything else.

Both arms are the same model, prompt, seed and temperature, so acceptance per draft is
identical by construction rather than by measurement.

Caveat on the absolute numbers. The DFlash sidecar shares `output.weight` with the target and
has to sit on the same device stack, and target plus draft do not fit on the 4070 alone, so
this ran with the model split across both cards. The second card is gen3 x4, about 3.5 GB/s,
which is why the host baseline is 4.8 t/s here against 12.7 t/s for the single-card run in
H5. The ratio is unaffected because both arms share the topology, but read the left two
columns as a slow-second-link case, not as the reference config.

That link asymmetry is worth its own note: a host-resident cache behind a gen3 x4 slot is
about 7x worse than behind gen4 x16, so on a mixed-slot machine the `--kv-gpu-layers`
ordering across devices matters far more than any item on this list.

### H4. Evict non-KV weights to host memory and buy KV residency with the VRAM

**Claim.** For a dense model, moving weights that are read rarely or in small slices to
host memory, and spending the freed VRAM on `--kv-gpu-layers`, is a net win at long context.

**Mechanism.** `token_embd.weight` is read by `GET_ROWS` for at most `n_ubatch` rows per
graph, so running it on the host costs almost nothing per token, and on a large vocabulary
it is several hundred MiB. A device-resident KV layer removes its whole delivery from every
decode step. The exchange rate is heavily in favour of the trade, and nothing currently
makes it for the user.

An untied output projection is a second candidate but a much weaker one: it runs for every
generated token, so it should be measured separately rather than assumed. A tied output
projection is not a candidate at all.

**Measure.** Hold total device memory constant across two arms:

```
-ot 'token_embd\.weight=CPU' --kv-gpu-layers N
                             --kv-gpu-layers N-k      (k chosen so both arms fit the same)
```

Compare tg at 16k and 32k, and pp to confirm the host-side gather is not hurting prefill.

**Kills it.** Prefill regressing measurably from the host-side gather at large `ub`.

### H5. Find the context length where host-side attention beats shipping the window

**Claim.** There is a crossover context length beyond which running attention on the CPU,
over the cache where it already lives, beats delivering the window to the accelerator.

**Mechanism.** `cparams.offload_attn_compute` is
`offload_kqv || (op_offload && kv_cpu_pinned)`, so toggling `--kv-cpu-pinned` on a
`-nkvo` run flips the entire attention region between the CPU and the accelerator. That
makes the crossover directly measurable today with no code at all. Host DRAM is
several times PCIe bandwidth, and decode attention is bandwidth-bound on both sides, so a
crossover should exist.

This is not a hypothesis about a speedup. It is the measurement that decides whether H10
and H14 are worth building, and it is the cheapest way to get it.

**Measure.** Arm A `-nkvo`; arm B `-nkvo --kv-cpu-pinned`. Sweep context 2k, 4k, 8k, 16k,
32k, 64k. Record tg for each, plus a host memory bandwidth number from any stream benchmark
so the crossover can be predicted rather than only observed.

**Kills it.** No crossover inside the usable context range, which would mean the CPU
attention implementation, not the link, is the limit. That is itself a useful result and
points at the CPU kernels rather than at H14.

**Measured: rejected.** RTX 4070, Qwen3.8-27B-UD-IQ2_M, all weights on the device, q8_0 K
and V, `-ub 512`, `-rso 1`, tg64 at depth, 3 reps.

| depth | device KV | host KV, device attn | host KV, host attn |
|---|---|---|---|
|  2048 | 36.84 | 32.93 | 29.78 |
|  4096 | 36.43 | 29.92 | 25.32 |
|  8192 | 35.57 | 25.14 | 19.41 |
| 16384 | 33.83 | 18.98 | 13.30 |
| 32768 | 30.69 | 12.71 |  8.13 |

There is no crossover. Shipping the window wins at every context, and the margin grows from
10.6% at 2k to 56.3% at 32k.

Fitting ms/token against depth gives one fixed cost and one slope per arm. All three arms
agree on the fixed cost to within 0.8 ms, which is what makes the slopes comparable:

| arm | fixed | slope, ms per 1k of context | implied bandwidth |
|---|---|---|---|
| device KV            | 26.78 ms | 0.177 | 197 GB/s of VRAM |
| host KV, device attn | 27.15 ms | 1.573 | 22.1 GB/s over the link |
| host KV, host attn   | 27.61 ms | 2.912 | 12.0 GB/s of host DRAM |

Against the ceilings measured on the same machine, the delivery reaches 90% of the 24.6 GB/s
pinned link, while the CPU attention path reaches 28% of the 42.8 GB/s host DRAM read. The
raw ratio the hypothesis assumes is only 1.74x here, and the CPU kernel gives away 3.6x of
it, so the host side ends up 1.85x slower than the link it was supposed to beat.

The host arm does not scale with threads, so the shortfall is not a matter of giving it more
of the machine. At depth 16384, `-t 4,8,16`: 13.32, 13.28, 13.31 t/s. One thread of host DRAM
read is already 16.3 GB/s, so this path is running at roughly single-thread speed no matter
how many threads it is given.

### H6. Decide whether pinning the whole cache pays for itself

**Claim.** `--kv-cpu-pinned` allocates the entire host cache with `cudaMallocHost`, and at
large `-c` the cost of that pinning may outweigh what the async copies gain.

**Mechanism.** The host buffer type is selected per layer for the whole cache, so at
`-c 262144` this pins multiple GiB of host memory for the process lifetime. Pinning is slow
to establish, is not swappable, and reduces what the rest of the machine can do. The
benefit is that copies can be genuinely asynchronous and reach full link bandwidth.

The flag currently defaults off and there is no guidance on when to set it, largely because
its two effects are entangled: it also flips attention placement (see H5). Those two effects
should be measured apart.

**Measure.** Startup time and RSS at `-c` of 32k, 128k, 256k with and without the flag.
Then, holding attention placement fixed, compare achieved copy bandwidth pinned against
pageable.

**Kills it.** Nothing; this produces guidance either way. If pinning does not pay at large
`-c`, the interesting follow-up is a bounded pinned staging window rather than a fully
pinned cache.

---

## Tier 1: contained changes on existing machinery

### H7. Stop round-tripping the rows this ubatch just wrote

**Claim.** Delivering only the rows this graph did not write, and writing the current rows
into the staged window device-side, removes one host round trip per attention layer and
lets every KV store be deferred to the end of the graph.

**Mechanism.** Today each attention layer runs: compute K/V on device, cast to the cache
type on device, copy the staged rows to host, run `SET_ROWS` on the CPU backend, copy the
whole window back to device, attend. The rows the graph just computed travel to the host
and come straight back.

Deliver `[0, n_kv - n_written)` from host into a staged window that has room for the tail,
and write the current rows into that tail on the device. The window the kernel reads stays
a single contiguous device tensor, so no attention change is needed.

The second-order effect is the larger one. Once the store is no longer a producer for this
graph's own attention, it has no in-graph consumer at all, so every `SET_ROWS` can be moved
to the tail of the graph and merged into one CPU split instead of one device-to-host
ping-pong per layer. Fewer splits also means CUDA graph capture covers much more of a
decode step.

The precondition is already proven elsewhere in the tree: `can_use_compact_causal_mask()`
verifies `sinfo->idxs[0][i] == physical_base + i`, that is, that this ubatch's cells are
contiguous at the tail of the window. Reuse that check rather than writing a second one.

**Touches.** `src/llama-kv-cache.cpp` (`get_k`, `get_v`, `build_kv_store`),
`src/llama-graph.cpp` (`build_attn`).

**Measure.** Split count per decode graph before and after; `tg` at 16k and 32k; byte-exact
greedy output against the ordered path.

**Risk.** The window tensor stops being a plain view of the cache, so anything that assumes
the delivered tensor is exactly the cache view needs auditing. Non-contiguous slot
assignment must fall back to the current path.

### H8. Give the compact causal mask a (lo, hi) pair so it survives batching and SWA

**Claim.** Extending the compact causal mask from a single upper bound to a bound pair
brings the multi-sequence, multi-stream and sliding-window cases back into scope.

**Mechanism.** `can_use_compact_causal_mask()` currently refuses when `n_seqs_unq != 1`,
`n_stream != 1`, `n_swa != 0`, or ALiBi is in use. The merged optimization therefore applies
to a single-stream run and switches itself off in the configuration people actually deploy.

One I64 bound per query row becomes two, the kernels compare against both ends instead of
one, and batched server traffic and sliding-window layers both become expressible. ALiBi
stays excluded; it needs a real slope per cell, not a bound.

The dense mask it replaces is `n_kv * n_tokens * n_stream * 2` bytes, which is 128 MiB of
compute buffer at `n_kv = 32768, ub = 2048`, written and read every prefill graph. This is
the largest single VRAM item still on the table for a server workload, and it compounds
with H1, which wants `ub` higher still.

**Touches.** `src/llama-kv-cache.cpp` (`can_use_compact_causal_mask`),
`ggml/src/ggml-cuda/fattn-vec.cuh`, `fattn-tile.cuh`, `fattn-mma-f16.cuh`, `fattn-common.cuh`.

**Measure.** Compute buffer size and pp/tg for a 4-slot and 8-slot server run at
`ub = 2048`; byte-exact output against the dense-mask path for every case newly admitted.

**Risk.** Kernel surface. Each admitted case needs a `test-backend-ops` case, and the
sliding-window bound interacts with the existing `KV_max` scan.

### H9. Let the accelerator write the KV store straight into pinned host memory

**Claim.** A CUDA `SET_ROWS` or `CPY` whose destination is a `cudaMallocHost` buffer can be
executed on the accelerator, removing the device-to-host staging copy and the CPU split
that follows it.

**Mechanism.** `ggml_backend_cuda_device_supports_buft()` accepts the CUDA host buffer type
only when the device reports `integrated`. On a discrete card under UVA on 64-bit Linux,
`cudaMallocHost` memory is directly addressable from a kernel, so that restriction is
conservative rather than necessary for these two ops. Admitting the host buffer type for
`SET_ROWS` and `CPY` only, gated on `--kv-cpu-pinned`, lets the store kernel write across
the link and deletes both the staging copy and the split.

Write traffic is small at decode, one row per token per layer, so this is a split-count and
latency win rather than a bandwidth one. It compounds with H7: together they take a decode
graph from roughly two splits per attention layer to a handful for the whole graph.

**Touches.** `ggml/src/ggml-cuda/ggml-cuda.cu` (`supports_buft`, `supports_op`),
`set_rows.cu`, `cpy.cu`.

**Measure.** Split count and tg, as H7. Also confirm the merged fused `f32 -> q8_0` pair CPY
still fires, since its fusion check requires both destinations to be in the same CUDA
buffer type.

**Risk.** Widening `supports_buft` unconditionally would change placement for unrelated
ops and could pull large computations onto host memory. The gate must be narrow: these ops,
this usage, this flag. Multi-GPU portability of the pinned allocation also needs checking.

### H10. Decide attention placement per layer, not once per context

**Claim.** The host-versus-accelerator attention decision belongs per layer and per current
window size, not as one context-wide bool.

**Mechanism.** `cparams.offload_attn_compute` is a single bool for the whole context, but
the right answer differs per layer. A layer whose KV is device-resident should always attend
on the accelerator. A layer behind a slow link should almost always attend on the host. A
layer behind a fast link depends on the current `n_kv`, because the transfer grows with
context while the attention compute does not.

Measure host-to-device bandwidth per device once at startup through the same path the cache
is delivered on, measure host memory bandwidth once, and the decision becomes arithmetic:
attend on the host when `window_bytes / host_bw < window_bytes / link_bw + kernel_time`.

This is the whole-layer approximation of H14 and needs no new op. H5 supplies the
calibration data.

**Touches.** `src/llama-graph.cpp` (`build_attn_mha`, the
`ggml_backend_sched_set_tensor_backend` call), `src/llama-context.cpp`.

**Measure.** tg across context lengths against both fixed placements; the per-layer policy
should track the better of the two everywhere and beat both where the layers differ.

**Parked on H5.** The policy has nothing to decide on this machine: the accelerator is the
right answer for every layer at every context in 2k to 32k, and the two fixed placements do
not cross. The item only becomes interesting on hardware where they do.

**Risk.** Mixing placement inside a graph adds splits, which is exactly what H7 and H9 are
removing. Land those first, or measure the split cost explicitly.

### H11. Teach `--fit` that a dense model would rather move its KV than its layers

**Claim.** `--fit` currently produces a bad plan for dense models, and adding the host-KV
axis fixes it without new inference machinery.

**Mechanism.** `common/fit.cpp` knows three levers: shrink `n_ctx`, reduce `ngl`, and
overflow MoE expert tensors to host memory. For a dense model the third does not exist, so
`--fit` either collapses the context or pushes whole dense layers to the host, both of
which are far worse than moving the attention KV.

The dense strategy should be: keep every weight resident, move the attention KV to host,
then spend the leftover device memory on `--kv-gpu-layers`, and only then start cutting
context. The projection machinery to size all of that already lives in the same file. The
output is a `-nkvo --kv-cpu-pinned --kv-gpu-layers N` line in the emitted arguments.

This is the item that decides whether anything else on this list reaches a user who has not
read this file.

**Touches.** `common/fit.cpp`.

**Measure.** For a dense model that does not fit: `--fit` output before and after, and the
tg and usable `-c` each plan delivers.

**Risk.** Needs a projection for host-resident KV compute buffers that is accurate enough
not to overcommit. `--kv-gpu-layers` ordering across devices is being handled elsewhere;
consume whatever policy lands rather than duplicating it.

### H12. Trim the padded tail out of the delivery

**Claim.** The scheduler copies the padded window including its empty tail; the shape must
stay padded but the copy length need not.

**Mechanism.** `get_n_kv()` pads to `max(n_pad, 256)` so the graph shape stays reusable
across batches. The scheduler then copies `ggml_nbytes()` of that view. Up to 255 rows per
layer per graph are empty. For the reference config that is up to about 8.5 MiB per decode
graph of pure waste on the critical path of every token.

The scheduler already has precedent for shortening an input copy: the MoE expert-subset
path in `ggml_backend_sched_compute_splits` copies only the used experts. This is the same
shape of change with a simpler predicate.

**Touches.** `ggml/src/ggml-backend.cpp` (`ggml_backend_sched_compute_splits`).

**Measure.** Delivered bytes per graph and tg at 4k, where the padding fraction is largest.

**Risk.** The destination tail must not be read. It is padding beyond `used_max_p1`, so the
mask excludes it, but the kernels round `KV_max` up to a stride and may touch it. Zero the
tail once at allocation rather than every graph, and confirm no NaNs reach the softmax.

### H13. Shift only the live cells

**Claim.** Context shift ropes the entire allocated cache rather than the live extent, and
with a host-resident cache that turns into a full-cache CPU pass.

**Mechanism.** `build_graph_shift()` builds its view over `get_size()*n_stream` rows, not
over the used extent. With a device-resident cache this is wasteful but fast. With a
host-resident cache the destination is a host tensor, so the ROPE node runs on the CPU
backend and reads and writes the whole allocated K cache. At a large `-c` that is gigabytes
of host traffic per shift, and the server triggers a shift on every cache-reuse
`seq_add`, not only on true context overflow.

Restricting the view to the live extent is a small change with a large constant factor
behind it. The follow-up, if the measurement justifies it, is to fold the shift into the
next delivery instead of running it as its own pass.

**Touches.** `src/llama-kv-cache.cpp` (`build_graph_shift`).

**Measure.** Time one context shift at `-c` of 32k, 128k and 256k with a host-resident
cache, live extent held small; then again with the cache nearly full. The gap between those
two is the whole hypothesis. Also time a server cache-reuse turn end to end.

**Risk.** Cells outside the live extent must genuinely never be read before they are
rewritten. Confirm against the defrag and `seq_cp` paths.

**Measured and implemented.** The risk turned out to be void. A cell at or past
`used_max_p1()` is empty, and `set_input_k_shift()` already wrote a shift of 0 for an empty
cell, so the old graph was applying an identity rotation to the whole tail. Skipping it is
exactly equivalent, and logits after a shift are bit-identical at every size tested.

Muse-Glimmer-30B, host-resident cache, 256 live cells, cost of the shift above a plain
decode:

| `-c` | plain decode | before | after |
|---|---|---|---|
|   8192 | 49.3 ms |  66.8 ms | 19.6 ms |
|  32768 | 49.1 ms | 197.8 ms | 16.1 ms |
| 131072 | 49.3 ms | 687.2 ms | 16.6 ms |

At `-c 131072` one shift cost 14x a plain decode for 256 live tokens. The remaining cost no
longer grows with `-c`, which is the whole point. Several streams keep the old behaviour,
because their per-stream extents are not contiguous in the cache and one rope view cannot
express them.

Note for anyone repeating this: Qwen3.8 cannot be used to measure it. It is an M-RoPE model,
and `seq_add()` asserts `n_pos_per_embd() == 1`, so context shift does not apply to it at all.

---

## Tier 2: structural

### H14. Split the KV between host and accelerator and merge the two partial softmaxes

**Claim.** Attending over the host-resident prefix on the CPU and over the
device-resident tail on the accelerator, then merging the two partial results, removes the
link from the decode critical path instead of hiding it behind compute.

**Mechanism.** The CPU reads the prefix at host DRAM speed and produces a partial
attention result. The accelerator attends over whatever is device-resident. The two are
merged with the standard numerically stable online-softmax combine over `(O, m, l)`.
Nothing crosses the link except a few hundred bytes of partial state per head.

Decode cost goes from `prefix_bytes / link_bw` to `max(prefix_bytes / host_bw, gpu_work)`.
The bandwidth ratio is the headline, but the concurrency may be worth as much: today the
accelerator idles on the link, and here both sides work at once.

What it needs that does not exist: `ggml_flash_attn_ext` able to emit the log-sum-exp
alongside its output, and a merge op that combines partial results. Both are small and
independently testable. Design the split index as a runtime input so the same pair of ops
serves a within-device split later.

**Measure.** H5 first: without a measured host-versus-link bandwidth ratio on the target
machine, this is not worth starting. Then a standalone benchmark of the two new ops before
any integration.

**Risk.** Largest item on the list. The merge must be exact to the last bit against the
single-path result, and the CPU side has to keep up. If H5 shows no crossover, H10 is the
version of this idea that is worth having and this one is not.

**Parked on H5.** H5 found no crossover. H14 replaces a 22.1 GB/s delivery with a 12.0 GB/s
host read, so built today it would be 1.85x slower on the prefix even with a free merge and
perfect overlap. It needs the CPU attention path to reach roughly 2x its current throughput
before the arithmetic turns, and even a perfect CPU kernel at the full 42.8 GB/s DRAM ceiling
only buys 1.9x. Re-open this only after the CPU attention numbers move, or on a machine whose
host-to-link bandwidth ratio is much larger than the 1.74x measured here.

### H15. Read K and V straight out of pinned host memory inside the attention kernel

**Claim.** If the attention kernel dereferences the host pointer directly, the copy is the
compute: no staging buffer, no staging VRAM, and overlap by construction.

**Mechanism.** FlashAttention reads K sequentially with high warp-level concurrency, which
is the access pattern PCIe handles best. If the kernel can sustain close to link bandwidth
reading across the bus, the whole staging apparatus disappears, along with the device memory
it costs, and the delivery stops being a separate graph node at all.

This addresses the same cost as the delivery-pipelining work in review, by a different
mechanism. Sequence it after that lands and re-measure the remaining headroom before
committing.

**Prerequisite measurement.** Before writing any of it: a `test-backend-ops`
`FLASH_ATTN_EXT` case with K and V allocated in `ggml_backend_cuda_host_buffer_type`,
reporting achieved GB/s against both the copy-engine number and the same case
device-resident. If a kernel reading across the bus lands well under what the copy engine
sustains, stop.

**Risk.** PCIe read latency needs enough concurrent warps to hide. Quantized block loads may
have a worse access pattern than a flat copy. Both are settled by the prerequisite
measurement, which is cheap.

### H16. Make the KV cache type a per-layer property

**Claim.** One cache type for the whole model means the most sensitive layer sets the
precision for every layer, and per-layer types recover both VRAM and link bytes.

**Mechanism.** `type_k` and `type_v` are a single choice in `llama_memory_params`. The cache
is already constructed in a per-layer loop that resolves the buffer type and store semantics
per layer, so carrying a type alongside them is structurally small. The work is mostly
measurement: a per-layer ablation to find which layers tolerate `q4_0`.

The cheap precursor, worth running first, is asymmetric K and V types, which the current
code already allows and which nothing has swept: `q8_0/q4_0` and `q4_0/q8_0` against both
symmetric pairs, on perplexity and on tg.

**Touches.** `src/llama-kv-cache.cpp` constructor, `llama_memory_params`, and the
serialization format if per-layer types are to survive a state save.

**Risk.** If the precursor shows K and V are similarly sensitive, per-layer types are likely
to disappoint too. Run the precursor before building anything.

---

## Tier 3: bets that change the output

These cannot be exact. They belong behind their own flags with published quality tables.

### H17. Screen the window with a heavily quantized shadow copy, ship only what matters

**Claim.** Block-sparse attention fits a transport-bound regime far better than it fits a
device-resident one, because the saving is in bytes moved rather than in FLOPs skipped.

**Mechanism.** Keep a very small shadow copy of K on the host, 2-bit or a per-block
summary. Run a cheap screening pass to rank KV blocks by an upper bound on their score.
Deliver only the top-k blocks at full precision for the exact pass. Transport becomes
`screen_bytes + k * block_bytes`, and `k = all` recovers current behaviour exactly, which
makes it a knob rather than a fork in the road.

**Risk.** The screening pass is itself a transfer. If the shadow copy is not small enough it
eats the saving outright. Sizing it is the first experiment.

### H18. Opt-in eviction, the only thing that stops per-token transport growing

**Claim.** A capped live window turns the per-token delivery from O(context) into O(1).

**Mechanism.** Sink-plus-window, or a scored policy, caps what the window covers. Every
other item on this list reduces a cost that still grows linearly with context. This one
bounds it, which at very long context is a difference in kind.

**Risk.** Changes output by construction. Needs its own flag and a quality table held to the
same standard as the exactness evidence elsewhere in the fork.

### H19. Half-precision activations through prefill

**Claim.** The compute buffer during prefill is dominated by F32 activation terms, and
halving them converts directly into resident KV layers.

**Mechanism.** `n_embd * ub * 4` and `n_ff * ub * 4` dominate. At `n_ff = 12288, ub = 4096`
the FFN intermediate alone is around 200 MiB, and H1 wants `ub` higher still. F16 or BF16
residuals halve both.

**Risk.** Touches every model graph and every backend. Listed last on purpose. Only worth
opening if H1 concludes that large ubatches are the prefill answer and the compute buffer is
what stops them.

---

## Suggested order

1. **H5** and **H1**. One decides whether Tier 2 is worth starting, the other may hand over
   a large prefill win for free.
   *Both done, both rejected.* H5 found no crossover, which parks H10 and H14. H1 found
   prefill already compute-bound at `ub 512`, which also parks H19.
2. **H2** and **H3**. Both test whether the weakest generation numbers are an artifact of
   measuring one stream and one token at a time.
   *Both done, and yes, they largely are.* H2: the delivery is a fixed per-graph cost as
   claimed, so the host-KV penalty falls from 29% to 13% between one and eight slots, though
   it does not scale linearly and never beats a device-resident cache. H3: confirmed, and it
   is the largest effect measured anywhere on this list. Speculation is worth 1.37x to 1.83x
   more under a host-resident cache, rising with draft length.

   Taken together these are the actionable result of the whole exercise, and it is a
   configuration result, not a code one: run host-resident KV with several slots, a unified
   cache and a draft model, and the configuration is far less bad than the single-stream,
   single-token numbers everyone quotes. Nothing in the defaults or the documentation says
   so.
3. **H11**. Until `--fit` knows about host-resident KV, none of the rest reaches a default
   configuration.
4. **H7**, **H9**, **H12**. Contained, compounding, and they clear the way for H10.
   H10 is now parked on H5, so judge these on their own split-count and byte savings.
5. **H8**. The largest remaining VRAM item once H1 pushes `ub` up.
   H1 did not push `ub` up, so this loses its main motivation for the single-stream case. It
   keeps the multi-slot one, which H2 says is the case that matters.
6. Everything else, gated on what the measurements above say.

Standing correction from H1: the compute buffer, not the mask and not the FFN intermediate
alone, is what caps `ub`. Whatever dominates that 1 MiB per ubatch token is the item worth
opening before H8 or H19.
