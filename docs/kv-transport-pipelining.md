# Pipelined delivery of a host-resident KV cache

With `--no-kv-offload` (optionally with `--kv-cpu-pinned`), the attention history lives in host RAM and has to reach the accelerator on every decode token. The backend scheduler used to issue that transfer on the consumer's own stream, right before the kernels that read it, so a token cost `copy + compute` in series.

The transfer and the attention arithmetic are the same as before, but the transfer is issued one split ahead, on a stream of its own, so the copy engine retires it underneath the kernels of the split before it.

`--kv-pipeline-depth N` controls it. It is on by default at `N = 1` and only has an effect where a host-resident cache produces the deliveries; `0` restores the ordered path exactly.

The staging it needs is bounded by `--kv-pipeline-budget` (default 128 MiB), so that a cache which lives on the host to keep device memory free never quietly spends that memory back. Past the cap the scheduler declines and the ordered path runs, at no cost. See [The budget](#the-budget).

## What it changes, and what it must not

R4 pipelines *deliveries*, not attention. Every byte and every attention operation is the same as on the ordered path; only the point at which the transfer is issued moves. Greedy server output is byte-identical, and that is a gate, not an aspiration -- see [Validation](#validation).

Three pieces make it work.

### 1. A stable prefix, so there is something safe to send early

The KV window a split reads is not stable for the whole graph: the same graph writes this ubatch's rows into it, and on a host-resident cache that write is a CPU split that runs *between* the attention of one layer and the attention of the next. Delivering the whole window ahead of that split would send rows that have not been written yet.

What *is* stable is everything below the lowest row this ubatch writes, which at decode depth is essentially the whole window. `ggml_tensor::stable_prefix` records that, in bytes, on the tensor that owns the storage; a view inherits the part of it that its own byte window covers. `llama_kv_cache::update_stable_prefixes()` sets it from the slot info in `apply_ubatch()` -- before the graph is built and allocated, so the scheduler's plan and the deliveries it then issues are decided against the same write position -- and `build_graph_shift()` clears it, because a shift rewrites the body in place.

The scheduler delivers `[0, stable_prefix)` early on the transfer stream and the remainder at the split, once every earlier split of the graph has run. At 18k tokens of context that split is about 620 MiB early against 1-5 MiB late.

The prefix is a hint about *this* graph. It has to be refreshed for every ubatch even when the graph is reused, which is why it is set from `apply_ubatch()` and not from graph construction. Where it cannot be established -- a transposed V cache, whose ubatch writes are scattered across the whole tensor -- it stays 0 and the input keeps the ordered path.

It also says nothing about the *next* graph. A delivery is still reading the host cache after the call that issued it has returned, and the next ubatch writes wherever its own slots fall, which can be below the window the previous graph is still delivering. The scheduler therefore waits for the transfer stream and for the consumer once at the top of each evaluation, before any split of the new graph can write the cache. The ordered path gets the same guarantee from its blocking copy, which pays for it once per split rather than once per graph.

### 2. A ring the graph allocator cannot reach

`ggml-alloc` is free to recycle a graph-owned input copy once its last graph-level consumer is done, and a look-ahead transfer is still in flight outside that lifetime. Writing split `k + 1`'s delivery into the scheduler's own input copies corrupts the split still reading them; that is the defect class the earlier cross-layer prefetch experiment hit (+1.38%, and not exact).

So the scheduler allocates its own ring and points the staged input copies at it before the graph is allocated. A tensor that already has `data` is left alone by `ggml_gallocr_init_tensor`, so the ring sits outside the allocator's reuse analysis rather than competing with it.

Each slot has one ownership cycle:

1. the transfer stream owns an idle slot and writes one future split's prefix into it;
2. it records the slot's `ready` event, which the consumer stream waits for before launching the split that reads the slot;
3. the consumer records `release` once every kernel that reads the slot has been enqueued, and the transfer stream waits for that before overwriting the slot for a later split.

Membership in the ring is decided once, when the ring is laid out, and execution goes by the recorded answer. How much of a staged input can go early moves with every ubatch; *which* input copies live in the ring must not, because their addresses were handed out at allocation time.

Two things disqualify an input that otherwise looks eligible:

- **A reader further down the graph.** The scheduler creates one input copy per (tensor, backend), not per split, so a later split can be pointed at the same copy without appearing to consume it -- and by then the ring may have recycled the slot. The plan scans the splits after the owner for such a reader and puts those inputs back on the ordered path. Attention does not produce this shape, but nothing in the scheduler forbids it.
- **No room on the device.** A slot holds one split's whole delivery, so the ring grows with the context: 27 MiB at 4k, 213 MiB at 32k, 1.7 GiB at 256k. The ring is allocated after the graph allocator has reserved its buffers, so it must not take the room those buffers may still have to grow into; it declines unless it can leave `GGML_SCHED_TRANSPORT_HEADROOM` (512 MiB) free, says so once, and stays on the ordered path.

### 3. A look-ahead that stays clear of the ring's tail

A delivery running `L` splits ahead recycles the slot of the split `L - n_slots` back. With `n_slots == L + 1` the ring is exactly full, so every delivery has to recycle the split that was enqueued a moment ago and is still running -- the ordered path with extra steps. The ring therefore keeps `GGML_SCHED_TRANSPORT_MARGIN` (2) slots behind the look-ahead, and `--kv-pipeline-depth N` allocates `N + 2` slots.

Two details matter as much as the margin:

- **Deliveries are issued after a split is enqueued, never before.** Issuing them first means the host can block on slot recycling while holding back work the consumer could already be running.
- **Slot recycling is ordered stream to stream, not through the host.** A host wait empties the transfer queue for as long as it blocks.

Getting either of these wrong costs the entire gain while still producing correct output, which is the failure mode worth knowing about: on this configuration the first attempt measured `+0.5%` and looked like "the copy simply does not overlap".

## Measurements

RTX 4070 (11,902 MiB usable, sm_89), driver 610.57.04 / CUDA 13.3, i5-13400F, `Qwen3.8-27B-UD-IQ2_M.gguf`, `-ngl 99 -sm none -mg 0 -t 3 -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512`, host residency `-nkvo --kv-cpu-pinned --recurrent-state-offload`, everything under `taskset -c 0,2,4`.

`llama-bench`, `--no-warmup`, A/B/A/B with reversed arm order (`docs/repro/r4-kv-pipeline-ab.sh`), at `--kv-pipeline-budget 512`, both passes shown:

| depth | ordered | pipelined | gain |
|---:|---|---|---:|
| 4,096 | 29.9557, 29.9585 | 34.7899, 34.8016 | **+16.2%** |
| 16,384 | 18.9675, 18.9793 | 29.7606, 29.7697 | **+56.9%** |
| 32,768 | 12.7079, 12.7093 | 15.2760, 15.2630 | **+20.2%** |

Re-measured on the current head as gate 2, and within 0.3% of the same table taken before the graph-boundary wait was added.

The 32,768 ring is 204 MiB at the full context, over the 128 MiB default, so that row needs `-kvpb 512`. `llama-bench` takes the option and the scripts pass it.

> These also need `-kvcp 1 -rso 1`, and for a while `llama-bench` did not have them: the repro scripts probed `--help`, found nothing, and quietly dropped both. The same commit then measures 19.43 -> 9.02 t/s ordered at 16,384 and the pipeline buys +6.7% instead of +60%, because a host-resident recurrent state costs more than the transport can win back. `llama-bench` takes them again, and the scripts now fail rather than drop an option the build does not have.

`llama-server`, one request, `temperature 0, top_k 1, seed 1234`, the four 18,432-prefill tasks of the exactness gate at `-c 32768`:

| task | prompt | ordered | pipelined | gain |
|---|---:|---:|---:|---:|
| prose | 14,821 | 19.738 | 29.955 | **+51.8%** |
| dialogue | 15,984 | 19.136 | 29.699 | **+55.2%** |
| records | 29,603 | 13.531 | 16.406 | **+21.2%** |
| code | 29,670 | 13.484 | 16.340 | **+21.2%** |

Re-measured on the current head as gate 1, and within 0.3% of the same table on the commit before the graph-boundary wait was added, which is the change that could have cost it.

The breakdown below was taken on an earlier head, at prompts of 19,246 and 48,042 against `-c 32768` and `-c 65536`:

| prompt | `-c` | ordered | pipelined | gain | copy ms | compute ms | ceiling | share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19,246 | 32,768 | 17.785 | 29.833 | **+67.7%** | 28.3 | 25.7 | 31.30 | 95.3% |
| 48,042 | 65,536 | 9.790 | 11.313 | **+15.6%** | 76.4 | 25.4 | 11.86 | 95.4% |

`copy` and `compute` are read off `GGML_SCHED_TRANSPORT_DEBUG=2` on each arm, not fitted: the ordered arm reports what it spends blocked in `ggml_backend_tensor_copy` and what it spends waiting for the consumer. The ceiling is `max(copy, compute)` plus the per-token work outside the split loop, which is on both arms.

**The pipeline is within 5% of that ceiling at both depths.** What is left is not a scheduling problem, and the section on the residual below says what it is.

Pinning is worth as much as the pipeline and is off by default. Behind a 13,128-token prompt:

| | ordered | pipelined |
|---|---:|---:|
| `--kv-cpu-pinned` | 21.582 | 32.252 |
| unpinned | 14.945 | 22.709 |

Look-ahead deeper than one split is worse at every depth measured. At 19,246: 29.83 t/s at `N = 1`, 28.44 at `N = 2`, 25.93 at `N = 4`. `N = 1` is the default for that reason. The exactness gate measures the same on every one of its four 18,432-prefill tasks: 29.955 against 27.031 on prose, 29.699 against 26.626 on dialogue, 16.406 against 15.926 on records, 16.340 against 15.864 on code.

### The link is the ceiling, so the lever is bytes

644 MiB in 28.3 ms is 22.0 GB/s, and `nvidia-smi` reports the card at gen4 x16. That is about 88% of what the link delivers in practice, so there is no room left in the transport itself. What is left is to send less. `-ctk q4_0 -ctv q4_0` halves the cache and therefore the traffic:

| prompt | KV | ordered | pipelined | delivered |
|---:|---|---:|---:|---:|
| 19,246 | q8_0 | 17.785 | 29.833 | 644.0 MiB |
| 19,246 | q4_0 | 22.904 | 31.502 | 343.2 MiB |
| 48,042 | q8_0 | 9.790 | 11.313 | 1602.5 MiB |
| 48,042 | q4_0 | 14.146 | 17.717 | 850.6 MiB |

Halving the traffic is worth +5.6% on the pipelined path at 19,246 and +56.6% at 48,042. The difference is the crossover: at 19,246 the pipeline has already brought the copy down to the compute floor, and the consumer wait is 27.31 ms at q8_0 against 27.26 ms at q4_0, the same number. Removing bytes there removes work nothing was waiting for. At 48,042 the copy still dominates and every byte removed is a byte off the token.

**Whether to spend a quantisation step on the cache is a depth question, and the two compound.** At 48,042, q4_0 with the pipeline is 17.717 against 9.790 for q8_0 without it.

### Across context depth, with device memory

`docs/repro/r4-kv-pipeline-context-sweep.sh`, A/B/A/B, peak device memory sampled with `nvidia-smi` across each arm. Both passes agreed to the digits shown.

| Context | ordered | pipelined | gain | peak device memory | delta | ring |
|---|---:|---:|---:|---|---:|---:|
| 4,096 | 29.90 | 34.72 | **+16.1%** | 10,121 -> 10,149 MiB | +28 MiB | 27 MiB |
| 16,384 | 18.98 | 29.72 | **+56.6%** | 10,111 -> 10,215 MiB | +104 MiB | 107 MiB |
| 32,768 | 12.70 | 15.26 | **+20.1%** | 10,113 -> 10,319 MiB | +206 MiB | 213 MiB |
| 65,536 | 7.63 | 8.66 | **+13.4%** | 10,115 -> 10,525 MiB | +410 MiB | 428 MiB |

The two deepest arms were taken on an earlier head and are not re-measured here: +9.1% at 131,072 for +818 MiB of ring, and at 262,144 the ring is declined and the two arms are the same.

**Every row above 20,556 rows of window is measured with the budget raised**, at `-kvpb 0` or `-kvpb 512`. They are what the ring costs and buys if you pay for it, not what a default run does: at the default 128 MiB the ring declines past 20,556 rows and those depths stay on the ordered path. See [The budget](#the-budget).

A finer sweep of the same configuration, `-kvpb 0` throughout so nothing declines, locates the peak between 16k and 24k:

| rows | ordered | pipelined | gain | ring | gain per MiB |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 32.76 | 35.42 | +8.1% | 13 MiB | 0.62 |
| 4,096 | 29.76 | 34.51 | +16.0% | 26 MiB | 0.62 |
| 8,192 | 25.02 | 32.71 | +30.8% | 51 MiB | 0.60 |
| 12,288 | 21.53 | 31.04 | +44.2% | 77 MiB | 0.57 |
| **16,384** | 18.90 | 29.44 | **+55.8%** | 102 MiB | **0.55** |
| 24,576 | 15.18 | 18.84 | +24.1% | 153 MiB | 0.16 |
| 32,768 | 12.65 | 15.19 | +20.1% | 204 MiB | 0.099 |
| 49,152 | 9.51 | 10.99 | +15.6% | 306 MiB | 0.051 |
| 65,536 | 7.63 | 8.66 | +13.6% | 408 MiB | 0.033 |
| 98,304 | 5.45 | 6.06 | +11.1% | 612 MiB | 0.018 |
| 131,072 | 4.26 | 4.66 | +9.5% | 816 MiB | 0.012 |

The gain per MiB is flat below the peak and falls off as `1/rows^2` above it, because the gain decays as `compute/copy` while the ring grows linearly. No depth measured here makes the pipeline slower -- only one past which the memory buys more elsewhere. That is one model on one link, not a general claim.

Two curves run in opposite directions here, and both matter.

**The gain narrows with depth.** A token is copy plus compute; as the context grows the copy grows with it while the compute per staged split does not, so the share of the token that can hide a transfer shrinks. At 16,384 compute still covers most of the copy; by 131,072 it covers a tenth of it. That is arithmetic, not an implementation limit, and no amount of look-ahead changes it.

**The ring's cost does not narrow.** It is `(depth + 2)` slots of one staged split, and a staged split is K and V of one attention layer over the whole context: it doubles every time the context doubles. At 131,072 it claims 818 MiB of an 11,902 MiB card to buy 9.1%.

At 262,144 the ring would need 1.7 GiB against 573 MiB free, so it declines and the run stays on the ordered path -- 2.25 against 2.24 t/s, inside the spread of the ordered arm's own two passes, and 62 MiB of device memory for the transfer backend's context. Declining is the intended outcome, not a failure: the +62 MiB and the unchanged throughput are what "the guard did its job" looks like.

**The ring beats `--kv-gpu-layers` per MiB, and the two barely add up.** Measured behind a 19,246-token prompt at `-c 32768`, where a device-resident layer costs about 68 MiB and the ring costs about 205 MiB:

| | no `--kv-gpu-layers` | `--kv-gpu-layers 4` | `--kv-gpu-layers 8` |
|---|---:|---:|---:|
| ordered | 17.785 | 20.334 | |
| pipelined | 29.843 | 30.263 | 30.640 |

Four device-resident layers are worth +14.3% on the ordered path and +1.4% on the pipelined one. The reason they stop paying is the point of the section above: the pipeline has already moved the bottleneck down to the compute floor, so removing a quarter of the traffic removes something that was no longer being waited for. Whether this still holds where the copy dominates by a wide margin has not been measured.

### Parallel sequences

`llama-batched-bench`, 2,048 prompt tokens per sequence, `-c 32768 -np 8`, generation t/s. Every cell is its own process, because the headroom guard reacts to what a process has already allocated rather than to the configuration: taken as the last step of a sweep that has already run 1, 2 and 4 slots, the 8-slot unified ring is refused and that arm reads 83.88 instead. Three passes, spread at most 0.05 t/s:

| `-npl` | unified ordered | unified pipelined | streams ordered | streams pipelined |
|---:|---:|---:|---:|---:|
| 1 | 32.62 | 35.34 | 32.62 | 35.36 |
| 2 | 51.51 | 58.48 | 34.09 | 58.45 |
| 4 | 72.04 | 86.24 | 49.44 | 85.45 |
| 8 | 83.95 | 105.22 | 70.92 | 105.63 |

A cache split into streams delivers a window per stream, so it moves more than a unified one for the same work, and before the per-stream delivery it could send almost none of it early: 6.6% at 8 slots, because the prefix stopped at the lowest stream's head. Both caches now pipeline to the same throughput, and which of them to use is a question about how the context is shared between sequences rather than about the transport. The ordered arm is the one that separates them: a non-unified cache scales much worse without the pipeline, so the pipeline is worth more there.

**The streams-pipelined column is lower than it was before the multi-stream span was fixed**, and the earlier numbers were wrong rather than better. The delivery sized one stream's range from `ne[2]*nb[2]`, which is one KV cell rather than the window, so it moved a fraction of the bytes and the attention read whatever the ring slot held before. Measured on the same machine, the predecessor reports 61.04, 90.81 and 108.23 at 2, 4 and 8 slots against 58.45, 85.45 and 105.63 here; the difference is the cost of copying the right amount.

**Concurrent slots can be gated on output, with a harness that fixes the batching.** The server cannot: its batching varies between runs, so the same build at the same depth gives different greedy output, and three runs at `N = 0` produced three different hashes. `llama-parallel` seeds its client schedule, so the batches repeat, and `docs/repro/r4-kv-pipeline-parallel-exact.sh` compares the transcripts of 8 concurrent sequences over a non-unified cache. Its clients ask different questions, which is what makes it a gate: with one prompt shared by every sequence the streams hold the same bytes and a cross-stream read is invisible. The predecessor above fails it at `N = 1` on the first sequence.

### The budget

The table above is what the feature costs uncapped, and it is the reason it is capped. A host-resident KV cache exists to keep device memory free; a transport that speeds it up by spending hundreds of MiB of that memory is working against the thing it is accelerating. `--kv-pipeline-budget` (default 128 MiB) is an absolute cap on the ring, not a fraction of what happens to be free:

- Under the cap the ring is allocated and the deliveries pipeline.
- Over it the scheduler declines and keeps the ordered path for that graph. Later graphs are evaluated again, so a smaller live window can use the ring.
- Declining costs nothing in steady state. Both the ring and the transfer backend's device context are released, and a graph that stages nothing at all releases them the same way.
- The value is in MiB, `0` removes the cap, and 65536 is the largest accepted: a slot holds one attention layer's K or V, so a cap past that is a typo rather than a budget.
- The budget and the headroom check are decided before the graph is allocated, so they can still leave the graph short. If graph reservation fails, the rings are released and the reservation is retried once on the ordered path; the devices that were holding a ring keep the ordered path from then on, the ones that held none keep their eligibility. An optional ring never turns a graph that fits into an allocation failure.

**The cap is applied to what the current graph needs, not to what the full context would need.** A run whose window stays small keeps the ring whatever `-n_ctx` says, which is the common case and the reason it is done this way: a staged input is a view of the cache tensor, so the full-context figure is there for the asking, but enforcing it would refuse the ring for every large `-c` even when the window never gets near it. The warning reports both numbers so that `--kv-pipeline-budget` can be sized against the one that matters.

The cost of deciding per graph is that a context which grows past the budget allocates a ring for the small early windows and gives it back once it outgrows them. That transient is bounded by the budget itself, which is the memory the user already authorised, so it is a property of the cap rather than a defect in it.

A window wider than the ring holds has to free the ring and allocate it again, which blocks the host on the device, and a prefill widens the window on nearly every ubatch. So a slot is allocated in powers of two, up to what the full context needs and never past the budget or the headroom check. The decision to decline still goes by what the graph needs, so the cap falls where it did; only the allocation is coarse.

At 32,768 the ring is 204 MiB at the full context, over the 128 MiB default. `--kv-pipeline-budget 512` buys 20.350 -> 31.463 t/s behind an 18,432-token prompt.

#### Why 128 MiB

The best the ring can do is hide the smaller of copy and compute behind the larger, so its value peaks where the two are equal. Writing `b` for the bytes one attention layer holds per row of window, the copy is `n_attn * b * rows / BW` and the peak is at

```
rows*  = compute * BW / (n_attn * b)
ring*  = n_slots * b * rows*  =  (n_slots / n_attn) * compute * BW
```

**`b` cancels.** The ring size at the peak does not depend on the cache type or on `n_embd_k_gqa`; it is set by the share of the traffic the ring holds, the compute a graph has to hide behind it, and the link. On this configuration -- 3 slots against 16 staged attention layers, 25.7 ms of consumer wait, 22.0 GB/s -- that is `0.1875 * 25.7e-3 * 22e9`, or **101 MiB against the 102 MiB measured at the +55.8% peak**.

The same sweep at `-ctk q4_0 -ctv q4_0` measures that cancellation rather than deriving it. Halving the bytes per row moves the whole curve to twice the window, and leaves the ring at the peak where it was:

| q8_0 rows | gain | q4_0 rows | gain |
|---:|---:|---:|---:|
| 2,048 | +8.1% | 4,096 | +8.9% |
| 4,096 | +16.0% | 8,192 | +16.4% |
| 8,192 | +30.8% | 16,384 | +30.9% |
| 12,288 | +44.1% | 24,576 | +43.2% |
| **16,384** | **+56.0%** | **32,768** | **+53.9%** |
| 24,576 | +24.0% | 49,152 | +26.4% |
| 32,768 | +20.1% | 65,536 | +22.4% |

The peak moves from 16,384 rows to 32,768, and the ring at it is 102 MiB against 108 MiB.

So the default is a size, not a depth, and it is the right kind of quantity to fix: a cache quantised to q4_0 doubles `rows*` and halves `b`, leaving the same budget. What does move it is `n_slots / n_attn` -- a model with 64 attention layers wants about a quarter of it -- and the compute and link of the machine. 128 MiB is a compromise across that spread with a little margin over this configuration's optimum, and `--kv-pipeline-budget` is there for the configurations it does not suit.

### Where the rest of the token goes

Per decode graph, `GGML_SCHED_TRANSPORT_DEBUG=2`, behind a 19,246-token prompt:

| | ordered | pipelined |
|---|---:|---:|
| total | 54.11 ms | 31.10 ms |
| blocked in the ordered `ggml_backend_tensor_copy` | 28.30 ms | 3.57 ms |
| blocked waiting for the consumer backend | 25.70 ms | 27.31 ms |
| issuing early deliveries | 0.00 ms | 0.04 ms |
| bytes delivered early / late | 0 / 0 MiB | 644.0 / 2.3 MiB |
| bytes left on the ordered path | 28.3 MiB | 0.4 MiB |

The blocking host-to-device copy is all but gone and the consumer wait is unchanged, which is the shape a working overlap has: the transfer left the host's critical path without being added to the consumer's. 644 MiB in the 28.3 ms the ordered arm reports for the same bytes is 22.0 GB/s, which is what this link does; the transfer cannot be made faster, only hidden.

The 3.57 ms that remains moves 0.4 MiB, and `GGML_SCHED_TRANSPORT_DEBUG=3` shows that almost all of it is one copy: `attn_inp_k_rot`, 256 KiB, 18 us on the ordered path and 3.4 ms behind one split of look-ahead. The 32 KV store copies cost 353 us between them.

It looks like latency and is not. A blocking copy shares the device's copy engine with the deliveries and waits for what is already queued there: two staged splits at 22.0 GB/s is 3.6 ms, which is the number. Two things were tried and neither helped. Issuing the delivery in pieces so the blocking copy can interleave does nothing -- the engine is FIFO across streams, `attn_inp_k_rot` stays at 3.4 ms at every piece size, and small pieces cost throughput (29.80 t/s whole, 28.73 at 4 MiB, 22.01 at 1 MiB). Putting the copy on the consumer's own stream so the host never blocks moves the time rather than removing it: the ordered copy falls from 3.57 ms to 0.16 ms, the consumer wait rises from 27.31 ms to 31.05 ms, and throughput does not move (29.808 against 29.834).

So this is not spare time. Those 256 KiB cross the same saturated link as the 644 MiB of deliveries, and on this configuration the link is the ceiling. A faster link, or a slower device behind it, moves that ceiling somewhere else.

Do not compare these numbers against runs on other models, prompts, cache settings, hardware, or commits.

## Validation

The gates, and what was run for them:

Gates 1, 2, 3 and 5 and `test-alloc` were run on the current head, on an RTX 4070 with a CUDA build, gate 5 also over both devices with `-sm layer`. The `llama-server` table and the parallel table under [Measurements](#measurements) are from those runs; the breakdowns marked as taken on an earlier head still are.

1. **Byte-identical greedy server output against the control.** Four fixed tasks at `temperature 0, top_k 1, seed 1234`, plus two tasks behind an 18,422-token prompt, hashed and compared against a build of the parent commit. Identical at `N = 0`, `N = 1` and `N = 4`, re-run on the current head. `docs/repro/r4-kv-pipeline-exact.sh` compares every requested depth with the first and fails on a hash difference. Two things keep the tasks independent of each other, and both were needed. Every task carries a nonce derived from its own name and length, so no two share a prefix the server could restore, and the harness fails a task whose `prompt_n` says one was reused anyway. Each request also sets `cache_prompt: false`, so a task never inherits what the previous one left in the cache.

   The second is what made `records@18432` a gate rather than a coin flip. Its prompt is about 29.6k tokens against a 32,768 context, and the task before it is about the same size, so the two do not both fit and placement depended on what was still resident. Two otherwise identical `N = 0` runs of it produced different hashes. Asked on its own with the cache off it is perfectly stable: the same hash three times running, at `-c 32768` and at `-c 65536`. With the flag set, two independent `N = 0` passes agree on all eight tasks, and `N = 0`, `N = 1` and `N = 4` agree on all eight.
2. **A/B/A/B at 4,096 / 16,384 / 32,768 with reversed arm order.** `docs/repro/r4-kv-pipeline-ab.sh`; the table above is its output, re-run on the current head.
3. **Device allocation high-water reported.** Above.
4. **Telemetry showing the deliveries actually converted.** `GGML_SCHED_TRANSPORT_DEBUG=1` reports the plan (staged splits, bytes per graph, how much of it goes early, and the source buffer type); `=2` adds the per-graph host-time breakdown above, as the mean over each 128 graphs, with depth stops and the number of recycle waits enqueued; `=3` names the tensors still on the ordered path. `ggml_backend_sched_get_transport_pipeline_stats()` exposes deliveries and early and late byte counts to callers.
5. **Byte-identical greedy output of concurrent sequences over a cache split into streams.** `docs/repro/r4-kv-pipeline-parallel-exact.sh` runs 8 concurrent sequences of 16 through `llama-parallel`, which seeds its client schedule so the batches repeat, and compares every depth's transcripts with the first. Identical at `N = 0`, `N = 1` and `N = 4` with `-sm none`, and at `N = 0` and `N = 1` with `-sm layer` over both devices, which is the case where each device carries its own ring. Gate 1 covers one sequence per ubatch, where a delivery is one range; this covers a ubatch spanning several streams, where it is one range per stream. Run against the commit before the multi-stream span fix it fails at `N = 1`, which is what makes it a gate rather than a smoke test.

A device-resident KV run is unaffected, and was measured to confirm it: 38.5612 t/s at depth 0 against 38.5240 at depth 1, `tg128 @ d4096`, with the transport never enabled because the scheduler is given a depth of 0.

## Scope and limits

- Only persistent host inputs marked with `GGML_TENSOR_FLAG_TRANSPORT` are candidates. The stable prefix remains a per-evaluation value. Unmarked inputs, weights, user inputs, transposed V, and copies with later readers stay ordered.
- CUDA is the only enabled backend. Meta, SYCL, WebGPU, and other backends stay ordered until their event behavior and transport path are validated.
- The ring costs `(depth + 2) x (largest staged split)` of device memory, and a staged split is both K and V of one attention layer over the whole context. That is linear in context length, and it is what bounds the feature at depth rather than anything about the transfer itself.
- **A cap is per graph, not per sequence.** `--kv-pipeline-budget` bounds the window one graph delivers, which is `n_kv * n_stream` over every sequence in the ubatch, so it cannot be applied to one sequence of a batch and not another.
- **A multi-stream window is delivered one range per stream**, keyed on the last dimension. A window whose streams are not on that dimension keeps the single flat range, which is correct but not accelerated.
- **A cache that shares cells with another one keeps the ordered path for the layers it shares.** [TAG_KV_CACHE_SHARE_CELLS] gives the borrowing cache the owner's K/V tensors, so their stable prefix would have two writers with two slot layouts. The borrower drops `GGML_TENSOR_FLAG_TRANSPORT` from the tensors it takes; the layers it allocates itself are unaffected.
- **One ring per accelerator.** A layer-split model pipelines on every device that qualifies; a device with no room within the budget falls back to the ordered path on its own without disabling the others.
- **The producer of a staged input must be the CPU or the consumer itself.** Neither part of a staged delivery is ordered against a third device: the stable prefix goes on the transfer stream and the rest on the consumer's own stream, where the ordered path would have synchronized the producer first. An input a second accelerator writes keeps the ordered path.
- **It turns graph-level pipeline parallelism off while it is delivering.** A graph that delivered has to block the host on its consumer before the next graph writes the host cache, because the host source of a delivery is read long after the call that issued it returned. That block is what `n_copies > 1` exists to avoid, so the two do not overlap: with `-sm layer` over several GPUs and `--kv-cpu-pinned`, `llama_context` enables both and the ring wins. Use `--kv-pipeline-depth 0` to keep the graph-level pipelining instead.
- **Tensor parallelism keeps the ordered path.** See [Tensor parallelism](#tensor-parallelism).
- **A host write to the cache waits for the delivery.** `llama_memory_clear(mem, true)` synchronizes the context before it clears the buffers, because a delivery the last decode issued can still be reading them. This was already needed without the transport: with a device-resident cache the same call cleared the buffers under the running graph, and `llama_decode` followed by that clear changed the logits of that decode on every trial.
- The scheduler must be configured with the device's own default buffer type. A scheduler built on a split or host buffer type keeps the ordered path.
- `GGML_KV_PIPELINE_DEPTH` and `GGML_KV_PIPELINE_BUDGET_MIB` provide scheduler defaults. Explicit scheduler settings and command-line options take precedence.

## Tensor parallelism

`-sm tensor` is not pipelined. The scheduler explicitly excludes meta devices. A host-resident cache needs a validated strided head-split write before this can be enabled.

Both sit behind a correctness problem that is not this feature's: **`-sm tensor` together with `--no-kv-offload` currently produces wrong output.** On one build and one prompt, `-sm layer --no-kv-offload` and `-sm tensor` with a device-resident cache agree exactly, while `-sm tensor --no-kv-offload` differs. It does not crash or warn; it generates fluent, different text.

The cause is the GQA head mapping. Tensor parallelism splits attention by head, but a host-resident cache is one undivided tensor, so the scheduler's copy of it is classified `MIRRORED` and the whole window goes to every device. With 24 query heads split 12/12 and 4 KV heads mirrored, the kernel derives the GQA ratio from the tensors it is handed -- 12/4 = 3 rather than 6 -- and the second device's queries, renumbered from 0, read the first device's keys. With an uneven split the same fault surfaces as a crash instead: `GGML_ASSERT(Q->ne[2] % K->ne[2] == 0)`, because 24 heads split 13/11 is not divisible by 4.

Head-splitting the copy rather than mirroring it fixes it. That was prototyped and reproduced the layer-split output byte for byte, and needs four coordinated changes: classify the scheduler's copy at all (it is a leaf in a compute buffer, so it never reaches the device's split-state callback), use the head axis for the permuted `[head_dim, n_kv, n_head_kv, 1]` shape rather than the cache tensor's own axis, express the granularity in heads aligned to the query split divided by the GQA ratio, and add a strided write because the heads are interleaved within each row rather than laid out end to end.

## Future work

- Fix `-sm tensor` with `--no-kv-offload` (above). Until then it should not be used: it is wrong rather than slow.
- Add the strided head-split delivery above, validate it, and then measure it.
