# Pipelined delivery of a host-resident KV cache

With `--no-kv-offload` (optionally with `--kv-cpu-pinned`), the attention history
lives in host RAM and has to reach the accelerator on every decode token. The
backend scheduler used to issue that transfer on the consumer's own stream, right
before the kernels that read it, so a token cost `copy + compute` in series.

This is the R4 item of the KV-offload roadmap
([PR #31](https://github.com/GenerelSchwerz/llama.cpp/pull/31)): the transfer and
the attention arithmetic are the same as before, but the transfer is issued one
split ahead, on a stream of its own, so the copy engine retires it underneath the
kernels of the split before it.

`--kv-pipeline-depth N` controls it. It is on by default at `N = 1` and only has
an effect where a host-resident cache produces the deliveries; `0` restores the
ordered path exactly.

The staging it needs is bounded by `--kv-pipeline-budget` (default 128 MiB), so
that a cache which lives on the host to keep device memory free never quietly
spends that memory back. Past the cap the scheduler declines and the ordered path
runs, at no cost. See [The budget](#the-budget).

## What it changes, and what it must not

R4 pipelines *deliveries*, not attention. Every byte and every attention
operation is the same as on the ordered path; only the point at which the
transfer is issued moves. Greedy server output is byte-identical, and that is a
gate, not an aspiration -- see [Validation](#validation).

Three pieces make it work.

### 1. A stable prefix, so there is something safe to send early

The KV window a split reads is not stable for the whole graph: the same graph
writes this ubatch's rows into it, and on a host-resident cache that write is a
CPU split that runs *between* the attention of one layer and the attention of the
next. Delivering the whole window ahead of that split would send rows that have
not been written yet.

What *is* stable is everything below the lowest row this ubatch writes, which at
decode depth is essentially the whole window. `ggml_tensor::stable_prefix` records
that, in bytes, on the tensor that owns the storage; a view inherits the part of
it that its own byte window covers. `llama_kv_cache::update_stable_prefixes()`
sets it from the slot info in `apply_ubatch()` -- before the graph is built and
allocated, so the scheduler's plan and the deliveries it then issues are decided
against the same write position -- and `build_graph_shift()` clears it, because a
shift rewrites the body in place.

The scheduler delivers `[0, stable_prefix)` early on the transfer stream and the
remainder at the split, once every earlier split of the graph has run. At 18k
tokens of context that split is about 620 MiB early against 1-5 MiB late.

The prefix is a hint about *this* graph. It has to be refreshed for every ubatch
even when the graph is reused, which is why it is set from `apply_ubatch()` and
not from graph construction. Where it cannot be established -- a transposed V
cache, whose ubatch writes are scattered across the whole tensor -- it stays 0 and
the input keeps the ordered path.

### 2. A ring the graph allocator cannot reach

`ggml-alloc` is free to recycle a graph-owned input copy once its last graph-level
consumer is done, and a look-ahead transfer is still in flight outside that
lifetime. Writing split `k + 1`'s delivery into the scheduler's own input copies
corrupts the split still reading them; that is the defect class the earlier
cross-layer prefetch experiment hit (measurements §5 of PR #31: `+1.38%`, and not
exact).

So the scheduler allocates its own ring and points the staged input copies at it
before the graph is allocated. A tensor that already has `data` is left alone by
`ggml_gallocr_init_tensor`, so the ring sits outside the allocator's reuse
analysis rather than competing with it.

Each slot has one ownership cycle:

1. the transfer stream owns an idle slot and writes one future split's prefix into it;
2. it records the slot's `ready` event, which the consumer stream waits for before launching the split that reads the slot;
3. the consumer records `release` once every kernel that reads the slot has been enqueued, and the transfer stream waits for that before overwriting the slot for a later split.

Membership in the ring is decided once, when the ring is laid out, and execution
goes by the recorded answer. How much of a staged input can go early moves with
every ubatch; *which* input copies live in the ring must not, because their
addresses were handed out at allocation time.

Two things disqualify an input that otherwise looks eligible:

- **A reader further down the graph.** The scheduler creates one input copy per
  (tensor, backend), not per split, so a later split can be pointed at the same
  copy without appearing to consume it -- and by then the ring may have recycled
  the slot. The plan scans the splits after the owner for such a reader and puts
  those inputs back on the ordered path. Attention does not produce this shape,
  but nothing in the scheduler forbids it.
- **No room on the device.** A slot holds one split's whole delivery, so the ring
  grows with the context: 27 MiB at 4k, 213 MiB at 32k, 1.7 GiB at 256k. The ring
  is allocated after the graph allocator has reserved its buffers, so it must not
  take the room those buffers may still have to grow into; it declines unless it
  can leave `GGML_SCHED_TRANSPORT_HEADROOM` (512 MiB) free, says so once, and
  stays on the ordered path.

### 3. A look-ahead that stays clear of the ring's tail

A delivery running `L` splits ahead recycles the slot of the split `L - n_slots`
back. With `n_slots == L + 1` the ring is exactly full, so every delivery has to
recycle the split that was enqueued a moment ago and is still running -- the
ordered path with extra steps. The ring therefore keeps
`GGML_SCHED_TRANSPORT_MARGIN` (2) slots behind the look-ahead, and
`--kv-pipeline-depth N` allocates `N + 2` slots.

Two details matter as much as the margin:

- **Deliveries are issued after a split is enqueued, never before.** Issuing them
  first means the host can block on slot recycling while holding back work the
  consumer could already be running.
- **Slot recycling is ordered stream to stream, not through the host.** A host
  wait empties the transfer queue for as long as it blocks.

Getting either of these wrong costs the entire gain while still producing correct
output, which is the failure mode worth knowing about: on this configuration the
first attempt measured `+0.5%` and looked like "the copy simply does not overlap".

## Measurements

RTX 4070 (11,902 MiB usable, sm_89), driver 610.57.04 / CUDA 13.3, i5-13400F,
`Qwen3.8-27B-UD-IQ2_M.gguf`, `-ngl 99 -sm none -mg 0 -t 3 -fa on -ctk q8_0
-ctv q8_0 -b 512 -ub 512`, host residency `-nkvo --kv-cpu-pinned
--recurrent-state-offload`, everything under `taskset -c 0,2,4`.

`llama-bench`, `--no-warmup`, A/B/A/B with reversed arm order
(`docs/repro/r4-kv-pipeline-ab.sh`):

| Depth | reps | ordered | pipelined | gain | `max(copy, compute)` ceiling from PR #31 §3 | share |
|---|---:|---|---|---:|---:|---:|
| 4,096 | 5 | 31.7324, 31.7363 | 37.0889, 37.0741 | **+16.9%** | 38.49 | 96.4% |
| 16,384 | 3 | 19.6765, 19.6854 | 31.5352, 31.5807 | **+60.4%** | 34.88 | 90.4% |
| 32,768 | 3 | 13.0264, 13.0254 | 15.5325, 15.5329 | **+19.3%** | 20.83 | 74.6% |

Server decode behind an 18,422-token prompt
(`docs/repro/r4-kv-pipeline-exact.sh`): **18.468 -> 30.685 t/s, +66.2%**.

### Across context depth, with device memory

`docs/repro/r4-kv-pipeline-context-sweep.sh`, A/B/A/B, peak device memory sampled
with `nvidia-smi` across each arm. Both passes agreed to the digits shown.

| Context | ordered | pipelined | gain | peak device memory | delta | ring |
|---|---:|---:|---:|---|---:|---:|
| 4,096 | 31.66 | 37.01 | **+16.9%** | 10,169 -> 10,197 MiB | +28 MiB | 27 MiB |
| 16,384 | 19.64 | 31.43 | **+60.1%** | 10,159 -> 10,263 MiB | +104 MiB | 107 MiB |
| 32,768 | 12.99 | 15.49 | **+19.2%** | 10,161 -> 10,367 MiB | +206 MiB | 213 MiB |
| 65,536 | 7.74 | 8.74 | **+12.9%** | 10,163 -> 10,573 MiB | +410 MiB | 428 MiB |
| 131,072 | 4.29 | 4.68 | **+9.1%** | 10,537 -> 11,355 MiB | +818 MiB | 855 MiB |
| 262,144 | 2.25 | 2.24 | **declined** | 11,329 -> 11,391 MiB | +62 MiB | not allocated |

Two curves run in opposite directions here, and both matter.

**The gain narrows with depth.** A token is copy plus compute; as the context
grows the copy grows with it while the compute per staged split does not, so the
share of the token that can hide a transfer shrinks. At 16,384 compute still
covers most of the copy; by 131,072 it covers a tenth of it. That is arithmetic,
not an implementation limit, and no amount of look-ahead changes it.

**The ring's cost does not narrow.** It is `(depth + 2)` slots of one staged
split, and a staged split is K and V of one attention layer over the whole
context: it doubles every time the context doubles. At 131,072 it claims 818 MiB
of an 11,902 MiB card to buy 9.1%.

At 262,144 the ring would need 1.7 GiB against 573 MiB free, so it declines and
the run stays on the ordered path -- 2.25 against 2.24 t/s, inside the spread of
the ordered arm's own two passes, and 62 MiB of device memory for the transfer
backend's context. Declining is the intended outcome, not a failure: the +62 MiB
and the unchanged throughput are what "the guard did its job" looks like.

**On a memory-constrained card, past roughly 64k the same device memory is
probably better spent on `--kv-gpu-layers`.** At 131,072 a staged split is
285 MiB, so the 818 MiB the ring takes is about three attention layers' worth of
K and V; making three of sixteen layers device-resident removes about 19% of the
host-to-device traffic against the 9.1% the ring buys. That comparison has not
been measured here and it will move with the model's layer count and the card, so
it is a pointer for whoever tunes a deployment, not a recommendation.

### The budget

The table above is what the feature costs uncapped, and it is the reason it is
capped. A host-resident KV cache exists to keep device memory free; a transport
that speeds it up by spending hundreds of MiB of that memory is working against
the thing it is accelerating. `--kv-pipeline-budget` (default 128 MiB) is an
absolute cap on the ring, not a fraction of what happens to be free:

- Under the cap the ring is allocated and the deliveries pipeline: 4,096 and
  16,384 in the table, at 28 MiB and 104 MiB.
- Over it the scheduler declines and keeps the ordered path, and the decision is
  latched, because a context only grows and a ring allocated for the small
  windows of early prefill would only have to be given back later.
- Declining costs nothing in steady state. Both the ring and the transfer
  backend's device context are released: at 32,768 with the default budget,
  device memory settles at 10,161 MiB, the same as the ordered path, and
  throughput matches it (12.965 against 12.984 t/s).

Raising the budget trades that memory back for speed where it is worth it:
`--kv-pipeline-budget 512` at 32,768 gives 15.487 t/s for 206 MiB.

**Known limitation.** The cap is applied per graph, so a run whose context grows
past it still allocates a ring for the early prefill graphs and releases it once
the window outgrows the budget -- at 32,768 that shows up as a transient peak of
+112 MiB even though the steady state is +0. Deciding against the context's final
size rather than the current graph's would remove it, and needs the KV geometry
the scheduler does not have.

Where the split-loop host time goes, per decode graph at 18.5k
(`GGML_SCHED_TRANSPORT_DEBUG=2`):

| | ordered | pipelined |
|---|---:|---:|
| total | 52.11 ms | 30.37 ms |
| blocked in the ordered `ggml_backend_tensor_copy` | 26.80 ms | 0.15 ms |
| blocked waiting for the consumer backend | 25.14 ms | 26.69 ms |
| issuing early deliveries | 0.00 ms | 0.04 ms |
| bytes delivered early / late | 0 / 0 MiB | 619.2 / 1.3 MiB |

The blocking host-to-device copy is gone and the consumer wait is unchanged,
which is the shape a working overlap has: the transfer left the host's critical
path without being added to the consumer's.

**32,768 is the weak point and is reported as such.** The gain there is 74.6% of
the probe ceiling, against 90-96% at shallower depths. At that depth the copy per
staged split (about 2.9 ms) exceeds the compute between staged splits (about
1.9 ms), so one split of look-ahead cannot cover it. Raising the look-ahead does
not help: at 32,768 `N = 2` measured 15.5274 and `N = 3` measured 15.1114 against
15.5273 for `N = 1`, and at 16,384 the same sweep gave 30.34 and 29.15 against
31.56. `N = 1` is the best setting at every depth measured, which is why it is
the default. Closing the 32,768 gap is a separate piece of work, not a knob.

Do not compare these numbers against runs on other models, prompts, cache
settings, hardware, or commits.

## Validation

The gates, and what was run for them:

1. **Byte-identical greedy server output against the control.** Four fixed tasks
   at `temperature 0, top_k 1, seed 1234`, plus two tasks behind an 18,422-token
   prompt, hashed and compared against a build of the parent commit. Identical at
   `N = 0`, `N = 1` and `N = 4`. `docs/repro/r4-kv-pipeline-exact.sh`.
2. **A/B/A/B at 4,096 / 16,384 / 32,768 with reversed arm order.**
   `docs/repro/r4-kv-pipeline-ab.sh`; the table above is its output.
3. **Device allocation high-water reported.** Above.
4. **Telemetry showing the deliveries actually converted.**
   `GGML_SCHED_TRANSPORT_DEBUG=1` reports the plan (staged splits, bytes per
   graph, how much of it goes early, and the source buffer type);
   `=2` adds the per-graph host-time breakdown above.
   `ggml_backend_sched_get_transport_pipeline_stats()` exposes the same counters
   to callers.

A device-resident KV run is unaffected, and was measured to confirm it: 39.13 t/s
on the parent commit against 39.10 t/s here at `tg128 @ d4096`, with the
transport never enabled because the scheduler is given a depth of 0.

## Scope and limits

- Only inputs carrying a stable prefix are eligible. Everything else -- weights,
  user inputs, a transposed V cache, an input copy with a reader in a later
  split, any backend that cannot transfer asynchronously or record events --
  keeps the ordered path untouched.
- The ring costs `(depth + 2) x (largest staged split)` of device memory, and a
  staged split is both K and V of one attention layer over the whole context. That
  is linear in context length, and it is what bounds the feature at depth rather
  than anything about the transfer itself.
- One ring, on the first backend that qualifies. Additional accelerators keep the
  ordered path; multi-GPU host-KV is not covered.
- The scheduler must be configured with the device's own default buffer type. A
  scheduler built on a split or host buffer type keeps the ordered path.
- `GGML_KV_PIPELINE_DEPTH` overrides the depth for tools that do not expose the
  command-line option, such as `llama-bench`.
