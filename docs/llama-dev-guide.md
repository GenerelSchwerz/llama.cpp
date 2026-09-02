# llama/dev fork - user guide to the extra features

This page lists everything the `llama/dev` branch adds on top of upstream
`master` that a user can see or set. It covers new command-line flags (what they
do, when to use them, how to use them), the optimizations that run on their own
with no flag, and the small extra knobs in `llama-bench`.

There are **no new build (CMake) options**. Everything here is set at runtime on
`llama-cli`, `llama-server`, `llama-completion` and friends. Every flag also has
an `LLAMA_ARG_*` environment variable, shown in each entry.

Two of the feature groups are still in open pull requests and are marked
**(PR #57)** and **(PR #39)**. The rest is already on `llama/dev`.

---

## 1. Quick reference

### New flags

| Flag | Value | Default | What it is for |
|---|---|---|---|
| `--kv-cpu-pinned` / `--no-kv-cpu-pinned` | on / off | off | Faster host <-> GPU copies for a host-resident KV cache |
| `--recurrent-state-offload` / `--no-recurrent-state-offload` | on / off | off | Keep the small recurrent state on the GPU when the KV cache is on host RAM (hybrid / Mamba / GDN models) |
| `--kv-gpu-layers` | `N` | `0` | With `--no-kv-offload`, keep `N` attention KV layers on the GPU(s) |
| `--kv-pipeline-depth` **(PR #39)** | `0`..`14` | `1` | Overlap the host->GPU KV transfer with compute; `0` = old serial path |
| `--kv-pipeline-budget` **(PR #39)** | MiB | `128` | Hard cap on GPU memory the overlap may use |
| `-as`, `--attn-split` **(PR #57)** | `N0,N1,...` | follow `-ts` | Split attention heads across GPUs by their own ratio under `-sm tensor` |
| `--phase-aware-workspace` / `--no-phase-aware-workspace` | on / off | off | Shrink the compute workspace between prompt and generation |
| `--live-context-workspace` / `--no-live-context-workspace` | on / off | off | Grow the compute workspace with the used context instead of the full `-c` |
| `--spec-draft-ubatch-size`, `--ubatch-size-draft`, `-ubd` | `N` | `0` (inherit) | Separate physical batch size for the draft context |
| `--spec-draft-kv-gpu-layers`, `--kv-gpu-layers-draft` | `N` | inherit target | `--kv-gpu-layers` for the draft context only |
| `--spec-mtp-rs-planes` | `N` | `0` (auto) | Cap the recurrent rollback planes for `draft-mtp` to save memory |

### Optimizations that run with no flag

- **Compact causal attention masks** - smaller attention mask at long context / large batch (needs flash attention).
- **Slowest link first** for `--kv-gpu-layers` **(PR #57)** - resident layers go to the GPU with the slowest host link first.
- **`-sm tensor` + `--no-kv-offload` is now correct** **(PR #57)** - this combination used to abort or return wrong text.
- **Quantization kept for host KV stores** - a `q8_0` / `q4_0` KV cache stays quantized when copied to host, so the copy is half or a quarter of the bytes.
- **Shared target / draft workspaces for MTP** - lower peak memory when running `draft-mtp` speculative decoding.

### Extra `llama-bench` options **(PR #39)**

`-kvcp` / `--kv-cpu-pinned`, `-rso` / `--recurrent-state-offload`,
`-kvpd` / `--kv-pipeline-depth`, `-kvpb` / `--kv-pipeline-budget`.

### Extra environment variables

`GGML_KV_PIPELINE_DEPTH`, `GGML_KV_PIPELINE_BUDGET_MIB` (same as the flags, **PR
#39**), `GGML_SCHED_TRANSPORT_DEBUG=1|2|3` (print KV transport timing, **PR #39**).

---

## 2. Running a large context: host-resident KV cache

### The problem this solves

When the context gets long, the KV cache stops fitting next to the model on the
GPU. Upstream already has `--no-kv-offload` (`-nkvo`), which puts the whole KV
cache in system RAM. That frees GPU memory but is slow: every decode token has to
copy the cache to the GPU and then run attention, one after the other.

This fork makes the host-resident cache usable: pin the memory, keep the hot part
on the GPU, keep the small recurrent state on the GPU, and overlap the copy with
compute. On an asymmetric two-GPU box the measured decode speed goes from
"unusable" to the fastest option for a long context.

All of the flags below only do something together with `--no-kv-offload` (or with
`--kv-cpu-pinned`, which implies a host-resident cache). A normal GPU-resident
run never creates the transfer and ignores them.

### `--kv-cpu-pinned`

**What it does.** Stores the host-resident KV cache in *pinned* (page-locked)
host memory instead of normal pageable memory. Pinned memory copies to the GPU
much faster and lets the copy overlap with compute.

**When to use it.** Any time you run `--no-kv-offload` and have the RAM to spare.
Pinned memory cannot be swapped out, so do not pin more than you can afford to
lock down.

**How.**

```bash
llama-cli -m model.gguf -c 65536 --no-kv-offload --kv-cpu-pinned
```

### `--recurrent-state-offload`

**What it does.** Hybrid models (Qwen3-Next / "qwen4exp", Nemotron-H, Jamba,
Falcon-H1, Mamba, gated-delta-net, ...) keep a small recurrent state (roughly
100-200 MiB) next to the attention KV cache. By default `--no-kv-offload` moves
that state to host RAM along with the KV cache. This flag keeps the recurrent
state on the GPU while the KV cache stays on host.

**When to use it.** Always set it for a hybrid or recurrent model run with
`--no-kv-offload`. The state is tiny but it is read on every single token, so
leaving it on host RAM is a large slowdown. In one measurement on a hybrid
27B model this flag alone moved decode from 7.6 to 17.2 tokens/s.

**How.**

```bash
llama-cli -m hybrid-model.gguf -c 65536 --no-kv-offload --kv-cpu-pinned --recurrent-state-offload
```

> Note on the name: enabling it *decouples* the recurrent state from the KV
> cache placement rule and keeps it on the accelerator. Under `-sm tensor` the
> fork keeps the recurrent state on the GPU automatically, so the flag is not
> needed there.

### `--kv-gpu-layers N`

**What it does.** With `--no-kv-offload`, keeps the KV cache of `N` attention
layers on the GPU(s) and sends the rest to host RAM. Every resident layer is one
layer of transfer you no longer pay per token.

**When to use it.** When the cache does not fit on the GPU but part of it does.
Start from `0` (full host cache), raise `N` until GPU memory is nearly full.
Under `-sm layer` (or single GPU), each extra layer helps. Under `-sm tensor`
each layer's cache is split across all GPUs, so a partial value still sends the
rest to the slow GPU - there it is worth setting to the **full** attention-layer
count or leaving at `0`, nothing in between.

**How.**

```bash
llama-cli -m model.gguf -c 65536 --no-kv-offload --kv-cpu-pinned --kv-gpu-layers 8
```

**(PR #57)** improvements, no extra flag:

- The budget now works for every cache layout (iSWA, DSA, DSV4, MSA, hybrid),
  not just the plain and direct-hybrid caches.
- Resident layers are spread one-per-GPU instead of filling GPU 0 first.
- Resident layers go to the GPU with the **slowest host link first**, because
  that is where removing a transfer is worth the most. GPUs whose link speed is
  within 15% of each other are treated as one group and share round-robin. The
  link speed is measured once at startup and printed:

  ```
  llama_pick_gpu_resident_layers: CUDA0: host-to-device 24.2 GB/s   (gen4 x16)
  llama_pick_gpu_resident_layers: CUDA1: host-to-device  3.3 GB/s   (gen3 x4)
  ```

### `--kv-pipeline-depth N` and `--kv-pipeline-budget MiB` **(PR #39)**

**What it does.** Issues the host->GPU KV transfer `N` splits ahead of the
kernels that read it, on a transfer stream of its own, so the copy runs
underneath the compute of the previous split. `N = 0` restores the exact
serial path (copy, then compute). Output is byte-identical to `N = 0`.

`--kv-pipeline-budget` is a hard cap in MiB on the GPU memory the staging ring
may use (one slot is one attention layer's K or V over the whole context, so it
grows with `-c`). Past the cap the scheduler declines and runs the serial path -
a host-resident cache never quietly spends back the GPU memory it exists to save.
`0` removes the cap.

**When to use it.** It is **on by default** (`N = 1`) and only engages with a
host-resident cache, so most users do not touch it. Set `N = 0` to compare
against the serial path. Do not raise `N` above `1`: every measured depth beyond
1 is slower. Raise `--kv-pipeline-budget` (e.g. `512`) if you run a very long
context and see a startup warning that the ring would need more than the budget
and is "staying on the ordered path (raise --kv-pipeline-budget to spend more
device memory on it)".

**How.**

```bash
# default: pipelining on
llama-cli -m model.gguf -c 32768 --no-kv-offload --kv-cpu-pinned

# bigger budget for a long context
llama-server -m model.gguf -c 131072 --no-kv-offload --kv-cpu-pinned --kv-pipeline-budget 1024

# turn it off to A/B
llama-cli -m model.gguf -c 32768 --no-kv-offload --kv-cpu-pinned --kv-pipeline-depth 0
```

Measured gain (single RTX 4070, `Qwen3-27B` hybrid, q8_0 KV, pinned host cache):
about +17% decode at 4k context, +60% at 16k, +19% at 33k. The gain narrows as
context grows because the copy eventually dominates and there is less compute
left to hide behind.

`GGML_SCHED_TRANSPORT_DEBUG=2` prints a per-graph breakdown of transfer time vs
consumer wait; `=3` also names the tensors still on the serial path.

### `-as`, `--attn-split N0,N1,...` **(PR #57)**

**What it does.** Under `--split-mode tensor`, sets the fraction of the
*attention heads* each GPU gets, separately from `--tensor-split`. The share a
GPU gets of the heads decides how much of a host-resident cache it receives and
how much attention work it does. Everything else in the model still follows
`--tensor-split`. Unset, it follows `--tensor-split` and nothing changes. The
share is rounded to whole heads.

**When to use it.** When you run `-sm tensor --no-kv-offload` on GPUs that
differ in host bandwidth or raw speed. Sending the slow GPU half the cache is
the bottleneck; giving it fewer (or zero) heads moves the cache onto the fast
GPU. In one test on a 4070 + 3060 pair, `-as 1,0` alone gave 3.1x decode, and
4.4x combined with `--kv-gpu-layers`. Setting it when it is not used costs
nothing.

**How.**

```bash
# give GPU 0 all attention heads, GPU 1 none; weights still split 50/50 by -ts
llama-cli -m model.gguf -sm tensor -ts 50,50 -as 1,0 --no-kv-offload --kv-cpu-pinned

# 3:1 head ratio
llama-server -m model.gguf -sm tensor -as 3,1 --no-kv-offload
```

### `-sm tensor` + `--no-kv-offload` correctness fix **(PR #57)**

Before this fork, `--split-mode tensor` together with `--no-kv-offload` either
aborted (`ggml-backend-meta.cpp` assert, or `fattn.cu` GQA assert) or silently
produced wrong text on any model with more than one KV head. The fork splits the
copied-in host cache by head like the rest of tensor-parallel attention, so the
combination is now correct and matches the GPU-resident output bit for bit.
Nothing changes where it already worked. No flag - it is just fixed.

### Recipes

**Single GPU, context too big for VRAM, dense model:**

```bash
llama-server -m model.gguf -c 65536 -ngl 99 \
  --no-kv-offload --kv-cpu-pinned --kv-gpu-layers 8 \
  -ctk q8_0 -ctv q8_0
```

`-ctk/-ctv q8_0` halves the bytes that cross the link, which here is a pure
transfer win: it is worth far more than the small kernel cost.

**Single GPU, hybrid model (e.g. Qwen3-Next):**

```bash
llama-server -m qwen3-next.gguf -c 65536 -ngl 99 \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  -ctk q8_0 -ctv q8_0
```

**Two asymmetric GPUs, long context, generation speed matters most:**

```bash
llama-server -m model.gguf -sm tensor -ts 50,50 -as 1,0 -c 65536 -ngl 99 \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  --kv-gpu-layers 16 -ctk q8_0 -ctv q8_0
```

Watch the startup line for the measured link speeds and, under `-sm tensor`,
set `--kv-gpu-layers` to the full attention-layer count or to `0`.

---

## 3. Compute workspace memory

The compute workspace is scratch GPU memory reserved for building the graph. By
default llama.cpp reserves it for the full `--ctx-size` and for the largest
phase up front. These two flags trade a little startup work for a smaller
reservation, which matters when you set a large `-c` but rarely fill it.

### `--phase-aware-workspace`

**What it does.** Uses a smaller compute workspace during token generation than
during prompt processing, and regrows the prompt reservation only when a later
prompt turn needs it.

**When to use it.** Long-context chat / server workloads where prompts are
processed in bursts and most of the time is spent generating. Off by default
because it adds reserve/regrow work on phase changes.

**How.** `llama-server -m model.gguf -c 131072 --phase-aware-workspace`

### `--live-context-workspace`

**What it does.** For supported attention caches, grows the workspace
reservation with the padded *used* KV extent instead of reserving the full
context up front. A session that only ever uses 8k of a 128k context only
reserves workspace for ~8k.

**When to use it.** You set a big `-c` as a ceiling but typical sessions are far
shorter. Off by default. Works together with `--phase-aware-workspace`.

**How.** `llama-server -m model.gguf -c 131072 --live-context-workspace`

---

## 4. Speculative decoding

These give the draft context its own settings instead of always inheriting the
target's, and add a memory cap for the MTP recurrent path. See
[docs/speculative.md](speculative.md) for the full speculative feature set.

### `--spec-draft-ubatch-size N` (`--ubatch-size-draft`, `-ubd`)

**What it does.** Physical maximum batch size for the draft context. `0` (the
default) inherits the target's ubatch. `draft-mtp` requires `0` or exactly the
target ubatch.

**When to use it.** When the draft model wants a different physical batch than
the target for memory or speed reasons. Most users leave it at `0`.

### `--spec-draft-kv-gpu-layers N` (`--kv-gpu-layers-draft`)

**What it does.** Like `--kv-gpu-layers` (section 2), but for the separate draft
context only. Keeps the first `N` draft attention KV layers GPU-resident.
Unset, the draft follows the target's KV placement. KV layers the draft shares
with the target follow their owner.

**When to use it.** When you run the target with a host-resident cache but want
the small draft cache to stay on the GPU (or vice versa).

### `--spec-mtp-rs-planes N`

**What it does.** Applies only to `draft-mtp`. Sets the total number of target
recurrent-state planes, including the current one. `0` (default) allocates
`--spec-draft-n-max + 1`. A smaller explicit value (in
`[2, --spec-draft-n-max + 1]`) turns on *capped replay*, which uses less
recurrent-state memory at the cost of replaying a verification batch on the
deeper partial-acceptance path.

**When to use it.** Long draft lengths where the recurrent-state planes cost
more memory than you want to spend. Capped replay cannot be combined with
Eagle3, DFlash, or DSpark. The server rejects an unsupported device layout at
startup rather than falling back. Details:
[docs/speculative.md#capped-mtp-recurrent-planes](speculative.md#capped-mtp-recurrent-planes).

### Shared target / MTP workspaces (no flag)

When running `draft-mtp`, the target and the MTP draft phases now share their
compute workspace allocations, lowering peak memory. This is automatic.

---

## 5. Optimizations that run on their own

### Compact causal attention masks

For a normal causal decode, the attention mask no longer spans the full context
width - it is trimmed to the live KV extent, which cuts the mask size and its
memory traffic at long context and large batch. It turns on automatically when
all of these hold: flash attention is on, attention is causal, the run is a
single sequence in a single stream, and the model does not use ALiBi, sliding
window attention, or the T5 layout. Otherwise the full mask is used, as before.
Output is unchanged.

### Quantization kept for host KV stores

When the KV cache type is quantized (`-ctk q8_0`, `-ctv q4_0`, ...) and the
cache is copied to host RAM, it now stays in its quantized form instead of
being expanded to `f16`. The host<->GPU copy is then half (`q8_0`) or a quarter
(`q4_0`) of the bytes. With a host-resident cache the link is the bottleneck, so
this is close to a direct speedup - a quantized KV cache under `--no-kv-offload`
can nearly double decode throughput versus `f16`.

---

## 6. `llama-bench` additions **(PR #39)**

`llama-bench` gained the switches needed to benchmark a host-resident cache
properly (it silently dropped unknown flags before, which produced misleading
numbers):

| Switch | Meaning | Default |
|---|---|---|
| `-kvcp`, `--kv-cpu-pinned` `<0\|1>` | pinned host KV buffers | `0` |
| `-rso`, `--recurrent-state-offload` `<0\|1>` | recurrent state on GPU | `0` |
| `-kvpd`, `--kv-pipeline-depth` `<0..14>` | transfer look-ahead depth | `1` |
| `-kvpb`, `--kv-pipeline-budget` `<MiB>` | GPU memory cap for the overlap | `128` |

Each takes a comma-separated list, like the other `llama-bench` axes, so you can
sweep them in one run:

```bash
llama-bench -m model.gguf -p 0 -n 128 -d 16384 \
  -fa 1 -nkvo 1 -kvcp 1 -rso 1 -ctk q8_0 -ctv q8_0 \
  -kvpd 0,1
```
