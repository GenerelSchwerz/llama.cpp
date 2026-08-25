# 245,760-context VRAM allocation attribution

This document records the allocation audit requested on 2026-08-23. It is a
diagnostic snapshot, not evidence that an uncommitted tree is a published
commit. No runtime feature or allocation policy was changed during the audit.

The audited source was `beellama/dev` at `8f34a3355`, plus the already intended
uncommitted compact-causal/native-Q8 lifetime repair in
`fattn-mma-f16.cuh` and its regression/doc updates. The recorded dirty-source
fingerprint was
`f6c22559c12e83866719f121dc9a9f65e1fb745eb860af573f13f7f79d5eab33`.
The model SHA-256 was
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.

## Method

A dedicated Release/SM120 build enabled the existing dormant
`GGML_ALLOCATOR_DEBUG` tracer without changing source. CUDA, FlashAttention,
native Q8 FlashAttention, and native CPU tuning were enabled; KVarN and the
expanded quant matrix were disabled. `llama-server` and `llama-bench` SHA-256
values were respectively
`60eedef3d1c4e83063284f9212781564618e96b8ec11721cbe79cdf500ccea8c`
and
`c3f602a26246857ae6417fc5949af5c8f81f11912c5eff649b88dd6ffe8baaf2`.

Every GPU process was fresh and enclosed by
`flock /tmp/beellama-single-gpu.lock -c '...'`. The four server processes used
one slot, context 245,760, Q8_0/Q8_0 pinned CPU KV, GPU recurrent state, all
model layers on CUDA, compact causal masking, native Q8 attention, phase-aware
workspace off, and live-context workspace off. Native affinity was
`--cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1`. A 250 ms sampler
recorded `nvidia-smi` process memory and `/proc/PID/status`; five-second
heartbeats, `/health`, and graceful `SIGINT` made lifecycle progress visible.

The focused Nsight Systems 2026.1.3 capture directly targeted
`llama-server`, enabled CUDA memory tracking and memory-API backtraces, and
used CUDA graph-node tracing. It loaded the same 245,760-capacity,
`b=ub=2048` server, evaluated 513 prompt tokens, generated one token, and
shut down cleanly. Profiler timing is diagnostic only.

Raw allocation logs and samples are under
`/home/gencoolpc/vram-results/2026-08-23-full-allocation-attribution`.

## Production server accounting

Normal one-slot serving sets `n_outputs_max=1`; all four processes therefore
held only a 0.947 MiB pinned output row. The measured allocations were:

| physical ubatch | model, device | context, device | compute, device | KV, pinned host | compute, pinned host | sampled process VRAM | steady RSS | transient VmHWM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 12,879.47 MiB | 158.12 MiB | 549.27 MiB | 8,160.00 MiB | 10.27 MiB | 13,842 MiB | 9,338.52 MiB | 14,196.62 MiB |
| 512 | 12,879.47 MiB | 166.62 MiB | 588.27 MiB | 8,160.00 MiB | 20.28 MiB | 13,890 MiB | 9,346.93 MiB | 14,196.78 MiB |
| 1,024 | 12,879.47 MiB | 183.62 MiB | 666.28 MiB | 8,160.00 MiB | 40.30 MiB | 13,984 MiB | 9,377.50 MiB | 14,196.78 MiB |
| 2,048 | 12,879.47 MiB | 217.62 MiB | 822.30 MiB | 8,160.00 MiB | 80.34 MiB | 14,174 MiB | 9,451.55 MiB | 14,194.70 MiB |

Device context is the fixed 149.625 MiB recurrent state plus a Q8 K/V
store-stage allocation that grows from 8.50 to 68.00 MiB. CUDA-owned pinned
host memory, including the output row, was 8,171.22, 8,181.23, 8,201.25, and
8,241.28 MiB. `VmPin` and `VmLck` remained zero, confirming again that those
Linux fields do not account CUDA page-locked mappings. Nsight independently
classified the 8,160 MiB KV, 80.336 MiB compute arena, and 0.947 MiB output as
pinned allocations.

The CPU-mapped remainder of the model was 644.14 MiB. Steady RSS includes
that mapping, CUDA/runtime state, executable pages, and other process memory;
it must not be relabeled as pinned memory. The roughly 14.2 GiB `VmHWM` was a
model-load/mapping transient, not steady host residency.

The internal memory reconciler closed the CUDA accounting. At `ub=2048`, for
example, 13,919 MiB was explicit llama ownership:
12,879 MiB model + 217 MiB context + 822 MiB compute. The remaining
approximately 293 MiB was CUDA runtime/module/graph/pool overhead. Comparing
the explicit byte totals to sampled `nvidia-smi` peaks gave a stable
254.6--255.6 MiB residual across the four clean processes; the one-MiB
difference is sampling/rounding, not a growing hidden arena.

## What is inside the 822.30 MiB device compute arena

The allocator's exact high-water live set at `ub=2048` contained:

| allocation or live range | size | scaling or ownership |
|---|---:|---|
| staged full-context Q8 K view | 255.00 MiB | 245,760 cells x 1,088 B |
| staged full-context Q8 V view | 255.00 MiB | 245,760 cells x 1,088 B |
| `ffn_out` | 40.00 MiB | 20 KiB per query token |
| `Qcur_full` | 96.00 MiB | 48 KiB per query token |
| attention result node | 48.00 MiB | 24 KiB per query token |
| compact causal descriptor | 0.016 MiB | 8 B per query token |
| rotary/input support | about 0.30 MiB | mostly fixed |
| allocator-plan holes between simultaneous ranges | 128.00 MiB | 64 KiB per query token |

The arena follows the measured formula
`510 MiB + ubatch * 156 KiB + about 0.27 MiB`. The named query activations
account for 92 KiB/token; the remaining 64 KiB/token consists of address gaps
left by the multi-graph reserve plan. Those gaps are a potential allocator-plan
target, not another live attention tensor.

The pinned-host compute arena contains two 20 KiB/token embedding ranges:
`inp_embd` and its scheduler copy `model.input_embed`, plus about 0.34 MiB of
small inputs. Its measured formula is therefore about
`ubatch * 40 KiB + 0.34 MiB`. Specializing reserve plans for token input could
remove pinned-host duplication, but it is not a VRAM saving by itself.

The persistent device store stage is independent of context depth. For this
layout it is
`16 attention layers * K/V * 1,088 B * ubatch`, yielding exactly 8.50,
17.00, 34.00, and 68.00 MiB in the sweep. It stores newly quantized rows back
to the host cache; it is separate from the 510 MiB pair that reloads the live
cache for attention.

## Why the earlier 2 GiB result was not an attention workspace

`llama-bench` leaves `n_outputs_max` at its upstream-style default of
`n_batch`. At `b=ub=2048`, it therefore reserves every output row. For this
248,320-token vocabulary:

- FP32 `result_output` logits are exactly 1,940 MiB;
- two 5,120-wide FP32 output/normalization rows total 80 MiB;
- their graph high-water is exactly 2,020 MiB (2,118,443,136 recorded bytes
  including small alignment/support storage).

That output graph exceeds the real server's 822.30 MiB attention/activation
arena and hides device-mask savings in the large-ubatch benchmark. It is not a
production server allocation. Perplexity legitimately needs all requested
logits and explicitly declares that capacity, so globally forcing one output
row would be incorrect. The server already uses the correct one-row capacity.

## Nsight allocation, VMM, transfer, and kernel findings

Nsight memory backtraces assigned the large allocations unambiguously:

| allocation | size | owning call path |
|---|---:|---|
| model device buffer | 12,879.473 MiB | model tensor loading |
| output buffer | 0.947 MiB pinned | `llama_context::output_reserve` |
| store stages | 68.000 MiB device | `llama_kv_cache` construction |
| persistent attention KV | 8,160.000 MiB pinned | `llama_kv_cache` construction |
| recurrent state | 149.625 MiB device | `llama_memory_recurrent` construction |
| compute arena | 822.298 MiB device | `ggml_backend_sched_reserve` |
| compute input arena | 80.336 MiB pinned | `ggml_backend_sched_reserve` |

The initial fit probe temporarily allocated and freed a 0.947 MiB output row
and 149.625 MiB recurrent state before final model construction. These are
bounded startup probes, not leaked duplicates.

The CUDA VMM pool mapped five physical chunks, `2+2+4+2+4 MiB`, for a 14 MiB
mapped high-water. All five mappings were released at context teardown. This
direct server result is much smaller than either the 510 MiB scheduler-copy
pair or the static compute arena. Existing source-matched `--kv-memory`
telemetry establishes the distinction between live and mapped pool bytes; this
Nsight server capture independently proves the physical 14 MiB mapping
sequence. VMM trimming can recover only the idle mapped tail for this shape,
not the 822 MiB graph arena.

Across the whole profiled lifecycle, H2D transfer was 12,941.334 MiB, dominated
by model load, and D2H was 168.619 MiB. The prompt/decode portion exposed the
expected cache traffic: 32 copies of 557,056 B (the 512-row K/V stage across
16 attention layers), followed by 32 copies of 835,584 B at the 768-cell
padded live extent. Two 993,280 B D2H copies are the single-row FP32 logits.

The active multi-token native-Q8 route was
`flash_attn_ext_f16<256,256,...Q8_0,Q8_0>` despite the historical function
family name; it launched once per attention layer. It used 255 registers per
thread, 33,792 B dynamic shared memory, and no reported per-thread local
spill. The one-token route used the Q8_0/Q8_0 vector kernel. The existing
source-matched NCU campaign for this exact compact/native-Q8 repair already
measured fewer executed instructions and shared loads than its dense control,
unchanged local loads/stores, unchanged occupancy limits, exact outputs, and
unchanged PPL. Replaying NCU cannot attribute graph-arena ownership, so it was
not duplicated for this allocation-only audit.

## Revised reduction priorities

The ranking below excludes lower-precision changes that could increase
perplexity. Any future candidate still requires a matched PPL gate; an observed
increase is automatic rejection.

1. **Rebase and close the canonical K/V store-stage pool candidate.** The
   capability/layout-based branch collapses 16 compatible per-layer K/V pairs
   to one K pool and one V pool. At `ub=2048`, the current 68 MiB would become
   4.25 MiB, saving 63.75 MiB at startup, prefill, decode, and next turn. Prior
   isolated evidence found identical kernels/copies, exact output and PPL, a
   16 MiB `ub=512` process saving, and neutral point estimates. It is not yet
   merge-certified: the composed branch needs a lower-variance 30K decode gate
   and multi-slot/interleaved lifetime coverage.
2. **Investigate reserve-plan fragmentation without changing graph math.** At
   `ub=2048`, 128 MiB of the device arena is spacing between simultaneous live
   ranges; the bound falls linearly to 16 MiB at `ub=256`. A maintainable
   solution would improve scheduler allocation/lifetime planning by graph
   shape or buffer capability. It must not add model-name checks or duplicate
   per-phase kernels. The 128 MiB is an upper bound, not a promised saving.
3. **Redesign full-context attention staging around the native Q8 consumer.**
   The 510 MiB K/V copy pair is the largest remaining non-model, full-context
   loss. A 32K pair would occupy about 68 MiB, a theoretical 442 MiB reduction
   at 240K. Do not revive the rejected pre-native vector prototype: it tripled
   vector launches, regressed 40K decode by about 3.34%, and added about 60 MiB
   to first-request VRAM. A viable design must reduce simultaneous residency
   without increasing total transfers/launches or changing reduction results;
   direct native-Q8 tiled/online attention is the relevant research direction.
4. **Use physical ubatch as the immediate operational lever.** On this real
   server, reducing `ub=2048` to 256 lowered process VRAM by 332 MiB and the
   device compute arena by 273 MiB; another 59.5 MiB came from the store stages.
   Existing long-depth screening found decode neutral while the larger ubatch
   improved prefill, so this is an explicit memory/speed tradeoff rather than a
   free code optimization. `ub=512` is the measured middle point at 13,890 MiB.
5. **Treat VMM trimming as cleanup, not the main reduction.** The measured
   server mapping high-water is 14 MiB. An all-slots-idle trim may improve
   next-turn residency, but repeated mapping can cost latency and the maximum
   benefit here is small.
6. **Leave the 149.625 MiB recurrent state on GPU unless a lossless design is
   proven.** Moving it to CPU saves VRAM but adds per-token transfer and harms
   decode. Changing its precision is outside the zero-PPL-regression contract.
7. **Do not optimize production around `llama-bench`'s 2,020 MiB all-logits
   arena.** Server already reserves one output row; perplexity needs its full
   output capacity. A benchmark-only control could improve diagnostic fit but
   would change the measured prefill workload and provide no serving VRAM
   benefit.
8. **Model weights remain the dominant 12,879.47 MiB allocation.** CPU layer
   offload or a different weight quantization can reduce it, but those are
   explicit performance or quality tradeoffs, not correctness-neutral allocator
   work.

The compact causal representation remains a real context-scaled win. The
large-ubatch `llama-bench` device graph merely hid it behind all-logits output
capacity; it still removed about 962 MiB of CUDA-owned host mask storage at
`ub=2048`, while earlier real-server and `ub=256` long-context evidence showed
the expected process-VRAM savings. The next work should therefore close the
small, exact store-stage pool first, then prototype allocator-plan compaction,
before committing to a new native-Q8 attention-staging design.
