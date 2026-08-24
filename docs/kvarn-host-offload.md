# KVarN and host-resident KV: open issue

**Status: open.** The guard described below is a stopgap. KVarN has not been
integrated into the KV-offload line at all, and until it is, KVarN and
`--no-kv-offload` are mutually exclusive on a discrete accelerator.

## What happens today

KVarN is inherited from BeeLlama. The host-KV-offload line — `--kv-cpu-pinned`,
`--kv-gpu-layers`, the `offload_attn_compute` separation of storage placement
from attention execution — is this fork's, and none of it was extended to
KVarN. BeeLlama v0.4.3 carries KVarN across 89 files and contains no occurrence
of `kv_cpu_pinned`, `offload_attn_compute` or `kv_gpu_layers`; this tree has
them in 9–11 files each.

The combination is currently permitted:
`llama_kvarn_backend_supports_ops(nullptr)` returns true on the grounds that
"the built-in CPU backend implements store + materialize". It does — correctly,
and very expensively.

`ggml_backend_cuda_device_supports_buft` accepts a host buffer only on an
integrated GPU. On a discrete one it refuses, so the scheduler places every node
that touches the host-resident KVarN records on the CPU backend and reserves
host workspace for their intermediates.

Scheduler assignment, `GGML_SCHED_DEBUG=2`, `-c 8192`, host-resident, RTX 4070:

| Cache | CPU-assigned nodes | host compute buffer |
|---|---|---:|
| `q8_0` | 9 × `GET_ROWS` | **20.28 MiB** |
| `kvarn5` | 576 `SET_ROWS`, 576 `KVARN_STOR`, 384 `MUL_MAT`, 384 `CONCAT`, 288 `KVARN_WHT` | **2,195.90 MiB** |

The large CPU tensors are `KVARN_WHT` and `GET_ROWS` at 640 MiB each,
`SOFT_MAX` at 414 MiB and `MUL_MAT` at 384 MiB.

A standard quantized cache avoids this because its K/V reach attention as
copyable split inputs, which the scheduler stages to the device — that staging
is the context-linear transfer measured in
[`pinned-host-kv-decode-experiments.md`](pinned-host-kv-decode-experiments.md).
KVarN's records reach attention through `KVARN_VIEW` chains instead, so there is
nothing for the scheduler to stage and it falls back wholesale.

## Cost

`kvarn5`, host-resident, reserved host compute workspace:

| Context | host workspace | host KVarN cache |
|---:|---:|---:|
| 8,192 | 2,195.90 MiB | — |
| 32,768 | 3,309.90 MiB | — |
| 65,536 | 6,541.90 MiB | — |
| 262,144 | 25,933.90 MiB | 5,528.00 MiB |

Between 32,768 and 262,144 that is about 0.099 MiB per context token. The same
cache GPU-resident at 65,536 reserves 84.65 MiB. The reservation is identical
for `kvarn8`, `kvarn6` and `kvarn5`, so it tracks context, not cache width, and
`--live-context-workspace` does not bound it (25,933.90 MiB with and without).

At 262,144 the workspace is roughly 4.7x the cache it compresses, and together
they exceed this host's 31 GiB. Attention also executes on the CPU, which
Experiment 004 measured as needing about a 6.6x CPU-attention speedup merely to
break even against the staged-transfer path.

So KVarN currently trades a real 36.5% cache reduction (`kvarn5` 5,528 MiB
against `q8_0` 8,704 MiB at 262,144) for a workspace and an execution placement
that cost far more than the reduction is worth.

## What this change does, and does not, do

It fails closed. When a layer's accelerator cannot operate on the buffer its
KVarN records would live in, context construction now raises an error naming the
device, the buffer type, and the two configurations that do exist — keep the
KVarN cache device-resident, or use a standard quantized type for host-resident
KV.

It does **not** make KVarN work with host offload. That needs KVarN brought into
the KV-offload line properly:

- give the KVarN body the same staged-input treatment standard KV gets, so
  attention stays on the accelerator while storage stays on the host, which is
  what `offload_attn_compute` already expresses for standard caches;
- decide whether `KVARN_VIEW` operands can be staged at all, or whether the
  records need a device-side mirror with its own residency policy;
- extend `--kv-gpu-layers` to KVarN layers so partial residency is expressible;
- then qualify it the way the standard types were: exactness, KLD, allocation
  high-water, and throughput at depth.

Integrated GPUs are unaffected — `supports_buft` accepts host buffers there, so
the guard does not trigger. CPU-only builds are unaffected for the same reason.
