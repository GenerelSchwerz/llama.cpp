# Reproduction scripts

Every script here reproduces one entry in
[`../kv-offload-defects.md`](../kv-offload-defects.md) or one measurement in
[`../kv-offload-measurements.md`](../kv-offload-measurements.md).

Set `LLAMA_KV_MODEL` to a model path before running anything; the reference
results were taken with `Qwen3.8-27B-UD-IQ2_M.gguf`. Set `LLAMA_KV_BUILD` to
override the build tree a script uses.

Scripts run one GPU job at a time behind `/tmp/beellama-single-gpu.lock`.
Running two of them concurrently produces CUDA out-of-memory failures that look
like defects and are not.

| Script | Reproduces |
|---|---|
| `d1-vector-d320.sh` | D1, head dimension 320 aborting in the vector kernel |
| `d2-barrier.sh` | D2, the divergent `__syncthreads()`, needs `build-li` |
| `d3-mtp-abort.sh` | D3, MTP + host-resident KV aborting mid-generation |
| `d4-test-kvarn.sh` | D4, `test-kvarn` on a default build |
| `d5-kvarn-workspace.sh` | D5, KVarN host-resident workspace, needs `build-kvarn` |
| `d6-softcap.sh` | D6, the softcap coverage gap |
| `m1-cost-structure.sh` | The decode cost structure, needs the telemetry patch |
| `m2-store-ceiling.sh` | The store-side ceiling, needs the store-ceiling patch |
| `m3-overlap-ceiling.sh` | The copy/compute overlap ceiling, needs the overlap patch |
| `m4-device-resident.sh` | The device-resident compute floor, no patch needed |

The patches under [`../probes/`](../probes/) are applied with `git apply` and
reverted with `git apply -R`. They are measurement instruments and produce
**incorrect output by construction** where stated; none of them is a candidate
implementation.
