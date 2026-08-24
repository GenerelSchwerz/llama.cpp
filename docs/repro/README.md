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

| Script | Reproduces | Verified how |
|---|---|---|
| `d1-vector-d320.sh` | D1, head dimension 320 aborting in the vector kernel | run against real binaries, produced the recorded output |
| `d2-barrier.sh` | D2, the divergent `__syncthreads()`, needs `build-li` | run against real binaries, produced the recorded output |
| `d3-mtp-abort.sh` | D3, MTP + host-resident KV aborting mid-generation | run against real binaries, produced the recorded output |
| `d4-test-kvarn.sh` | D4, `test-kvarn` on a default build | run against real binaries, produced the recorded output |
| `d5-kvarn-workspace.sh` | D5, KVarN host-resident workspace, needs `build-kvarn` | both arms run against real binaries |
| `d6-softcap.sh` | D6, the softcap coverage gap | run against real binaries, produced the recorded output |
| `d7-kvarn-build.sh` | D7, the historical KVarN build failure | the base failure and the patched build were both reproduced; PR #34 carries the repair |
| `d8-q2-fa64.sh` | D8, `q2_0s` FlashAttention wrong output at `hsk=64` | run against real CUDA binaries at three seeds |
| `d9-kvarn-device.sh` | D9, KVarN refused for a device-resident cache on a default build | run against real binaries; requires `LLAMA_KV_MODEL` |
| `d10-agents-static.sh` | D10, `test-upstream-merge-keepers-static` failure | run against a clean build; no GPU or model |
| `m1-cost-structure.sh` | The decode cost structure, needs patches 01+02 | argument handling verified against a stub; the recorded numbers were taken with an equivalent ad-hoc command, not this script |
| `m2-store-ceiling.sh` | The store-side ceiling, needs patch 03 | as m1 |
| `m3-overlap-ceiling.sh` | The copy/compute overlap ceiling, needs patch 04 | as m1 |
| `m4-device-resident.sh` | The device-resident compute floor, no patch needed | run against real binaries |

**On that last column.** The `d*` scripts were each executed end to end and the
output in [`../kv-offload-defects.md`](../kv-offload-defects.md) is what they
printed. The `m*` scripts are a tidied-up form of the ad-hoc commands that
produced the numbers in
[`../kv-offload-measurements.md`](../kv-offload-measurements.md); their argument
handling is verified but, except for `m4`, they have not themselves been run
against a patched build to regenerate those numbers. Treat the measurements as
sound and the `m*` scripts as convenience wrappers that have had one round of
fixes.

The patches under [`../probes/`](../probes/) are applied with `git apply` and
reverted with `git apply -R`. They are measurement instruments and produce
**incorrect output by construction** where stated; none of them is a candidate
implementation.
