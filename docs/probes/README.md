# Measurement probes

Patches, not proposals. Each is applied to a clean `946c1e5b6` tree, measured,
and reverted. Three of the five produce **incorrect output by construction** and
say so; none is a candidate implementation.

```bash
git apply docs/probes/01-transport-telemetry.patch
cmake --build build-clean --parallel
# ... measure ...
git apply -R docs/probes/01-transport-telemetry.patch
```

| Patch | What it does | Output correct? |
|---|---|---|
| `01-transport-telemetry.patch` | Counts and times blocking H2D copies and backend synchronizations behind `GGML_KV_TRANSPORT_STATS=1` | yes, no-op when unset |
| `02-transport-telemetry-d2h.patch` | Adds D2H counters and CUDA split counts to the same instrument | yes, no-op when unset |
| `03-store-ceiling.patch` | Retargets the KV store at a scratch device tensor, removing the device-to-host copies, the CPU `set_rows` nodes and the per-layer splits | **no** -- the cache is never written |
| `04-overlap-ceiling.patch` | Delivers the pinned host KV on a copy stream that nothing waits on | **no** -- attention reads the previous token's buffer |
| `05-pinned-alloc-modes.patch` | Adds write-combined, huge-page and no-huge-page modes to the pinned host allocator | yes, off unless the env var is set |

Apply order matters for 03 and 04: both were authored on top of 01 and 02.
Apply 01 and 02 first, or expect context conflicts.
