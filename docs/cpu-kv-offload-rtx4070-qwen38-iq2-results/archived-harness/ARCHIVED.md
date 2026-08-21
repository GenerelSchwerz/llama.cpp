# Historical harness snapshot

These non-executable files are preserved byte-for-byte from the 2026-08-19 RTX
4070 session. They contain machine-specific paths, a whole-process `taskset`
wrapper, no shared GPU lock, and in Pass 1 a later-rejected MTP ubatch geometry.
They are evidence of how the archived artifacts were produced, not maintained
or current runnable protocol.

For new work, use
[`cpu-kv-offload-current-testing.md`](../../cpu-kv-offload-current-testing.md),
including llama-native affinity, visible progress, and the exact
whole-lifecycle `flock /tmp/beellama-single-gpu.lock -c 'COMMAND'` form.
