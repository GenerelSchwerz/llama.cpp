# BeeLlama INI presets

Presets store reusable llama.cpp arguments in an INI file. Use them with the
router server when several models need different paths, cache policies, or
speculative settings:

```powershell
llama-server --models-preset .\models.ini
```

The exact preset argument and preset-only defaults are listed in the
[BeeLlama argument reference](beellama-args.md#presets).

## File format

- Write option names without leading dashes: `ctx-size`, not `--ctx-size`.
- Put shared values in `[*]`.
- Put each named model in its own section.
- Later command-line arguments override values loaded from the selected preset.
- `load-on-startup` and `stop-timeout` are preset-only keys; they are not CLI
  arguments.

```ini
[*]
mmap       = 1
kv-unified = 1
parallel   = 1

[Qwen-DFlash-KVarN]
model                  = D:/models/qwen.gguf
spec-type              = draft-dflash
spec-draft-model       = D:/models/qwen-dflash.gguf
spec-dm-controller     = profit
cache-type-k            = kvarn4
cache-type-v            = kvarn4
flash-attn              = on
reasoning-loop-guard    = force-close
load-on-startup         = 1
stop-timeout            = 10
```

This example omits `spec-draft-n-max`, so DFlash uses
`dflash.block_size - 1` and the default-on profit controller adapts within that
limit. Add `spec-draft-n-max = N` when a fixed upper bound is required; add
`spec-dm-controller = off` when the resolved or explicit depth must remain
static.

## Router model selection

The section name is the router model name. A shared section can define ordinary
server defaults, while each model section supplies its own model source and
overrides:

```ini
[*]
ctx-size    = 32768
batch-size  = 1024
ubatch-size = 512
# Optional for non-MTP model-backed speculation: keep the target ubatch above
# while reducing only the separate draft context's physical ubatch.
spec-draft-ubatch-size = 128
# MTP must omit that setting or explicitly use 512 so recurrent prompt
# synchronization retains clean Bee's physical batch geometry. Use the
# phase-aware workspace setting below for MTP decode-workspace reduction.
# Optional: override target KV placement for independently owned draft cache
# layers. Omit this key to inherit the target policy.
# spec-draft-kv-gpu-layers = N
# Optional: retain only the active prompt or generation workspace. This regrows
# automatically when a later request needs prompt processing and is the MTP
# workspace-reduction control.
phase-aware-workspace = true
# Optional and independent of phase-aware sizing: grow supported standard
# attention workspace reservations with the padded live physical KV extent.
# This is default-off; unsupported memory layouts keep full reservation.
live-context-workspace = true

[qwen-local]
model        = D:/models/qwen.gguf
cache-type-k = kvarn4
cache-type-v = kvarn4

[gpt-oss-hf]
hf          = ggml-org/gpt-oss-20b-GGUF
temp        = 1.0
top-p       = 1.0
top-k       = 0
```

`load-on-startup = 1` autoloads a section. The total number of startup sections
must not exceed `--models-max` unless that upstream limit is configured as
unlimited.

`live-context-workspace` is the preset spelling of
`--live-context-workspace`. It is opt-in and independent of
`phase-aware-workspace`: the former bounds supported standard or hybrid
attention reservations by live physical KV placement, while the latter changes
prompt/generation token geometry. Omit it or set it to `false` to retain the
full-context upfront reservation. Unsupported memory layouts and
fit/no-allocation contexts also retain full reservation and the established
upfront decode order even when it is enabled. At full live demand, the bounded
reservation reaches the configured capacity.

## Remote presets

A Hugging Face preset repository contains `preset.ini` at its root and points
to the actual model repositories from its named sections. Load it through the
normal `-hf` flow:

```powershell
llama-server -hf user/preset-repository
```

Remote presets can select models and server options. Use only repositories you
trust, and inspect `preset.ini` before deployment.

Do not place `hf-token` in a preset. It is a sensitive option and is omitted
from preset serialization and public router metadata. Supply `HF_TOKEN` in the
router environment (or `--hf-token` at startup); child model processes receive
only the environment value, never a token-bearing argv entry.

## BeeLlama migration rules

New presets must use `draft-dflash`, standard q cache names, or `kvarnN` target
cache names. Do not carry forward TurboQuant/TCQ formats,
`spec-dflash-cross-ctx`, tree-verifier settings, `GGML_DFLASH_*` variables, or
`GGML_CUDA_FA_HALF_QUANTS`. The complete redirect and removal list is in
[Migration from earlier versions](beellama-args.md#migration-from-earlier-versions).
