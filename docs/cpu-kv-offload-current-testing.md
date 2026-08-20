# CPU KV-offload current testing and setup

This is the sole runnable protocol for current work on `exp/kv-cpu-offload`.
Use it before starting a build, correctness run, profiler capture, or
performance comparison. The local source and each binary's recorded identity
remain authoritative if this document ever disagrees with behavior.

The companion documents have different roles:

- [`cpu-kv-offload-development.md`](cpu-kv-offload-development.md) is the
  progress and decision journal. It preserves earlier protocol editions,
  rejected paths, and the evidence behind current choices.
- [`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md) is the
  immutable evidence ledger. Commands in an older entry reproduce that entry;
  they are not current command templates.
- Focused reproduction documents explain their named investigation. Treat
  their commands as historical unless this document references them.

When the protocol changes, update this document first. Add the reason and the
superseded behavior to the development journal without rewriting old evidence.

## Current implementation under test

The retained host-KV configuration uses the supported CLI controls below.

| Control | Current meaning |
|---|---|
| `--no-kv-offload` | Enables host-resident attention KV. |
| `--kv-cpu-pinned` | Allocates supported host KV through accelerator-visible pinned host buffers. |
| `--recurrent-state-offload` | Keeps supported hybrid-model recurrent state on the accelerator while attention KV remains on the host. |
| `--kv-gpu-layers N` | Under host KV, keeps the first `N` target-owned attention-KV layers on the accelerator. Zero leaves all target-owned layers on the host. |
| `--spec-draft-kv-gpu-layers N` | Overrides target KV residency for a separate draft context and keeps the first `N` independently owned draft attention-KV layers on the accelerator. Omission inherits the target policy; zero explicitly selects host residency. Shared layers follow their owner. |
| `--phase-aware-workspace` | Uses compact generation reservations, grows them for prompt work, and shrinks them again for generation. |
| `--spec-mtp-rs-planes N` | Caps total target recurrent planes for MTP, including the current plane. |

The residency implementation consumes the common per-owned-layer KV placement
plan; it is not tied to one model's owned-layer count. Determine `N` from the
layout being tested and record it. The integrated Qwen MTP setup below happens
to have one independently owned draft attention-KV layer, so its full
draft-owned-GPU candidate uses `N=1`; that value is not a general
recommendation.

MTP must retain the target's physical ubatch geometry. Omit
`--spec-draft-ubatch-size`, or set it equal to the target `--ubatch-size` only
when an explicit-value parser test requires that spelling. A different MTP
draft ubatch is rejected. Use phase-aware workspace to reduce decode backing
without changing physical ubatch geometry.

Standard quantized host KV uses the canonical accelerator quant-store path.
That keeps stored Q8 bytes independent of host versus device residency and is
part of the current exactness contract.

`llama-perplexity` declares its full logical batch as the maximum output-row
requirement before context creation. Therefore matched PPL runs may enable or
disable `--phase-aware-workspace` without reducing the all-logits capacity the
tool needs. Treat an output-capacity assertion as a failed run, not as quality
evidence.

## Retired controls are historical only

Do not put removed controls into a current command, including defensive
`env -u` entries. In particular, `GGML_KV_CPU_PINNED` and
`GGML_RECURRENT_STATE_OFFLOAD` are retired historical environment variables.
Use `--kv-cpu-pinned` and `--recurrent-state-offload` directly.

Prefer explicit CLI controls for every measured configuration. Use a current
`LLAMA_ARG_*` environment variable only when the environment-variable path is
itself under test, and record it as part of the configuration. Do not copy a
legacy environment-clearing preamble from the journal or experiment ledger.

Also exclude the superseded MTP draft-ubatch 128 layout from current tests.
Historical measurements that used it remain evidence for why the mismatch is
now rejected, not evidence for the integrated implementation.

## Current local layout

- Experimental source: `/home/gencoolpc/beellama-kv-offload`, branch
  `exp/kv-cpu-offload`.
- Known baseline source: `/home/gencoolpc/beellama.cpp`. Keep it unchanged
  unless a task explicitly changes the baseline.
- CUDA build: `build-cuda-all`, Release, native CPU, CUDA FlashAttention,
  compute architecture 120, default quant-pair matrix.
- GPU: NVIDIA GeForce RTX 5070 Ti, 15,880 MiB usable process memory, compute
  capability 12.0.
- CPU: Intel Core Ultra 9 285K. Controlled decode uses CPUs 0-2; batch work may
  use CPUs 0-23.
- Target and integrated-MTP model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.
- Multimodal projector for the original serving layout:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf`.
- `llama-benchy` tokenizer source: `Qwen/Qwen3.5-27B`. Version 0.4.0 pulls or
  reuses that tokenizer independently of the served alias.
- Benchmark corpus URL:
  `https://www.gutenberg.org/files/1661/1661-0.txt`. The current cached file is
  `/home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt`,
  606,662 bytes, SHA-256
  `8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2c`.

These paths describe this benchmark host, not portable project defaults.
Recheck hardware and file identities instead of assuming they are unchanged.

## Preflight and binary identity

Run these checks before a current measurement:

```bash
cd /home/gencoolpc/beellama-kv-offload
git status --short --branch
git rev-parse HEAD
build-cuda-all/bin/llama-server --version
llama-benchy --version
sha256sum build-cuda-all/bin/llama-server \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  /home/gencoolpc/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt
nvidia-smi
```

Confirm that the binary version corresponds to the source being measured.
Documentation-only dirt does not change a binary, but an unbuilt source change
does. Record the build options, compiler, model sizes and hashes, GPU driver,
and relevant hardware. Do not start a controlled performance run while an
unrelated GPU workload is active.

## Canonical server layout

The original multimodal performance layout currently uses:

```bash
build-cuda-all/bin/llama-server \
  --model /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf \
  --mmproj /home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-F16.gguf \
  --no-mmproj-offload \
  --n-gpu-layers 999 --n-gpu-layers-draft 999 \
  --fit off --split-mode none --main-gpu 0 --flash-attn on \
  --no-kv-offload --kv-cpu-pinned --recurrent-state-offload \
  --kv-gpu-layers 0 \
  --ctx-size 32768 --parallel 1 --cont-batching --kv-unified \
  --batch-size 1024 --ubatch-size 512 \
  --phase-aware-workspace \
  --spec-type draft-mtp --spec-draft-n-max 6 \
  --spec-mtp-rs-planes 3 --spec-draft-p-min 0.85 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --threads 3 --threads-batch 24 \
  --cpu-range 0-2 --cpu-range-batch 0-23 --cpu-strict 1 --poll 100 \
  --reasoning-loop-guard force-close --seed 1234 --cache-ram 0 \
  --alias qwen38-kv-test \
  --host 127.0.0.1 --port 8080
```

For the inherited-host draft baseline, do not add a draft residency override.
For a draft-owned-GPU candidate, add
`--spec-draft-kv-gpu-layers N`, where `N` is the tested draft-owned layer
count. Keep the same alias on both servers so the request payload is identical.
Change only the intended variable; restart the server between configurations.

Use a context size large enough for the largest requested depth, prompt, chat
template overhead, and generated tokens. The 32,768 setting covers the current
30K-depth, 512-prompt, 64-generation screen. Restore 140K only for a test that
actually requires very deep context and has an explicit progress plan.

## Matched `llama-benchy` performance protocol

The current screen measures 512-token prefill and 64-token exact generation at
depths 4,096 and 30,000, once per fresh server. The benchmark must expose
progress with `--emit-progress` and preserve its result file.

`llama-benchy` 0.4.0 has two prompt-randomization details that matter across
separate invocations:

1. `--no-cache` appends a fresh UUID to the prompt and can change tokenization.
2. Corpus offsets use NumPy's process-random generator, for which the CLI has
   no seed option.

Therefore, do not use `--no-cache` for a matched cross-process pair. Seed NumPy
before entering the same `llama-benchy` CLI in each process, and send
`cache_prompt=false` as a request field instead. The server also has
`--cache-ram 0`, and every configuration starts in a fresh process.

```bash
python3 -c \
  'import numpy as np; np.random.seed(1234); from llama_benchy.__main__ import main; main()' \
  --base-url http://127.0.0.1:8080/v1 \
  --model Qwen/Qwen3.5-27B \
  --served-model-name qwen38-kv-test \
  --book-url https://www.gutenberg.org/files/1661/1661-0.txt \
  --pp 512 --tg 64 --depth 4096 30000 \
  --runs 1 --no-warmup --skip-coherence --no-adapt-prompt \
  --latency-mode none --exact-tg \
  --extra-body temperature=0,seed=1234,cache_prompt=false \
  --emit-progress - \
  --save-result RESULT.json --format json
```

Use the same seed, tokenizer revision/cache, corpus URL/cache, alias, CLI order,
and request body for both configurations. Save or hash the corpus cache used by
the accepted comparison. Before calculating a delta, require equal observed
prompt-token counts at every depth. Also compare generated-token counts,
acceptance/replay work, server timing fields, and errors. A nominally equal
depth is not a matched input if the observed prompt count differs.

Record prefill throughput, decode throughput at 4K and the long-context depth,
and peak process VRAM. Sample VRAM at least once per second from server startup
through completion and retain the timestamped sample log. Record configured
host and pinned-memory allocations separately; `nvidia-smi` process memory does
not include page-locked host allocation.

A single run is acceptable for the requested long screen, but it is a screen,
not a noise-resistant performance claim. Repeat clean alternating pairs before
claiming a small improvement or regression.

## Full live MTP decode protocol

Use the maintained 5,000-token live matrix when the short `llama-benchy`
generation is not representative enough. It sends one stochastic chat
completion with the original host-resident multimodal projector, MTP depth 6,
temperature 0.8, seed 1234, phase-aware workspace, three recurrent planes, and
target/effective-draft physical ubatch 512. A fresh inherited-host-draft server
is the live reference; a fresh draft-owned-GPU server is compared against it.

```bash
cd /home/gencoolpc/beellama-kv-offload
python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-draft-residency-live-mmproj-q8-5k.json
```

The runner reports decoded-token progress and process VRAM every five seconds,
captures server timings and allocation logs, and requires exact prompt tokens,
request semantics, output token IDs, and response bytes. The configured output
directory must not already exist; use a new dated output-directory suffix for
a later run rather than overwriting prior evidence.

Use server-reported `predicted_per_second` for the decode comparison. Confirm
equal acceptance, generated-draft count, replay cycles/tokens, token count, and
hashes before attributing a throughput change to residency. This remains a
single ordered pair unless the experiment explicitly repeats alternating
configurations, so small differences are screens rather than stable claims.

## Current exactness oracle

MTP placement comparisons use a clean, same-MTP-geometry reference, not
target-only decoding. Match model bytes, prompt, sampler and seed, MTP depth and
threshold, cache formats, context, batch, target ubatch, effective draft
ubatch, and request semantics. Residency, recurrent-plane count, and
phase-aware backing may differ only when those are the variables being tested.

The maintained draft-residency gates are:

```bash
cd /home/gencoolpc/beellama-kv-offload
python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-partial-draft-residency-q8-1k.json
python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-partial-draft-residency-q8-5k.json
```

The runner provides native `/slots` token progress, starts every live case in a
fresh sterile server process, captures request/response/log/progress artifacts,
samples process VRAM, and verifies the manifest's identity contract. Do not use
`--allow-mismatch` for an acceptance gate.

The current manifests are text-only exactness tests. The performance protocol
above retains the original multimodal projector layout. Do not silently compare
throughput across those layouts.

## Regression and artifact requirements

Run coverage proportional to the change. The current draft-residency merge was
gated by:

```bash
python3 scripts/test-mtp-exactness.py
ctest --test-dir build-cuda-all --output-on-failure \
  -R 'test-(std-kv-tail-static|server-loop-guard-checkpoint-static|spec-cession-static|kvarn-rollback-static|batch-alloc|generate-models|recurrent-state-rollback|arg-parser|kv-cache-tail|server-loop-guard|server-prompt-checkpoint|backend-sampler)'
```

Also run the relevant CPU and CUDA `SET_ROWS` matrices when cache store or
quantization behavior changes, and the full maintained 1K/5K exactness gates
when MTP scheduling, recurrent state, workspace geometry, or KV residency can
affect output.

Every accepted experiment record must contain:

- base and candidate source commits plus binary versions and hashes;
- exact commands, environment, model/projector/tokenizer/corpus identities,
  request body, and prompt-token counts;
- hardware, driver, compiler, and build configuration;
- prefill, 4K decode, long-context decode, process VRAM, system-memory, and
  pinned-memory measurements as applicable;
- the progress mechanism and retained progress/log/result artifacts;
- correctness coverage, resource tradeoffs, and disposition.

If a run violates the matching contract, preserve it as an invalid or
diagnostic artifact and explain the failure in the development journal. Do not
publish its directional numbers as a candidate-versus-baseline result.

## Reading previous editions

Use the development journal to understand why a control or protocol changed,
then follow its references into the experiment ledger or Git commits for exact
evidence. Historical names and commands are intentionally searchable. Their
presence does not make them supported today.

When recovering an older experiment:

1. Identify its dated journal section and ledger entry.
2. Reproduce it only when historical reproduction is the task.
3. For new evidence, translate the intended variable into this current
   protocol and document every necessary deviation.
4. If the translation changes the oracle, geometry, prompt, or measured
   implementation, treat it as a new experiment rather than extending the old
   result.
