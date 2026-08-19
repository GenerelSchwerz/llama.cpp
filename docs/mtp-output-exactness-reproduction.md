# MTP output exactness: complete investigation and reproduction record

This document records the oracle decision, baseline checks, diagnostic
sequence, rejected approaches, retained implementation, automated harness,
exact commands, results, and limitations for making BeeLlama's MTP output
independent of target/draft Q8 KV residency and phase-aware workspace policy.
It complements the compact Experiment 017 and Experiment 018 entries in
[`cpu-kv-offload-experiments.md`](cpu-kv-offload-experiments.md).

## Final result

For the tested Qwen3.8 27B CUDA configuration, the candidate now matches clean
BeeLlama MTP token-for-token and response-byte-for-byte when:

- target and draft Q8_0/Q8_0 KV move between GPU and pinned CPU storage;
- phase-aware target/draft workspace is off or on;
- full-plane GPU MTP depth is 1, 2, 3, 5, 6, or 8;
- MTP-3, MTP-5, and MTP-8 use 2-, 3-, 4-, and full-plane policies, while
  MTP-6 also covers the retained three-plane CPU serving candidate;
- one live server reuses, shrinks, and regrows its prompt cache across four
  requests and two real sleep/unload/wake/reload cycles;
- a router preset is explicitly rescanned, unloads the
  changed live model, and autoloads it again on the next identical request;
- requests include the 1,000-token greedy and 5,000-token stochastic orbital
  cases, a stochastic number-game prompt at seeds 7 and 42, and an independent
  stochastic website prompt at seed 2026;
- acceptance edges include rejection-heavy `p_min=0` replay beyond a zero-token
  direct horizon and high-confidence `p_min=0.999` full acceptance with no
  replay; and
- standard exact-tail attention is enabled in the 128-token control.

The final 5K MTP-6 matrix produced one common token SHA-256,
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`,
for clean GPU, candidate GPU, candidate pinned-CPU full-plane, and candidate
pinned-CPU three-plane cases. CPU full decode measured 57.200 t/s versus
63.223 t/s for candidate GPU. CPU three-plane decode measured 55.363 t/s and
reduced sampled peak process VRAM from 15,006 MiB to 13,926 MiB.

These are exactness gates and matched single-run characterizations, not
throughput confidence intervals. The final matched Nsight matrix independently
measures the whole-process H2D/D2H costs reported below.

## Correct oracle

The initial goal said every MTP variant should match target-only decoding.
Investigation showed that this is not the upstream behavioral contract.
Multi-token MTP verification can choose a different floating-point execution
shape from one-token target decoding, so a fixed seed does not require the MTP
stream to equal the target-only stream.

The correct compatibility oracle is clean BeeLlama at the same:

- MTP depth and acceptance threshold;
- model, prompt, sampler, and seed;
- target and draft cache types;
- target and effective draft physical ubatch;
- prompt-cache policy; and
- relevant execution geometry.

The sibling clean llama.cpp checkout provided an independent baseline at
`af5172627d3513a7efed526b206dca9cd6536452`. Its 1K Q8 token hashes were:

| Mode | Token SHA-256 |
|---|---|
| target-only | `cd8d20d1270ee556a5035994abe08e55fab1a38600a89208becbc9e348e8d283` |
| MTP-2 | `14e47fb5c35897bfe818339bd163ef1be1f630b558055fe0d323cad922c36217` |
| MTP-6 | `842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00` |

Those are the same hashes produced by clean Bee and the final candidate. The
upstream control server SHA-256 was
`710538e3219625f43531037128e0357621b5ba6c6898e5bbd4ec5682ed094b27`;
its artifact is `/tmp/llama-upstream-depth-sweep-1k-20260819`.

## Source lineage and isolation

| Purpose | Path | Identity |
|---|---|---|
| untouched clean Bee oracle | `/home/gencoolpc/beellama.cpp` | `ba27edad2a84ff045a556df06661e821285c2fab` |
| experimental candidate | `/home/gencoolpc/beellama-mtp-exact` | branch `exp/mtp-bit-exact`, base `7febdc06a795002bf9e82f4b84026fd3740a3a12`; retained implementation `3693c6119` and `d4b50c5cc`; harness `85eef4a89` |
| clean llama.cpp control | `/home/gencoolpc/llama.cpp` | `af5172627d3513a7efed526b206dca9cd6536452` |
| CPU-KV integration tree | `/home/gencoolpc/beellama-kv-offload` | held unchanged during implementation and measurement; used as the integration target afterward |

The measurements were intentionally completed against the documented dirty
patch before a commit was requested. The retained source was subsequently
committed without runtime changes as `3693c6119` (MTP ubatch geometry) and
`d4b50c5cc` (canonical quantized host storage); the harness and maintained
manifests were committed as `85eef4a89`. The final candidate server SHA-256 is
`da7374a9c26c4ddf89136c67c5927fc6f27d09dcee36fe365c04c59eb07f6be3`.
The SHA-256 of
`git diff -- common ggml src tests tools/server` after the final audit is
`12b51655d51ab26c16ac527eeef1cdf3cc4d884d8b1bfbe06e208801406bb49c`.
The runtime-bearing diff at the final server rebuild was
`28c5eedaf35c2c776b574ffeb8a35f76346e11e71b430478a15514a026f98f32`;
the later difference is only the semantic upstream-merge static guard described
in Verification, so the server binary did not require another rebuild.
The manifests and runner were separate untracked reproduction inputs at
measurement time, are individually captured in each artifact's provenance,
and are now retained by `85eef4a89`.

## Build and shared ccache

The candidate build used Ninja, Release mode, CUDA FlashAttention, native CPU,
the default quant matrix, and CUDA architecture 120 (compiled as 120a):

```bash
cd /home/gencoolpc/beellama-mtp-exact

cmake -S . -B build-mtp-exact -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-mtp-exact \
  --target llama-server llama-bench test-arg-parser \
           test-backend-ops test-kv-cache-tail -j 32
```

All related worktrees use a shared ccache configuration:

```text
cache_dir = /home/gencoolpc/.cache/ccache
base_dir  = /home/gencoolpc
hash_dir  = false
max_size  = 50G
```

This normalizes sibling worktree paths. A public-header content change still
requires every dependent translation unit to be considered; ccache can reuse
only objects with an identical dependency hash.

## Hardware, model, and common runtime geometry

- GPU: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, driver 610.57.04,
  16,303 MiB physical and about 15,880 MiB usable process capacity.
- CPU: Intel Core Ultra 9 285K, 24 cores, no SMT, one NUMA node.
- CUDA toolkit: 13.3.
- Model:
  `/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`.
- Model size: 14,437,471,712 bytes.
- Model SHA-256:
  `ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.
- Target and draft K/V: Q8_0/Q8_0.
- Batch/ubatch/effective draft ubatch: 1,024/512/512.
- Decode threads and strict affinity: three threads on CPUs 0-2.
- Batch threads and strict affinity: 24 threads on CPUs 0-23.
- GPU layers: 999 for target and MTP context; split mode `none`, main GPU 0.
- FlashAttention on, unified KV, one parallel slot, continuous batching.
- Prompt cache disabled with `--cache-ram 0` for the 1K and 5K throughput gates.
  The lifecycle matrix explicitly sets `cache_prompt=true` per request and
  `--sleep-idle-seconds 2`.

The final 1K request used greedy sampling and seed 1234. The final 5K request
used seed 1234, temperature 0.8, MTP-6, and `p_min=0.85` in context 8,192.

## Environment isolation

The exactness runner constructs a sterile subprocess environment. It does not
inherit ambient `LLAMA_*`, `GGML_*`, CUDA visibility, tuning, preset, thread,
or affinity variables from the interactive shell. Each maintained manifest
adds only:

```json
"environment": {"CUDA_PATH": "/opt/cuda"}
```

All behavior-changing choices appear explicitly in `common_args` or a case's
`args`. This matters for the CPU/GPU comparison: an old shell export cannot
silently enable operation offload, change a cache type, change MTP geometry, or
select a different device for only one case.

The runner records the effective manifest, request and input hashes, executable
and linked-library hashes, Git state, CMake cache, CUDA/driver/CPU inventory,
ccache statistics, and start/completion timestamps. It starts one fresh server
per case, polls `/health`, polls `/slots` at the manifest interval for visible
progress, samples per-process VRAM with `nvidia-smi`, saves token IDs and
response bytes, and stops the server before the next case. A manifest may run
one completion or a typed same-process sequence. The maintained sequence types
are completion, bounded `/props` sleep/awake observation, and explicit router
model reload. Reload can only replace a harness-owned preset with a declared,
hashed fixture; unknown actions fail closed.

## Automated exactness contract

[`scripts/mtp-exactness.py`](../scripts/mtp-exactness.py) compares more than
rendered text. A candidate must match its explicit reference for:

- required identity fields from the manifest;
- request semantics;
- tokenized prompt hash;
- output token count and every token ID; and
- generated response bytes.

MTP manifests must declare `effective_draft_ubatch`. The runner independently
resolves target and draft ubatch from CLI and environment inputs and rejects a
false declaration. Explicit reference edges prevent an unrelated case from
becoming the accidental golden. It also binds a declared sampler seed and
temperature to the request body actually sent for every completion, so two
matching identity labels cannot conceal different sampler inputs.

For lifecycle tests, `prompt_from` builds the next prompt from the earlier
request's exact prompt-token IDs, an explicitly bounded prefix of its output
token IDs, and suffix text tokenized with `add_special=false`. The comparison
contract checks every step independently. Thus a mismatch in one response
cannot silently turn the next request into a comparison with different input
tokens, and a matching final request cannot hide an earlier mismatch.

The harness unit test is:

```bash
cd /home/gencoolpc/beellama-mtp-exact
python3 scripts/test-mtp-exactness.py
```

It passes 35/35, including manifest/environment disagreement, missing identity,
prompt mismatch, sampler mismatch, token mismatch, content mismatch, valid
multi-reference cases, lifecycle ordering, unknown-action rejection,
continuation-token construction, bounds checks, and later-step mismatch
localization. Router coverage also checks typed reload parsing, strict model
status lookup, and within-case before/after equivalence references.

## Investigation chronology

### 1. Target-only was rejected as the universal MTP oracle

Clean Bee and clean llama.cpp showed that same-depth MTP agrees across the two
trees while target-only can be a different stream. The goal was narrowed to
matching clean Bee MTP at identical execution geometry.

### 2. Physical MTP ubatch caused a delayed divergence

The inherited clean-Bee draft ubatch is 512. Candidate draft ubatches 128 and
32 passed a short screen but first changed the 1K MTP-2 visible stream at token
100. Trace instrumentation showed an earlier acceptance regrouping near token
60 after the 149-token prompt was synchronized as `128 + 21` rather than one
physical batch. Phase awareness off and on produced the same mismatch.

The retained validator rejects a nonzero MTP draft ubatch different from target
ubatch through CLI, `LLAMA_ARG_SPEC_DRAFT_UBATCH`, and rendered INI/preset
paths. Omitted and explicit-equal values pass; other model-backed speculative
modes retain their independent setting. Experiment 017 contains the complete
matrix and trace artifact.

### 3. CPU/GPU Q8 storage first differed at token 5

With ubatch geometry matched, clean/candidate GPU Q8 agreed but candidate
pinned-CPU Q8 first differed at generated token 5. The F16 CPU/GPU control was
exact. Raw state comparison found sparse byte differences in Q8 rows produced
by the CPU and CUDA converters. That proved the source was quantized persistent
KV construction, not sampling, recurrent rollback, MTP acceptance, or phase
workspace.

### 4. Low-level rounding synchronization was rejected

Temporary CPU/CUDA rounding edits attempted to force the two converters to
match. This was not a durable abstraction: each backend and cache type would
need coupled numerical maintenance, and the approach did not define which
backend owned the expected bytes. These edits were reverted.

### 5. Indexed full-stage storage was rejected

A temporary device `SET_ROWS` stage coupled scratch to cache extent and allowed
partial microbatches to expose unwritten stage rows. It was also reverted.

### 6. Retained canonical conversion and byte scatter

For host-resident quantized standard KV belonging to an accelerator layer, the
cache allocates bounded quantized per-layer stage tensors on the accelerator.
The graph converts only current F32 rows on that accelerator, transfers those
quantized rows to the CPU backend, and scatters them into persistent host KV
with same-type `SET_ROWS`.

The CPU same-type path uses `memcpy` of `ggml_row_size(type, width)` bytes. It
never dequantizes or requantizes. The layer/type route is capability-probed and
fails closed if the accelerator cannot perform the required direct and staged
conversions. No architecture-name or model-name exception is used.

### 7. Exact-tail planning had conflated storage and execution

After body rows were canonical, the exact-tail CPU case still differed at
token 5. Pinned host KV is storage-owned by CPU, but with operation offload its
attention runs on CUDA. The old route probe derived both decisions from the
buffer owner and therefore selected the wrong numerical execution route.

The route descriptor now separates storage buffer type from execution backend.
Writes are checked against storage capability; attention/math are checked
against execution capability. A native final-tail operation is assigned to the
scheduler backend for the planned execution device, refusing fallback if it is
missing. `--no-op-offload` remains a verified CPU route.

### 8. Fused-op probes exposed a one-token stage-sizing bug

`llama-bench` with ubatch one constructs a synthetic 16-token Gated Delta Net
graph while resolving fused operations. A stage sized only to physical ubatch
failed with source `[256,4,16,1]` versus stage `[1024,1]`. A shared
`LLAMA_MAX_FUSED_OP_PROBE_TOKENS_PER_SEQ=16` contract now sizes persistent
per-layer graph scratch to the maximum of real ubatch and fused probe geometry.
The one-token CPU-pinned Q8 benchmark then completed at 16.25 t/s; the
`--no-op-offload` form completed at 16.21 t/s and logged a CPU tail route.

## Retained data path and cost model

```text
accelerator layer output (F32 current rows)
                |
                v
accelerator CPY conversion (F32 -> Q8 stage)
                |
                v
scheduler transfer (only current Q8 rows, D2H)
                |
                v
CPU same-type SET_ROWS (byte-preserving scatter)
                |
                v
persistent pinned-host Q8 KV
```

At ubatch 512 on this model, 16 host-resident target attention layers allocate
512-row K and V store stages totaling 17.00 MiB of device memory. The MTP
context's one host-resident attention layer allocates another 1.06 MiB, for an
18.06 MiB device-stage total in the measured CPU-resident MTP cases. The path
transfers each new quantized row D2H. It does not allocate a hidden full-context
GPU KV copy. Final Nsight accounting finds 214.858 MB more whole-process D2H in
the CPU-resident full-plane case than the otherwise matched GPU-resident case.
That increase is consistent with the canonical row-store mechanism, although
the placement comparison includes every residency-dependent transfer and
therefore is not treated as a stage-only micro-measurement. Serialized
checkpoint payload counters remain unrelated to PCIe transfer measurement.

## Exact reproduction commands

Run each manifest from the candidate worktree. Each has its own output path;
change `output_dir` first if preserving the recorded artifacts.

```bash
cd /home/gencoolpc/beellama-mtp-exact

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-tail-cross-residency-q8-128.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-target-cross-residency-q8-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp-cross-residency-q8-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-gpu-q8-full-depths-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-gpu-q8-plane-cap-depths-1k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-cross-residency-q8-5k.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-lifecycle-q8.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp6-router-reload-q8.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp8-rejection-seeds-q8.json

python3 scripts/mtp-exactness.py \
  scripts/mtp-exactness-manifests/qwen38-mtp8-high-confidence-q8.json

scripts/mtp-nsys-profile.sh gpu-full gpu 0
scripts/mtp-nsys-profile.sh cpu-full cpu 0
scripts/mtp-nsys-profile.sh cpu-planes3 cpu 3
```

All request files are maintained under
`scripts/mtp-exactness-manifests/requests/`. The committed 1K request is
byte-identical to the original temporary input (file SHA-256
`c91ffa3ff2ca9958a7835004e3d5a6ffa7a242a9f74bfce3e7935d52161dce4c`),
so the older artifact remains reproducible without `/tmp`. The 5K manifest
uses a rendered prompt request because the runner deliberately requires a
prompt string or token list. An earlier attempt passed an obsolete
chat-message object and was rejected before generation;
`/tmp/qwen38-mtp6-cross-residency-q8-5k-20260819` is a setup-error artifact,
not a performance result.

## Final result tables

### Target-only 1K

| Case | Prefill | Decode | Peak VRAM | Exact |
|---|---:|---:|---:|---|
| clean Bee GPU Q8 | 1,015.914 t/s | 50.055 t/s | 13,560 MiB | golden |
| candidate GPU Q8 | 977.607 t/s | 50.021 t/s | 13,560 MiB | yes |
| candidate pinned-CPU Q8 | 879.466 t/s | 47.532 t/s | 13,442 MiB | yes |

Common token SHA-256:
`cd8d20d1270ee556a5035994abe08e55fab1a38600a89208becbc9e348e8d283`.

### Exact-tail 128

| Case | Prefill | Decode | Peak VRAM | Exact |
|---|---:|---:|---:|---|
| clean Bee GPU Q8 | 975.201 t/s | 47.773 t/s | 13,584 MiB | golden |
| candidate GPU Q8 | 953.027 t/s | 47.649 t/s | 13,584 MiB | yes |
| candidate pinned-CPU Q8 | 820.516 t/s | 45.632 t/s | 13,464 MiB | yes |

Common token SHA-256:
`31e64405e471ffe000382e76ddc4a6ba75b82ed19f0f52c057b9967764f9d5b0`.

### MTP 1K

| Case | Prefill | Decode | Accepted/generated | Replay cycles/tokens | Peak VRAM | Exact |
|---|---:|---:|---:|---:|---:|---|
| clean GPU MTP-2 | 848.548 t/s | 61.918 t/s | 365/390 | 0/0 | 14,256 MiB | golden |
| candidate GPU MTP-2 | 848.273 t/s | 62.047 t/s | 365/390 | 0/0 | 14,256 MiB | yes |
| candidate CPU MTP-2, phase off | 808.140 t/s | 58.836 t/s | 365/390 | 0/0 | 14,124 MiB | yes |
| candidate CPU MTP-2, phase on | 771.713 t/s | 58.416 t/s | 365/390 | 0/0 | 13,890 MiB | yes |
| clean GPU MTP-6 | 835.586 t/s | 65.321 t/s | 428/470 | 0/0 | 14,854 MiB | golden |
| candidate GPU MTP-6 | 840.549 t/s | 65.335 t/s | 428/470 | 0/0 | 14,854 MiB | yes |
| candidate CPU MTP-6, phase off | 804.809 t/s | 62.091 t/s | 428/470 | 0/0 | 14,722 MiB | yes |
| candidate CPU MTP-6, phase on | 751.091 t/s | 61.812 t/s | 428/470 | 0/0 | 14,488 MiB | yes |
| candidate CPU MTP-6, phase on, 3 planes | 756.602 t/s | 60.790 t/s | 428/470 | 10/56 | 13,890 MiB | yes |

MTP-2 and MTP-6 common token hashes are respectively
`14e47fb5c35897bfe818339bd163ef1be1f630b558055fe0d323cad922c36217`
and
`842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00`.

### Depth, plane-cap, seed, prompt, and acceptance edges

The final candidate binary also matched clean Bee for the complete maintained
GPU full-plane depth sweep:

| MTP depth | Common 1K token SHA-256 |
|---:|---|
| 1 | `f7b4d96cb96b758db87171318f49d3989b18e4ee49a215e116a0224ad708126b` |
| 2 | `14e47fb5c35897bfe818339bd163ef1be1f630b558055fe0d323cad922c36217` |
| 3 | `1ac772fb57519d94113bb9fcb74620955718557aa107e33d0ea1c6c44735cd81` |
| 5 | `0bf4ed79f3b5c2f1748cf7b800f3f0c75b5c89fd864d97b7087990f025cd02ce` |
| 6 | `842b39c1982b2ef8aabf1c70a3f6dc5576ba3f90d80e35704c7c47c499e1de00` |
| 8 | `1cb3a7995019c0d2589fb8fe5c157537937bdfbceb7c4b8ef0b88d448d956320` |

At depths 3, 5, and 8, every 2-, 3-, 4-, and full-plane candidate in the
manifest matched its same-depth clean oracle. The capped rows exercised 8 to
27 replay cycles and 42 to 127 replay-batch tokens, depending on depth and
plane count; full-plane rows used no replay. These artifacts use the same final
candidate server SHA-256 reported above, so they were not rerun after later
harness-only changes.

Two additional clean-process matrices vary inputs independently of that 1K
prompt:

| Prompt / edge | Case | Accepted/generated | Replay cycles/tokens | Decode | Peak VRAM | Exact |
|---|---|---:|---:|---:|---:|---|
| number game, `p_min=0`, seed 7 | clean GPU full | 202/424 | 0/0 | 106.376 t/s | 15,154 MiB | golden |
| same | candidate GPU, 2 planes | 202/424 | 43/387 | 65.117 t/s | 13,860 MiB | yes |
| same | candidate pinned CPU, 2 planes | 202/424 | 43/387 | 64.420 t/s | 13,754 MiB | yes |
| number game, `p_min=0`, seed 42 | clean GPU full | 178/614 | 0/0 | 73.679 t/s | 15,154 MiB | golden |
| same | candidate GPU, 2 planes | 178/614 | 69/621 | 43.273 t/s | 13,860 MiB | yes |
| same | candidate pinned CPU, 2 planes | 178/614 | 69/621 | 42.635 t/s | 13,754 MiB | yes |
| website, `p_min=0.999`, seed 2026 | clean GPU full | 31/31 | 0/0 | 49.185 t/s | 15,154 MiB | golden |
| same | candidate GPU full, phase on | 31/31 | 0/0 | 48.545 t/s | 14,920 MiB | yes |
| same | candidate pinned CPU full, phase on | 31/31 | 0/0 | 46.982 t/s | 14,794 MiB | yes |
| same | candidate pinned CPU, 3 planes | 31/31 | 0/0 | 47.073 t/s | 13,892 MiB | yes |

The seed-7 and seed-42 common hashes are respectively
`b1cdcec6763a8335d171825bc4a33d73feb96956dc4090599ed6cbc28763b028`
and `5a5450968f1862765c77170595dda7632f242fa9747e62f67ea7242c1e6cc4d0`.
The independent high-confidence prompt hash is
`f8e70e80e321d5d31efe9b47dc69a87f7d120886f67b6d3c19b2ed68b4386a13`.
All output IDs and response bytes matched, and every checkpoint counter, time,
and serialized payload remained zero. `p_min=0` is the rejection-heavy case:
it drafts to full depth and the two-plane policy has a zero-token ordinary
rollback horizon, forcing selected GPU replay. At `p_min=0.999`, all actual
drafts were accepted and no replay or checkpoint restore was needed.

### Stochastic MTP-6 5K

| Case | Prefill | Decode | Accepted/generated | Replay cycles/tokens | Peak VRAM | Exact |
|---|---:|---:|---:|---:|---:|---|
| clean GPU, full planes | 847.371 t/s | 63.281 t/s | 2,077/2,435 | 0/0 | 15,006 MiB | golden |
| candidate GPU, full planes | 841.466 t/s | 63.223 t/s | 2,077/2,435 | 0/0 | 15,006 MiB | yes |
| candidate CPU, phase-aware, full planes | 751.182 t/s | 57.200 t/s | 2,077/2,435 | 0/0 | 14,514 MiB | yes |
| candidate CPU, phase-aware, 3 planes | 750.641 t/s | 55.363 t/s | 2,077/2,435 | 90/443 | 13,926 MiB | yes |

All rows contain exactly 5,000 output tokens and identical response bytes.
Acceptance is 2,077/2,435 = 85.298%. All checkpoint capture/restore counts,
times, and payload bytes are zero. The capped row uses 90 selected GPU replay
cycles containing 443 actual replay-batch tokens.

### Same-process prompt-cache and sleep/wake lifecycle

The MTP-6 lifecycle sequence generated 96 tokens from the original 149-token
prompt, observed the server asleep, woke it with a 255-token continuation,
generated 96 tokens, shrank to a 192-token branch for 32 tokens, observed a
second sleep, then woke and regrew a 365-token branch for 64 tokens. Server logs
confirm that each wake recreated the model contexts. All candidates matched
clean Bee token-for-token and response-byte-for-byte at every completion:

| Case | Peak VRAM | Exact four-step sequence |
|---|---:|---|
| clean GPU, phase off | 14,854 MiB | golden |
| candidate GPU, phase off | 14,854 MiB | yes |
| candidate GPU, phase on | 14,614 MiB | yes |
| candidate pinned CPU, phase off | 14,722 MiB | yes |
| candidate pinned CPU, phase on | 14,488 MiB | yes |
| candidate pinned CPU, phase on, 3 planes | 13,896 MiB | yes |

The four common token SHA-256 values, in sequence order, are
`83388a4200b4dc6292a1a88f4f1e5cdcb223d853f6c851ee63342670d5bd9447`,
`0c68b2382550f4e53b662682e5f2900cda42cfad24f4abd8b5717b0354b6d0e9`,
`d5fe0a774d226db31b79a29843a2e3e6639d7a078c72b302edcc49a43360f31e`,
and `b23508c341f7afd50bf72c089c2011de52a45e1e0aaa852e69e8fcb5b8738287`.
Candidate full-plane cases accepted 160/183 draft tokens and used zero
checkpoint or replay work. The three-plane case had the same acceptance and
used five selected-GPU replay cycles containing 30 actual replay-batch tokens;
its checkpoint counts, times, and payload bytes remained zero.

### Explicit router model reload

The router matrix starts each model through a declared INI preset, generates
96 tokens, atomically replaces only the case-owned preset with a second hashed
fixture whose informational tag differs, and calls `POST /models/reload`.
Every case was observed as `loaded` before the call and `unloaded` afterward;
the router log confirms `source updated` as the reason. The next identical
request autoloaded a new child process. Its token IDs, response bytes, request
semantics, and prompt-token IDs matched the pre-reload request within the case,
and every candidate step also matched clean Bee:

| Case | Peak child-process VRAM | Exact before/after and to clean Bee |
|---|---:|---|
| clean GPU, phase off | 14,854 MiB | golden / yes |
| candidate GPU, phase on | 14,614 MiB | yes |
| candidate pinned CPU, phase on | 14,488 MiB | yes |
| candidate pinned CPU, phase on, 3 planes | 13,890 MiB | yes |

Both sides of every reload emitted token SHA-256
`83388a4200b4dc6292a1a88f4f1e5cdcb223d853f6c851ee63342670d5bd9447`.
The runner follows the router's descendant process tree for GPU accounting, so
these peaks are the serving child allocations rather than the zero-GPU router
parent. Full-plane candidates used zero replay/checkpoint work; the capped
candidate repeated 3 replay cycles/20 replay-batch tokens before and after the
reload, with zero checkpoint payload.

### Authoritative CPU/GPU and replay transfer accounting

Nsight Systems 2026.1.3 profiled three clean candidate processes with the same
5K request, phase-aware workspace, MTP-6, Q8_0/Q8_0 target and draft KV,
8,192-token context, 1,024/512/512 batch geometry, seed 1234, and temperature
0.8. Only KV residency changed between the first two rows; only total recurrent
planes changed between the last two. The harness SHA-256 at measurement time
was `3f53b7698e1d2621c679aa9c12738ab6e5bbc02c7c51a4104c5659047ac74df5`,
and every artifact records base commit
`7febdc06a795002bf9e82f4b84026fd3740a3a12` plus dirty fingerprint
`556f683c9faad89c98160e604e2743ae16782cf641d4f94131b055b49e01e426`.

| Measurement | GPU full | Pinned CPU full | Pinned CPU, 3 planes |
|---|---:|---:|---:|
| Prefill | 751.827 t/s | 719.122 t/s | 726.574 t/s |
| Decode under Nsight | 61.001 t/s | 55.622 t/s | 54.065 t/s |
| Request wall | 82.167 s | 90.103 s | 92.688 s |
| Sampled peak VRAM | 14,778 MiB | 14,532 MiB | 13,944 MiB |
| H2D total/count | 17,485.510 MB / 97,840 | 327,709.736 MB / 185,590 | 337,755.028 MB / 189,820 |
| H2D aggregate time | 0.886 s | 6.549 s | 6.710 s |
| D2H total/count | 5,700.920 MB / 21,699 | 5,915.778 MB / 131,647 | 6,381.261 MB / 134,887 |
| D2H aggregate time | 0.156 s | 0.213 s | 0.223 s |
| Replay cycles/tokens | 0/0 | 0/0 | 90/443 |

Pinned CPU full versus GPU full saved 246 MiB of profiled peak VRAM but added
310,224.226 MB H2D and 214.858 MB D2H, and decode was 8.82% slower. This is the
important CPU/GPU distinction: `--no-kv-offload --kv-cpu-pinned` changes
persistent storage residency, while normal operation offload still executes
attention on CUDA and must make host KV available to that accelerator path.
The resulting context-dependent H2D traffic is structural, not an output
correctness failure. `--no-op-offload` selects a different CPU execution route
and is not represented by this serving-performance matrix.

Three planes versus CPU full saved another 588 MiB but added 10,045.292 MB H2D
and 465.483 MB D2H through selected GPU replay; decode was 2.80% slower. All
three profiles emitted 5,000 tokens with canonical token SHA-256
`1a19d5ac5189b1a9d7822833794aaa9e0a4585b4e143f88917dc066ce8924b1c`
and content SHA-256
`afd0208aaaf57cd003c1b0a8d8f29a83c73fa8a8264b547fc6cba0093e1cbe5c`.
Acceptance was 2,077/2,435 in every row; every checkpoint count, time, and
payload remained zero.

Profiler artifacts:

| Case | Directory | Trace SHA-256 | Provenance SHA-256 |
|---|---|---|---|
| GPU full | `/tmp/mtp-exact-nsys-gpu-full-20260819` | `2a89d5977705e5502c8cbb38026ba875f41105a989f0afa8e8fc0ce57fab4ee7` | `ecf675c279b9426c927dc1b296fa6065c5ae3087997e33d2d887cc2091bc065b` |
| pinned CPU full | `/tmp/mtp-exact-nsys-cpu-full-20260819` | `6552e47910792a2c375d8d6f8e7987eb30074b2661793f47ce3c453de624191c` | `98373df9e9ac171015490fa39304909ab37333dc8f921d642858af2883fdd093` |
| pinned CPU, 3 planes | `/tmp/mtp-exact-nsys-cpu-planes3-20260819` | `b8b42f0e05fdc231ebb82bc365546919d40aee130338c243af5cb9179ffa65b0` | `a4152610ed7491ac52d43b206f1abab8ac0cf59a9328ab307859f2a38eb69141` |

## Artifact ledger

| Matrix | Artifact directory | Manifest SHA-256 | Provenance SHA-256 | Comparisons SHA-256 | Summary SHA-256 |
|---|---|---|---|---|---|
| exact tail 128 | `/tmp/qwen38-tail-cross-residency-q8-128-v3-20260819` | `e2f1cbe8524bd60541019b99297b711b10ffd51337e083082425b4a3b25b6dad` | `52c523c90ea5e499c9f9bee1ba76431afcc37e50cb6ab7a94326d560303955ca` | `d43fad0f228c61ce85d7d99d23534209a0db06eb811cd2290ecedf5162d287c3` | `f8d37c1f6137aa28137482113f650907a306abcee924013aaa86b7378b02a7a1` |
| target 1K | `/tmp/qwen38-target-cross-residency-q8-1k-final-v2-20260819` | `72ce3de204c9fee4b13c8c9b28d13413b7bc80fc541de9fcee97c56949ae4735` | `8656d1a1539aafbd920894f55854c30619db641631a6f7d950e746885bc4d7d0` | `e8118f4f690ffa519b5396c57fd5c14842a7067c24524830d21e4bb83eaf4b33` | `4d5ddb2f81861005df31b8a77fe3c867131911481e002a02ec51cd83115d974e` |
| MTP 1K | `/tmp/qwen38-mtp-cross-residency-q8-1k-final-v2-20260819` | `fa056f5fea141e627fb0d8101c1a24b204e4cde2ee0ae3e625d78cb6fdddfe2a` | `92b6ee20ed9832d0a44d2de7c08c4a7fe0276fd152c89f483671658232ee8a10` | `eb6d3f275310a5855f12067e4a15931cae6abb00d923468125f9af4695733cab` | `fe4d12e2691962f82cf886846f821942dca9d375be78043142c1d1ed04aebe5f` |
| full-plane depth sweep | `/tmp/mtp-golden-gpu-q8-full-depths-1k-20260819` | `0cd15951f029869a5497e7ebba169259d262757c73eb140fcc61252e77f77700` | `fda16144ae164c8399036b5361d7251f6d04e3a072101034457540023c2190f8` | `bbc72c9aabb1e9808b03f3032017180b3714e7a0275f324ef32b2dda16c086c6` | `f3d51aae99e3b74aa392ffd71ce95c8835b6f46af4ec46c4f4ec8363dcc02951` |
| recurrent-plane depth sweep | `/tmp/mtp-gpu-q8-plane-cap-depths-1k-20260819` | `df6847f50bdbbe779b2a94f43166de5a2bcc5500b8d3c96107402df704b43aef` | `740cdc2f75408430fc6804930dba5009f65d771d894885273905956f86a3c684` | `7119dbd4283b6013e0c1d3a0e17b6feb9f5215e07ff1b2aac00d318b96ff4965` | `2607c124009105180d6d59c547b6760140c2df1db98806aee01352ed0f9f3514` |
| MTP-8 rejection/seeds | `/tmp/qwen38-mtp8-rejection-seeds-q8-20260819` | `0f605d28c782439bfbe16b42e824b9d6769def5ab5c8ce7383f0ceea3f28b148` | `312613aa2803222bbfcc4591845460e249896386c8d0213daf0e7de68ce644ba` | `e2f177efe29a30bb0ee9ec16e204d0aa527816784ab2031c3f47c824cb713bd9` | `03176b06fdd2816833c93286cef679d294539c357575017ad85e42c52fe209dd` |
| MTP-8 high confidence | `/tmp/qwen38-mtp8-high-confidence-q8-20260819` | `7d542a5a514e6d0db24b08ba30c0dcce0ff6833b3f02888dd4d35932f38dba35` | `8283338a3a734770c1e3a3bddf57fab46aaee0faf3347b3e5e9489c12fc9b398` | `15c28dd9db58bec25f1b79e80846c1153dcdfda0bfcc757e050b680844a1d4fd` | `b1c9bb20bc0cb33476e821fad5c2a9d03ae380ec7aacba8b18067f963997610c` |
| MTP-6 5K | `/tmp/qwen38-mtp6-cross-residency-q8-5k-v2-20260819` | `7bb54c0bbabcf16b44be11ae1016f55880b55a4f47d08f5113b160dd38477105` | `63385922b8fb63be331a9e467350bdaa155803fc1ddac07be00dae1edec7a0de` | `5f38d35db1a63b979c355115978d469f1d0dac0f8928e7ea099c4331073595bd` | `23490d7c5a2b69b79bb1386fead97976069b434e420bb7d349f129ae851b83cb` |
| MTP-6 lifecycle | `/tmp/qwen38-mtp6-lifecycle-q8-20260819` | `bd974faf394447b3d71eacf3ac2a3e4597dc9f9f75377b15e21ffff39387cf69` | `fe3a6781a81daf4d3e1b1a4da58921100c1f3eb74a57c83943f17b17fe18dc2d` | `4acf083d85b1ea64c1504ab2c043a5ec5ae481c0389da9cb6d61e4b195c43556` | `77d84efddc79f702978ecdeb4e2bf660941dc0aa970703a700e6b59e89d71723` |
| MTP-6 router reload | `/tmp/qwen38-mtp6-router-reload-q8-20260819` | `94b47d3eac42696fb288124131703036ffad082d879976188e82e9e7ab683b0f` | `12f4f5744521a753633fedd2417e16903184fdf948b059c02dff8ffd875eeb90` | `5d194ac098b7aed6f63dab80d9832a06b58492771df79f4d9ca4b7ea328d2983` | `f6942b8ec35fac297f881eee50557e3970ce84dc79634505bc8484f4b0c0d40e` |
| clean llama.cpp 1K control | `/tmp/llama-upstream-depth-sweep-1k-20260819` | `e779ac3851e04838c729199d1b96283729134c1bbba52c80ff60b269c63a3c89` | `c627a685b15a1101d6dc4cd8933a433a5c22c9b081b204544029977b4db5fe67` | `298550ab6668918fd5933b0fc0994b588ddb67d3e211d9c215771e876aa35438` | `6bbf48004436aa2a989a3e3fc078b9e40603c270dc405fcf049bcd66befef6b7` |

Important diagnostic artifacts:

- `/tmp/qwen38-target-residency-diagnostic-16-20260819`: original Q8 CPU/GPU
  mismatch and raw state comparison;
- `/tmp/qwen38-target-cross-residency-f16-16-20260819`: exact F16 control;
- `/tmp/qwen38-tail-cross-residency-q8-128-v2-20260819`: body canonicalized
  but tail execution still planned from CPU storage ownership;
- `/tmp/qwen38-tail-cross-residency-q8-128-20260819`: first route-probe context
  was too small for the added probe tensor and aborted before benchmarking;
- `/tmp/qwen38-cpu-mtp2-ubatch-trace-130-20260819`: draft-ubatch trace; and
- `/tmp/mtp-cpu-q8-mtp2-draft-ubatch-1k-20260819`: token-100 mismatch matrix.

## Verification commands

After the final source freeze:

```bash
cd /home/gencoolpc/beellama-mtp-exact

build-mtp-exact/bin/test-arg-parser
build-mtp-exact/bin/test-kv-cache-tail
python3 scripts/test-mtp-exactness.py

build-mtp-exact/bin/test-backend-ops test \
  -o SET_ROWS -b CPU

build-mtp-exact/bin/test-backend-ops test \
  -o SET_ROWS -b CUDA0

git diff --check
```

The same-type/standard `SET_ROWS` matrix passed 721/721 cases on the CPU test
selection and 267/267 on the CUDA selection. The exactness runner passed 35/35.
The parser suite covers default/equal/mismatched MTP ubatch through CLI,
environment, and INI as well as non-MTP use. The broader focused CTest list and
its generated-model dependency passed 13/13 after `test-llama-archs` was
provisioned. That set included standard-tail static validation, loop-guard
checkpoint static validation, speculative-boundary static validation, KVarN
rollback static validation, batch allocation, recurrent-state rollback, parser,
KV-tail policy, loop guard, prompt checkpoint, backend sampler, and allocator
sharing. The first CTest invocation reported recurrent rollback as not run
because its `test-generate-models` dependency had not yet been built; building
`test-llama-archs` and rerunning both dependency and test passed. This was a
missing test artifact, not a code failure.

The final audit reran the focused recurrent/parser/KV-tail/loop-guard/prompt-
checkpoint/sampler set and its generated-model dependency: 11/11 passed. Its
first invocation also included the generic `test-backend-ops` CTest, which was
stopped when it expanded into the complete unrelated operation matrix; the two
targeted `SET_ROWS` commands above then passed. One upstream-merge static guard
initially failed because it counted four copies of the pre-refactor CPU-owner
assignment. The implementation still resolved CPU storage correctly; the
guard was revised to check the retained semantic invariant directly: host
storage resolves to CPU, writes are probed on the storage backend, and
attention is probed on the separately declared execution backend. It passed on
rerun.

## What is and is not generalized

The implementation is structurally general for standard quantized KV:

- it is driven by cache type, buffer residency, model-layer device, and backend
  capability probes;
- it is threaded through ordinary, hybrid, iSWA, and hybrid-iSWA standard cache
  construction;
- same-type CPU `SET_ROWS` copies any supported row type byte-for-byte; and
- exact-tail route planning separates storage and execution for every model.

The empirical claim is deliberately narrower. It currently covers one Qwen3.8
27B model, homogeneous Q8_0/Q8_0, one NVIDIA CUDA device, coupled full CPU
versus full GPU target/draft KV, phase awareness, full-plane MTP depths
1/2/3/5/6/8, recurrent-plane capping at depths 3/5/6/8, three stochastic
seeds, three prompts, and the acceptance/rejection edges above. It does not
yet establish exactness or performance for:

- lower standard cache quants or asymmetric K/V;
- KVarN, whose native structured-cache path is separate;
- partial GPU KV layer mixes;
- multi-GPU/meta devices;
- Vulkan, HIP/ROCm, MUSA, or CPU-only model execution;
- other architectures, SWA layouts, or transposed-V non-FlashAttention paths;

Those require fresh manifests, clean processes, output hashes, resource
measurements, and—where transfer claims are made—Nsight or the corresponding
backend profiler. They must not be inferred from serialized checkpoint bytes
or from `nvidia-smi` process VRAM.

## Disposition and next work

Retain the MTP draft-ubatch gate, canonical accelerator quant-store stage,
byte-preserving host scatter, storage/execution tail split, explicit native-tail
scheduling, and contract-based fused-probe sizing. The final 5K gate removes
CPU/GPU placement as a known source of output divergence for the supported
Q8/CUDA configuration.

The live prompt-cache shrink/regrow, sleep/wake, explicit router-reload, and
Nsight CPU/GPU/replay matrices are complete for the supported Q8/CUDA
configuration. The profiler makes clear that output-exact CPU residency is not
transfer-free: normal operation offload pays growing-cache H2D traffic, and a
capped plane policy adds replay traffic. Broader format, backend, and model
claims must expand one dimension at a time rather than adding model- or
prompt-specific checks.
