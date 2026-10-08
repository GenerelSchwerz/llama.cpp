# Experimental generic CPU/GPU MoE execution

The source executor runs resident experts on CUDA and assigns cache misses to CUDA transfers or a persistent CPU pool. It uses the existing grouped cache as the single residency owner. Target, draft and MTP keep separate execution certificates. Model layout, operators and backend capabilities determine eligibility; there is no architecture whitelist. Unsupported capabilities retain the existing pre-effects fallback or required-execution failure contract.

Complete gated SiLU/GELU and ungated squared-ReLU bodies can use the existing ggml CPU arithmetic and owned CUDA matrix/activation operations when the native arithmetic is unavailable. CPU intermediates stay on the CPU until the body result is published. Other body layouts retain checked projection execution or the existing capability fallback. Two cache-enabled source contexts use independent workspaces; uncached heads and ordinary execution retain the existing shared phase-workspace contract. Prepared source variants include output count and use the declared decode capacity, including MTP catch-up batches without outputs.

This is an explicit experimental mode. Normal execution remains the default. Supported prompt turns use the existing cached prefill path; sequential prompt rows are not treated as independent decode rows.

## Build

Follow [CUDA build instructions](build.md#cuda), with tests enabled for profile collection and qualification:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server test-moe-cache test-llama-archs test-arg-parser -j 18
```

Use matching executables and llama/ggml/CUDA libraries from one build. The generic implementation keeps live shared resource/capture and CPU helpers; the standalone specialized reference is not a runtime dependency.

## Run

Example settings from the qualified single-GPU configuration; choose context, slots, threads and miss split for your model and hardware:

```sh
./build/bin/llama-server -m model.gguf --moe-hybrid on \
  -ngl all -fa on -c 16384 -b 2048 -ub 2048 -np 1 -t 16 \
  --moe-expert-cache-size 64 --moe-expert-cache-host-pinned-mb 0 \
  --load-mode none --lazy-mode on --fit off \
  --decode-overlap --decode-boundary-overlap --ple-prefetch --phase-aware-workspace
```

### Activation and tuning

| Control | Default and meaning |
|---|---|
| `--moe-hybrid on` | Opt into the generic source executor with checked failures. Ordinary execution remains the default; `off` overrides inherited hybrid activation. |
| `--moe-gpu-miss-fraction F` | Optional hardware tuning, default 0.17, finite in [0,1]. Fraction of distinct cache misses transferred to GPU, at 1/256 resolution; resident hits already use GPU. |
| Shared GPU overlap | Enabled for source execution. Move independent operations ahead of expert completion only when dependencies permit. |
| CPU runtime allocation permission | Enabled for source execution, including OpenMP runtime behavior. It is not a RAM/VRAM limit; backing, capacity and completion checks still apply. |

No hybrid environment variables are required. Host pinning value 0 requests automatic registration with bounded staging fallback; it does not guarantee every source is pinned or fits every machine. Choose cache capacity and loading mode for the available memory.

Configuration is selected at process startup, before model/context preparation; the split is shared by contexts in that process. In a `models.ini` preset, use `moe-hybrid = on` and optionally `moe-gpu-miss-fraction = 0.17`. Target and auxiliary contexts retain their existing eligibility checks.

Legacy environment activation remains supported. `GGML_MOE_HYBRID=required` defaults to the source executor; explicit older executor selection remains available. CLI activation and miss fraction override their legacy environment settings. The older `fidelity` plus `reference-conversion` activation remains compatible; do not combine source mode with a different legacy pipeline.

Advanced `GGML_MOE_SOURCE_SHARED_OVERLAP=0` disables shared overlap. `GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS=0` requests strict allocation-free CPU admission, which can reject OpenMP/NUMA builds. Both overrides accept only 0 or 1; these are optional diagnostic controls.

Look for `provider=source-core`, complete copy/GPU/CPU/publication counts and zero failures in the server log. A flag alone does not prove effective execution.

[CUDA 12.3+](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html) normally enables CUDA module lazy loading by default. On older compatible installations, `CUDA_MODULE_LOADING=LAZY` can request it explicitly. This is independent of the `--lazy-mode on` model-loading option.

## MTP and graph reuse

Add a supported MTP draft model with the normal speculative options:

```sh
--spec-type draft-mtp --model-draft mtp.gguf --spec-draft-n-max 3 --spec-draft-p-min 0.5
```

`--backend-sampling` is a separate optional target sampling choice. `--moe-source-graph-capacity` reserves reusable source shapes from actual capacity; it is opt-in, may change outputs and does not apply to normal full-GPU execution. It is not fixed to four shapes. Keep context, microbatch, sampling, head precision, graph-capacity setting and profile/adaptation identical in comparisons. See [optional draft vocabulary policy](moe-draft-vocab.md).

## Profiles and adaptation

### Profile-aware allocation

The default `--moe-cache-allocation auto` keeps the existing allocation without a profile. With a profile and generic source execution, it distributes fixed capacities across routed groups using actual storage costs and profile priorities, within the existing per-device cache budget. Each group retains a positive execution floor. Statistics profiles rank by expected benefit per storage byte; STRP files retain their global pair order.

Use `--moe-cache-allocation uniform` to keep the existing common capacity even with a profile. Draft and MTP contexts have the separate `--spec-draft-moe-cache-allocation auto|uniform` option.

```sh
--moe-hybrid on --moe-expert-cache-size 64 --moe-expert-profile calibration.gguf
```

The slot option establishes the original storage budget; profile allocation can give individual groups more or fewer than 64 slots. Fixed and shared allocation costs remain reserved. Capacities are selected during context construction and stay fixed during generation; online adaptation changes cached identities within them. This does not resize caches during requests or change the compute kernels. Look for `moe-cache-capacity` and effective grouped payload counters in the logs.

Without generic source execution, allocation stays uniform. A backend that cannot publish group capacities rejects profile allocation; select `uniform` to retain its previous behavior.

Keep three decisions separate:

| Decision | Input/control |
|---|---|
| Offline expert statistics | Model-bound source occurrence statistics in GGUF |
| Hardware budget and miss split | Cache budget/slots, CPU threads and GPU miss fraction |
| Online residency adaptation | `--moe-profile-adapt off`, `occurrence` or `occurrence-sync` |

### Generate a weighted corpus profile

Use the in-tree collector; no Python package or separate repository is required. Build `test-llama-archs` with the matching engine libraries. This example gives general and coding requests equal workload weight, independently of their output lengths:

```json
{
  "version": 1,
  "weights": {"general": 0.5, "code": 0.5},
  "requests": [
    {"family": "general", "prompt_file": "general.txt", "max_tokens": 256},
    {"family": "code", "prompt_file": "code.txt", "max_tokens": 256}
  ]
}
```

Save this as `corpus.json`. Prompt file paths are relative to that file. Each request accepts either `prompt_file` or inline `prompt`. Use raw model input, including its chat template if required. Use representative calibration prompts and separate held-out prompts for evaluation.

```sh
./build/bin/test-llama-archs --collect-moe-profile model.gguf \
  --profile-corpus corpus.json --profile-output calibration.gguf \
  --profile-threads 16 --replay-cache-slots 64 --replay-load-mode none
```

The collector automatically enables generic hybrid execution; no activation environment or `--moe-hybrid` flag is needed. It accepts the same optional `--moe-gpu-miss-fraction F` tuning flag. Unset `GGML_MOE_EXPERT_PROFILE` for fresh collection. Choose slots, threads and miss fraction for your hardware; they are not properties learned into the profile. Collection loads one model and uses a fresh context for each serial request. Greedy generation stops at EOG or its requested limit. A request that ends before any observed decode row fails explicitly; the collector does not continue past EOG to invent observations.

The score for each original source/expert is:

```text
request rate = measured expert count / actual target decode rows
workload rate = mean(request rates in that workload)
score = sum(normalized workload weight * workload rate)
```

This reproduces the normalized workload-mixture method. It does not give longer requests more weight. Raw counts remain separate from scores. Cache groups combine complete-bank scores using their existing expert payload-byte weights; equal scores prefer the lower expert ID. Profile generation does not encode a fixed cache size, PCIe split or model architecture.

The profile is metadata-only GGUF version 2 with optional source-indexed F64 `moe.profile.ranking_scores`. Raw version 1 profiles remain readable. The adjacent `calibration.gguf.calibration/` directory records the corpus, per-request raw GGUF counts, routing JSONL, emitted token IDs, actual decode rows, stop reasons, hashes and `REPORT.json`. Prompt file contents are identified by hash; keep the original files to reproduce collection. The final profile appears only after all requests finish and the codec roundtrip succeeds. Use fresh output paths.

Bounds: 128 requests, 32 workload labels, an 8 MiB corpus file, 64 KiB/4096 tokens per prompt, 2..4096 requested output tokens and 64 MiB of workload-rate storage. Routing evidence is capped at 256 MiB per request; exceeding a bound aborts collection without publishing a final profile. Collection covers target decode logical accesses for raw and fused expert bodies, including all CPU/GPU ownership classes. Prefill, draft/MTP accesses and acceptance are excluded. The first token sampled from prefill has no decode row; the denominator counts actual subsequent target decode calls. Different hardware/placement can change greedy continuations. This method alone establishes neither held-out quality nor a speed gain for every model.

The older fixed-row diagnostic remains available through `--test-moe-replay` and `GGML_TEST_MOE_PROFILE_EXPORT`. It can continue past EOG and should not replace natural-EOG corpus calibration.

### Load a profile and enable adaptation

```sh
./build/bin/llama-server -m model.gguf --moe-hybrid on \
  -ngl all -fa on -c 16384 -b 2048 -ub 2048 -np 1 -t 16 \
  --moe-expert-cache-size 64 --moe-expert-cache-host-pinned-mb 0 \
  --moe-expert-profile calibration.gguf --moe-profile-adapt occurrence \
  --load-mode none --lazy-mode on --fit off \
  --decode-overlap --decode-boundary-overlap --ple-prefetch --phase-aware-workspace
```

Load the model-bound statistics with `--moe-expert-profile calibration.gguf`. Default adaptation is off; add `--moe-profile-adapt occurrence` for the asynchronous occurrence policy or `occurrence-sync` for its synchronous control. Ranked STRP input remains supported when its geometry matches. Source names/domains, counts, types, shapes and complete bank coverage are checked before use. Binding does not fingerprint weight contents; regenerate or reevaluate profiles when model weights or quantization change. A separately loaded draft can use `--spec-draft-moe-expert-profile` and `--spec-draft-moe-profile-adapt`; target and draft configurations are independent.

Observers are private test hooks. Without an observer, the runtime skips source-access enumeration. Profile loading uses the canonical residency owner; it does not create a second expert cache.

## Validation and limits

Focused existing fixtures cover routed/native CPU service, original quantization, source ownership, publication, graph/state reuse, cancellation, backing retention, auxiliary contexts, partial-pin transport and logical observation. LFM and Nemotron controls compare matching mixed CPU/GPU arithmetic; a GPU-only reference is not exact hybrid arithmetic.

Current serving qualification is Linux, one RTX5070Ti, Flash with8194 prompt tokens and2048 output tokens, no-MTP and MTP3. Default/off observation preserves exact outputs/work/acceptance in the qualified pairs. Timings are observations, not statistical speed guarantees. Profile/adaptation-enabled no-MTP output is not deterministic on the unchanged control either; exact long-output parity for that cohort is not established.

The same machine also passed Ornith with8194 prompt/2048 output tokens, MTP3 and a Q8_0 head, and Nemotron with8425 prompt/2048 output tokens without MTP. Ownership-matched ordinary CPU/GPU controls cover their expert types and body operations. Nemotron additionally passed64512 pretokenized real-text prompt tokens plus64 output tokens with64K Q8 K/V context. That short capacity check does not establish maximum-context throughput or near-64K Ornith fit.

Native Windows partial pinning, physical multi-GPU generic execution, broader parallel/staggered/cancellation serving, all auxiliary/MTP families, live profile replacement and held-out generated-profile quality remain unqualified. Existing [grouped-cache multi-GPU checks](moe-grouped-multigpu.md) do not establish those generic hybrid gates. This release does not claim every MoE or concurrency configuration is qualified for production.
