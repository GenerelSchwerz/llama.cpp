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
CUDA_MODULE_LOADING=LAZY \
GGML_MOE_HYBRID=required \
GGML_MOE_HYBRID_EXECUTOR=source \
GGML_MOE_SOURCE_GPU_MISS_FRACTION=0.17 \
GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS=1 \
GGML_MOE_SOURCE_SHARED_OVERLAP=1 \
./build/bin/llama-server -m model.gguf \
  -ngl all -fa on -c 16384 -b 2048 -ub 2048 -np 1 -t 16 \
  --moe-expert-cache-size 64 --moe-expert-cache-host-pinned-mb 0 \
  --load-mode none --lazy-mode on --fit off \
  --decode-overlap --decode-boundary-overlap --ple-prefetch --phase-aware-workspace
```

`GGML_MOE_SOURCE_GPU_MISS_FRACTION` is required in source mode and must be finite in [0,1]. It sets the transfer share of distinct cache misses, not a percentage of all model layers or resident experts. It uses the existing 1/256 split resolution. It is a hardware tuning choice, not a learned expert profile. The explicit runtime-allocation setting permits ordinary prepared CPU operators whose allocation freedom has not been proven; backing and completion checks still apply. Host pinning value0 requests automatic registration with the existing bounded staging fallback; it does not guarantee every source is pinned or fit on every machine.

The old `fidelity` executor plus `reference-conversion` pipeline remains a compatibility activation. The canonical `source` alias selects that same generic provider without requiring research activation names. Do not combine source mode with a different legacy pipeline. Source mode does not change the ordinary executor default.

Look for `provider=source-core`, complete copy/GPU/CPU/publication counts and zero failures in the server log. A flag alone does not prove effective execution.

## MTP and graph reuse

Add a supported MTP draft model with the normal speculative options:

```sh
--spec-type draft-mtp --model-draft mtp.gguf --spec-draft-n-max 3 --spec-draft-p-min 0.5
```

`--backend-sampling` is a separate optional target sampling choice. `--moe-source-graph-capacity` reserves reusable source shapes from actual capacity; it is opt-in, may change outputs and does not apply to normal full-GPU execution. It is not fixed to four shapes. Keep context, microbatch, sampling, head precision, graph-capacity setting and profile/adaptation identical in comparisons. See [optional draft vocabulary policy](moe-draft-vocab.md).

## Profiles and adaptation

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
CUDA_MODULE_LOADING=LAZY \
GGML_MOE_HYBRID=required GGML_MOE_HYBRID_EXECUTOR=source \
GGML_MOE_SOURCE_GPU_MISS_FRACTION=0.17 \
GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS=1 \
./build/bin/test-llama-archs --collect-moe-profile model.gguf \
  --profile-corpus corpus.json --profile-output calibration.gguf \
  --profile-threads 16 --replay-cache-slots 64 --replay-load-mode none
```

Unset `GGML_MOE_EXPERT_PROFILE` for fresh collection. Choose slots, threads and miss fraction for your hardware; they are not properties learned into the profile. Collection loads one model and uses a fresh context for each serial request. Greedy generation stops at EOG or its requested limit. A request that ends before any observed decode row fails explicitly; the collector does not continue past EOG to invent observations.

The score for each original source/expert is:

```text
request rate = measured expert count / actual target decode rows
workload rate = mean(request rates in that workload)
score = sum(normalized workload weight * workload rate)
```

This reproduces the normalized workload-mixture method. It does not give longer requests more weight. Raw counts remain separate from scores. Cache groups combine complete-bank scores using their existing expert payload-byte weights; equal scores prefer the lower expert ID. Profile generation does not encode a fixed cache size, PCIe split or model architecture.

The profile is metadata-only GGUF version2 with optional source-indexed F64 `moe.profile.ranking_scores`. Raw version1 profiles remain readable. The adjacent `calibration.gguf.calibration/` directory records the corpus, per-request raw GGUF counts, routing JSONL, emitted token IDs, actual decode rows, stop reasons, hashes and `REPORT.json`. Prompt file contents are identified by hash; keep the original files to reproduce collection. The final profile appears only after all requests finish and the codec roundtrip succeeds. Use fresh output paths.

Bounds: 128 requests, 32 workload labels, an 8 MiB corpus file, 64 KiB/4096 tokens per prompt, 2..4096 requested output tokens and 64 MiB of workload-rate storage. Routing evidence is capped at 256 MiB per request; exceeding a bound aborts collection without publishing a final profile. Collection covers target decode logical accesses for raw and fused expert bodies, including all CPU/GPU ownership classes. Prefill, draft/MTP accesses and acceptance are excluded. The first token sampled from prefill has no decode row; the denominator counts actual subsequent target decode calls. Different hardware/placement can change greedy continuations. This method alone establishes neither held-out quality nor a speed gain for every model.

The older fixed-row diagnostic remains available through `--test-moe-replay` and `GGML_TEST_MOE_PROFILE_EXPORT`. It can continue past EOG and should not replace natural-EOG corpus calibration.

### Load a profile and enable adaptation

```sh
CUDA_MODULE_LOADING=LAZY \
GGML_MOE_HYBRID=required GGML_MOE_HYBRID_EXECUTOR=source \
GGML_MOE_SOURCE_GPU_MISS_FRACTION=0.17 \
GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS=1 \
GGML_MOE_SOURCE_SHARED_OVERLAP=1 \
./build/bin/llama-server -m model.gguf \
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
