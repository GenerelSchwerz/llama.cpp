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

The bounded diagnostic collector uses logical source accesses for raw and fused expert bodies. It does not need numerical projection outputs or a standalone executor. Example collection:

```sh
GGML_MOE_HYBRID=required \
GGML_MOE_HYBRID_EXECUTOR=source \
GGML_MOE_SOURCE_GPU_MISS_FRACTION=0.17 \
GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS=1 \
GGML_TEST_MOE_PROFILE_EXPORT=calibration.gguf \
./build/bin/test-llama-archs --test-moe-replay model.gguf \
  --replay-reference calibration.bin --replay-prompt-file calibration.txt \
  --replay-rows 256 --replay-split calibration --replay-cache-slots 64 --replay-load-mode none
```

Use new output paths. The collector also writes a routing JSONL sidecar with caller/source identity. Prompt text is bounded to64KiB/4096 tokens and continuation to4096 rows. This diagnostic uses a fixed number of rows, including past EOG; production serving respects EOS normally. Collection covers post-prefill decode source accesses, not prefill, drafts, MTP acceptance or representative held-out quality. Its corpus split label records provenance, not a quality guarantee.

Load the model-bound statistics with `--moe-expert-profile calibration.gguf`. Default adaptation is off; add `--moe-profile-adapt occurrence` for the asynchronous occurrence policy or `occurrence-sync` for its synchronous control. Ranked STRP input remains supported when its geometry matches. Source identity, counts, types and complete bank coverage are checked before use. A separately loaded draft can use `--spec-draft-moe-expert-profile` and `--spec-draft-moe-profile-adapt`; target and draft configurations are independent.

Observers are private test hooks. Without an observer, the runtime skips source-access enumeration. Profile loading uses the canonical residency owner; it does not create a second expert cache.

## Validation and limits

Focused existing fixtures cover routed/native CPU service, original quantization, source ownership, publication, graph/state reuse, cancellation, backing retention, auxiliary contexts, partial-pin transport and logical observation. LFM and Nemotron controls compare matching mixed CPU/GPU arithmetic; a GPU-only reference is not exact hybrid arithmetic.

Current serving qualification is Linux, one RTX5070Ti, Flash with8194 prompt tokens and2048 output tokens, no-MTP and MTP3. Default/off observation preserves exact outputs/work/acceptance in the qualified pairs. Timings are observations, not statistical speed guarantees. Profile/adaptation-enabled no-MTP output is not deterministic on the unchanged control either; exact long-output parity for that cohort is not established.

The same machine also passed Ornith with8194 prompt/2048 output tokens, MTP3 and a Q8_0 head, and Nemotron with8425 prompt/2048 output tokens without MTP. Ownership-matched ordinary CPU/GPU controls cover their expert types and body operations. Nemotron additionally passed64512 pretokenized real-text prompt tokens plus64 output tokens with64K Q8 K/V context. That short capacity check does not establish maximum-context throughput or near-64K Ornith fit.

Native Windows partial pinning, physical multi-GPU generic execution, broader parallel/staggered/cancellation serving, all auxiliary/MTP families, live profile replacement and held-out generated-profile quality remain unqualified. Existing [grouped-cache multi-GPU checks](moe-grouped-multigpu.md) do not establish those generic hybrid gates. This release does not claim every MoE or concurrency configuration is qualified for production.
