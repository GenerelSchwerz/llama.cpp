# MoE cache: configurable partial host pinning

Status: experimental implementation. Windows validation is still required.

## Experimental usage

`--moe-expert-cache-host-pinned-mb N` sets a model-wide MiB cap on CUDA registrations and pinned staging/control allocations managed by the expert cache. It requires a positive `--moe-expert-cache-size`. It does not cap total RAM, other backend pinned buffers, driver-internal allocations, KV memory, or GPU cache storage.

```sh
llama-server -m model.gguf --moe-expert-cache-size 48 --moe-expert-cache-host-pinned-mb 4096 --no-mmap -fit off -lv 4
```

The same option works with `--mmap`. Use `-lv 4` to see allocation diagnostics. Omit the budget to retain the existing allocation and dispatch defaults. Explicit zero is rejected because this implementation needs pinned staging for GPU misses. `GGML_CUDA_NO_PINNED` also rejects this mode. No automatic CPU expert placement or allocation beyond the cap is allowed.

The environment spelling is `LLAMA_ARG_MOE_EXPERT_CACHE_HOST_PINNED_MB`. The old `--moe-expert-cache-l2-pinned-mb` and `LLAMA_ARG_MOE_EXPERT_CACHE_L2_PINNED_MB` spellings are deprecated aliases with the new semantics, including rejection of zero. Supplying both names, including a CLI/environment combination, is an error. A CLI value overrides the environment value for the same spelling.

The loader reserves conservative staging headroom before selecting a fixed prefix of source storage for in-place registration. Its batch size is capped by the model's maximum number of selected experts per layer and by 16, and is reduced further when the budget cannot hold that many copies of the one-miss staging minimum. Larger route sets retain batched handling. Extra staging reduces the registered-source allowance under the same cap; the minimum accepted budget is unchanged. Sources keep their original backing, without a retained duplicate of all weights. A prefix can cover only part of a bank; complete registered experts use their CUDA aliases and other experts use staging. Read-only mmap registration is capability-checked and falls back to staging without changing file protections. Set `GGML_CUDA_MOE_HOST_REGISTER=0` to force staging-only validation.

Grouped decode retains the GPU slot planner. A pinned control copy supplies miss IDs to captured host callbacks; each callback copies a batch of missing experts across all needed banks into stable pinned storage, followed by the fused GPU gather. Larger route sets use multiple batches, including a partial final batch. Callbacks and gathers run in stream order, so staging is not overwritten before use. Fully mapped groups keep the original callback-free gather. Existing prefill and unsupported-routing fallback paths remain in place.

`-DGGML_CUDA_MOE_STREAMING_COPY=ON` enables an optional x86-64 host-copy optimization at build time. It defaults to `OFF`. It uses SSE2 streaming stores for staging copies of at least 256 KiB whose destination is 16-byte aligned and whose size is a multiple of 64 bytes. The copy loop requests source cache lines 1 KiB ahead, within the current expert bank. A store fence completes those writes before callback readiness is published. Other copies and non-x86-64 hosts retain `memcpy`. This option does not add allocations, threads, or CUDA APIs, and does not change the pin cap or source prefix. Performance depends on the CPU, memory system, and copy sizes; enabling it is not a universal speedup.

Allocations and registered ranges are charged in 64 KiB blocks under a shared model budget. Contexts share the cap, but own separate staging resources. Additional contexts or graph resources can fail if the remaining budget is insufficient; they do not silently expand it. The load log reports conservative staging headroom, successful source registration, and cap/source/staging/pending/peak-reserved bytes. These counters describe cache-owned CUDA requests, not an OS measurement of locked physical memory.

`-DGGML_CUDA_MOE_PARALLEL_COPY=ON` enables an experimental CPU staging helper independently of the streaming-store option. It defaults to `OFF` and remains disabled when more than one CUDA device is visible. One helper is created lazily per bounded-pin model owner and sleeps between jobs. A callback can give the helper the second half of its expert range when both halves need at least 512 KiB of actual staging copies. The callback copies the first half, waits for its accepted job, and publishes readiness only if both halves succeeded. If the helper is busy or cannot start, the callback uses serial copying; there is no pending-job queue. Small copies, fully registered sources and empty miss batches retain serial or callback-free handling as applicable.

The helper uses host pointers only and makes no CUDA calls. It does not change the source prefix, expert staging buffers, GPU gather or pin cap, but adds a thread and CPU coordination state. Each submitting callback owns its job until it observes completion, and existing resource completion ordering keeps the backing alive through that callback and its GPU consumer. The model owner joins the helper when its last retained resource is released. No CPU affinity policy is added. Multiple physical GPUs and native Windows CUDA remain unvalidated; this option does not enable unsupported routing or concurrent use of a shared staging buffer.

A too-small budget fails at model loading or grouped-resource creation. Source-registration failure releases its reservation and uses staging. Staging allocation failure releases its reservation and reports an execution error. An invalid captured staging result traps the GPU graph before incomplete slots can be consumed; this is a fatal backend error and requires restarting the process. It is not recoverable request cancellation.

CUDA 12.8 is the current validation target, per the revised implementation request. The new baseline uses CUDA 11-era host callbacks and ordinary kernels, with a CUDA 11.1 guard around read-only registration. The feature translation unit compiled with CUDA 11.0 during development, but this is not full-build or runtime certification. Unrelated stock CUDA 11 compatibility fixes are intentionally excluded.

### Validation and limitations

Windows, multiple physical GPUs, and CUDA 11 runtime behavior are not certified by the Linux single-GPU tests. Each group uses one reusable staging batch and a fixed source prefix, not adaptive hot-expert selection. Host callbacks still execute on cache-hit steps for groups that require staging, so throughput must be measured independently of correctness and capacity. The pin cap is not a total-RAM limit: non-mmap source backing still requires RAM or swap even where it is not registered.

Linux validation used an RTX 5070 Ti, driver 610.57.04, CUDA 12.8, GCC 14.3.1, Release, architecture `120a-real`, and `GGML_CUDA_NCCL=OFF`. The linked backend runtime and cuBLAS were version 12; the system NCCL library was excluded because it would also load CUDA 13.

- Full `test-moe-cache` passed, including the original default-policy tests.
- The bounded-pinning suite also passed with `GGML_CUDA_MOE_FREQUENCY=0` (LRU policy).
- Added tests reuse existing fixtures for Q4_0, Q4_K, Q8_0, mixed quantization, separate/fused groups, exact output and raw F32/F16/BF16/NVFP4 copies, dynamic routes, multi-row replay, prefill transitions, concurrent contexts, owner release, partial source registration, and read-only mmap fallback.
- Boundary tests cover zero, 65535/65536-byte budgets, allocation-failure rollback, overflow rejection, and `GGML_CUDA_NO_PINNED`.
- A Qwen model load with a 1 MiB cap failed explicitly before inference, reporting a 75 MiB staging minimum.
- The isolated `--host-pinning-fault` test captured and replayed a graph, injected a host callback failure, and terminated with the expected CUDA launch error before returning output.
- Compute Sanitizer memcheck on `--host-pinning-only` reported zero errors. The sanitizer executable was from CUDA 13.3; the tested backend still linked CUDA 12.8.
- `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT` passed 2959/2959 cases.
- `test-arg-parser` passed on both CUDA and CPU-only builds, including CLI/environment precedence and deprecated-alias conflicts.
- Small-model and Qwen3.6-35B-A3B-Q4_K_M server tests completed with both mmap and non-mmap loading. The default model-fit dry run was enabled, exercising zero-allocation model ownership. Cache pool sizes still need manual tuning because fit does not account for them.
- A Qwen3.6-35B-A3B-NVFP4-Q8-NVFP4 server comparison completed with full pinning and a 512 MiB cap. All eight 64-token outputs matched exactly; bounded peak reservation was 491.4375 MiB and sampled process GPU memory was 5702-5704 MiB. Warm-request median generation was 91.73 tokens/s with full pinning and 22.18 tokens/s bounded, with a wide 20.71-32.35 bounded range. This used the same server geometry and prompt as the Q4_K_M comparison below, except for `n_predict=64`.

### Batched staging update

The batching change passed the full `test-moe-cache` suite, bounded-pinning LRU tests, Compute Sanitizer memcheck (zero errors), and 880/880 CUDA `MUL_MAT_ID` backend tests. Existing fixtures now check captured callback counts with one-, three-, and sixteen-miss budgets, partial final batches, mixed registered/staged sources, and dense/NVFP4 copies. The injected batched callback failure during graph replay still stopped execution with the expected CUDA error.

An Nsight Systems comparison on the same Linux CUDA 12.8 system used Qwen3.8-Flash-Next-UD-Q3_K_XL, 80 GPU slots, a 28610 MiB host-pin cap, context 12288, batch 4096, ubatch 512, 12 threads, Q8_0 KV, non-mmap loading, lazy PLE, and `LLAMA_ATTN_ROT_DISABLE=1`. Both profiles used the same 158-token chat prompt and produced the same 1024-token output, including identical content and reasoning hashes.

| Measurement | One miss per callback | Batched callbacks |
| --- | ---: | ---: |
| Generation tokens/s | 25.00 | 27.97 |
| Decode callbacks | 245520 | 25575 |
| GPU-idle time in intervals containing callbacks | 15.251 s | 11.783 s |
| Cache-owned peak pin reservation | 28551.3125 MiB | 27711.625 MiB |

This is one before/after profiled pair, not a repeated benchmark or a Windows result. The observed throughput improvement was 11.9%; callbacks fell by 89.6%. The batched run reserved 1748 MiB of conservative staging headroom, registered 26862 MiB of sources, and allocated 849.625 MiB of staging/control. The smaller source prefix caused 25 groups to require staging instead of 24. All 48 groups completed 49104 grouped calls with zero fallback or errors and unchanged transfer counts/bytes.

The run used a hard job memory limit, no job swap, disabled core dumps/backtraces, and an independent 800 MiB system-available-RAM watchdog. Minimum sampled available RAM was 2219 MiB, peak job memory was 58134 MiB, and peak device memory was 14305 MiB. There were no job OOM, memory-limit, or watchdog events. These test safeguards are not implemented by the pin-budget flag.

The Qwen3.6 measurements below are the original one-miss staging results, not measurements of the batching update.

### Optional streaming-copy update

The opt-in streaming-copy build passed the full `test-moe-cache` suite, bounded-pinning LRU tests, Compute Sanitizer memcheck (zero errors), the injected captured-callback failure test, and 880/880 CUDA `MUL_MAT_ID` cases. The disabled build passed the bounded-pinning suite. An existing Q4_K fixture now exercises copies above the 256 KiB threshold. MSVC 14.51 compiled the extracted helper with the option both enabled and disabled using `/W4 /WX`; both executables passed 208 byte-copy and guard-byte cases under Wine. This does not validate a Windows CUDA build or WDDM execution.

A fresh profiled pair used the Flash Next geometry above, identical requests, and verified the loaded CUDA library hashes. The baseline was the saved batched backend without streaming stores. Both runs produced identical content and reasoning, made 25575 decode callbacks, and retained the same 27711.625 MiB cache-owned peak pin reservation.

| Measurement | Batched baseline | Streaming copy enabled |
| --- | ---: | ---: |
| Generation tokens/s | 28.11 | 29.22 |
| Callback execution time | 8.945 s | 6.304 s |
| GPU-idle time in intervals containing callbacks | 11.698 s | 8.503 s |
| Prompt evaluation | 3.336 s | 7.343 s |
| Whole HTTP request | 39.744 s | 42.427 s |

Callback execution fell by 29.5% and observed decode throughput rose by 3.9%, but the whole request was 6.8% slower. No staging callbacks ran during prefill; that window had similar GPU busy time and more sampled file reads in the enabled run. Other decode idle time also increased. This single pair does not isolate the cause of those changes or establish a repeatable end-to-end gain. Both runs had zero job swap, OOM, memory-limit, or watchdog events; minimum sampled available RAM exceeded 2149 MiB and peak sampled GPU memory was 14305 MiB. The build option remains off by default.

### Source read-ahead follow-up

The optional streaming-copy loop now requests source cache lines 1 KiB ahead, without crossing the current copy extent. A captured callback/gather microbenchmark checked exact GPU results for three Flash bank strides, 1/3/10 misses and three repetitions. Median CPU copy bandwidth improved about 5% across the nine geometries with both inherited and P-core-only placement. The final implementation passed the full MoE suite, LRU tests, memcheck, injected-failure test and 880 CUDA `MUL_MAT_ID` cases. MSVC helper checks also passed; Windows CUDA remains unvalidated.

A matched Flash profile with streaming stores enabled in both arms measured callback execution at 6.083 s before and 5.833 s after (4.1% lower). Generation was 32.614 versus 32.726 tokens/s, and whole-request time was 34.137 versus 33.985 s. Other GPU-idle time increased, so the 0.34% throughput difference is not evidence of a repeatable end-to-end gain. Output hashes, 25575 callbacks, transfer counters and the 27711.625 MiB peak pin reservation matched. Both jobs had zero swap, OOM/limit events and watchdog trips. PLE, affinity, threads, buffers and the default-off setting are unchanged.

A separate callback-only P-core placement experiment reduced callback body time by 2.9% but reduced generation throughput by 4.2% in one matched pair. It did not justify adding a production affinity policy.

### Bounded CPU helper follow-up

With streaming stores and source read-ahead enabled in both arms, one matched Flash profile measured 6.292 s of callback execution with the helper off and 5.329 s with it on (15.3% lower). Generation was 28.580 versus 32.635 tokens/s; whole-request time was 40.135 versus 34.112 s. Output hashes, all 25575 callbacks, transfer counters and the 27711.625 MiB peak pin reservation matched. The helper accepted 10239 jobs. Both jobs had zero swap, OOM/limit events and watchdog trips. This is one profiled pair: sampled file reads and other idle time also fell, and prefill improved despite having no staging callbacks. The full 14.2% generation improvement cannot be attributed to parallel copies alone.

A separate Qwen3.6 Q4_K_M test used 48 GPU slots, a 512 MiB pin cap, context 8192 and four rounds of 128 output tokens per request. Excluding the first round, single-request median HTTP throughput was 40.807 versus 42.810 tokens/s (4.9% higher), with matching output hashes. Four simultaneous requests completed in each round without errors or hangs; median aggregate throughput was 56.311 versus 63.491 tokens/s. Outputs differed across arms and across repeats within the control, so the four-request timing is observational, not an equivalent-work speed comparison. Source/staging reservation and sampled GPU memory matched within each pair. The four-request job had zero swap and memory-limit/OOM events. The single-request servers also exited successfully with zero sampled swap, but their original reporting script failed on one shutdown sample without a `VmSwap` field; completed results were retained, not rerun, and cgroup event counters were not saved for that pair.

The enabled build passed the full MoE suite, bounded-pinning LRU tests, CUDA memcheck, the expected fatal injected-failure test and 880 `MUL_MAT_ID` cases. The disabled build passed bounded-pinning tests. An exact extraction of the worker passed ThreadSanitizer and an MSVC/Wine host-only check covering concurrent initialization, busy fallback, completion ownership, repeated jobs and joined teardown. A captured callback/gather microbenchmark checked 72 exact GPU results across three bank strides and four miss counts; some larger-copy geometries regressed, so the helper remains experimental and off by default. Native Windows CUDA and physical multi-GPU operation remain unvalidated.

### Model-sized staging and full-pinning comparison

Model-sized staging uses the model's maximum selected-expert count, capped at 16 and reduced further by the budget. The old reservation API remains a 16-miss compatibility wrapper. In a Flash comparison with 80 GPU slots and a 28610 MiB pin cap, capacity 16 -> 10 increased registered source from 26862 to 27517.5 MiB and reduced actual staging/control from 849.625 to 516.9375 MiB. Diagnostic copy traffic fell from 385.044 to 374.217 GiB across four 1024-token requests and one different 512-token request. Warm median throughput was 33.219 -> 33.388 tokens/s, only 0.5% and below run variation; all outputs matched. The temporary diagnostic instrumentation used for those counters is not part of the PR candidate.

A subsequent same-build comparison used fresh servers in partial/full/full/partial order, with streaming copies and the CPU helper compiled in, profiling off, default source-prefix placement and no overlap pipeline. Every server ran four identical 158-token chat requests with 1024 output tokens, followed by a different 512-token database request. The table excludes each server's first request and the database request. All 20 corresponding output hashes matched. Commands, environment, libraries and memory ceilings matched except for the partial-pinning budget flag.

| Mode | First server warm median tokens/s | Second server warm median tokens/s | Pooled six-request warm median tokens/s |
| --- | ---: | ---: | ---: |
| Partial pinning, 28610 MiB cap | 39.608 | 35.778 | 36.294 |
| Default full pinning | 42.569 | 45.994 | 42.932 |

Full pinning was 18.3% faster by the pooled medians. One full-pinned first request reached 46.500 tokens/s; neither 40 partial nor 47 full was a steady rate across repeats. The partial runs registered 27517.5 MiB of expert source plus 516.9375 MiB staging/control, while full pinning allocated 53237.5 MiB of pinned expert source. Other backend allocations and lazy PLE backing were unchanged. Sampled peak GPU memory was 14293 versus 14343 MiB.

Each job used a 59200 MiB memory ceiling, no permitted job swap, admission requiring 60000 MiB system-available RAM and an 800 MiB available-RAM guard. No server error, job swap, OOM or watchdog stop occurred. The second full-pinning load triggered 1421 memory-limit reclaim events before requests began; its strict post-run check stopped the controller after all requests completed. Only the unperformed final partial leg was then resumed with the same limits. No completed result was overwritten or rerun.

Filesystem caches were not flushed. The faster partial session's last two 1024-token requests had no sampled file reads or major faults; the slower partial session still incurred both. Full-pinning requests also incurred file-backed page activity. These are request-level observations, including prefill, not a complete causal explanation of decode variation. The historical 47.00 tokens/s wiki run used another build/toolkit and is not a matched arm here.

### Excluded experiments

A shifted registration window removed some callbacks but did not establish a throughput gain. A two-chunk CPU/GPU staging pipeline regressed the matched Flash pair from 40.083 to 32.331 tokens/s, adding streams/events and doubling callbacks for predominantly small miss batches. Both prototypes and their diagnostic controls were archived separately and removed from the PR candidate. They are not needed by model-sized staging, streaming stores or the CPU copy helper.

The cleaned candidate passed the full MoE suite with both copy options ON and with both OFF, plus optimized-build LRU, forced-staging, memcheck, injected-failure, argument-parser and 880 CUDA `MUL_MAT_ID` checks. A final Flash server completed four 1024-token requests and the different 512-token request with all five output hashes matching the earlier reference. Warm median generation was 34.970 tokens/s, with zero job swap, memory-limit/OOM events or watchdog stops. This final run checks correctness after cleanup; it is not another controlled performance comparison.

### Qwen3.6 Q4_K_M comparison

Measured on 2026-09-08 with `ggml-org/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf`: 40 layers, 256 experts, 8 experts used, 2048 embedding width. Every run used 48 GPU slots, one server slot, context 2048, batch/ubatch 256, flash attention on, F16 KV, `-ngl 99`, and the default frequency policy. CUDA-visible process memory was 5888-5892 MiB across the configurations. No build ran concurrently with this comparison.

Each configuration ran four identical greedy requests: 170 prompt tokens, 128 generated tokens, `seed=123`, `ignore_eos=true`, and `cache_prompt=false`. The table reports medians of requests 2-4, excluding the first request, and the observed generation range. All 20 generated strings matched exactly across the five configurations. Filesystem pages were not flushed between runs; this is not a cold-storage benchmark.

| Host mode | Source pins MiB | Staging/control MiB | Peak cache reservation MiB | Prompt tokens/s | Generation tokens/s (range) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default full pin, no mmap | 17280 model backing | No new bounded staging | Not tracked by the new ledger | 578.63 | 96.52 (96.33-97.01) |
| 512 MiB cap, fixed source prefix | 437 | 68.875 | 505.875 | 245.82 | 30.64 (29.80-36.38) |
| 512 MiB cap, forced staging only | 0 | 70 | 70 | 239.62 | 31.01 (29.57-33.57) |
| 512 MiB cap, mmap | 0 | 70 | 70 | 239.71 | 27.28 (26.04-35.10) |
| 8192 MiB cap, fixed source prefix | 8117 | 38 | 8155 | 311.94 | 51.02 (50.72-51.52) |

The loader reserved 75 MiB of conservative staging headroom. This GPU did not support read-only source registration, so the mmap case correctly used staging only. The bounded modes reduced cache-owned pins substantially but were slower than full pinning. The data does not establish that partial pinning preserves full-pinning throughput, nor does it compare an unmodified parent binary against this branch's default path. Small differences among the 512 MiB modes are not evidence of a repeatable ranking.

A separate 512 MiB diagnostic run with `--experimental-logs` and 16 generated tokens reported 40 registered/covered groups, 600 ready/completed grouped calls per request, 14 plan reuses, and zero fallback, rollback, prepare errors, or finish errors. Each request reported 6963 transferred banks and 4106944512 gathered bytes. This establishes grouped execution on the real model; the diagnostic run is not part of the timing table. Per-callback CPU copy and synchronization durations were not separately profiled.

Observed process RSS high-water marks were 18369, 18266, 18242, 19968, and 18199 MiB respectively in table order. These `/proc` samples include model/application residency but do not account for all driver or OS memory. `VmLck` was zero even in the full CUDA-pinned case and is not a valid CUDA pin counter here. NVML memory was sampled after inference, not continuously, so the 5888-5892 MiB figures are not guaranteed transient VRAM peaks.

Reproduce the server geometry with:

```sh
llama-server -m Qwen3.6-35B-A3B-Q4_K_M.gguf --host 127.0.0.1 --port 18791 -c 2048 -b 256 -ub 256 -ngl 99 --moe-expert-cache-size 48 --parallel 1 -fa on --no-mmap -lv 4
```

Add the budget flag for bounded runs; replace `--no-mmap` with `--mmap` for mmap, or set `GGML_CUDA_MOE_HOST_REGISTER=0` for forced staging. The original measurements left fit enabled. Use this standard-library Python client four times after `/health` reports ready:

```python
import json
import urllib.request

payload = {
    "prompt": "Explain why the sky is blue and how sunlight interacts with the atmosphere. " * 12,
    "n_predict": 128,
    "temperature": 0,
    "seed": 123,
    "ignore_eos": True,
    "cache_prompt": False,
    "stream": False,
}
request = urllib.request.Request("http://127.0.0.1:18791/completion", json.dumps(payload).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    result = json.load(response)
print(result["timings"])
print(result["content"])
```

### Windows test handoff

Build branch `moe-cache-partial-pinning-experimental` using the [CUDA build instructions](../build.md). CUDA 12.8 is the tested target; select a toolkit and architecture supported by your GPU. Run `test-moe-cache --host-pinning-only`, then compare the same model, slots, prompt, and KV settings with the budget omitted, with an explicit budget, and with mmap. On PowerShell, forced staging is `$env:GGML_CUDA_MOE_HOST_REGISTER = "0"`; remove it with `Remove-Item Env:GGML_CUDA_MOE_HOST_REGISTER` before testing source registration again.

Record the Windows/driver/toolkit versions, exact model and command, allocation diagnostics at `-lv 4`, process RAM, measured GPU memory, prompt/generation timings, and whether repeated output matches the default run. Choose a budget with room for staging and other process allocations; this branch does not infer a universal Windows pinning limit. A failed registration should fall back within the cap. Report a staging error or crash with its complete log; do not treat a successful load alone as an inference pass.

## Original research and approved plan

The following records the original planning state. Its CUDA 11.0 certification target was subsequently revised to CUDA 12.8 validation while keeping the feature CUDA 11-friendly.

Research date: 2026-09-07. Base: `GenerelSchwerz/llama.cpp`, branch `moe-cache`, commit `b46f7f7a436f990932d3da3ec53380e2b9effc89`. Planning branch: `codex/moe-partial-pinning`.

Expanded research: inspected precedent allocator implementations, transfer-pool synchronization, residency tests, loader failure handling, and NVIDIA registration requirements. The resulting changes are a staging-first implementation order, a strict accounting invariant, read-only mmap capability checks, and explicit progress/failure contracts.

## Objective and agreed direction

Run models whose host expert banks exceed the available CUDA pinning capacity, with an explicit user-controlled budget on every supported operating system. Windows motivated the investigation; Windows-only detection or a hardcoded fraction of RAM is not the feature.

- Introduce one host-pinning budget, provisionally `--moe-expert-cache-host-pinned-mb N`, measured in MiB.
- Cover both mmap-backed and non-mmap expert sources.
- Make the baseline compile and operate with CUDA 11.0, not just 11.8. Newer toolkit optimizations are optional and must retain a tested CUDA 11 path. This requirement applies to partial pinning and grouped staging, not just successful model loading.
- Keep the host budget independent of GPU expert-cache slots and CPU/GPU execution placement. Leaving an expert unpinned must not implicitly request CPU execution.
- Target grouped GPU decode with bounded staging for unpinned misses. Preserving correctness, grouped execution, CUDA graph replay, and throughput are separate acceptance criteria.
- Replace the mmap-only `--moe-expert-cache-l2-pinned-mb` interface once the replacement covers its use cases. Use a temporary deprecated alias with a warning and reject conflicting old/new settings.

At planning time the flag did not exist, and defaults, zero semantics, failure policy, and ownership still required the decisions described below. The implementation summary above supersedes those open questions. No speedup or cross-platform validation is implied by the research.

The CUDA minimum is a design requirement, not a completed compatibility certification. GPU architecture, model/quantization support, host compiler, driver, and operating-system support still have to match the chosen toolkit. An old toolkit cannot provide support for hardware introduced after it.

## What the external research established

### Pinning limits: plausible diagnosis, not a universal constant

NVIDIA describes the commonly observed Windows limit near half of physical RAM as Windows-managed, but explicitly says the fraction is not universal. Treat the reported 31.9 GB ceiling on a 64 GB machine as a machine-specific observation, not an API guarantee. Splitting allocations alone does not solve an aggregate limit. See the [NVIDIA discussion](https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235/8).

Linux is not an unlimited `mlock` alternative in this fork: the current CUDA expert allocator calls `cudaMallocHost` there too. OS locking and CUDA registration are distinct capabilities; OS-locked memory does not automatically have a GPU-accessible alias. NVIDIA warns that pinned memory is scarce and excessive use can hurt system performance; WSL also has pinned-memory limitations. See the [CUDA guidance](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-c-best-practices-guide/index.html#pinned-memory) and [WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

The FreeToken discussion contains an important correction: one apparent persistent driver-memory problem was an orphaned Python child retaining host pins and VRAM, not a driver-level zombie context. Its author withdrew the claim that the affected experiment isolated the quota. Check process ownership and achieved allocations before attributing every failure to WDDM. See the [correction](https://github.com/FlashML-org/FreeToken/pull/27#issuecomment-5389679333). This research did not independently reproduce the reported 64 GB system's ceiling.

### FreeToken: follow the closed proposal to its replacement

[FreeToken PR #27](https://github.com/FlashML-org/FreeToken/pull/27) closed without merging. Its proposed `--pin-exempt-layers` skipped pinning selected layers, decoded them on CPU, and used whole-layer transfers for prefill. The [closing comment](https://github.com/FlashML-org/FreeToken/pull/27#issuecomment-5389514466) points to [PR #112](https://github.com/FlashML-org/FreeToken/pull/112), merged as `e0a3bbc04652ec0622d932adcbf2772848e3c2e2`.

PR #112 implements per-layer residency: CUDA-pinned, OS-locked, or pageable. Selected `--moe-cpu-layers` use non-pinned storage. This is a useful residency and loader precedent, but it is not an implementation of arbitrary unpinned experts feeding our grouped GPU decode.

The implementation and subsequent guidance were inspected at FreeToken commit `af71ba43206e124f5ff6419b47ee36c6e9981078`:

| Source | Finding and implication for this fork |
| --- | --- |
| [engine.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/engine/engine.py) | `FREETOKEN_PIN_BUDGET_GB` supplies an explicit budget. Automatic selection estimates bank bytes and chooses CPU layers, with backend-support gates. The inspected version subtracts an existing host-table reservation. Reuse the accounting principle, not automatic CPU placement or an estimate that can still fall back to pinning everything. |
| [host_banks.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/host_banks.py) | Explicit residency, allocate/fill/register ordering, a pinning pipeline, and reporting actual residency are useful. OS-lock failure can downgrade to pageable. Anonymous mmap allocation here is not equivalent to our file-backed GGUF mmap. Process-lifetime buffer registries and ambient residency policy are not suitable ownership patterns for multiple llama models and contexts. |
| [offload_cache.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/offload_cache.py) | Unpinned layers do not expose usable GPU source aliases. Whole-layer materialization requires different indexing from LRU-remapped expert slots, with guards against mixing them. Preserve this distinction across prefill/decode transitions. |
| [pinned_tensor.cpp](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/kernel/csrc/pinned_tensor.cpp) | Explicit mapped registration and device-pointer lookup distinguish actual CUDA aliases from host addresses. Do not assume pointer identity for partially registered ranges or multiple devices. |
| [cpu_executor.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/cpu_executor.py) | Stable pinned I/O buffers and persistent tasks support captured host work. The optimized handshake uses capability-probed stream memory operations, with host callbacks as a fallback. This informs synchronization design, not a ready-made expert staging implementation. |

The inspected automatic budget detection recognizes WSL through its kernel name but misses native Windows; [issue #120](https://github.com/FlashML-org/FreeToken/issues/120) tracks that gap. Our explicit budget must work independently of OS detection. FreeToken's head/tail layer heuristic also needs workload evidence before reuse; it is not a universal expert-hotness rule.

The [FreeToken paper](https://arxiv.org/html/2608.16157v1), especially sections 3.3, 4.1, and 4.2, provides related guidance: keep dynamic decisions in fixed-shape graph data, maintain consistent expert identities across banks, and account for transfer bandwidth. Its performance model cannot be copied unchanged when our design adds pageable-to-pinned CPU copies. Replacing this fork's cache architecture with FreeToken's global cache is outside scope.

### Other precedents

[AnimaLoraStudio PR #511](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/pull/511) reduces PyTorch host allocator rounding waste by packing tensor views into appropriately sized blocks. That can reduce demand, but does not bypass an aggregate pin limit. Our allocator already uses raw `cudaMallocHost`, so the same PyTorch rounding fix is not directly applicable.

[cubie PR #697](https://github.com/cubiepy/cubie/pull/697), inspected at `3bef4d76ac4aae1c8d23fda4b401e9b9c261e64a`, provides a cumulative reserve/release budget and bounded-staging precedent. Retained allocations must remain charged until their actual lifetime ends. Some paths permit forced allocations beyond the budget; do not copy that exception into a strict cap. This is allocator guidance, not a grouped MoE implementation.

### Deeper precedent review and resulting changes

#### cubie: admission, retention, and forward progress are different concerns

The [memory manager implementation](https://github.com/cubiepy/cubie/blob/3bef4d76ac4aae1c8d23fda4b401e9b9c261e64a/src/cubie/memory/mem_manager.py#L1251) reserves under a lock, rolls back failed allocations, and separates live bytes from bytes retained by CuPy's pool. Under pressure it flushes retained blocks before refusing admission. However, its ledger charges `prod(shape) * itemsize`; it does not establish exact underlying allocator granularity. Combine its ownership pattern with explicit backing-byte accounting, rather than assuming its counters measure OS-locked pages exactly.

The [chunk pool](https://github.com/cubiepy/cubie/blob/3bef4d76ac4aae1c8d23fda4b401e9b9c261e64a/src/cubie/memory/chunk_buffer_pool.py#L102) reuses matching shape/dtype buffers and waits when it cannot deepen the pipeline. Its first matching allocation can use `force=True`. Consequently, removing that exception without another progress guarantee would be unsafe: a request might wait when no compatible buffer can ever become available. Our replacement must reserve a minimum usable staging set upfront and distinguish temporary contention from an impossible request.

The [tests](https://github.com/cubiepy/cubie/blob/3bef4d76ac4aae1c8d23fda4b401e9b9c261e64a/tests/memory/test_memmgmt.py#L2237) explicitly cover cumulative admission, a view retaining its parent's reservation, concurrent attempts, pool reuse, and forced overshoot. Adapt the first four properties; invert the overshoot case into rejection for this feature. These are test precedents, not tests run in this checkout.

The [review questioned overlapping configuration layers](https://github.com/cubiepy/cubie/pull/697#discussion_r3700383321); the [final revision removed indirect spill-policy lookup](https://github.com/cubiepy/cubie/pull/697#discussion_r3700540228). Keep one explicit owner and one budget here. Avoid an unrelated generic memory-manager framework or separate per-bank budget knobs.

#### Anima: exact layouts and two-way transfer ordering

At commit `e8e602239e6a2608d4fe55c1d403f3a73fcbd379`, [PinnedPacker](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/e8e602239e6a2608d4fe55c1d403f3a73fcbd379/runtime/training/block_swap.py#L84) rounds the initial plan to 64 MiB, divides it into power-of-two blocks, aligns tensor views to 256 bytes, and can allocate additional blocks during placement. It improves utilization but is not a strict-cap allocator. Its constructor-only failure comment also does not cover the overflow allocation in `_reserve`. Compute and validate the complete layout before publishing graph addresses; never rely on that comment as a no-allocation guarantee.

The same file's [transfer path](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/e8e602239e6a2608d4fe55c1d403f3a73fcbd379/runtime/training/block_swap.py#L337) waits for the prior consumer before overwriting a GPU slot and records readiness before computation reads it. Borrow both dependency directions. Merely waiting for a copy to finish does not protect the next copy from overwriting an active reader.

The [packer tests](https://github.com/WalkingMeatAxolotl/AnimaLoraStudio/blob/e8e602239e6a2608d4fe55c1d403f3a73fcbd379/tests/test_pinned_packer.py) inject an allocator and exercise alignment, dtype conversion, overflow, empty tensors, and failure messages without allocating model-sized buffers. Use this technique within the existing MoE test infrastructure to verify byte arithmetic and forced failures cheaply.

#### FreeToken: validate achieved residency, not just requested policy

The [residency tests](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/tests/moe/test_offload.py#L712) reject unpinned layers without CPU placement, reject their overlapping prefill path, keep their GPU source descriptors zero, and reject LRU-indexed use of a whole-layer copy. They also verify that failed locking reports pageable residency. The transferable lesson is to validate the actual source and addressing contract at every consumer; the CPU-placement restriction itself is not our target design.

The [pinning worker](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/host_banks.py#L284) carries the creator's CUDA device into its thread, drains pending entries after a failure, joins, and propagates the first error. Its queue is unbounded, so it is not itself a bounded-memory staging design. Prefer synchronous initialization first; if registration overlap is later justified, cap queued work and preserve device binding, cleanup, and cancellation.

[expert_banks.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/expert_banks.py#L460) avoids automatic parallel loading when shard-sized temporary buffers do not fit beside final banks. It also warns if a loader ignores the requested residency and pins everything. For our explicit cap, such a bypass must be impossible or fail before allocation, not merely warn after overspending. Measure peak memory during loading as well as during decode.

Two additional reports constrain the design, without establishing performance for this fork:

- [Issue #194](https://github.com/FlashML-org/FreeToken/issues/194) describes mixed-quantization GGUF rows where padding every layer to the largest geometry inflates host banks from about 85 GiB to 212 GiB. It remains a reported proposal, not a verified result here. Size our residency and staging from each actual bank stride and auxiliary layout, not a uniform bytes-per-layer estimate or a padded model-wide expert size.
- [Issue #239](https://github.com/FlashML-org/FreeToken/issues/239) reports a CPU-MoE configuration that fits a 4 GB GPU by reducing a two-layer prefill buffer to one. Its short-context CPU-decode throughput is not evidence for grouped GPU partial pinning. It supports treating pipeline depth as a capacity/performance tradeoff rather than requiring double buffering everywhere.

#### CUDA contracts that change the mmap and graph plan

The fork's [llama-mmap.cpp](../../src/llama-mmap.cpp) uses `PROT_READ` on POSIX and `FILE_MAP_READ` on Windows. NVIDIA requires `cudaHostRegisterReadOnly` for CPU-read-only mappings on devices without host-page-table access, with a separate support attribute. Check registration and read-only capabilities before selecting direct mmap sources. Unregister with the original registration base, not an expert's interior pointer. On systems using host page tables, registration may populate pages without locking them; keep policy-accounted bytes distinct from claims about physically locked bytes. See the [CUDA registration contract](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

An `Async` copy from pageable memory may block and use driver staging. It is therefore not a substitute for a controlled pinned staging path or a guarantee of overlap. See [CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

NVIDIA also warns that stream-memory-operation ordering is invisible to its scheduler unless CUDA-visible dependencies express the ordering too. Capability support alone is insufficient: preserve graph/event dependencies, use valid device aliases for flags, and verify replay. See the [stream memory operations contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html). The CUDA 11.0 host-callback path remains the first target; faster handshakes are a separate optimization, including on compatible CUDA 11 systems.

### FreeToken's speed paths: what to preserve and what not to import

This review follows executable paths at `af71ba43206e124f5ff6419b47ee36c6e9981078`, rather than treating every optimization mentioned in a comment as active everywhere.

1. GPU-side cache admission avoids a routing round trip. [offload_kernels.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/offload_kernels.py#L19) invokes the device slot cache and produces miss indices, destination slots, and a device-side valid count. Keep our existing GPU planner. Only staged misses need host descriptors; do not move LRU decisions or fully mapped routing to the CPU.
2. Fusing bank copies reduces launch overhead. [fast_index_copy.cuh](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/kernel/csrc/jit/fast_index_copy.cuh#L467) copies all banks in one launch with aligned `uint4` accesses and a runtime valid count. This fused path uses ordinary vector loads/stores, not the single-bank path's special PTX cache hints. Our `moe_grouped_gather_decode` already fuses banks; extend it to safe staged/direct sources instead of restoring a launch per bank.
3. Tune enough outstanding requests to fill the link. [fast_index_copy.py](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/kernel/fast_index_copy.py#L153) defaults to 1024 threads and eight blocks per bank, based on its own PCIe measurements. Our gather uses 256 threads with its own block-count policy. Benchmark both total concurrency and empty-launch cost before changing that policy; a wider grid is not automatically faster, and FreeToken's measured saturation point is not a hardware constant.
4. Remove callback overhead only where the handshake is supported. The [CPU executor implementation](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp#L568) resolves `cuStreamWriteValue64`/`cuStreamWaitValue64` dynamically, trying `_v2` and older symbols. It resets completion before publishing readiness, then a persistent CPU coordinator performs work and publishes completion. Its eager probe does not prove capture support; capture errors currently produce a diagnostic suggesting flag-sync opt-out. Our optional path should probe capture/replay as well and choose callbacks automatically before publishing a production graph.
5. Separate decode acceleration from prefill acceleration. FreeToken's [prefill split](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/offload_cache.py#L723) gathers cache hits on GPU and sends coalesced miss runs through a copy stream with release/ready events. Its [batch-copy binding](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/kernel/batch_memcpy.py#L42) requires CUDA 13.0 and probes an actual transfer. That binding is not required for fused decode. It also does not provide arbitrary pageable-expert grouped decode.

The [CPU wrapper](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/moe/cpu_executor.py#L32) reports callback latency as its reason for flag synchronization, while noting that polling consumes a CPU core during traffic and GPU spin kernels hurt power-coupled machines. Treat those timings as source-reported motivation, not our benchmark results. A single bounded CPU-copy callback per tile is a better first baseline here than automatically copying its two-callback submit/sync protocol. Consider a persistent copy worker only when overlap or measured callback cost justifies it.

The modern helper [utils.cuh](https://github.com/FlashML-org/FreeToken/blob/af71ba43206e124f5ff6419b47ee36c6e9981078/python/freetoken/kernel/csrc/include/freetoken/utils.cuh) uses C++20 facilities, `cudaLaunchKernelEx`, and optional programmatic dependent launch. None belongs in the CUDA 11.0 baseline. Reuse this fork's normal kernel launches, language level, and graph compatibility wrappers. The kernel algorithm and its launch wrapper have different compatibility requirements.

## CUDA 11 compatibility contract

The oldest target is CUDA 11.0 headers, compiler, and runtime. Distinguish compile-time API availability, runtime driver support, device capability, and graph-capture support. A runtime `if` cannot hide a declaration or PTX instruction from an older compiler.

| Mechanism | CUDA 11.0 baseline | Optional faster/newer path |
| --- | --- | --- |
| Pinned staging and writable source registration | `cudaHostAlloc`/`cudaMallocHost`, `cudaHostRegister`, and explicit device aliases; allocate before capture. | Retained registered subsets under the same budget, selected by capability and measured benefit. |
| Host work and graph replay | `cudaLaunchHostFunc` or host graph nodes, pinned descriptor copies, explicit stream/graph dependencies, and ordinary kernel launches. | Persistent worker with a functionally tested stream-memory handshake; callbacks remain available. |
| Read-only file registration | Stage from read-only GGUF mappings. Do not require a read-only registration flag absent from 11.0. | CUDA 11.1+ headers expose `cudaHostRegisterReadOnly`; also check runtime device support before using it. |
| Multi-bank gather | Aligned vector copies, device-side miss count, and ordinary kernel parameters. | Compiler/architecture-gated parameter annotations or cache hints, only after an A/B win. |
| Batched host-to-device API | Explicit bounded staging plus ordinary async copies or the fused gather. | `cudaMemcpyBatchAsync` under CUDA 12.8+ guards, with the correct 12.x versus 13.x signature. |
| Graph lifecycle | Existing five-argument `cudaGraphInstantiate` and pre-12 graph-update handling. | Keep newer graph APIs behind their own guards; no conditional graph nodes or graph allocation nodes are needed for correctness. |

The [CUDA 11.0 memory API](https://docs.nvidia.com/cuda/archive/11.0/cuda-runtime-api/group__CUDART__MEMORY.html), [execution API](https://docs.nvidia.com/cuda/archive/11.0/cuda-runtime-api/group__CUDART__EXECUTION.html), and [graph API](https://docs.nvidia.com/cuda/archive/11.0/cuda-runtime-api/group__CUDART__GRAPH.html) document the baseline building blocks. Host callbacks cannot call CUDA, cannot depend on later CUDA work, and may serialize across otherwise independent streams. Fixed graph dependencies remain necessary; host callback availability alone does not prove a deadlock-free design.

Specific compatibility boundaries:

- Read-only registration appears in the [CUDA 11.1 memory API](https://docs.nvidia.com/cuda/archive/11.1.0/cuda-runtime-api/group__CUDART__MEMORY.html), not the 11.0 API. Guard newer flags and attribute identifiers with the appropriate header-version condition; do not invent numeric enum substitutes. CUDA 11.0 mmap support must continue through staging, and can use budgeted retained pinned copies if that policy later proves worthwhile.
- Stream memory operations already exist in [CUDA 11.0](https://docs.nvidia.com/cuda/archive/11.0/cuda-driver-api/group__CUDA__MEMOP.html), but its documentation describes them as disabled by default and enabling them only on Linux. They are a capability-based optimization, not a mandatory CUDA 12/13 feature and not a portable CUDA 11 baseline. No privileged driver reconfiguration may be required for normal operation.
- `__grid_constant__` is a [CUDA 11.7 addition](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-toolkit-release-notes/index.html), with [SM 7.0+ requirements](https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-programming-guide/index.html#grid-constant). FreeToken's fused kernel uses it; use ordinary parameters below those compiler/architecture requirements.
- FreeToken's single-bank `ld.global.L1::no_allocate` needs the eviction-priority qualifiers introduced in [PTX 7.4 / CUDA 11.4](https://docs.nvidia.com/cuda/archive/11.4.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld), with SM 7.0+ requirements. Its fused kernel does not depend on these hints. Keep them out of the baseline rather than raising the minimum for an unmeasured tuning change.
- The [CUDA 12.8 batch-copy API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__MEMORY.html) includes `failIdx`; FreeToken's CUDA 13 binding uses the later signature. Existing fork code already has version guards for this difference. Reuse them where appropriate; do not import the CUDA 13-only wrapper or make this prefill optimization a decode dependency.

The CUDA 11 memory contract also forbids a copy region spanning CUDA-registered and unregistered allocations. Partition DMA requests at residency boundaries or CPU-copy the requested bytes into one fully pinned staging range. This applies to prefill and fallback copies as well as decode. Do not issue one whole-bank DMA after registering only part of that bank.

For optional runtime resolution, use declared driver types and calling conventions, preserve device aliases, and handle missing symbols normally. Test write/wait ordering on scratch storage, then capture, instantiate, and replay a scratch graph before selecting that mechanism for a real graph. Probe allocations count against the budget or must be freed before admitting resident allocations. An invalidated capture must be discarded; do not continue it after a failed optional call. Unsupported optimization means a logged baseline selection, not legacy CPU decode or an unsupported-API crash.

The checked-out fork already guards graph-update signatures in [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu) and newer batch-copy/legacy stream-memory paths in [moe-cache.cu](../../ggml/src/ggml-cuda/moe-cache.cu). Preserve those boundaries. Auditing this feature's APIs is not proof that the entire repository builds on CUDA 11.0; an actual oldest-toolkit build is an acceptance gate.

## Current moe-cache constraints

The links below resolve within the repository. Symbol names and line numbers refer to the base commit, not future revisions.

| Area | Current behavior | Required change |
| --- | --- | --- |
| [moe-cache.cu](../../ggml/src/ggml-cuda/moe-cache.cu), `ggml_cuda_moe_cached_pinned_malloc`, approximately line 10962 | Attempts to pin the entire requested allocation. On failure, the buffer allocator returns an ordinary CPU buffer, losing the CUDA MoE cache buffer marker. | Preserve cache scheduling identity separately from physical residency. A pageable fallback must not silently leave the GPU cache path. |
| [llama-model.cpp](../../src/llama-model.cpp), backend allocation, approximately lines 1700-1790 | Non-mmap tensors can share a large context allocation; mmap uses wrappers around file mappings. | Do not assume one allocation per expert or per layer. Track owned storage, mapping lifetime, and registered subranges explicitly. |
| [common.cpp](../../common/common.cpp), approximately line 1418 | The legacy L2 setting is published after model/context creation. | A source-pinning budget must reach the model loader before its first expert allocation; merely renaming the existing setter is too late. |
| `device_alias` and `group_source_mapped`, approximately lines 4485-4565 | Resolve aliases from the buffer base and require mapped sources for every participating bank and auxiliary tensor. | Introduce range-aware source eligibility and aliases; a partially registered allocation cannot be treated as entirely mapped. |
| `moe_grouped_gather_decode`, approximately line 3184 | The GPU reads `source + expert * expert_stride` directly from a bank's mapped source. | Supply a safe direct or staged source for every miss. Removing the eligibility check would permit invalid GPU reads. |
| Graph-plan certification, approximately lines 7190-7220 | An unsupported materialization group can make the entire decode slice select legacy execution. | Pinning only some layers does not automatically preserve grouped execution for the remaining layers. Integrate staging into the grouped contract. |
| `moe_cache_l2_acquire` and `ggml_cuda_moe_cache_l2_source`, approximately lines 509 and 9508 | Legacy mmap-only L2 uses CPU copies into pinned LRU slots; eviction can synchronize the copy stream. Budget is apportioned across registered mmap banks. | Reuse useful ownership/staging logic, but remove the mmap-only assumption and avoid treating this path as grouped decode. |
| `ggml_cuda_moe_cache_record_expert_access`, approximately line 9549 | Detailed host-side expert access counters require debug instrumentation. | Existing profiling is not yet a production partial-pin placement policy. Measure misses remaining after the GPU cache, not just raw routing frequency. |

Until the new path passes validation, retain the current tuning guidance: prefer load mode `none` for grouped decode; use legacy L2 only when mmap is necessary, with its slower decode path. This plan does not change that recommendation.

## Proposed design

### One bounded host-memory policy

Account for all expert-source CUDA pins and feature-owned pinned staging/control buffers in the same budget. Charge allocation or registration granularity, including alignment and shared pages, rather than only logical tensor bytes. Reserve before pinning, roll back failed reservations, and release only after all in-flight consumers and graph references finish.

For an explicit cap, maintain this invariant throughout loading, capture, execution, and cleanup:

```text
source backing + staging backing + pinned control backing + pending reservations <= configured bytes
```

Count each backing once, including idle reusable blocks; do not add its active views again. A reservation becomes a backing charge on successful allocation rather than being counted twice. Use overflow-checked arithmetic and reject an impossible reservation without entering a wait loop. Driver-internal allocations and other components remain outside this feature ledger and require headroom.

The budget limits this feature's consumption, not the entire machine's CUDA allocations. Other models, processes, backend allocations, and driver requirements still need headroom. The effective available capacity can be below the configured maximum; report requested, reserved, successfully pinned, staging, and peak bytes separately.

Keep residency separate from ownership and execution placement. Represent pageable backing, CUDA-registered ranges and their device aliases, and staging leases explicitly using existing buffer/resource infrastructure. Optional OS locking is not a substitute for registration and should not become a prerequisite for partial pinning.

Reserve enough staging capacity before assigning the remainder to directly mapped sources. Registering selected source ranges in place avoids duplicating those weights. A separate retained pinned copy is another option, but a 52 GiB pageable model plus 30 GiB of copied pinned weights already needs roughly 82 GiB before other allocations. A pin budget alone is not a total-RAM budget.

Begin with fixed placement and stable addresses. Consider adaptive hot-expert placement only after correctness and profiling: optimize residual GPU-cache miss bytes/cost, since the most frequently routed experts may already stay in VRAM. Avoid per-token registration churn.

The first correctness milestone should use pageable source backing plus a fixed pinned staging arena, with no retained subset. This isolates bounded transport from selection policy and works even when direct file registration is unavailable. Add selected in-place registered ranges next, without changing the user-facing budget. Treat copied pinned residency as a measured alternative, not a mandatory second host copy of all retained weights.

Build registration ownership at model load time and share it with contexts. Register only valid mapped ranges after source layout is known. Coordinate the loader's mapping trimming with registration lifetime, and unregister before the backing mapping disappears. A background loader must not register unfinished bytes or publish a usable alias before registration succeeds. Keep source ownership alive through cancellation and graph destruction.

Model loading can precede knowledge of the final context and graph shapes. Establish the budget owner during loading, but defer retained source registration until the required staging reservation is known, or reserve a documented conservative minimum. A later context needing more staging must obtain capacity under the same owner or fail clearly; it cannot silently multiply the cap. Any rebalancing that changes registered addresses requires a safe point and invalidation of affected graphs.

### Preserve grouped decode through explicit staging

For an unpinned miss, the proposed sequence is:

```text
GPU miss plan
  -> bounded descriptors copied to host
  -> CPU copies required pageable expert bytes into reserved pinned staging
  -> GPU gathers staged/direct sources into GPU cache slots
  -> existing grouped expert computation
```

Keep counts and expert IDs in fixed-capacity graph data, with stable buffers and dependencies. All weight and quantization auxiliary banks must refer to the same selected experts and destination slots. A stage buffer cannot be overwritten until its GPU readers finish. Tile requests larger than the available staging capacity or reject an unsupported minimum budget explicitly; never exceed the cap to make the first request succeed.

Make that capacity contract concrete before capture:

- For each group, calculate the bytes needed for one expert across its participating banks and auxiliaries, using actual strides and alignment. The first implementation can require capacity for at least one complete expert group; sub-expert tiling is optional later work.
- Reserve control storage and at least one staging lease before retained source pins. Additional leases are optional and must fit the same cap. Serialize competing users when a single lease is sufficient; reject a requested graph shape if its minimum cannot fit.
- The current grouped planner requires `n_routes <= n_slots`. Preserve that separate GPU-capacity invariant. Host staging does not make an otherwise unsupported routing shape valid.
- Capture a bounded number of tile steps derived from the maximum supported miss count and tile capacity. Replay changes valid counts and descriptor contents, not allocation sizes or graph topology. No tile may reuse its arena until its gather has completed.
- Describe a miss with its expert ID, destination slot, bank geometry, and direct/staged source selection. CPU code checks descriptor bounds before copying. Never form a device address by adding an expert offset to a partly mapped bank base.

Publish failure state as well as success. If host staging fails or is cancelled, prevent downstream consumers from accepting incomplete slots, invalidate affected cache state, and return an error through the owning execution path. Do not signal normal completion just to unblock a wait. Test failure during replay, not only during allocation.

CUDA host callbacks may perform CPU work but must not call CUDA APIs, directly or indirectly. Allocate/register outside callbacks and capture; enqueue transfers as graph/stream operations with explicit ordering. Do not wait for future work on the same blocked stream. See the [CUDA execution API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html).

Start with a portable synchronization path. Stream memory operations can be a later capability-gated optimization, not an OS-wide assumption. Avoid GPU spin kernels as the generic wait mechanism. A captured callback may still cost time on an all-hit step; demonstrate any claimed zero-overhead bypass rather than assuming it.

For CUDA 11.0, start with one CPU-copy callback per tile covering all staged banks, followed by a fused gather into existing GPU slots. Keep descriptors and storage stable across replay. Preserve a callback-free fully mapped path. After correctness, benchmark whether independent direct-source gather can overlap CPU staging of disjoint misses; extra streams and workers are justified only by a measured improvement with valid dependencies.

Preserve the existing fully mapped fast path when no staged source can be selected. For mixed-source graphs, measure the cost of empty tile callbacks separately. A smaller pin budget can increase CPU-copy work, host-memory traffic, and cold mmap page faults even when grouped GEMM remains enabled; grouped execution alone does not imply full-pin speed.

Reuse the existing grouped planner, slot maps, generation checks, resource lifetimes, and graph fingerprints. Handle concurrent requests, shared models, multiple devices, and target/draft contexts without reusing an active staging lease or stale graph pointer. An eager/split-graph prototype can validate copies first, but it does not establish final replay support or grouped performance.

### Retire legacy L2 only after replacement coverage

The final interface should expose the host-pinning budget, not two competing host caches. Retain the old spelling temporarily as a deprecated compatibility alias, including its environment-variable counterpart. Document that the new budget covers source residency and staging, not just mmap L2 slots.

Resolve CLI/environment precedence and detect both names being supplied. Do not silently reinterpret legacy zero/default settings until zero semantics are settled. Remove legacy allocation/dispatch code only after mmap correctness, bounded memory, and grouped execution have passed the same acceptance tests as non-mmap.

## Decisions before implementation

1. Defaults and zero: distinguish an omitted option from an explicit zero. Recommended direction is to preserve existing full-pin defaults initially; define whether zero requests fully unpinned operation with an explicit fallback or rejects GPU staging that cannot fit.
2. Budget scope: choose a model-owned budget shared across its contexts or an explicitly configured wider owner. Prevent accidental multiplication per bank, context, or GPU, and define accounting for shared registered pages and target/draft models.
3. Allocation failure: choose between shrinking the mapped subset while preserving minimum staging, an explicit slower fallback, or a clear error. Never silently exceed the requested budget.
4. `GGML_CUDA_NO_PINNED`: define precedence. Recommended direction is to honor the prohibition and report why a positive pinned budget cannot take effect.
5. Initial placement: use staging-only as the first correctness milestone, then compare fixed in-place registration against copied pinned residency. Keep staging available when read-only file registration is unsupported; do not change file protections just to force it.
6. Backend scope: make the policy OS-independent for the existing CUDA MoE path. This is not a claim that CUDA grouped execution or host registration already works on every backend; HIP/MUSA and other backends require capability checks and separate validation.

CUDA 11.0 baseline support is a requirement, not an open tradeoff. Optional speed features may change transport or synchronization, but not the budget, expert selection, output semantics, or availability of the baseline.

Review these choices with the owner before making the substantial loader/dispatch changes. Do not import unrelated experimental branches as part of this work.

## Implementation sequence and acceptance gates

1. Establish a CUDA 11.0 build baseline, then add explicit budget plumbing before model allocation, choose the owning lifetime, and expose accounting diagnostics. Establish unchanged behavior when the feature is not requested. Review any required model-parameter/API addition before implementation.
2. Separate cache buffer identity from residency. Implement pageable cached backing and a strictly bounded staging/control allocation for both loading modes. Verify minimum-capacity rejection and cleanup with injected failures.
3. Extend grouped source descriptors and validate staged miss copies, expert identity, auxiliary banks, and prefill-to-decode transitions. Initially use one reusable lease and a simple synchronization path.
4. Integrate bounded tiling, CUDA 11.0 capture/replay, cancellation, and lease lifetime handling. Require correct grouped execution with all source banks unregistered before adding source-selection policy. This milestone must not require stream memory operations, batched-copy APIs, or newer graph nodes.
5. Add fixed in-place partial registration with version/capability checks and safe fallback to staging. Benchmark mixed direct/staged misses, optional extra leases, and retained-copy alternatives. Evaluate optional stream-memory synchronization and newer copy APIs independently; keep the baseline force-selectable for validation. Add adaptive residency only if measured gains justify it.
6. Add the compatibility alias, retire the legacy path after coverage, and update flag help plus the wiki tuning page. Do not advertise an unfinished flag as recommended setup.

Extend [test-moe-cache.cpp](../../tests/test-moe-cache.cpp) and relevant cases in [test-backend-ops.cpp](../../tests/test-backend-ops.cpp); do not add a new test subsystem.

- Force small budgets on Linux, native Windows, and WSL so tests do not depend on discovering a real OS ceiling.
- Compile with CUDA 11.0 and 11.8, then representative CUDA 12.8 and 13.x toolkits to cover baseline declarations and the batch-copy signature boundary. Use toolkit-supported host compilers and GPU targets. Runtime testing must use compatible driver/OS/hardware combinations; a CUDA 11 binary running on a new driver alone does not validate old-driver behavior.
- Run the callback/fused-gather baseline on every tested toolkit, not only when optional features happen to be missing. Test missing driver symbols, unsupported read-only registration, eager-success/capture-failure, and scratch-probe cleanup. Verify automatic baseline selection before production capture.
- Exercise zero, below-minimum, exact-boundary, overflow, page alignment, overlapping ranges, allocation/registration failure, and `GGML_CUDA_NO_PINNED`.
- Test a request that cannot fit any lease separately from one waiting for a busy lease. Cover different bank geometries, zero-sized auxiliary ranges, and failure after some allocations have succeeded; prove neither indefinite waits nor budget overshoot occurs.
- Compare outputs for cold misses, all hits, thrashing, multiple routed experts, supported quantization auxiliaries, prefill/decode transitions, batches, and repeated graph replay.
- Exercise read-only mmap with direct registration enabled and forced unavailable, distinct host/device aliases, and truncated final pages without accessing outside the mapping. Verify that model-load cancellation does not leave registered mappings or a live worker behind.
- Exercise multiple contexts/devices, cancellation, teardown, and repeated model loads. Check that reservations and registrations return to baseline only after their last consumer finishes.
- Retain a tensor view or captured graph after its original allocator handle is dropped and verify that the backing remains charged. Race reservation attempts against the same owner, and simulate host-stage failure during replay without accepting partial cache contents.
- Record grouped versus legacy coverage, source/staging pinned bytes, peak resident RAM, GPU-cache miss rates, descriptor transfer time, CPU copy time, synchronization time, GPU gather time, throughput, and latency.
- Benchmark real prompts at both short and populated contexts with identical model, quantization, cache slots, KV settings, hardware, and load mode. Successful startup alone does not establish correct serving or equivalent performance.

Use a controlled comparison matrix: full pinning, staging-only, fixed partial registration plus staging, and legacy mmap L2. Sweep budgets and one versus two leases; distinguish warm resident mmap pages from first-touch/page-fault behavior. Record total load peak as well as steady state. Do not copy any precedent's reported tokens/second into this fork's expected results.

For speed work, compare baseline callbacks against the optional handshake at the same pin budget and routing workload. Measure all-hit overhead, bytes per staged miss, effective PCIe bandwidth, CPU memory-copy bandwidth, gather grid size, worker CPU use, and end-to-end latency. Benchmark fused gather versus independent async copies for useful transfer sizes; isolate prefill batch-copy gains from decode gains. Verify small auxiliary banks do not turn an apparently asynchronous optimization into host-blocking work. No single microbenchmark establishes the fastest end-to-end policy.

At the original planning checkpoint, no runtime code had changed and no inference tests or new measurements had run. The implementation and validation summary above supersedes that checkpoint; the linked research alone is not performance or compatibility evidence.
