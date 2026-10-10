# Prefill setup and first-token delivery attribution

Evidence root: /home/gencoolpc/moe-cache-tests/results/generic-prefill-gap-20261009. Retained generic control is frozen100. These observations do not close the approximately1700 tok/s end-to-end prefill goal.

## Direct measurements

Temporary RTLD_NEXT probes162/164/166 call the original functions unchanged. Compilation and CPU preload checks pass before model serving. Nested durations include inner calls and must not be summed twice. Diagnostic timings are not performance qualification. Frozen binary hashes, exact requests, source/probe closure, path counters and whole-tree teardown pass for163/165/167.

All requests use2049 saved prompt tokens, cache_prompt=false, cache_n=0, no MTP, ubatch2048, cache64 and the same profile/occurrence adaptation. Both requests in each fresh context repeat the whole prompt and return846,198,7734,264.

| Component | Cold observation | Repeated full prompt | Evidence |
| --- | ---: | ---: | --- |
| Main graph allocation | 3.09ms | 2.76ms | 163 |
| Main context region finalization/preparation, outer | 9.28ms | 7.01ms | 163 |
| CPU input embedding graph | 3.85ms | 2.39ms | 165 |
| Main workspace reservation | First decode contains nested startup reservations | 46.95ms | 167 |
| One-row tail workspace reservation | 32.21ms | 27.88ms | 167 |
| Next queued decode before first output | 99.29ms | 94.00ms | 167 |

Main region finalization is only5.29ms in163; configuration1.23ms and scheduler preparation2.34ms are nested in the9.28ms context total. The remaining full-graph reader scan is not a major measured bottleneck. Keep the checks intact. CPU embedding placement also lacks evidence as the primary remedy.

Cold input setting varies:5.04ms in165 versus198.98ms in167, with repeated4.76ms in167. This does not establish expensive CPU mask generation or a driver defect; input upload can encounter prior asynchronous work. Do not assign the entire outlier to CPU arithmetic.

## First-token delivery defect

server-context.cpp post_decode queues llama_decode_sampled_batch_async before accepting and sending the preceding token. The queue is eligible for SLOT_STATE_DONE_PROMPT. Hybrid source execution can block inside this call. The first token is already ready, but its delivery waits for preparation/execution of the next decode. The first-generation branch then updates prompt_last after that call, so reported prompt time includes this real client latency.

In167's repeated request, reported prompt1454.13ms minus main logical decode1255.90ms and tail102.99ms leaves95.24ms. The next queued decode takes94.00ms. This source/timing agreement isolates the delivery dependency; it does not prove the complete prefill kernel pipeline matches the specialized engine.

Candidate168 adds one eligibility guard using the existing generated-token count: do not queue ahead until at least one generated token has been processed. Subsequent overlap eligibility, backend sampling, MTP exclusion, cancellation and rollback paths are otherwise unchanged. This changes no arithmetic, residency owner or model geometry. Build j18 plus CPU discovery/body checks pass. Frozen169 retains the exact source/binary manifest; copied frozen files' read-only mode initially interrupted freezing, corrected only in the new169 copies before any model launch.

Diagnostic170 passes both full requests with unchanged output IDs and source routing. Repeated first-token latency1.45714 ->1.36229s; reported prefill1409.09 ->1512.54 tok/s. Total request1.50267 ->1.50391s is effectively unchanged in these observations. The short reported decode rate drops because preparation now occurs after the first token, within the decode timing interval. Do not report this as a compute speedup or infer a steady decode regression from four tokens.

## Longer comparison rejected

Unprofiled ABBA171-174 completes two full2049-token requests and128 outputs per process, with successful source routing and whole-tree teardown in all four arms. Sampled peak is12363MiB in every arm. Cold main-prefill routing/paid bytes match exactly; repeated residency differs, including between the two controls. Initial exact-output and repeated-residency analysis assertions fail; these failures are preserved in FIRST-TOKEN-RELEASE-ABBA-REPORT.json rather than weakening the comparison into a qualification.

| Observation mean | Retained100 control | First-token guard169 |
| --- | ---: | ---: |
| Cold reported prefill tok/s | 1145.74 | 1073.03 |
| Cold first token | 1.79556s | 1.91584s |
| Cold full128-token request | 4.14845s | 4.39045s |
| Repeated full prefill tok/s | 1427.47 | 1523.60 |
| Repeated first token | 1.43855s | 1.34800s |
| Repeated full128-token request | 3.67240s | 3.65684s |

Runs171/172/173 match all128 output IDs in both requests. Candidate174 diverges from zero-based token index97 in both requests. This does not by itself establish numerical corruption; sampled-input graph changes, CPU/GPU arithmetic and adaptive state require attribution. Cold main preparation averages295.23ms control versus350.70ms candidate, before the modified first-token branch executes. It explains part of the cold difference but not its cause. Do not attribute the entire difference to the guard, discard slow arms or accept the observed regression.

The guard is rejected and source is restored to retained100, including the exact pre168 server file. Restore175 passed its build/CPU harness but binary verification caught a stale server library: copy2 preserved the old source mtime, so the build reused the experimental object. Forcing the restored source timestamp and rebuilding176 passes j18 and both CPU discovery/body checks. All13 retained source hashes, the exact pre168 server file and all10 mutable binary hashes now match frozen100. No experiment remains in production source. No promotion, commit or push.

## Next investigation

The first-token delivery dependency remains source-backed, but the tested workaround does not qualify. A later change should preserve the same sampled-input graph/execution while publishing the already-ready token before a blocking next submission; acceptance, sampler checkpoints, stop/cancellation and mixed-slot batches need explicit review. This is distinct from making the expert pipeline faster.

Do not optimize the remaining full-graph finalization scan or relocate embeddings as the primary remedy: their measured costs are minor. Use the preserved GPU trace and checked source to separate MMQ execution, compact-chunk preparation, actual bank transfers and copy/compute gaps. Quantify those costs before adding a cross-layer issuer or changing scratch capacity. Cold resource capture, phase reservations and tail/decode preparation remain separate contributors. Keep the original1700 end-to-end objective and live-resource comparison open.

Machine-readable evidence: SETUP-EXECUTION-ATTRIBUTION-REPORT.json;163/SETUP-TIMES.json;165/EXECUTION-TIMES.json;167/PHASE-TIMES.json;170/PHASE-TIMES.json, with their full directory names under the evidence root; FIRST-TOKEN-RELEASE-ABBA-REPORT.json includes failed comparison assertions and per-arm work.
