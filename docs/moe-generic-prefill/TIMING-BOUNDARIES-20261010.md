# Prefill timing boundaries, 2026-10-10

## Verified source mismatch

The specialized52 frozen manifest records tools/moe-native-runtime/vendor/strata-kernels/src/program/generate.cpp SHA25680b611bd8d6cb19e53f038d2a362e980b95fd1728459ad3b54a149b07e59354a. Current vendor file matches that exact hash. Its serving branch builds cuts ending at n-1 at9873, refills lent slots, records prompt_ms at9935, then processes the final prompt token in the first verify window at9938-9940. DONE reports n prompt tokens and read_n=fresh=n-resume at10923-10924, even though the measured conditioning interval stops before that final token. This is the actual frozen serving path, not the separate non-serving loop near5282.

Generic includes final-token evaluation in prompt completion. server-context.cpp4841-4872 can additionally enqueue a sampled decode for SLOT_STATE_DONE_PROMPT before first-token handling; first-generation prompt_last updates at4972. The existing167 probe measures main1255.90ms, tail102.99ms and following queue94.00ms inside reported1454.13ms. Those are archived retained100 observations, not timings for198. Source-core classifies only sequential MAIN n_rows>1 as prefill at1588-1590; the one-row tail is the existing hybrid decode path and does not imply CPU-off prefill incorrectly governs it.

Therefore specialized1766.53 versus generic approximately1430 headline prompt tok/s compares different intervals. The one-token numerator discrepancy is small; the final-token and queued-decode work is material. Real client TTFT still differs: archived specialized52 is1.20304s, while latest candidate190 repeated first tokens are approximately1.435s. This is real latency to improve, not permission to relabel a metric or declare parity. Different run sets, output length, adaptive residency and live resource budgets remain qualifications.

## Next measurement and implementation boundaries

Frozen198 has only numerical qualification so far.199 queues two full uncached2049/128,noMTP observations.200 is prepared, not launched, for the same two specialized saved-ID requests and128 outputs, explicit CPU-share off, no MTP (--mtp absent) or suffix draft. Run sequential under locks after199 terminal teardown. Preserve actual TTFT, whole-request wall, original timing counters and memory lending. Record boundaries beside rates; do not redefine the1700 end-to-end objective around a narrower compute interval.

The rejected169 guard skipped first queueing and changed the sampled-input execution path; do not blindly restore it. Existing llama_get_sampled_token_ith can retrieve the preceding backend sample, but a later publication-before-submission change must retain the same sampled-input graph, checkpoint/acceptance order, stop/cancellation and mixed-slot semantics. Review existing interfaces first. Down-route reuse, source preparation and transfer overlap are independent compute costs.
