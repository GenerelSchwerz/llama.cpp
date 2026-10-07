# MTP input uploads

With `--decode-boundary-overlap`, eligible single-CUDA MTP contexts stage token, embedding and hidden-state inputs in the existing two event-protected host buffers. Uploads use the compute backend stream. The context copies caller inputs into owned storage and waits for the upload event before buffer reuse or growth.

This extends the existing sampled-input staging path. It changes no arithmetic kernel, expert-cache policy, profile format or public API. The option remains opt-in. Device capability discovery is cached per context.

## Eligibility and fallback

The optimized path requires one model device, a CUDA backend with async/event support and a host-buffer provider. Pipeline, evaluation-callback and shared-workspace contexts use ordinary input setters. Unsupported device input layouts also use ordinary setters. CUDA host allocation retains its existing pageable-memory fallback when pinning is unavailable.

## Validation

The existing `test-llama-archs --test-phase-workspace` fixture covers f16/q8_0 KV, changing batch sizes, alternating and simultaneous sequences, immediate caller-input overwrite, logit agreement and exact serialized KV. Async-upload counts verify execution. Pageable host storage and events-disabled fallback pass; the unchanged library fails the expected missing-upload regression assertion.

Focused cache lifetime and pin-fallback checks pass. A staggered two-request MTP serving check completes 128 tokens per request with clean teardown. Native Windows and live multi-GPU are not qualified by these Linux single-GPU checks.

## Bounded serving observation

RTX 5070 Ti, RCO IQ3_XXS target, q2-f16 MTP, 8194 input tokens and 2048 output tokens, MTP depth 3/p-min 0.5, 64 expert-cache slots per layer and 16 CPU participants. Own-control order: unchanged, patched, patched, unchanged. Both shared locks cover startup, generation and teardown.

| Build | First tok/s | Second tok/s | Mean tok/s |
| --- | ---: | ---: | ---: |
| Unchanged | 71.204 | 74.594 | 72.899 |
| Patched | 75.958 | 73.320 | 74.639 |

The observed mean difference is +2.387%. All four emitted token sequences differ, including same-build repeats; MTP acceptance, routes and transfers vary. This is not a demonstrated causal speedup. All four arms complete 2048 outputs with zero grouped errors/fallback/rollback and clear process, port and GPU teardown. This change addresses input uploads; it does not establish removal of the wider execution gap.
