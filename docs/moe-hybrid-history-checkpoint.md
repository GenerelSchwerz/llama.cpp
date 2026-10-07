# Generic MTP history checkpoint - 2026-10-07

## Activation

History drafting remains opt-in:

```sh
llama-server [...] --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5 --spec-lookup-chain 2
```

Use the model's existing embedded or separate MTP setup. `--spec-lookup-chain-min` defaults to 3. See [speculative decoding](speculative.md#history-extension-after-mtp) and [generic hybrid execution](moe-hybrid.md).

## Changes

- Use probabilities already provided by the MTP sampler to evaluate the current proposal, without another device read or synchronization.
- Learn history outcomes separately for replacement and extension, grouped by logarithmic match length. Compare expected committed tokens with observed draft-through-target-acceptance costs.
- Replace an MTP proposal with committed-history tokens when its first token agrees with MTP, or append a continuation. The initialized MTP limit, configured extra capacity and request limits bound every window.
- Keep full-row MTP catch-up, accepted target hidden carry, target sampling, rollback, overlap and expert residency unchanged. Original MTP acceptance counts the matching prefix; history replacement counts all selected history tokens. These two statistics are not additive.

The cost timer includes existing full-row catch-up but excludes deferred finish-accept, server delivery and next-round preparation. Candidate probabilities retain the existing sampler normalization; they are not independently calibrated acceptance probabilities. Unknown-cost probes are limited to strong matches. Selection remains a heuristic.

## Validation

Linux, one RTX 5070 Ti, MTP depth 3 and p-min 0.5. These are observational serving results:

| Case | Previous history selector | New history selector | History accepted/offered |
|---|---:|---:|---|
| Flash real 8,194-token dossier, 2,048 output, two runs each |77.68 tok/s mean|87.40 tok/s mean|3/32 and 3/31 -> 22/50 both runs|
| Ornith embedded MTP,8,194/2,048, calibrated profile/adaptation,64K q8 configured context |104.01 tok/s|104.34 tok/s|107/186 -> 296/402|
| Gemma shared-KV MTP,8,440/2,048 repeated text |124.27 tok/s|125.10 tok/s|658/694 -> 1,673/1,702|
| Flash repeated text,8,655/2,048 |86.60 tok/s|87.52 tok/s|6/8 -> 3/5|

Enabled outputs/work differ; no statistical or universal speed gain is claimed. Default-off Flash matches exact output, MTP counts and logical source work at 83.91 versus 84.73 tok/s. Fifteen serving runs complete with no source/CUDA/grouped fallback/rollback/preparation/finish errors and clean process/port/GPU teardown. Short simultaneous and staggered Flash requests verify independent histories, not parallel throughput gains. PP/TTFT/VRAM and exact hashes are retained in the private qualification artifacts; disk/page warmth prevents a prefill speed claim.

Existing argument, speculative-limit and source-overlap fixtures pass. MTP fixture coverage includes first-token mismatch rejection, separate sequences, confidence selection, match-conditioned learning and rejected/partially accepted/fully accepted replacement hidden-state equality. No new test file or kernel was added.

## Remaining work

Accepted-prefix-only MTP catch-up is not implemented. Native Windows, physical multi-GPU, successful capped recurrent snapshots, new live multi-head coverage and universal model qualification are not established by these tests. Configured 64K Ornith context was tested with 8,194 prompt tokens, not at its maximum. Existing ordinary execution and history-off defaults remain unchanged.

Future prefill work must preserve the canonical expert residency owner, verified capacity/materialization/graph rules and these MTP hidden-carry tests.
