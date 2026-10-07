# Optional MTP draft vocabulary

Use `--spec-type draft-mtp --mtp-draft-vocab FILE` to restrict a supported MTP final projection. Default selection is empty and means unrestricted execution. `LLAMA_ARG_MTP_DRAFT_VOCAB` is the environment equivalent. This policy may change drafts, acceptance and output schedules; it is not a proven default speed improvement.

The input is a metadata-only GGUF sidecar, version1, with exactly ten `mtp_draft_vocab.*` keys: `version`, `vocab_type`, `tokenizer_model`, `tokenizer_pre`, `token_bytes`, `token_offsets`, `token_attributes`, `special_tokens`, `eog_tokens` and `selected_tokens`. Binding checks ordered token bytes/attributes and special/EOG identity against the loaded model. Selected IDs must be nonempty, unique, in range and include every loaded EOG token. Types, counts, offsets and file extent are bounded before publication. Invalid or mismatched policies fail configuration.

The staging extension API `llama_write_mtp_draft_vocab` exports a loaded vocabulary and a selected-ID list. The policy is immutable after the first processing call, including after a performance reset. Compact heads retain original quantized rows and projection precision. Unsupported projection/storage/backend capabilities use the full projection with a once-per-context diagnostic. Scaled, biased, adapter, custom or unproved split/repacked outputs retain full projection.

Existing `test-llama-archs --test-mtp-draft-vocab` covers selected logits and masks, graph reuse, malformed/late/repeated configuration and unsupported projection fallback. Native Windows, multi-GPU and parallel-request qualification remains open. Keep this option separate from expert residency, static profiles and graph-capacity reuse.
