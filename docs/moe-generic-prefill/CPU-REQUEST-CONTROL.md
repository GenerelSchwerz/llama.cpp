# Request-scoped CPU control

This is a shared prerequisite for [features 1-4](GOAL-1-4.md), not a completed CPU-assisted prefill implementation.

## Contract and source ownership

The additive `ggml_backend_moe_cpu_controlled_execute_v1` procedure in `ggml-backend-moe.h` carries per-call abort and diagnostic callbacks to the existing synchronous CPU execution service. It supports combined regions and the existing original-route ownership mask. Control fields are validated before admission; callback state is borrowed until return. Worker callbacks must be thread-safe and must not throw or reenter the service.

`ggml-cpu.cpp` reuses the prepared region, admission limit, private output publication, execution mutex and existing worker pool. Scoped calls consult their request's abort callback rather than the legacy through-epoch watermark. Whole-service close still cancels every call. Existing v1 callers retain their watermark and diagnostic behavior. Additional prepared lanes bound admitted work; they do not imply parallel CPU arithmetic.

`moe-source-core.cu` resolves the companion from the same registered CPU provider as the configured service. Its per-program control reads the program cancellation flag and carries its own hook. Stop, hook changes and completed synchronous drain no longer mutate unrelated service-wide controls when the companion is available. The scheduler's existing module retention keeps the provider alive. Older standalone providers retain the serialized legacy path. Resource logs expose `cpu_control=scoped` or `legacy`, and scoped program state reports its own active CPU calls.

## Qualification

`CPU-REQUEST-CONTROL-QUALIFICATION.json` in the established result directory records exact source/binary identities and guard proofs.

- Actual CPU service: seven invalid controls, three paired admission/commit cancellation cases, four routed ownership/cancellation/reuse cases, and mixed scoped/legacy service close pass. Unaffected calls match the ordinary CPU reference; rejected or canceled unpublished routes keep their sentinels. Legacy watermark behavior remains checked.
- Actual source executor: 14 tiny programs use scoped CPU control; 16 canonical owner cold restores and 16 controller continuations pass. Eight subsequent 128-row ordinary-prefill transitions match the CUDA reference exactly.
- Complete CPU targets and CUDA backend/library/test linkage pass. The CUDA candidate predates only later edits to the unlinked CPU test translation unit; production source identities agree.
- Jobs use verified hard memory limits, zero swap, finite timeouts, ordered shared locks and whole-tree teardown. A CPU test attempt expired on the outer lock before launching tests; it is preserved as deferred, not as a source failure. The final test started only after a fresh lock check.

## Remaining boundaries

This does not implement main sequential CPU/GPU prompt partitioning, cross-layer scheduling or cache loans. Independent asynchronous source submissions still need retained reader/completion contracts; the shared execution mutex remains. Full context/file profile reload, pending adaptation, MTP, concurrent serving, physical multi-GPU, native Windows, multi-model performance and held-out quality remain open. Model-server/original Flash replay remains blocked and its root defect remains unproven.
