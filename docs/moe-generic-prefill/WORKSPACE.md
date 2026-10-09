# Generic hybrid feature workspace

| Item | Current value |
| --- | --- |
| Owned branch | design/generic-hybrid-features-20261007 |
| Owned worktree | /home/gencoolpc/llama-moe-generic-hybrid-features-20261007 |
| Owned build | build-prefill-control inside this worktree |
| Generic integration control | 98a3f31943a073a072fa135f3dca7271304ca44a |
| Codex task | Generic hybrid feature parity |
| Shared locks | /tmp/beellama-cuda-build.lock, then /tmp/beellama-single-gpu.lock |

The owner requested renaming the generic implementation workspace on 2026-10-08. The branch name, worktree path, current mission/goal paths and reference-review filename were updated. The earlier generic integration commit and frozen controls are unchanged. The optimization session owns its separate efficiency checkout.

CMake metadata was regenerated for the relocated own build. Its next build must produce current binaries at the new path. Relocation does not qualify new binaries; use a matching source/binary manifest for all tests. Plain 18-job builds retain the ordered locks and finite deadlines. Preserve serving fatal-error whole-tree teardown and the current owner-approved resource policy.

Historical commands, failed trials, frozen builds and manifests retain their original absolute paths as provenance. They are not current launch instructions. Do not blindly relaunch an archived harness with an obsolete source path. Current work must use this worktree and a fresh explicit artifact directory; historical evidence is not renamed or rewritten to simulate new validation.
