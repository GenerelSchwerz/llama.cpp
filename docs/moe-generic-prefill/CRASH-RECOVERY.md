# Crash recovery and containment

## Current bounded component qualification

The containment prerequisites below now pass, including an actual cgroup OOM with group teardown. All model-server/original Flash attempts remain blocked by GPU-RUNS-BLOCKED.json; the original CUDA/root allocation chain is unproven. Scoped tiny corrected component runs are allowed only with verified3072MiB hard RAM, zero swap,120s runtime, both ordered locks, fatal-log watcher and exact process-tree teardown.

The separately frozen candidate-paired-owner-certified passes32 paired arithmetic plus12 actual canonical owner source/read/retirement cases. Outputs match independent ordinary CUDA exactly. Peak326.36MiB, elapsed8.16s, no memory-limit/OOM events and no interval kernel Xid/OOM; the whole tree is gone. The held-copy case establishes a pending stream and retained owner at submission entry, not every queued reader's timing. Serving selection and speed remain unqualified. CANONICAL-PAIRED-PREFILL-QUALIFICATION.json records command/source/manifest lineage; the failed uncertified-warmup attempt remains separately preserved with source diagnosis and clean teardown. No frozen control was replayed or edited.

2026-10-07. The owner requires CPU-only source diagnosis of the identified failing workload. No GPU reproduction, model-server run or memory-heavy parallel build is authorized during this investigation. Original incident and run files are preserved. Do not kill unrelated processes or change global settings.

## Confirmed evidence

- /tmp/codex-recovery-oom-20261007-h72ya3k7/incident.json identifies this session's run_models.py PID2753308 and frozen control llama-server PID2753309.
- Kernel Xid43 at17:01:38 EDT names PID2753309. At17:10:46 the global OOM task table reports10509872 resident pages for this server, about40.09GiB. Codex sessions and system/network services were killed. These facts identify the failed workload; they do not prove an allocation leak, bad CUDA source pointer or driver root cause.
- Frozen control's loaded-maps.txt identifies control/bin/libggml-cuda.so.0.25.3. Its library SHA256 is4a1684fa7cb394951fd1c1be218792b3e2861719573208e3c54f75500b2da6e2. The executable hash alone cannot distinguish the control and candidate shared-library paths.
- The frozen occurrence-adaptation control completed its first2049-input/16-output request, then failed at ffn_moe_gate-0 on the second prefill. The first response's1095.94 prompt tok/s and41.79 decode tok/s are partial failed-run observations, not a valid paired result. The candidate fails at the same request transition. MTP is not necessary for this failure.
- Logs show asynchronous adaptation completion for window16 before the second-prefill failure. This weakens a simple claim that an unfinished copy alone explains the failure; it does not certify every alias, clock, graph or source lifetime.
- These RUN.json files never recorded terminal teardown. RECOVERY.json sidecars mark them interrupted/failed without rewriting the originals. They are excluded from successful serving comparisons.

## Harness defects and repairs

The former runner had only sampled MemAvailable, with a2048MiB threshold. Its sample thread invoked nvidia-smi before evaluating the threshold and did not contain subprocess failures. A blocked/erroring driver query could delay or terminate protection. There was no kernel-enforced memory/swap limit, no service-wide runtime limit and no immediate fatal-compute log guard. Process-group signals alone do not cover descendants that create another session. This is a source-backed harness weakness, not proof of exactly which path failed during the incident.

The isolated evidence runner now requires a dedicated transient user service. Before executing the workload, guarded_job.py reads its actual cgroup and service properties and rejects absent/mismatched controls: MemoryMax, MemorySwapMax=0, memory.oom.group=1, OOMPolicy=kill, finite RuntimeMaxSec, TimeoutStopSec=3 and KillMode=control-group. Future model jobs are capped at32768MiB and600s, including harness/helpers, unless a smaller budget is supplied. These are containment limits, not evidence that Flash or DeepSeek can fit. An8192MiB host reserve is checked without any driver query. Actual CUDA/error log markers trigger immediate kill of this exact owned unit. GGML_NO_BACKTRACE=1 suppresses GGML automatic debugger forks; LimitCORE=0 prevents core generation on this verified host. The kernel can still invoke its configured system crash-reporting helper outside the workload cgroup; that limitation is recorded below. Workload helpers and detached descendants remain in the same cgroup.

The systemd manager enforces time and group teardown even if the external launcher is killed. Final records require an empty/removed workload cgroup; uncertain teardown blocks further jobs. GPU-RUNS-BLOCKED.json remains present. Containment qualification does not automatically clear the incident block or authorize a GPU replay.

## CPU-only validation

Artifacts: /home/gencoolpc/moe-cache-tests/results/generic-strata-prefill-20261007/CONTAINMENT-VALIDATION.json. No CUDA/model was used.

| Case | Effective limit and result | Teardown |
| --- | --- | --- |
| Simulated fatal CUDA log |128MiB, zero swap; fatal watcher kills group in0.64s |Escaped-session child, which ignored SIGTERM, is gone |
| Finite wall time |2s runtime plus3s stop grace; timeout result |Entire tree gone in5.41s |
| Allocation pressure |memory.max reached exactly128MiB, memory.events max incremented, swap.current remained0 |Bounded allocation stalls; manager timeout kills tree in6.35s; this did not exercise OOM kill |
| Actual cgroup OOM |128MiB, zero swap,20s runtime; dirty-page allocation causes CONSTRAINT_MEMCG and result oom-kill |Kernel kills allocating parent and detached child; whole tree gone in0.23s |
| External launcher SIGKILL |Runtime/group limits remain active in the user manager |Escaped-session child gone in5.33s after launcher death |

All recorded workload cgroups are empty/removed and all recorded leaf PIDs are gone. The first setup rejected an unsupported MemoryOOMGroup property; the corrected service uses OOMPolicy=kill and verifies memory.oom.group directly. Preserve both artifacts. Cgroup limits bound accounted workload memory; they are not proof that every GPU-driver allocation is charged or that a driver fault cannot leave an uninterruptible task. Keep the GPU block while diagnosing.

containment-memory-oom-02 qualifies actual group OOM teardown. The CPU fixture verifies the limits and waits for its detached, SIGTERM-ignoring child, then allocates192MiB and writes each page. The preserved kernel window names the exact owned cgroup with constraint=CONSTRAINT_MEMCG, usage/limit131072kB, memory.oom.group and both killed PIDs. The manager records oom-kill/SIGKILL and128MiB peak; elapsed launcher-to-verified-teardown time is0.229s. Both PIDs and the cgroup are gone. The0.1s sampler missed the brief peak, so peak evidence comes from the kernel and manager, not from that sampler. The earlier containment-memory-oom-01 zero-filled allocation did not establish dirty residency and ended by an explicit assertion plus child cleanup; it is preserved separately and is not OOM proof. This test strengthens containment only. The original CUDA defect remains unproven and GPU-RUNS-BLOCKED.json remains present.

## Source diagnosis and next steps

1. Review the source-core -> ordinary-prefill transition: source program destruction, shared resource serial/generation, completed adaptation publication, completion stream/event, source lease and pinned alias validity. prepare_prefill_group calls prepare_residency then reads selected slots back; current logs identify only the first failed operator, not the exact failing CUDA launch.
2. Audit prepare_residency's planner and materialization bounds when decode-created resources are reused for a64-route prefill selection. prefill_ids and plan are allocated for snapshot.n_slots, so a simple top-k-sized prefill_ids overflow is not supported by source. Materialization callback geometry and source descriptors still need examination.
3. release_source_transport frees transport events and fallback staging. Device resources retain their source owner; do not claim model weights are freed merely because a source program is destroyed. Audit the actual retained owner and registration aliases.
4. Keep numerical, performance, profile-quality and physical-platform gates open. The current generic implementation also lacks Strata's complete-expert pipeline, cross-layer stream-ahead and native fused arithmetic. Containing a crash does not close that performance gap.

An initial stale-ready-plan hypothesis is weakened by prepare_source_group: after its completion dependency, a changed residency token causes moe_grouped_reconcile_admission before host-map readback. That kernel commits the ready plan and clears status/counts to BUILDING. begin_decode also assigns a new residency token for an ordinary mutable transaction, invalidating the prior cached-map token. Do not claim profile replacement normally replays a stale plan without finding a path that bypasses these boundaries. The general planner can explicitly __trap on an invariant failure, so an unspecified launch failure/Xid is not proof of an out-of-bounds pointer. No preserved kernel status/stack identifies which invariant or launch failed.

The materialization constructor allocates copies for full slot capacity and copy spans for its actual tile capacity; callbacks clamp count to that tile capacity. Thus merely reusing a resource created with a smaller top-k does not establish callback overflow. Keep the source/bank span and transition audit open rather than importing a top-k-sized-buffer fix.

No speculative source-lifetime/top-k buffer fix or new model admission rule was applied. The subsequent source-proven ready-age correction is documented below. No shared checkout edit, commit, push or PR was made during recovery.

## Kernel clone evidence and failure amplification

The preserved kernel.log lines3383-3430 and3500-3544 identify PID2753309 in copy_page_range -> dup_mmap -> copy_process -> kernel_clone -> __x64_sys_clone during the OOM. The recorded clone flags0x1200011 do not include CLONE_VM0x100. This is evidence of address-space duplication, not an ordinary shared-address-space worker thread. It does not identify the userspace caller or prove which pinned pages were copied.

The exact frozen base081cf1d792596a2cac5aa06090751d0850aadaab has ggml_print_backtrace at ggml/src/ggml.c:180. Unless GGML_NO_BACKTRACE is set, it creates a pipe, calls fork at219, attempts gdb/lldb in the child, and waits in the parent at257. ggml_abort calls this handler before abort at291. The failed control log prints the ggml_cuda.cu:128 abort marker at server.log:93; the kernel clone stack is consistent with that crash-handler path. This is a stronger source/kernel-supported explanation for post-failure memory amplification, but the exact fork caller and original CUDA failure remain unproven. No completed debugger child is required for failure while cloning.

The guarded run_models.py sets GGML_NO_BACKTRACE=1 before server exec; the containment unit sets LimitCORE=0. These are diagnostic-harness choices. Frozen binaries and product behavior were not edited. Keep immediate fatal-log group kill, zero swap and hard accounted-memory controls as separate defenses.

## Audited transition boundaries

- llama-context.cpp:3810 resets the selected source graph before replacing graph metadata. moe-source-core.cu:2482 completes adaptation after drain and before releasing source transport, graph leases or host control storage.
- moe-cache.cu:16390 checks adaptation resource identity, device serial, pending state, active transaction and active caller count before publication. It publishes reciprocal maps and records completion, waits for the job stream, clears pending state, invalidates the residency token and then destroys the job stream. The completion event remains part of the device resource.
- moe-cache.cu:8917 retains the registered-source owner in the device resource;7237 releases it at device-resource destruction. Clearing program resource leases alone does not prove original weights or their registration are freed.
- The ordinary prefill planner uses full-slot-capacity storage and waits on the prior resource completion event. Prior plan commit, route bounds, reciprocal map checks, frequency epoch and clock checks can trap. No saved plan status or launch-level trace narrows the incident to one such check.
- The auxiliary single-row hybrid admission path returns its plan to BUILDING at moe-cache.cu:5406. The current source rows path instead receives an explicit plan buffer; its constructor/import initializes that buffer. Do not conflate that program plan with the canonical residency device.plan reused by prefill. Reconciliation at4343 and writer token invalidation remain the relevant boundaries for the stale-READY-plan hypothesis; a bypass is not proved.

A future bounded regression should preserve the transition itself: same canonical resource, static placement, several source decode windows including asynchronous swaps, source graph retirement, then ordinary prefill selecting all available slots. Check map reciprocity, plan status/producer, frequency epoch, device serial, completion dependency and source registration before each transition. Use small synthetic tensors and an existing cache fixture, with the verified containment wrapper and locks, only after the GPU investigation restriction is lifted. Current work remains CPU-only.

## Actual frozen-library fatal-path validation

containment-ggml-abort-01 loads only control/bin/libggml-base.so.0.25.3 in a small Python process, verifies no CUDA libraries in its maps, verifies GGML_NO_BACKTRACE=1 and both core limits0, starts the existing detached child fixture, then calls the frozen library's ggml_abort. Effective memory.max128MiB and memory.swap.max0 are verified before execution. Expected SIGABRT occurred; systemd removed the SIGTERM-ignoring descendant after its3s grace. Both PIDs are gone, the cgroup is empty/removed, sampled memory.peak17944576bytes (17.11MiB), and full elapsed time3.29s. QUALIFICATION.json records the expected failure separately from the launcher's nonzero exit.

This host's core_pattern is piped to systemd-coredump. The journal at17:47:17 EDT names the fixture PID2765431 and confirms resource limits disabled core generation, then confirms the helper service deactivated. The unit result is still core-dump, which alone must not be interpreted as a dumped memory image. LimitCORE=0 does not prevent invocation of the external system crash-reporting helper. No global core setting was changed; no helper was killed. GPU/model execution remains blocked and the original CUDA defect remains unproven.

## Resolved userspace instruction and fail-closed evidence writing

The OOM RIP0x7fe09f903e32 minus saved libc base0x7fe09f800000 is0x103e32. The saved mapping inode346578 matches the current libc inode; its mtime is2026-08-12 and build ID5e3b85a3eaa72d02eaae3882b22410ebda6b5555. Current disassembly places this instruction exactly at _Fork+0x42, immediately after syscall at0x103e30, with clone flags0x1200011 loaded at0x103e1f. __libc_fork calls that routine. This corroborates actual process-fork/address-space duplication; it still does not identify the higher-level caller or original CUDA defect. Details are in KERNEL-CLONE-EVIDENCE.json.

The owned fatal callback previously wrote RUN.json and fatal-guard.json before killing the group. A write/rename failure could bypass immediate teardown and leave only the manager's finite runtime bound. Both writes now use finally to guarantee an attempt to kill the exact verified unit. Host-memory sampler exceptions also trigger fatal handling instead of silently stopping the sampler. These changes affect only the isolated diagnostic harness.

## Source-proven profile age mismatch

PROFILE-AGE-FIX.md records the frozen release contradiction between profile/adaptation writers that publish ready ages as0 and the ordinary planner that traps on any resident age0. A bounded CPU probe extracts the actual frozen predicate and existing scalar setter, establishes the counterexample across several capacities and verifies the source correction to age1 at both writers. This follows the existing ordinary static placement policy without assigning usage credit. The isolated product source is changed and its guarded serial CUDA rebuild now passes, and the later24-request retained-owner device transition qualifies it within the scoped tiny-fixture exception. Exact incident assertion and original CUDA root cause remain unproven; keep GPU-RUNS-BLOCKED.json present. No full model replay is qualified.

containment-fatal-record-failure-01 deliberately makes fatal-guard.json a directory, so the evidence rename fails. The CUDA marker is synthetic text only. The128MiB/zero-swap unit still terminates with SIGKILL, both parent and detached SIGTERM-ignoring child are gone, the cgroup is empty/removed, sampled peak16531456bytes and full elapsed time0.63s. The remaining temporary JSON proves the fatal marker was observed before the intentionally failed rename. QUALIFICATION.json records the expected failure; CONTAINMENT-VALIDATION.json includes this case. No GPU run or model load occurred.

## Scoped synthetic device investigation

The corrected test-only candidate passes the guarded CPU command/manifest preflight:3072MiB hard accounted memory, zero swap,120s finite runtime and verified whole-tree removal. test_profile_transition.py permits only test-moe-cache --hybrid-metadata-only with source-core-profile/adapt/async; it validates frozen candidate hashes, acquires both ordered locks nonblocking, installs the fatal-log group-kill watcher before child launch and requires all eight retained-owner prefill checks plus four source fixtures. No server or faulty frozen GPU control is launched. GPU-RUNS-BLOCKED.json remains present and continues to block run_models.py. Its explicit tiny-fixture scope follows the now-verified containment prerequisites; it does not release the original Flash workload or establish its root cause. The previous block file is preserved. That preflight-stage lock wait ended before the completed small device transitions recorded below; it is historical, not current process state.

## Completed small device transition

PROFILE-TRANSITION-QUALIFICATION.json qualifies the corrected candidate on static placement, synchronous adaptation and held asynchronous copies:24 retained-owner ordinary-prefill requests, separate/fused bank layouts and pinned/pageable transport, forced waves and exact ordinary-CUDA output agreement. All owner keys and bank allocations remain retained; error counters are zero. Each job uses3072MiB hard memory, zero swap,120s runtime and both ordered locks; RAM peaks423-432MiB, all test PIDs/trees are gone and the interval's kernel log contains no Xid/OOM. This closes only the small age-publication/transition qualification. No model replay, faulty frozen GPU control, default promotion or matched serving-performance claim follows. The exact original incident assertion/root cause remains unproven.

Async command setup initially used0.5, violating the pending all-CPU-miss assertion, then0, violating the seed-transfer assertion. Both failed jobs terminated cleanly and remain preserved. Source gates require floor(8*n/256)>0 and floor(3*n/256)=0; numerator64/fraction0.25 satisfies both. The test assertions and product source were unchanged. Static and synchronous cases use0.5. Follow the actual fixture recipe rather than treating a shared environment as interchangeable across these tests.

The shared production region-method refactor is now qualified on12 contained device cases/24 production preparations, following124416 CPU control-flow/certificate/lifetime scenarios against extracted frozen code. Every GGML backend library remains byte-identical to the preceding candidate; only llama-region orchestration/test code changed. SHARED-REGION-LIFETIME-DEVICE-QUALIFICATION.json records the final exact source. These tests are separate from the profile age correction and do not establish the original Flash/Xid/global OOM root or release the model-server block.
