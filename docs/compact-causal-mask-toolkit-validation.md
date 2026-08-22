# Compact causal mask validation with the feature-performance toolkit

Date: 2026-08-22

This is the evidence report for a frozen comparison of compact-causal PR 7
commit `f92c2de62bcc11abed0cff8b10a848fc8d97f29d` against its exact KV-offload
base `f6341a15779eb58fe6ad9e1b890e331c32b676c7`. It is not evidence for the
current PR head. During this study PR 7 advanced to
`3fd817df9dc373c4ab7f4732d3565a4d1786144f`; that newer source is not covered.

The exact full-ladder preregistration used for this evidence is
[compact-causal-f92-vs-f634.full.manifest.json](../examples/feature-performance-validation/compact-causal-f92-vs-f634.full.manifest.json).
The shorter sibling manifest is a later fast-screen protocol and does not
retroactively identify these results.
Large raw artifacts remain outside Git under
`/tmp/beellama-fpv-causal-f92-vs-f634-20260821`. No commit, push, PR action, or
ref change was made for this validation.

## Result

The frozen candidate passes the preregistered scope exercised here:

- deterministic 16-token CLI output was byte-identical;
- matching-batch perplexity was identical at `2.1674 +/- 0.03849`;
- weighted early prefill improved by `+0.5262%` with a 95% paired interval of
  `[+0.2847%, +0.7683%]`;
- production 29,398-token prefill was equivalent at `+0.1154%`, interval
  `[-0.0737%, +0.3048%]`;
- the mandatory 30K-depth, 128-token decode gate improved by `+0.4727%`,
  interval `[+0.3988%, +0.5466%]`;
- process VRAM was equal at the production and final sampled peaks, while the
  candidate reduced the allocator-classified accelerator-host causal buffer by
  28.75 MiB at production prefill and 29.5 MiB at final decode;
- the observed VMM allocation lifecycle returned to zero live/mapped bytes and
  zero pools after context teardown.

`status=acceptance_complete` here means that the final gate both executed and
passed its preregistered policy. Completion alone cannot produce this status.
The evidence does not cover the ledger's 128K CPU-KV composition, direct
page-locked host bytes, default-KVarN compatibility build, or the PR's newer
head.

## Frozen source and artifact provenance

The comparison used detached, independent source copies in `/tmp`, not the
concurrently active causal worktree. `git diff --check` passed for the frozen
range. The exact range contains 26 changed files, 3,428 insertions, and 116
deletions, including its investigation ledger and test tooling. The binary
source-only patch identity, excluding `docs/**` and `scripts/**`, is:

```text
85ec27bba2dfb9416c0325b60482fec8bd91eb1c206285ae69a5e4bba010ee6
```

The source audit found the compact causal representation, production consumer
changes, and bounded tests, without native-Q8 implementation import,
architecture/model-family dispatch, public flags, or optional dependencies.
The builds explicitly used Q8_0/Q8_0 KV, `-nkvo 0`, CUDA FlashAttention, and
`GGML_CUDA_KVARN=OFF`. Historical native-Q8 permission was absent and CPU-KV,
recurrent-state, pinned-KV, and shared-workspace controls were disabled.

The model was
`Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf`, size 14,437,471,712 bytes, SHA-256
`ca5c3fab5c68a00a7c4fc04a0467946e2069f3cdb073601e7158ae7977e73f6c`.
The perplexity corpus SHA-256 was
`8a2f79a2f4601cfe6e25830c29c1a25c7a3d906285a989948117568f8077ab2`.
The preregistration SHA-256 was
`0f10b55152ac5798f8f204a145057e1f325747755865866ed6fdad6d53c4b037`;
its provenance fingerprint was
`28454f1c2b144136e3c6fe048601455a2eacf157047cfa48b8e67714ee67fb16`.

Exact pre-existing Release, CUDA, SM120, CUDA-FA, native builds with KVarN off
were reused after commit, cache, executable, and linked-library verification.
No configure or build was performed. This saved two complete builds, but their
elapsed time was not recorded, so no numerical time saving is claimed.

| artifact | base SHA-256 | candidate SHA-256 |
|---|---|---|
| `llama-bench` | `497de09c880d0c0c9539f6e8347ce789023b3de76a25bcf45c211672f43ad1d2` | `990139351acda670b8867e919401245d1086ade3d553e6a04e4f95c4e611e4d7` |
| `llama-cli` | `fc26d0198eef19a21f8f79cd585178dc7feb497f656df2ff05442e78366ad5f0` | `2e832f8f79106315cd22a8cbd34debed1b7cb24136ec5b13d578e0bd26a97469` |
| `llama-perplexity` | `40c809576713eb7d82a29a418dd3a471bc8496894362f10b0dd203480d576b6b` | `43c86f0f65813b4cd5f2a1ba7b31121ab54fa30d14bd6f3b5eb8744a0200a195` |
| `llama-server` | `2152baa65085baf9f5df419a42c772a82e1f6f4f7295af59628124bc3d2c25c6` | `6c1aec477e7f8aecded4a1582daa0228a2cc417e0b013370babfeb36c74daf5a` |
| `libggml-cuda.so` | `b6a035b252ee1b1765261e2e079967994075065dd08117d6a1e8e02a9553b196` | `99463282a9ee08474356bee153ca9663217e94fcf357d5505b782583bcf2b58c` |
| `CMakeCache.txt` | `4daf76513c5fc367db1439b5f013de19ce9675747a0aa874dcb9ebb6dcae04fb` | `12eee7b2959ae8db22d8ecce7be3156a520e672464ce4875fbd95c1bfadd665b` |

The candidate executable's embedded display label is
`f6341a-pr7-combined-direct-tile`, not a Git object name. The frozen ledger ties
the exact hashes above to `f92c2de62`; the manifest independently binds the
source commit, tree state, binaries, libraries, model, and build cache.

The host was an RTX 5070 Ti, UUID
`GPU-36e6dd4d-4d34-f8a7-dd17-500c1791f28f`, driver `610.57.04`, with an Intel
Core Ultra 9 285K. The installed profilers were NSYS 2026.1.3.425 and NCU
2026.2.1.0.

## Preregistered policy

The performance policy used three independent matched pairs in balanced order
`AB, BA, AB`; pairs four and five were permitted only if the three-pair
confidence interval was inconclusive. The benefit direction was higher-is-
better. A stage was an improvement only if the interval lower bound was at
least `+0.25%`, a regression if the interval upper bound was at most `-0.5%`,
and equivalent only when its complete interval stayed within `+/-1%`.
Thresholds were not changed after seeing results. No run was omitted and no
five-pair extension was triggered.

The stages were deterministic output exactness, short prefill, short depth-4K
decode with graph replay, a real-ubatch-weighted low/mid/high prefill screen,
29,398-token production prefill, and a separate mandatory depth-30K decode
acceptance gate. Every target was a fresh process under the whole-command lock,
with native llama affinity and visible `llama-bench --progress` output.

## Raw paired performance results

All throughput values are tokens/second. Percent changes are candidate over
base. The reported interval is the preregistered two-sided Student-t interval
on paired log ratios.

| stage | order | base | candidate | pair change |
|---|---:|---:|---:|---:|
| short prefill p128 | AB | 1477.668938 | 1474.341232 | -0.22520% |
|  | BA | 1473.279878 | 1478.951007 | +0.38493% |
|  | AB | 1473.313378 | 1475.050162 | +0.11788% |
| short decode d4096/n16 | AB | 49.280521 | 49.118300 | -0.32918% |
|  | BA | 49.280640 | 49.126488 | -0.31280% |
|  | AB | 49.254152 | 49.095380 | -0.32235% |
| weighted prefill screen | AB | 1557.092167 | 1563.714839 | +0.42532% |
|  | BA | 1550.842228 | 1560.450244 | +0.61954% |
|  | AB | 1549.606894 | 1557.881241 | +0.53396% |
| production prefill p29398 | AB | 1618.173251 | 1618.675716 | +0.03105% |
|  | BA | 1616.314414 | 1618.512353 | +0.13598% |
|  | AB | 1615.679308 | 1618.573624 | +0.17914% |
| final decode d30000/n128 | AB | 44.051650 | 44.244784 | +0.43843% |
|  | BA | 44.057755 | 44.274396 | +0.49172% |
|  | AB | 44.040441 | 44.255353 | +0.48799% |

| stage | effect | 95% interval | decision |
|---|---:|---:|---|
| short prefill | +0.0922% | [-0.6648%, +0.8550%] | equivalent |
| short decode | -0.3214% | [-0.3419%, -0.3010%] | equivalent |
| weighted prefill | +0.5262% | [+0.2847%, +0.7683%] | improvement |
| production prefill | +0.1154% | [-0.0737%, +0.3048%] | equivalent |
| final decode | +0.4727% | [+0.3988%, +0.5466%] | improvement |

The short decode is consistently a small negative point estimate but remains
inside the preregistered equivalence band. The final gate's positive interval,
not mere process completion, is what permits acceptance.

## Correctness and perplexity

The base and candidate deterministic CLI outputs had the same SHA-256:
`22df427dfe743170ed81bed484cd1ea3eafcb442844e73fab2c2bb4eb9abd6b9`.

Matching perplexity processes used the same model/corpus, Q8_0/Q8_0,
`-c 4096 -b 512 -ub 256`, and four chunks. Both emitted
`[1]1.9315,[2]2.1279,[3]2.2498,[4]2.1674` and
`PPL = 2.1674 +/- 0.03849`. Their outer walls were 11.98 and 12.02 seconds.
This closes the scoped no-perplexity-increase gate.

## Memory and lifecycle

The roughly 1 Hz sampler found no foreign compute process. Each target had one
persistent `nvidia-smi -q -x -l 1` process, lightweight `/proc/PID/status`
sampling, and a reaped sampler. Xorg's 4 MiB graphics context was retained as
ambient non-compute load.

| checkpoint | process VRAM base/candidate | ordinary RSS candidate delta | accelerator-host causal buffer delta |
|---|---:|---:|---:|
| short decode | 13,956 / 13,952 MiB | transient/noise-scale | not used for acceptance |
| production prefill | 14,780 / 14,780 MiB | about +1.15 MiB | -28.75 MiB |
| final decode | 14,810 / 14,810 MiB | about +1.8 MiB | -29.5 MiB |

At production and final depth, synchronized CUDA-used context, post-prefill,
peak, and post-context values were equal between variants. Device compute was
529,530,880 bytes for base and 529,827,968 for candidate. The causal buffer's
allocator classification moved from accelerator-host 51,417,120 to 21,270,560
bytes at production and from 52,203,552 to 21,270,560 bytes at final depth.
This is allocation-ledger evidence, not a direct physical pin counter.

The VMM ledger peaked at 14,632,960 live and 14,680,064 mapped bytes for both.
After the workload it showed zero live bytes, 14,680,064 mapped bytes, and one
pool; after context destruction both variants showed zero live bytes, zero
mapped bytes, and zero pools.

Direct per-process page-locked/pinned bytes were unavailable and remain
`null`. They are not inferred from `VmLck`, `VmPin`, ordinary RSS/HWM, process
VRAM, or the accelerator-host allocator label. A dedicated 128K CPU-KV run is
still required to reproduce the frozen ledger's deepest pinned-host claim.

## Profiler screen

The intended graph-node discovery was first attempted with explicit NSYS
`--cuda-graph-trace=node` support verification. The installed tool supported
the option, but this selected production-prefill command emitted no graph node
IDs. The graph-required attempt failed closed; it was not silently treated as
graph evidence. Because the production workload was a direct non-graph
prefill, the corrected manifest preregistered graph tracing as not applicable.

The exact candidate NSYS run observed 580,420 CUDA launches and 3,712 matching
F16 causal-mask launches. The plan selected the semantic last occurrence,
global launch index 580,398, grid `[140,1,1]`, block `[32,4,1]`. The subsequent
NCU process filtered by discovery-derived name and shape rather than replaying
the NSYS global skip. Its exported report was hashed and re-imported; exactly
one expected capture was observed and verification returned `verified` with no
issues. The report was 2,157,936 bytes, SHA-256
`9da2bc16d27fc9615f8b46209deb9ded701546de256b5111dc56ba9e29bb8807`,
and the selected kernel duration was 3.371104 ms.

This is a candidate-only implementation screen. It is not a kernel A/B speed
claim and does not substitute for output, memory, production, or long-context
evidence.

## Timing and token efficiency

The toolkit is entirely local and consumed **zero LLM/API tokens**. Agent
reasoning used to design and interpret this investigation is not test-suite
runtime and is not attributed to the suite.

| activity | observed outer wall | target runtime | toolkit/lock detail |
|---|---:|---:|---|
| validate manifest/provenance | 6.26 s | none | includes a 4.04 s model hash |
| early ladder, cold | 292.91 s (4m52.91) | 280.75 s | 4.31 s runner + 1.29 s lock; 32 processes |
| early ladder, warm resume | 6.26 s | none | all 32 successful runs reused |
| production increment | 455.36 s (7m35.36) | 447.86 s | 0.85 s runner + 0.23 s lock |
| acceptance increment | 220.74 s (3m40.74) | 213.20 s | 0.85 s runner + 0.23 s lock |
| staged main total | 969.01 s (16m09.01) | 941.81 s | 6.00 s runner + 1.75 s lock across 44 runs |
| perplexity pair | 24.00 s | about 23.6 s | separate exactness gate |
| NSYS discovery + filtered NCU | 170.70 s | 157.58 s | remaining time is export/parse/verify/prep |
| optional state-reuse experiment | 60.74 s | 60.36 s | four clean server lifecycles |

The three staged invocations repeat about 6.2 seconds of provenance
preparation apiece. A derived single-invocation cold estimate from recorded
components is about 955.78 seconds (15m55.78), but it was not executed and is
not presented as a measurement. Main run plus perplexity and profiling was
1,163.71 seconds (19m23.71); including the optional state experiment was about
1,224.45 seconds (20m24.45).

The early screen missed its 30-90 second design target by 3.25x. This was not a
Python/telemetry bottleneck: target processes accounted for 280.75 of 292.91
seconds, while measured runner plus lock overhead was 5.60 seconds. The 1 Hz
sampler startup accounted for 3.04 seconds across all 32 runs. Initial
preparation was 6.21 seconds, with the 14.4 GB model hash alone taking 4.04
seconds. Stat-guarded identity reuse already prevents repeated per-process
rehashing.

The immediate optimization is therefore to preregister fewer early processes:
one exactness pair, one short smoke pair per workload, and a reduced low/mid/
high screen before invoking the three-pair statistics only for the most
decision-relevant weighted aggregate. Model/library hashing and telemetry
should remain fail closed. The measured follow-up is recorded below.

### Fast early-screen follow-up

That optimization was subsequently implemented and measured with
[compact-causal-f92-vs-f634.manifest.json](../examples/feature-performance-validation/compact-causal-f92-vs-f634.manifest.json).
The original full evidence manifest remains preserved separately with its
unchanged SHA-256. The fast manifest preregisters one matched pair for each
smoke and the weighted screen, alternates `AB`/`BA` stage order, and samples
representative 2K/8K/16K context strata. Production remains at 29,398 tokens
and final acceptance remains at depth 30K with the global 3-to-5 policy.

The fresh run completed in **76.27 seconds**, within the 30-90 second target,
versus 292.91 seconds for the original cold early ladder: 216.64 seconds and
74.0% less wall time. It launched 12 instead of 32 clean locked processes.
The fast manifest SHA-256 was
`025e726b1338cfbe1ba70bf94c23e860703440cb480bf4c4b4bab748279ceec5`.
Target processes used 67.81 seconds, initial provenance preparation 6.20
seconds, recorded runner overhead 1.65 seconds, lock-wrapper handoff 0.46
seconds, and sampler startup 1.17 seconds. All samplers were reaped and no
foreign compute process was observed.

The outputs were exact and all three fail-fast signals cleared the
preregistered 1% regression threshold: short prefill `-0.0594%`, short decode
`-0.3074%`, and the weighted screen `+0.0945%`. Each report deliberately has
`confidence_interval=null` and `confidence_claim=none`; these points do not
replace or reinterpret the full three-pair statistics above. The speedup is a
measured workflow improvement, not new feature-performance evidence.

### State reuse

`llama-bench` already caches an exact depth in memory between repetitions via
`llama_state_seq_get_data`/`llama_state_seq_set_data`; that existing behavior,
not the toolkit, makes repeated depth samples inexpensive within one process.
An optional direct-server experiment tested provenance-guarded on-disk state
reuse separately for each exact build.

| variant | cold fill | state save | cold total | restore | restored total | decode cold/restored |
|---|---:|---:|---:|---:|---:|---:|
| base | 18.110 s | 0.278 s | 24.133 s | 0.330 s | 6.084 s | 2.977 / 2.980 s |
| candidate | 18.101 s | 0.275 s | 24.089 s | 0.327 s | 6.053 s | 2.963 / 2.964 s |

Restore removed about 18.04 seconds, or 74.7%, from each fresh 29,398-token
server lifecycle while preserving all 128 generated token/content hashes. Each
snapshot was 1,181,121,088 bytes and both had SHA-256
`eec76e8f548db8725ff8bdac0e8e908b164a8e66573dd4e494634f1ec6134b82`.
They were nevertheless kept per build and guarded by model, prompt, server,
and CUDA-library hashes. State reuse is suitable for targeted decode screens;
it must not be used for prefill timing, startup/resource acceptance,
perplexity, VMM/allocation lifecycle, or as a cross-build compatibility claim.

## Comparison with the causal ledger

The frozen investigation ledger records the final validation coverage but does
not record an elapsed wall time for every final stage. The user-reported
`2h35m` is the whole development turn, including design, edits, builds,
debugging, interpretation, and documentation; it is not an actual validation
runtime and cannot be used as a suite-speed denominator.

The final frozen ledger's validation consisted of the source-guarded direct
CUDA harness at 30K/64K/128K, focused CPU/CUDA oracle, matching PPL, full 29K
prefill, 30K output/decode, 4K lifecycle, 128K CPU-KV, NCU A/B, and a default-
KVarN compatibility build. Its only explicit final setup wall is the accepted
harness compile at 2.545 seconds. Therefore a complete like-for-like final
runtime cannot be reconstructed.

Two older, superseded process gates do have enough timing detail: Gate A's ten
processes total about 18m45.6 and Gate B's ten processes about 39m28.6, or
58m14.2 combined. They used an older candidate and unlike coverage, so the
toolkit's 19m23.71 core workflow must not be called a 3x speedup. The defensible
conclusion is that automation compresses the repeatable, overlapping subset to
about 19m24 with auditable artifacts; no total speedup versus the final causal
validation is measurable from the ledger.

## Coverage gained, lost, and workflow changes

Automated coverage gained:

- exact source/binary/model/library/build provenance and safe whole-command
  locking for every run;
- balanced clean processes, all raw paired samples, preregistered thresholds,
  and fail-closed executed-versus-passed acceptance;
- persistent GPU/proc telemetry, contamination evidence, sampler ownership,
  and synchronized allocation/VMM checkpoints;
- deterministic resume and measured target, lock, telemetry, and runner time;
- NSYS discovery converted to an NCU plan and post-capture verification without
  assuming cross-process launch-order stability.

Coverage lost relative to the frozen causal ledger:

- the source-guarded direct CUDA correctness/performance harness at 30K/64K/
  128K and the focused backend oracle;
- 4K production server lifecycle as a separately preregistered gate;
- 128K CPU-KV composition and direct pinned-host instrumentation;
- paired NCU A/B metrics and default-KVarN compatibility build/tests.

Recommended workflow:

1. Keep the current manifest/provenance/statistics machinery as the common
   evidence spine.
2. Split early screening into a sub-90-second minimal gate and run the full
   three-pair weighted screen only after it passes.
3. Add a small source-identity-guarded hook for the existing production-aligned
   causal harness rather than copying CUDA into the toolkit.
4. Use per-build on-disk state only for decode iteration; keep cold prefill and
   lifecycle acceptance independent.
5. Retain one candidate NSYS discovery plus one verified filtered NCU capture
   per investigation stage, and add base NCU only when a kernel A/B conclusion
   is required.
6. Record stage wall times in future ledgers, separately from development-turn
   duration, configuration/build, lock wait, profiling, and documentation.

## Validation commands and failures

The main toolkit commands were:

```bash
python3 scripts/feature-performance-validation.py validate \
  examples/feature-performance-validation/compact-causal-f92-vs-f634.manifest.json
python3 scripts/feature-performance-validation.py run --through early MANIFEST
python3 scripts/feature-performance-validation.py run --through production --resume MANIFEST
python3 scripts/feature-performance-validation.py run --through acceptance --resume MANIFEST
python3 scripts/feature-performance-validation.py profile --execute-ncu MANIFEST
PYTHONWARNINGS=error::ResourceWarning python3 scripts/test-feature-validation.py -v
git diff --check
```

Every generated GPU run-spec records the exact safely quoted form
`flock /tmp/beellama-single-gpu.lock -c 'COMMAND'`; no `taskset` or profiler
wrapper target was used. The final CPU-only suite passed 49 tests in 8.084
seconds with `ResourceWarning` promoted to an error.

An initial artifact set is invalid and excluded: while it was running, the
concurrent causal worktree's linked `libggml-cuda.so` changed. The toolkit
recorded the unexpected library identity, but the first manifest had not
preregistered that library hash. All accepted evidence was rerun as `r2` from
the independent exact candidate build. This failure is why both executable and
linked-library identity are now mandatory in this manifest.

The first profiler attempt also failed closed when graph-node IDs were absent;
the non-graph applicability correction was tested before the accepted capture.
These failed attempts are workflow evidence only and contribute no performance
samples.

No other worktree was edited, built, cleaned, committed, or otherwise mutated
by this task. The assigned worktree remains on its original tooling branch
commit with only this task's uncommitted manifest, report, and bounded toolkit
test/fail-closed fixes.
