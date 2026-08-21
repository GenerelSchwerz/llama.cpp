# RTX 4070 / Qwen3.8 IQ2_M results archive

This directory holds the raw artifacts backing Characterizations 020-026 in
[`cpu-kv-offload-experiments.md`](../cpu-kv-offload-experiments.md): a
baseline/`draft-mtp`/`draft-dspark` comparison run on a second, weaker
machine (RTX 4070 12 GiB, Intel i5-13400F) than this branch's primary RTX
5070 Ti / Core Ultra 9 285K development host, using
`Qwen3.8-27B-UD-IQ2_M.gguf` and `Qwen3.8-27B-DSpark-Q8_0.gguf`.

Two measurement passes are included, against two different points on this
branch:

- **Pass 1** (tags `A-`, `B-`, `C-`, `D-`, `E-`, `F-`): measured against
  commit `53adab814` ("docs: record phase-aware KV merge"), the PR #1 head
  at the time. This predates the MTP draft-ubatch/target-ubatch matching
  fix and the canonical accelerator-quant-store fix that later landed as
  Experiments 017/018.
- **Pass 2** (tags `A2-`, `G-`, `H-`): re-measured against commit `c9f727c1e`
  ("docs: record live draft KV decode comparison"), the Pass 2 source tip,
  after rebasing onto that work. The `H-` run additionally postdates the
  DSpark `n_max` fix in this PR and verifies that the omitted default resolves
  to the same trained depth and runtime behavior as explicit depth 6. MTP
  configurations in this pass omit
  `--spec-draft-ubatch-size` entirely (the flag now hard-rejects any value
  that does not equal the target ubatch for `draft-mtp`).

This is an immutable historical-results archive, not a current runnable
protocol. Do not execute the archived command snapshots or scripts as current
work: they predate the native-affinity and whole-lifecycle GPU-lock contract,
and some intentionally preserve the now-rejected MTP ubatch geometry. Use
[`cpu-kv-offload-current-testing.md`](../cpu-kv-offload-current-testing.md)
for every new run.

Do not mix rows across the two passes when comparing absolute throughput:
the underlying binary changed between them. Pass 1 rows are retained because
several of them isolate a single variable (`--spec-draft-ubatch-size`,
`--kv-gpu-layers`, `--spec-mtp-rs-planes`, `--phase-aware-workspace`) with
both sides of the comparison measured on the *same* Pass 1 binary, which
keeps the relative delta meaningful even though the absolute numbers are
superseded.

## Contents

- `SUMMARY.txt` — the harness's per-run digest: request timings, VRAM/RSS
  monitor summary, and the tail of each server's own log (`print_timing`,
  `memory breakdown`, checkpoint/replay lines).
- `logs/` — full server log for every run (`--log-verbosity 4`), plus the
  recorded `llama-server` arguments in the matching `.cmd` file. The historical
  whole-process affinity wrapper is preserved separately in `archived-harness/`.
- `monitor/` — the 0.5 s-interval VRAM (`nvidia-smi --query-compute-apps`)
  and RSS/VmHWM (`/proc/<pid>/status`) CSV sampled for the lifetime of each
  server process.
- `archived-harness/` — non-executable snapshots of the bash/python scripts
  used for the 2026-08-19 session; they are evidence, not maintained harnesses
  (`run_one.sh`, `run_synth.sh`, `monitor.sh`, `summarize.py`,
  `driver.sh` for Pass 1, `driver2.sh` for Pass 2) and the three prompt text
  files (`prompt_short.txt`, `prompt_creative.txt`, `prompt_coding.txt`).

## Full results table

All 51 tabulated runs from both passes, in run order. "Status" is `ok` (request
completed), `CRASHED` (server died mid-generation with a CUDA illegal
memory access — see Characterization 021), or `OOM/REJECTED` (failed before
serving any tokens, either a CUDA allocation failure or a CLI validation
rejection). A `-` in a numeric column means that value could not be
recovered for that run (typically because the run crashed before its final
timing summary printed).

| Group | Config | Base commit | Prompt tok | Prefill t/s | Output tok (req) | Decode t/s | Draft acceptance | Peak proc. VRAM (MiB) | Peak RSS (MiB) | Status |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| A | baseline, short prompt | 53adab814 (pre-rebase) | 72 | 267.69 | 64 | 39.86 | - | 9782 | 2033 | ok |
| A | MTP-5 (ubatch=128, pre-fix), short prompt | 53adab814 (pre-rebase) | 72 | 248.64 | 64 | 69.42 | 75.0% (45/60, len 3.65) | 10876 | 2170 | ok |
| A | DSpark-6 (CPU draft), short prompt | 53adab814 (pre-rebase) | 72 | 186.93 | 64 | 16.88 | 100.0% (30/30, len 2.88) | 11246 | 3951 | ok |
| A | baseline, creative prompt | 53adab814 (pre-rebase) | 89 | 325.34 | 400 | 39.00 | - | 9780 | 2036 | ok |
| A | MTP-5 (ubatch=128, pre-fix), creative prompt | 53adab814 (pre-rebase) | 89 | 299.60 | 400 | 46.04 | 90.6% (144/159, len 2.69) | 10874 | 2172 | ok |
| A | DSpark-6 (CPU draft), creative prompt | 53adab814 (pre-rebase) | 89 | 234.21 | 400 | 11.41 | 93.9% (77/82, len 2.64) | 11254 | 3957 | ok |
| A | baseline, coding prompt | 53adab814 (pre-rebase) | 135 | 450.22 | 1200 | 37.79 | - | 9782 | 2047 | ok |
| A | MTP-5 (ubatch=128, pre-fix), coding prompt | 53adab814 (pre-rebase) | - | - | - | - | - | 10868 | 2284 | **CRASHED** |
| A | DSpark-6 (CPU draft), coding prompt | 53adab814 (pre-rebase) | 135 | 327.38 | 1200 | 10.68 | 95.0% (228/240, len 2.40) | 11258 | 4033 | ok |
| A | baseline, 12K synthetic prefill | 53adab814 (pre-rebase) | 12000 | 1136.02 | 25 | 23.54 | - | 9822 | 2081 | ok |
| A | MTP-5 (ubatch=128, pre-fix), 12K synthetic | 53adab814 (pre-rebase) | 12000 | 1088.86 | 25 | 71.08 | 100.0% (20/20, len 6.00) | 10890 | 2311 | ok |
| A | DSpark-6 (CPU draft), 12K synthetic | 53adab814 (pre-rebase) | 12000 | 1007.35 | 25 | 3.64 | 100.0% (11/11, len 2.57) | 11312 | 4783 | ok |
| B | DSpark, no `--spec-draft-n-max` (silently n_max=3) | 53adab814 (pre-rebase) | 135 | 326.69 | 1200 | 10.90 | 94.9% (185/195, len 2.36) | 10800 | 4067 | ok |
| B | DSpark, `--spec-draft-n-max 6` (correct) | 53adab814 (pre-rebase) | 135 | 325.34 | 1200 | 10.72 | 95.0% (228/240, len 2.40) | 11258 | 4069 | ok |
| C | MTP-5, draft ubatch 512 (=target, now the only valid value) | 53adab814 (pre-rebase) | 72 | 248.80 | 64 | 68.83 | 75.0% (45/60, len 3.65) | 10948 | 2220 | ok |
| C | MTP-5, draft ubatch 128 (now CLI-rejected) | 53adab814 (pre-rebase) | 72 | 249.77 | 64 | 68.78 | 75.0% (45/60, len 3.65) | 10876 | 2196 | ok |
| C | MTP-5, draft ubatch 32 (now CLI-rejected) | 53adab814 (pre-rebase) | 72 | 248.00 | 64 | 69.06 | 75.0% (45/60, len 3.65) | 10858 | 2190 | ok |
| C | DSpark-6, draft ubatch 128 (still valid for DSpark) | 53adab814 (pre-rebase) | 72 | 187.37 | 64 | 16.95 | 100.0% (30/30, len 2.88) | 10892 | 3967 | ok |
| D | baseline, `--kv-gpu-layers 0` | 53adab814 (pre-rebase) | 72 | 266.91 | 64 | 39.64 | - | 9782 | 2062 | ok |
| D | MTP-5, `--kv-gpu-layers 0` | 53adab814 (pre-rebase) | 72 | 249.39 | 64 | 69.01 | 75.0% (45/60, len 3.65) | 10876 | 2196 | ok |
| D | DSpark-6, `--kv-gpu-layers 0` | 53adab814 (pre-rebase) | 72 | 187.52 | 64 | 16.98 | 100.0% (30/30, len 2.88) | 11246 | 3977 | ok |
| D | baseline, `--kv-gpu-layers 8` | 53adab814 (pre-rebase) | 72 | 270.66 | 64 | 39.89 | - | 10052 | 1789 | ok |
| D | MTP-5, `--kv-gpu-layers 8` | 53adab814 (pre-rebase) | 72 | 251.78 | 64 | 71.16 | 78.9% (45/57, len 3.65) | 11140 | 1898 | ok |
| D | DSpark-6, `--kv-gpu-layers 8` | 53adab814 (pre-rebase) | 72 | 187.28 | 64 | 16.96 | 100.0% (30/30, len 2.88) | 11504 | 3711 | ok |
| D | baseline, `--kv-gpu-layers 16` | 53adab814 (pre-rebase) | 72 | 301.10 | 64 | 40.45 | - | 10290 | 1517 | ok |
| D | MTP-5, `--kv-gpu-layers 16` | 53adab814 (pre-rebase) | 72 | 253.09 | 64 | 71.02 | 78.9% (45/57, len 3.65) | 11378 | 1628 | ok |
| D | DSpark-6, `--kv-gpu-layers 16` | 53adab814 (pre-rebase) | 72 | 186.56 | 64 | 16.98 | 100.0% (30/30, len 2.88) | 11742 | 3444 | ok |
| E | MTP-6, `--spec-mtp-rs-planes 0` (full, 7 planes) | 53adab814 (pre-rebase) | - | - | - | - | - | 10980 | 2334 | **CRASHED** |
| E | MTP-6, `--spec-mtp-rs-planes 4` (capped) | 53adab814 (pre-rebase) | - | - | - | - | - | 10532 | 2340 | **CRASHED** |
| E | DSpark-6 + `--spec-mtp-rs-planes 4` | 53adab814 (pre-rebase) | - | - | - | - | - | - | - | OOM/REJECTED (CLI) |
| F | MTP-5, `--phase-aware-workspace` off | 53adab814 (pre-rebase) | 135 | 422.27 | 1200 | 43.41 | 87.7% (398/454, len 2.61) | 10868 | 2209 | ok |
| F | MTP-5, `--phase-aware-workspace` on | 53adab814 (pre-rebase) | 135 | 392.25 | 1200 | 42.95 | 87.7% (398/454, len 2.61) | 10702 | 2169 | ok |
| F | DSpark-6, `--phase-aware-workspace` off | 53adab814 (pre-rebase) | 135 | 327.15 | 1200 | 10.75 | 95.0% (228/240, len 2.40) | 11258 | 4047 | ok |
| F | DSpark-6, `--phase-aware-workspace` on | 53adab814 (pre-rebase) | 135 | 312.77 | 1200 | 10.74 | 95.0% (228/240, len 2.40) | 11206 | 4021 | ok |
| smoke | MTP-5 + draft ubatch 128 on rebased binary | c9f727c1e (Pass 2) | - | - | - | - | - | - | - | OOM/REJECTED (CLI) |
| A2 | baseline, short prompt (rebased) | c9f727c1e (Pass 2) | 72 | 253.59 | 64 | 39.69 | - | 9800 | 1959 | ok |
| A2 | MTP-5 (ubatch inherited=512, Pass 2), short prompt | c9f727c1e (Pass 2) | 72 | 251.80 | 64 | 70.58 | 78.9% (45/57, len 3.65) | 10966 | 2126 | ok |
| A2 | DSpark-6 (CPU draft), short prompt (rebased) | c9f727c1e (Pass 2) | 72 | 179.80 | 64 | 17.00 | 100.0% (30/30, len 2.88) | 11266 | 3877 | ok |
| A2 | baseline, creative prompt (rebased) | c9f727c1e (Pass 2) | 89 | 329.46 | 400 | 38.92 | - | 9798 | 1965 | ok |
| A2 | MTP-5 (Pass 2), creative prompt | c9f727c1e (Pass 2) | 89 | 307.08 | 400 | 46.86 | 86.7% (157/181, len 2.54) | 10964 | 2128 | ok |
| A2 | DSpark-6 (CPU draft), creative prompt (rebased) | c9f727c1e (Pass 2) | 89 | 234.23 | 400 | 11.65 | 95.4% (83/87, len 2.98) | 11276 | 3889 | ok |
| A2 | baseline, coding prompt (rebased) | c9f727c1e (Pass 2) | 135 | 453.46 | 1200 | 37.69 | - | 9800 | 1965 | ok |
| A2 | MTP-5 (Pass 2), coding prompt | c9f727c1e (Pass 2) | 135 | 422.58 | 1200 | 43.27 | 84.4% (410/486, len 2.47) | 10958 | 2132 | ok |
| A2 | DSpark-6 (CPU draft), coding prompt (rebased) | c9f727c1e (Pass 2) | 135 | 326.54 | 1200 | 10.52 | 93.8% (211/225, len 2.34) | 11278 | 3950 | ok |
| A2 | baseline, 12K synthetic (rebased) | c9f727c1e (Pass 2) | 12000 | 1147.83 | 25 | 23.50 | - | 9840 | 2010 | ok |
| A2 | MTP-5 (Pass 2), 12K synthetic | c9f727c1e (Pass 2) | 12000 | 1106.70 | 25 | 72.21 | 100.0% (20/20, len 6.00) | 11012 | 2310 | ok |
| A2 | DSpark-6 (CPU draft), 12K synthetic (rebased) | c9f727c1e (Pass 2) | 12000 | 1014.98 | 25 | 3.52 | 100.0% (11/11, len 2.57) | 11332 | 4706 | ok |
| G | MTP-6, planes=0, crash re-check 1/3 | c9f727c1e (Pass 2) | 135 | 422.93 | 1200 | 43.48 | 84.3% (420/498, len 2.57) | 11108 | 2159 | ok |
| G | MTP-6, planes=0, crash re-check 2/3 | c9f727c1e (Pass 2) | - | - | - | - | - | 11108 | 2252 | **CRASHED @954** |
| G | MTP-6, planes=0, crash re-check 3/3 | c9f727c1e (Pass 2) | - | - | - | - | - | 11108 | 2250 | **CRASHED @956** |
| G | MTP-6, planes=0, `CUDA_LAUNCH_BLOCKING=1` | c9f727c1e (Pass 2) | 135 | 383.32 | 1200 | 39.01 | 84.3% (423/502, len 2.57) | 11108 | 2166 | ok |
| H | DSpark, fixed binary, `--spec-draft-n-max` omitted (resolves to 6) | 85552567e (fix applied) | 135 | 325.80 | 1200 | 10.53 | 93.8% (211/225, len 2.34) | 11278 | 4058 | ok |

The following is the exact historical common-flag record, not a current command:
`--no-kv-offload --kv-cpu-pinned
--recurrent-state-offload --ctx-size 16384 --flash-attn on --cache-type-k
q8_0 --cache-type-v q8_0 --n-gpu-layers 99 --split-mode none --parallel 1
--cont-batching --threads 3 --threads-batch 3 --poll 100 --seed 1234
--cache-ram 0`, whole process under `taskset -c 0,2,4` (three distinct
physical P-cores on this i5-13400F; CPUs 0-1/2-3/4-5/... are SMT pairs).
DSpark rows add `--n-gpu-layers-draft 0` (see Characterization 022: DSpark's
draft model does not fit GPU-resident on this 12 GiB card). Exact per-row
commands are in `logs/<tag>.cmd`.

## Historical crash attempts

The `G-` and `smoke-mtp-ubatch-mismatch` rows are historical crash/rejection
attempts referenced in Characterization 021. The representative command snapshot
(against `c9f727c1e`, `--spec-mtp-rs-planes 0` so the plane-cap/sparse-replay
code path is not involved) is `logs/G-mtp-coding-crashcheck-2.cmd`. It
crashed 2 of 3 initial unmodified attempts. Across the full corrected-geometry
Pass 2 set, 5 of 6 non-blocking inference runs crashed. One of three eligible
`CUDA_LAUNCH_BLOCKING=1` inference runs crashed; two additional blocking
artifacts failed during model loading and are excluded from that denominator.
The `compute-sanitizer --tool memcheck` run was started but not completed (see
Characterization 021). The MTP crash remains unresolved and PR 3 does not claim
to fix it.
