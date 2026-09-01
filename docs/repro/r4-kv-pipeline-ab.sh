#!/bin/bash
# R4: pipelined delivery of a host-resident KV cache, A/B/A/B with reversed arm order.
# The two arms are the same binary: --kv-pipeline-depth 0 is the ordered path.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-ab.sh [depth ...]
set -euo pipefail
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
BUDGET="${LLAMA_KV_BUDGET:-512}"
LOCK=/tmp/beellama-single-gpu.lock

# An unpinned host cache and a host-resident recurrent state both cost more than the transport can
# win back, and without a budget the ring is declined at the larger contexts, so a build without
# these options does not measure what the doc reports. Fail rather than measure something else.
if ! HELP="$("$BUILD/bin/llama-bench" --help 2>&1)"; then
  echo "cannot run $BUILD/bin/llama-bench:" >&2
  echo "$HELP" >&2
  exit 1
fi
for opt in kvcp rso kvpb; do
  if ! grep -q -- "-$opt," <<< "$HELP"; then
    echo "$BUILD/bin/llama-bench has no -$opt option" >&2
    exit 1
  fi
done

run () { # $1 label, $2 pipeline depth, $3 context depth, $4 reps
  local out err rc
  out="$(mktemp)"
  err="$(mktemp)"
  rc=0
  taskset -c "$PIN" "$BUILD/bin/llama-bench" -m "$MODEL" --kv-pipeline-depth "$2" \
    --kv-pipeline-budget "$BUDGET" -ngl 99 -sm none -mg 0 -t 3 -nkvo 1 -kvcp 1 -rso 1 \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --no-warmup -p 0 -n 128 -d "$3" -r "$4" -o json \
    > "$out" 2> "$err" || rc=$?
  if [ "$rc" -eq 0 ]; then
    python3 -c "import json,sys;d=json.load(sys.stdin);print('  %-12s %.4f +- %.4f'%('$1',d[0]['avg_ts'],d[0]['stddev_ts']))" \
      < "$out" 2>/dev/null || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  $1: FAILED" >&2
    cat "$err" >&2
  fi
  rm -f "$out" "$err"
  return "$rc"
}

DEPTHS=(4096 16384 32768)
if [ $# -gt 0 ]; then
  DEPTHS=("$@")
fi
for D in "${DEPTHS[@]}"; do
  R=3
  if [ "$D" -le 4096 ]; then
    R=5
  fi
  echo "== context depth=$D reps=$R"
  flock "$LOCK" bash -c "set -euo pipefail; $(declare -f run); BUILD='$BUILD'; MODEL='$MODEL'; PIN='$PIN'; BUDGET='$BUDGET'
    run ordered    0 $D $R
    run pipelined  1 $D $R
    run ordered2   0 $D $R
    run pipelined2 1 $D $R"
done
