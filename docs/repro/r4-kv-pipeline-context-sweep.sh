#!/bin/bash
# R4 across context depth: throughput and device allocation high-water, ordered against
# pipelined, on the same binary. The ring holds one split's whole delivery per slot, so its
# cost grows with the context; this is what measures where that stops being affordable.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-context-sweep.sh [depth ...]
set -euo pipefail
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
NGEN="${LLAMA_KV_NGEN:-64}"
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

arm () { # $1 pipeline depth, $2 context depth, $3 reps
  local vram out err rc ts
  vram="$(mktemp)"
  out="$(mktemp)"
  err="$(mktemp)"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > "$vram" 2>/dev/null &
  local sampler=$!
  rc=0
  taskset -c "$PIN" "$BUILD/bin/llama-bench" -m "$MODEL" --kv-pipeline-depth "$1" \
    --kv-pipeline-budget "$BUDGET" -ngl 99 -sm none -mg 0 -t 3 -nkvo 1 -kvcp 1 -rso 1 \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --no-warmup -p 0 -n "$NGEN" -d "$2" -r "$3" -o json \
    > "$out" 2> "$err" || rc=$?
  kill $sampler 2>/dev/null || true
  wait $sampler 2>/dev/null || true
  ts=""
  if [ "$rc" -eq 0 ]; then
    ts="$(python3 -c "import json,sys;d=json.load(sys.stdin);print('%.4f'%d[0]['avg_ts'])" < "$out" 2>/dev/null)" || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  depth=$1: FAILED" >&2
    cat "$err" >&2
    rm -f "$vram" "$out" "$err"
    return "$rc"
  fi
  printf '  %-10s %-10s %s MiB\n' "depth=$1" "$ts" "$(sort -n "$vram" | tail -1)"
  rm -f "$vram" "$out" "$err"
}

DEPTHS=(4096 16384 32768 65536 131072 262144)
if [ $# -gt 0 ]; then
  DEPTHS=("$@")
fi
for D in "${DEPTHS[@]}"; do
  R=3
  if [ "$D" -gt 32768 ]; then
    R=1
  fi
  echo "== context depth=$D reps=$R  (t/s, peak device memory)"
  flock "$LOCK" bash -c "set -euo pipefail; $(declare -f arm); BUILD='$BUILD'; MODEL='$MODEL'; PIN='$PIN'; BUDGET='$BUDGET'; NGEN='$NGEN'
    arm 0 $D $R
    arm 1 $D $R
    arm 0 $D $R
    arm 1 $D $R"
done
