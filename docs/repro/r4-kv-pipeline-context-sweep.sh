#!/bin/bash
# R4 across context depth: throughput and device allocation high-water, ordered against
# pipelined, on the same binary. The ring holds one split's whole delivery per slot, so its
# cost grows with the context; this is what measures where that stops being affordable.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-context-sweep.sh [depth ...]
set -u
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
NGEN="${LLAMA_KV_NGEN:-64}"
LOCK=/tmp/beellama-single-gpu.lock

# An unpinned host cache and a host-resident recurrent state both cost more than the transport
# can win back, so a run without these does not measure the same thing. Older llama-bench builds
# do not have them; skip rather than fail, and the numbers are then not comparable to the doc.
BENCH_KV_OPTS=""
for opt in kvcp rso; do
  "$BUILD/bin/llama-bench" --help 2>&1 | grep -q -- "-$opt," && BENCH_KV_OPTS="$BENCH_KV_OPTS -$opt 1"
done

arm () { # $1 pipeline depth, $2 context depth, $3 reps
  local vram; vram=$(mktemp)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > "$vram" 2>/dev/null &
  local sampler=$!
  local ts
  ts=$(taskset -c "$PIN" "$BUILD/bin/llama-bench" -m "$MODEL" --kv-pipeline-depth "$1" \
        -ngl 99 -sm none -mg 0 -t 3 -nkvo 1 $BENCH_KV_OPTS \
        -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --no-warmup -p 0 -n "$NGEN" -d "$2" -r "$3" -o json \
        2>/dev/null \
      | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print('%.4f'%d[0]['avg_ts'])
except Exception:
    print('FAILED')")
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  printf '  %-10s %-10s %s MiB\n' "depth=$1" "$ts" "$(sort -n "$vram" | tail -1)"
  rm -f "$vram"
}

DEPTHS=(4096 16384 32768 65536 131072 262144); [ $# -gt 0 ] && DEPTHS=("$@")
for D in "${DEPTHS[@]}"; do
  R=3; [ "$D" -gt 32768 ] && R=1
  echo "== context depth=$D reps=$R  (t/s, peak device memory)"
  flock "$LOCK" bash -c "$(declare -f arm); BUILD='$BUILD'; MODEL='$MODEL'; PIN='$PIN'; BENCH_KV_OPTS='$BENCH_KV_OPTS'; NGEN='$NGEN'
    arm 0 $D $R
    arm 1 $D $R
    arm 0 $D $R
    arm 1 $D $R"
done
