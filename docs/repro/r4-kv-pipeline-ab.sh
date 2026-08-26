#!/bin/bash
# R4: pipelined delivery of a host-resident KV cache, A/B/A/B with reversed arm order.
# The two arms are the same binary: GGML_KV_PIPELINE_DEPTH=0 is the ordered path.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-ab.sh [depth ...]
set -u
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
LOCK=/tmp/beellama-single-gpu.lock

# llama-bench does not expose every host-KV option that llama-server does; pass only what it takes
BENCH_KV_OPTS=""
for opt in --kv-cpu-pinned --recurrent-state-offload; do
  "$BUILD/bin/llama-bench" --help 2>&1 | grep -q -- "$opt" && BENCH_KV_OPTS="$BENCH_KV_OPTS $opt"
done

run () { # $1 label, $2 pipeline depth, $3 context depth, $4 reps
  GGML_KV_PIPELINE_DEPTH=$2 taskset -c "$PIN" "$BUILD/bin/llama-bench" -m "$MODEL" \
    -ngl 99 -sm none -mg 0 -t 3 -nkvo 1 $BENCH_KV_OPTS \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --no-warmup -p 0 -n 128 -d "$3" -r "$4" -o json \
    2>/dev/null \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('  %-12s %.4f +- %.4f'%('$1',d[0]['avg_ts'],d[0]['stddev_ts']))"
}

DEPTHS=(4096 16384 32768); [ $# -gt 0 ] && DEPTHS=("$@")
rc=0
for D in "${DEPTHS[@]}"; do
  R=3; [ "$D" -le 4096 ] && R=5
  echo "== context depth=$D reps=$R"
  flock "$LOCK" bash -c "$(declare -f run); BUILD='$BUILD'; MODEL='$MODEL'; PIN='$PIN'; BENCH_KV_OPTS='$BENCH_KV_OPTS'
    run ordered    0 $D $R
    run pipelined  1 $D $R
    run ordered2   0 $D $R
    run pipelined2 1 $D $R" || rc=$?
done
exit $rc
