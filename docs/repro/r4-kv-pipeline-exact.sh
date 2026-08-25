#!/bin/bash
# R4 gate 1: greedy server output must be byte-identical to the ordered path.
# Compare the hashes across pipeline depths, and against a build of the parent commit.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-exact.sh [depth ...]
set -u
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
PORT="${LLAMA_KV_PORT:-18099}"
NTOK="${LLAMA_KV_NTOK:-256}"
NPARA="${LLAMA_KV_NPARA:-400}"
HERE="$(cd "$(dirname "$0")" && pwd)"

DEPTHS=(0 1 4); [ $# -gt 0 ] && DEPTHS=("$@")
rc=0
for D in "${DEPTHS[@]}"; do
  echo "== pipeline depth=$D"
  LOG=$(mktemp /tmp/r4-kv-pipeline.XXXX.log)
  GGML_KV_PIPELINE_DEPTH=$D taskset -c "$PIN" "$BUILD/bin/llama-server" -m "$MODEL" \
    -ngl 99 -sm none -mg 0 -t 3 -nkvo --kv-cpu-pinned --recurrent-state-offload \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -c 32768 --parallel 1 \
    --host 127.0.0.1 --port "$PORT" --no-warmup > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  python3 "$HERE/r4-kv-pipeline-exact.py" "$PORT" "$NTOK" "$NPARA" || rc=$?
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  rm -f "$LOG"
done
exit $rc
