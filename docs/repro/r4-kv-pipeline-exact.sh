#!/bin/bash
# R4 gate 1: greedy server output must be byte-identical to the ordered path, across several
# prefill corpora and prefill lengths. The script compares every requested depth with the first.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-exact.sh [pipeline-depth ...]
#   LLAMA_KV_LENGTHS=2048,18432,65536 selects the prefill lengths (default 2048,18432).
set -u
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
PORT="${LLAMA_KV_PORT:-18099}"
LENGTHS="${LLAMA_KV_LENGTHS:-2048,18432}"
CTX="${LLAMA_KV_CTX:-32768}"
HERE="$(cd "$(dirname "$0")" && pwd)"

DEPTHS=(0 1 4); [ $# -gt 0 ] && DEPTHS=("$@")
rc=0
BASE=""
for D in "${DEPTHS[@]}"; do
  echo "== pipeline depth=$D  ctx=$CTX  prefill lengths=$LENGTHS"
  LOG=$(mktemp /tmp/r4-kv-pipeline.XXXX.log)
  GGML_KV_PIPELINE_DEPTH=$D taskset -c "$PIN" "$BUILD/bin/llama-server" -m "$MODEL" \
    -ngl 99 -sm none -mg 0 -t 3 -nkvo --kv-cpu-pinned --recurrent-state-offload \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -c "$CTX" --parallel 1 \
    --host 127.0.0.1 --port "$PORT" --no-warmup > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  OUT=$(mktemp /tmp/r4-kv-pipeline.XXXX.hashes)
  if python3 "$HERE/r4-kv-pipeline-exact.py" "$PORT" "$LENGTHS" "$OUT"; then
    if [ -z "$BASE" ]; then
      BASE="$OUT"
    elif ! cmp -s "$BASE" "$OUT"; then
      diff -u "$BASE" "$OUT"
      rc=1
    fi
  else
    rc=$?
  fi
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
  rm -f "$LOG"
  [ "$OUT" = "$BASE" ] || rm -f "$OUT"
done
[ -z "$BASE" ] || rm -f "$BASE"
exit $rc
