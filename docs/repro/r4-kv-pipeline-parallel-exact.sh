#!/bin/bash
# R4 gate 2: greedy output of concurrent sequences over a cache split into streams must be byte-identical to the ordered path.
# Gate 1 covers one sequence per ubatch, where a delivery is a single range. This covers a ubatch that spans several streams, where a delivery is one range per stream and the cells between them are never read.
# llama-parallel is used rather than the server because it seeds its client schedule, so the batches are the same run to run and the outputs can be compared directly.
#
#   LLAMA_KV_MODEL=/path/model.gguf docs/repro/r4-kv-pipeline-parallel-exact.sh [pipeline-depth ...]
#   LLAMA_KV_NP=8 sets the concurrent sequences, LLAMA_KV_NS=16 the total.
#   LLAMA_KV_SM=layer spreads the model over every device, which gives each of them its own ring.
set -euo pipefail
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build}"
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
CTX="${LLAMA_KV_CTX:-16384}"
BUDGET="${LLAMA_KV_BUDGET:-512}"
NP="${LLAMA_KV_NP:-8}"
NS="${LLAMA_KV_NS:-16}"
SM="${LLAMA_KV_SM:-none}"

DEPTHS=(0 1 4); [ $# -gt 0 ] && DEPTHS=("$@")

# Without a pinned host cache, a host-resident recurrent state and a budget the run does not exercise the path the doc reports on.
if ! HELP="$("$BUILD/bin/llama-parallel" --help 2>&1)"; then
  echo "cannot run $BUILD/bin/llama-parallel:" >&2
  echo "$HELP" >&2
  exit 1
fi
for opt in --kv-cpu-pinned --recurrent-state-offload --kv-pipeline-depth --kv-pipeline-budget --no-kv-unified; do
  if ! grep -q -- "$opt" <<< "$HELP"; then
    echo "$BUILD/bin/llama-parallel has no $opt option" >&2
    exit 1
  fi
done

# Keep only what the clients produced: drop the log timestamps, the colour codes and the timing summary, all of which differ between runs by design.
transcript () {
  sed -e 's/\x1b\[[0-9;]*m//g' \
      -e '/^[0-9][0-9.]* [A-Z] /d' \
      -e '/speed:/d' -e '/^Cache misses/d' "$1"
}

rc=0
BASE=""
for I in "${!DEPTHS[@]}"; do
  D="${DEPTHS[$I]}"
  echo "== pipeline depth=$D  ctx=$CTX  np=$NP  ns=$NS  sm=$SM  streams (non-unified)"
  RAW="$(mktemp /tmp/r4-kv-parallel.XXXX.log)"
  OUT="$(mktemp /tmp/r4-kv-parallel.XXXX.txt)"
  arm_rc=0
  taskset -c "$PIN" "$BUILD/bin/llama-parallel" -m "$MODEL" \
    --kv-pipeline-depth "$D" --kv-pipeline-budget "$BUDGET" \
    -ngl 99 -sm "$SM" -mg 0 -t 3 -nkvo --kv-cpu-pinned --recurrent-state-offload \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -c "$CTX" -no-kvu \
    -np "$NP" -ns "$NS" --temp 0 > "$RAW" 2>&1 || arm_rc=$?
  if [ "$arm_rc" -ne 0 ]; then
    echo "  depth $D: FAILED (exit $arm_rc)" >&2
    tail -20 "$RAW" >&2
    rm -f "$RAW" "$OUT"
    exit "$arm_rc"
  fi
  transcript "$RAW" > "$OUT"
  if [ ! -s "$OUT" ]; then
    echo "  depth $D: FAILED (no client output)" >&2
    rm -f "$RAW" "$OUT"
    exit 1
  fi
  echo "  $(sha256sum < "$OUT" | cut -c1-16)  $(wc -l < "$OUT") lines"
  if [ "$I" -eq 0 ]; then
    BASE="$OUT"
  else
    if ! cmp -s "$BASE" "$OUT"; then
      diff -u "$BASE" "$OUT" | head -40
      rc=1
    fi
    rm -f "$OUT"
  fi
  rm -f "$RAW"
done
rm -f "$BASE"
exit $rc
