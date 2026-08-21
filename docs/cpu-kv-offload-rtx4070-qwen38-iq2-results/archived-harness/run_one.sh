#!/usr/bin/env bash
# run_one.sh <tag> <port> <prompt_file> <max_tokens> <extra server args...>
# Starts a clean llama-server, waits for health, starts VRAM/RAM monitor,
# sends one chat completion request, records timings + monitor summary,
# then shuts the server down. Emits everything under $OUTDIR/<tag>.*
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/results"
mkdir -p "$OUTDIR"

TAG="$1"; PORT="$2"; PROMPT_FILE="$3"; MAX_TOKENS="$4"; shift 4
EXTRA_ARGS=("$@")

BIN=/home/piggidragon/Services/llama.cpp/build/bin/llama-server
MODEL=/home/piggidragon/Services/models/llama-cpp/Qwen3.8-27b/Qwen3.8-27B-UD-IQ2_M.gguf
LOG="$OUTDIR/${TAG}.log"
RESP="$OUTDIR/${TAG}.response.json"
MONPREFIX="$OUTDIR/${TAG}.mon"

echo "=== $TAG ===" | tee -a "$OUTDIR/SUMMARY.txt"
echo "cmd: $BIN --model $MODEL --port $PORT ${EXTRA_ARGS[*]}" >> "$OUTDIR/${TAG}.cmd"

taskset -c 0,2,4 "$BIN" \
  --model "$MODEL" \
  --n-gpu-layers 99 --split-mode none --flash-attn on \
  --parallel 1 --cont-batching \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 3 --threads-batch 3 --poll 100 \
  --seed 1234 --cache-ram 0 \
  --host 127.0.0.1 --port "$PORT" \
  --log-file "$LOG" --log-verbosity 4 \
  "${EXTRA_ARGS[@]}" &
SPID=$!
echo "$SPID" > "$OUTDIR/${TAG}.pid"

# wait for health
UP=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    UP=1
    break
  fi
  if ! kill -0 "$SPID" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [ "$UP" -ne 1 ]; then
  echo "FAILED to start (see $LOG)" | tee -a "$OUTDIR/SUMMARY.txt"
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null
  exit 1
fi

bash "$SCRIPT_DIR/monitor.sh" "$SPID" "$MONPREFIX" &
MONPID=$!

PROMPT_CONTENT=$(cat "$PROMPT_FILE")
PAYLOAD=$(jq -nc --arg content "$PROMPT_CONTENT" --argjson max_tokens "$MAX_TOKENS" \
  '{model:"bench", messages:[{role:"user", content:$content}], max_tokens:$max_tokens, seed:1234, stream:false, cache_prompt:false}')

START=$(date +%s.%N)
curl -sS -o "$RESP" -H 'Content-Type: application/json' --data-binary "$PAYLOAD" "http://127.0.0.1:${PORT}/v1/chat/completions"
END=$(date +%s.%N)
WALL=$(awk -v a="$END" -v b="$START" 'BEGIN{printf "%.3f", a-b}')

echo "wall_s: $WALL" >> "$OUTDIR/${TAG}.cmd"
python3 -c "
import json
try:
    d = json.load(open('$RESP'))
    t = d.get('timings', {})
    u = d.get('usage', {})
    print('prompt_tokens:', u.get('prompt_tokens'))
    print('completion_tokens:', u.get('completion_tokens'))
    print('prompt_per_second:', t.get('prompt_per_second'))
    print('predicted_per_second:', t.get('predicted_per_second'))
    print('predicted_n:', t.get('predicted_n'))
    if 'error' in d:
        print('ERROR:', d['error'])
except Exception as e:
    print('parse error:', e)
" | tee -a "$OUTDIR/SUMMARY.txt"

kill "$MONPID" 2>/dev/null
wait "$MONPID" 2>/dev/null
echo "vram/rss summary:" >> "$OUTDIR/SUMMARY.txt"
python3 "$SCRIPT_DIR/summarize.py" "${MONPREFIX}.csv" >> "$OUTDIR/SUMMARY.txt" 2>&1

kill "$SPID" 2>/dev/null
wait "$SPID" 2>/dev/null
sleep 1
echo "server print_timing / memory breakdown:" >> "$OUTDIR/SUMMARY.txt"
grep -E "print_timing|memory breakdown|DFlash: omitted|invalid speculative|replay|checkpoint" "$LOG" | tail -20 >> "$OUTDIR/SUMMARY.txt"
echo "" >> "$OUTDIR/SUMMARY.txt"
