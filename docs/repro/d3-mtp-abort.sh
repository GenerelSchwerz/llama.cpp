#!/bin/bash
# D3: MTP with a host-resident KV cache aborts during sustained generation.
# Intermittent: expect at least one of three runs to die with an illegal memory access.
. "$(dirname "$0")/common.sh"
PORT="${LLAMA_KV_PORT:-8091}"
RUNS="${LLAMA_KV_RUNS:-3}"
for run in $(seq 1 "$RUNS"); do
  LOG=$OUT/d3-run$run.log
  gpu bash -c "
    taskset -c $PIN '$BUILD/bin/llama-server' -m '$MODEL' --port $PORT \
      -ngl 99 -sm none -mg 0 -t 3 -tb 3 --seed 1234 \
      -nkvo --kv-cpu-pinned --recurrent-state-offload -fa on \
      -ctk q8_0 -ctv q8_0 -c 8192 -b 512 -ub 512 --parallel 1 --cache-ram 0 \
      --spec-type draft-mtp --spec-draft-n-max 5 --draft-p-min 0.85 \
      --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 > '$LOG' 2>&1 &
    P=\$!
    for i in \$(seq 1 400); do sleep 2
      [ \"\$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PORT/health || echo 000)\" = 200 ] && break
      kill -0 \$P 2>/dev/null || break
    done
    echo -n 'run$run: '
    python3 '$(dirname "$0")/request.py' $PORT 900
    kill \$P 2>/dev/null; wait \$P 2>/dev/null
  "
  grep -m1 -oE "illegal memory access" "$LOG" && \
    grep -m1 -oE "n_decoded = *[0-9]+" "$LOG" | tail -1
  sleep 3
done
