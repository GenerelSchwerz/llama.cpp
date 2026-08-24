#!/bin/bash
# D2: divergent __syncthreads() in upstream's MMA FlashAttention kernel.
# Needs a -lineinfo build; --show-backtrace device is what yields file:line.
# Runs ~40 minutes under the sanitizer.
. "$(dirname "$0")/common.sh"
BUILD="${LLAMA_KV_BUILD:-build-li}"
LOG=$OUT/d2-barrier.log
PORT="${LLAMA_KV_PORT:-8093}"
gpu bash -c "
compute-sanitizer --tool synccheck --launch-timeout 600 --show-backtrace device \
  '$BUILD/bin/llama-server' -m '$MODEL' --port $PORT -ngl 99 -sm none -mg 0 -t 3 -tb 3 \
  --seed 1234 -nkvo --kv-cpu-pinned --recurrent-state-offload -fa on \
  -ctk q8_0 -ctv q8_0 -c 8192 -b 512 -ub 512 --parallel 1 --cache-ram 0 \
  --spec-type draft-mtp --spec-draft-n-max 5 --draft-p-min 0.85 \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 > '$LOG' 2>&1 &
P=\$!
for i in \$(seq 1 900); do sleep 2
  [ \"\$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PORT/health || echo 000)\" = 200 ] && break
  kill -0 \$P 2>/dev/null || break
done
python3 '$(dirname "$0")/request.py' $PORT 900
kill \$P 2>/dev/null; wait \$P 2>/dev/null
"
echo "--- barrier errors: $(grep -c 'Barrier error' "$LOG")"
echo "--- distinct sites:"
grep -A1 "Barrier error" "$LOG" | grep "at void" | sed 's/.*in //' | sort | uniq -c
echo "--- distinct instantiations:"
grep -A1 "Barrier error" "$LOG" | grep -o "flash_attn_ext_f16_process_tile<[^>]*>" | sort -u
