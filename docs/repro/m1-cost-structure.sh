#!/bin/bash
# The decode cost structure. Needs patches 01 and 02.
# Differences tg128 against tg64 at fixed depth so prefill and cache population cancel.
. "$(dirname "$0")/common.sh"
for D in "${@:-4096 16384 32768}"; do
  for N in 64 128; do
    echo "== d=$D n=$N"
    gpu env GGML_KV_TRANSPORT_STATS=1 taskset -c $PIN "$BUILD/bin/llama-bench" -m "$MODEL" \
      -ngl 99 -sm none -mg 0 -t 3 -nkvo 1 --kv-cpu-pinned --recurrent-state-offload \
      -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --no-warmup -p 0 -n $N -d $D -r 1 2>&1 \
      | grep -E "KV-TRANSPORT"
  done
done
echo "Subtract the n=64 line from the n=128 line and divide by 64 for per-token figures."
