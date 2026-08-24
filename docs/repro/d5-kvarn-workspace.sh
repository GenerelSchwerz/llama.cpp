#!/bin/bash
# D5: host-resident KVarN falls to the CPU backend and reserves host workspace
# proportional to context. Needs a GGML_CUDA_KVARN=ON build.
# Compare the two arms' "compute buffer" lines.
. "$(dirname "$0")/common.sh"
BUILD="${LLAMA_KV_BUILD:-build-kvarn}"
CTX="${LLAMA_KV_CTX:-8192}"
for ct in kvarn5 q8_0; do
  LOG=$OUT/d5-$ct-c$CTX.log
  echo "=== -ctk $ct -ctv $ct  -c $CTX"
  gpu env GGML_SCHED_DEBUG=2 taskset -c $PIN "$BUILD/bin/llama-cli" -m "$MODEL" \
    -ngl 99 -sm none -mg 0 -t 3 -nkvo --kv-cpu-pinned --recurrent-state-offload \
    -fa on -ctk $ct -ctv $ct -c $CTX -n 1 -p hi --no-warmup -v > "$LOG" 2>&1
  echo "   exit=$?"
  grep -iE "compute buffer size|CPU_Mapped|KV self size" "$LOG" | head -5
  echo "   CPU-assigned node ops:"
  grep -oE "^[A-Z_]+ +#" "$LOG" | sort | uniq -c | sort -rn | head -5
done
