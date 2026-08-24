#!/bin/bash
# D5: host-resident KVarN falls to the CPU backend and reserves host workspace
# proportional to context. Needs a GGML_CUDA_KVARN=ON build, which in turn needs
# the D7 fix (docs/probes/06-kvarn-build-fix.patch) to compile at all.
#
# Compares the reserved compute buffers for a KVarN cache against a standard
# quantized one at the same context.
. "$(dirname "$0")/common.sh"
BUILD="${LLAMA_KV_BUILD:-build-kvarn}"
CTX="${LLAMA_KV_CTX:-8192}"
# GGML_SCHED_DEBUG=2 is what names the CPU-assigned nodes, but it emits
# gigabytes. Set LLAMA_KV_SCHED_DEBUG=1 only when you want the node breakdown,
# and expect a very large log.
SCHED="${LLAMA_KV_SCHED_DEBUG:-0}"
for ct in kvarn5 q8_0; do
  LOG=$OUT/d5-$ct-c$CTX.log
  echo "=== -ctk $ct -ctv $ct -c $CTX"
  if [ "$SCHED" = 1 ]; then
    gpu env GGML_SCHED_DEBUG=2 taskset -c $PIN "$BUILD/bin/llama-cli" -m "$MODEL" \
      -ngl 99 -sm none -mg 0 -t 3 -nkvo --kv-cpu-pinned --recurrent-state-offload \
      -fa on -ctk $ct -ctv $ct -c $CTX -n 1 -p hi --no-warmup -v > "$LOG" 2>&1
  else
    gpu taskset -c $PIN "$BUILD/bin/llama-cli" -m "$MODEL" \
      -ngl 99 -sm none -mg 0 -t 3 -nkvo --kv-cpu-pinned --recurrent-state-offload \
      -fa on -ctk $ct -ctv $ct -c $CTX -n 1 -p hi --no-warmup 2>&1 \
      | grep -iE "compute buffer size|KV self size|cannot|error" | head -8 > "$LOG"
  fi
  echo "   exit=$?"
  grep -iE "compute buffer size|KV self size" "$LOG" | sort -u | head -6
done
echo
echo "The CUDA_Host compute buffer is the figure of interest. Divide the"
echo "difference between two contexts by the token delta for MiB per context token."
