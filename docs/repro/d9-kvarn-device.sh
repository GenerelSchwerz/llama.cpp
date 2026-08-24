#!/bin/bash
# D9: on a default build KVarN is refused for a device-resident cache and
# accepted only where the cache lands on the CPU backend.
# Expect: -ngl 99 fails context creation, -ngl 0 and -ngl 99 -nkvo 1 succeed.
. "$(dirname "$0")/common.sh"
run() {
  echo "--- $*"
  gpu "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n 4 -r 1 --no-warmup \
    -fa 1 -ctk kvarn4 -ctv kvarn4 -v "$@" 2>&1 \
    | grep -E "cannot enable|enabling structured|failed to create context|\| *tg4 *\|" \
    | sed -E 's/^ *//'
}
run -ngl 99
run -ngl 0
run -ngl 99 -nkvo 1
echo "--- standard cache at -ngl 0, for the CPU-arm comparison"
gpu "$BUILD/bin/llama-bench" -m "$MODEL" -p 0 -n 4 -r 1 --no-warmup \
  -fa 1 -ctk q8_0 -ctv q8_0 -ngl 0 2>&1 | grep -E "\| *tg4 *\|"
