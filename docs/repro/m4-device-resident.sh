#!/bin/bash
# The compute floor: same model, cache type and depth with the KV device-resident.
# No patch needed. Subtracting this from the host-resident arm gives the copy term.
. "$(dirname "$0")/common.sh"
for D in "${@:-4096 16384 32768}"; do
  echo "== depth=$D"
  gpu taskset -c $PIN "$BUILD/bin/llama-bench" -m "$MODEL" -ngl 99 -sm none -mg 0 -t 3 \
    -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 --no-warmup -p 0 -n 128 -d $D -r 3 -o json 2>/dev/null \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('  device_kv %.4f +- %.4f'%(d[0]['avg_ts'],d[0]['stddev_ts']))"
done
