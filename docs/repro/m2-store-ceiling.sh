#!/bin/bash
# Upper bound on removing the host KV store. Needs patch 03. Output is incorrect.
. "$(dirname "$0")/common.sh"
run () {
  env $2 taskset -c $PIN "$BUILD/bin/llama-bench" -m "$MODEL" -ngl 99 -sm none -mg 0 -t 3 \
    -nkvo 1 --kv-cpu-pinned --recurrent-state-offload -fa on -ctk q8_0 -ctv q8_0 \
    -b 512 -ub 512 --no-warmup -p 0 -n 128 -d $3 -r $4 -o json 2>/dev/null \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('  %-10s %.4f +- %.4f'%('$1',d[0]['avg_ts'],d[0]['stddev_ts']))"
}
for D in "${@:-4096 32768}"; do
  R=3; [ "$D" -le 4096 ] && R=5
  echo "== depth=$D reps=$R"
  gpu bash -c "$(declare -f run); BUILD='$BUILD'; MODEL='$MODEL'; PIN='$PIN'
    run store    'GGML_KV_NONE=1'            $D $R
    run removed  'GGML_KV_STORE_CEILING=1'   $D $R
    run store2   'GGML_KV_NONE=1'            $D $R
    run removed2 'GGML_KV_STORE_CEILING=1'   $D $R"
done
