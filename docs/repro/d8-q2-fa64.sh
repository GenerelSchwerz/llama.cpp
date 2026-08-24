#!/bin/bash
# D8: q2_0s FlashAttention returns wrong values at hsk=64, kv=256, nb=2.
# Expect one FAIL with ERR around 1.7 among 2,673 hsk=64 cases, and the same
# failure at three different seeds. No model is needed.
. "$(dirname "$0")/common.sh"
CASE='hsk=64,hsv=64,nh=4,nr23=\[2,1\],kv=256,nb=2,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q2_0s,type_V=q2_0s'

echo "--- full hsk=64 sweep"
gpu "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT -b CUDA0 -p "hsk=64," 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -E "ERR =|selected_cases|Backend CUDA0:"

echo "--- the failing case alone, three seeds"
for seed in 0 1 42; do
  printf 'seed=%-3s ' "$seed"
  gpu "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT -b CUDA0 -p "$CASE" --seed "$seed" 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "ERR = [0-9.]+ > [0-9.]+" | head -1
done

echo "--- same shape, every other quantized pair the suite generates"
gpu "$BUILD/bin/test-backend-ops" test -o FLASH_ATTN_EXT -b CUDA0 -p "hsk=64," 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -E "kv=256,nb=2," \
  | sed -E 's/.*type_K=([a-z0-9_]*),type_V=([a-z0-9_]*).*: (OK|FAIL|not supported).*/\1 \2 => \3/' \
  | sort -u
