#!/bin/bash
# D6: the quantized + logit_softcap coverage gap.
# Prints how many softcap cases exist and how many of them use a quantized cache.
. "$(dirname "$0")/common.sh"
LOG=$OUT/d6-softcap.log
gpu "$BUILD/bin/test-backend-ops" -o FLASH_ATTN_EXT -p "logit_softcap=10.000000" > "$LOG" 2>&1
echo "exit=$?"
echo "softcap cases:            $(grep -cE '^ *FLASH_ATTN_EXT' "$LOG")"
echo "of those, quantized K:    $(grep -E '^ *FLASH_ATTN_EXT' "$LOG" | grep -c 'type_K=q')"
grep -E "^ *FLASH_ATTN_EXT" "$LOG" | grep 'type_K=q' | cut -c1-190
