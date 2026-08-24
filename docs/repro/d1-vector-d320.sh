#!/bin/bash
# D1: head dimension 320 is routed to a vector kernel that has no D320 instance.
# Expect exit 134 and "fattn.cu:412: fatal error" in ggml_cuda_flash_attn_ext_vec.
. "$(dirname "$0")/common.sh"
LOG=$OUT/d1-vector-d320.log
gpu "$BUILD/bin/test-backend-ops" -o FLASH_ATTN_EXT > "$LOG" 2>&1
rc=$?
echo "exit=$rc  (expected 134)"
echo "--- aborting case:"
grep -E "^ *FLASH_ATTN_EXT" "$LOG" | tail -1 | cut -c1-200
echo "--- abort site:"
grep -E "fattn\.cu:[0-9]+: fatal error|flash_attn_ext_vec" "$LOG" | head -2
