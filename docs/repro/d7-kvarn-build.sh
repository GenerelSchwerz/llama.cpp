#!/bin/bash
# D7: GGML_CUDA_KVARN=ON does not compile on 946c1e5b6.
# Expect 82 instances of "no instance of function template" at
# fattn-mma-kvarn-case.cuh(178) and a non-zero exit.
. "$(dirname "$0")/common.sh"
TREE="${LLAMA_KV_KVARN_TREE:-build-kvarn-probe}"
LOG=$OUT/d7-kvarn-build.log
rm -rf "$TREE"
cmake -B "$TREE" -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_KVARN=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_KV_ARCH:-89}" -DCMAKE_BUILD_TYPE=Release > "$LOG" 2>&1
echo "configure=$?"
cmake --build "$TREE" --parallel "${LLAMA_KV_JOBS:-8}" >> "$LOG" 2>&1
echo "build=$?  (expected non-zero)"
echo "--- template mismatch errors: $(grep -c 'no instance of function template' "$LOG")"
echo "--- call sites:"; grep -oE "fattn-mma-kvarn-case\.cuh\([0-9]+\)" "$LOG" | sort -u
echo
echo "The fix is two lines: docs/probes/06-kvarn-build-fix.patch"
