#!/bin/bash
# D4: test-kvarn aborts on a default build (GGML_CUDA_KVARN off) with a CUDA device.
. "$(dirname "$0")/common.sh"
gpu ctest --test-dir "$BUILD" -R "^test-kvarn$" --output-on-failure 2>&1 \
  | grep -E "Passed|Subprocess|not supported|GGML_ASSERT|tests (passed|failed)"
