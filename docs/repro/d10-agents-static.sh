#!/bin/bash
# D10: test-upstream-merge-keepers-static fails on a clean checkout.
# Needs no GPU, no model and no build; runs in well under a second.
set -u
BUILD="${LLAMA_KV_BUILD:-build-clean}"
ctest --test-dir "$BUILD" -R "^test-upstream-merge-keepers-static$" --output-on-failure 2>&1 \
  | grep -E "AssertionError|Passed|Failed|tests passed"
echo "--- the string the test requires, as AGENTS.md carries it:"
grep -n -A1 "50 standard vector" AGENTS.md
