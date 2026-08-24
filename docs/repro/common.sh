# shared settings; source this from the scripts in this directory
set -u
MODEL="${LLAMA_KV_MODEL:?set LLAMA_KV_MODEL to a .gguf path}"
BUILD="${LLAMA_KV_BUILD:-build-clean}"
LOCK=/tmp/beellama-single-gpu.lock
PIN="${LLAMA_KV_TASKSET:-0,2,4}"
OUT="${LLAMA_KV_OUT:-$(pwd)/kv-offload-repro-out}"
mkdir -p "$OUT"
gpu() { flock "$LOCK" "$@"; }
