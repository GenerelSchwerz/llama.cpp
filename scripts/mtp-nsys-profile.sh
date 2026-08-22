#!/usr/bin/env bash

set -euo pipefail

if (( $# != 3 )); then
    echo "usage: $0 LABEL gpu|cpu PLANES" >&2
    exit 2
fi

label=$1
residency=$2
planes=$3

if [[ ! $label =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "LABEL may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi
if [[ $residency != gpu && $residency != cpu ]]; then
    echo "residency must be gpu or cpu" >&2
    exit 2
fi
if [[ ! $planes =~ ^[0-9]+$ ]] || (( planes != 0 && (planes < 2 || planes > 7) )); then
    echo "PLANES must be 0 or a total recurrent-plane count in 2..7 for MTP-6" >&2
    exit 2
fi

worktree=/home/gencoolpc/beellama-mtp-exact
server=$worktree/build-mtp-exact/bin/llama-server
model=/home/gencoolpc/llm_models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ4_XS-IQ3_S.gguf
request_source=$worktree/scripts/mtp-exactness-manifests/requests/qwen38-orbital-5k.json
artifact_dir=/tmp/mtp-exact-nsys-$label-20260819

for required in "$server" "$model" "$request_source"; do
    if [[ ! -f $required ]]; then
        echo "missing required input: $required" >&2
        exit 1
    fi
done
if [[ -e $artifact_dir ]]; then
    echo "artifact path already exists: $artifact_dir" >&2
    exit 1
fi
mkdir -p "$artifact_dir"
cp "$request_source" "$artifact_dir/request.json"

port=$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)

residency_args=()
if [[ $residency == cpu ]]; then
    residency_args+=(
        --no-kv-offload
        --kv-cpu-pinned
        --recurrent-state-offload
    )
fi

server_args=(
    --model "$model"
    --n-gpu-layers 999
    --n-gpu-layers-draft 999
    --fit off
    --split-mode none
    --main-gpu 0
    --flash-attn on
    --ctx-size 8192
    --parallel 1
    --cont-batching
    --kv-unified
    --batch-size 1024
    --ubatch-size 512
    --cache-type-k q8_0
    --cache-type-v q8_0
    --threads 3
    --threads-batch 24
    --cpu-range 0-2
    --cpu-range-batch 0-23
    --cpu-strict 1
    --poll 100
    --seed 1234
    --cache-ram 0
    --verbosity 4
    --phase-aware-workspace
    --spec-type draft-mtp
    --spec-draft-n-max 6
    --spec-mtp-rs-planes "$planes"
    --spec-draft-p-min 0.85
    --spec-draft-type-k q8_0
    --spec-draft-type-v q8_0
    --alias "$label"
    --host 127.0.0.1
    --port "$port"
    "${residency_args[@]}"
)

# Keep Nsight collectors and helpers unrestricted. The server arguments above
# apply affinity only to llama.cpp worker pools; do not wrap either side of
# this command in taskset.
profile_command=(
    nsys profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --cpuctxsw=none
    --force-overwrite=true
    --output="$artifact_dir/trace"
    "$server"
    "${server_args[@]}"
)

printf '%q ' env -i "PATH=$PATH" LC_ALL=C CUDA_PATH=/opt/cuda "${profile_command[@]}" \
    > "$artifact_dir/nsys-command.txt"
printf '\n' >> "$artifact_dir/nsys-command.txt"

{
    printf 'started_utc=%s\n' "$(date --utc --iso-8601=seconds)"
    printf 'label=%s\nresidency=%s\nplanes=%s\nport=%s\n' \
        "$label" "$residency" "$planes" "$port"
    printf 'worktree=%s\n' "$worktree"
    printf 'head=%s\n' "$(git -C "$worktree" rev-parse HEAD)"
    printf 'branch=%s\n' "$(git -C "$worktree" branch --show-current)"
    printf 'dirty_diff_sha256='; git -C "$worktree" diff --binary HEAD -- | sha256sum | cut -d' ' -f1
    sha256sum "$server" "$model" "$request_source" "$worktree/scripts/mtp-nsys-profile.sh"
    nsys --version
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,compute_cap \
        --format=csv,noheader,nounits
    lscpu
    ccache --show-config
    ccache --show-stats
} > "$artifact_dir/provenance.txt"
git -C "$worktree" status --porcelain=v1 --untracked-files=all \
    > "$artifact_dir/git-status.txt"
ldd "$server" > "$artifact_dir/server-ldd.txt"

nsys_pid=
server_pid=
progress_pid=

cleanup() {
    if [[ -n ${progress_pid:-} ]]; then
        kill "$progress_pid" 2>/dev/null || true
        wait "$progress_pid" 2>/dev/null || true
    fi
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
    fi
    if [[ -n ${nsys_pid:-} ]] && kill -0 "$nsys_pid" 2>/dev/null; then
        wait "$nsys_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

env -i "PATH=$PATH" LC_ALL=C CUDA_PATH=/opt/cuda \
    "${profile_command[@]}" > "$artifact_dir/server.log" 2>&1 &
nsys_pid=$!
printf '%s\n' "$nsys_pid" > "$artifact_dir/nsys.pid"
echo "[$label] nsys pid=$nsys_pid port=$port; waiting for health" >&2

ready=0
for second in {1..600}; do
    if curl --fail --silent "http://127.0.0.1:$port/health" \
        > "$artifact_dir/health.json" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$nsys_pid" 2>/dev/null; then
        echo "[$label] profiler/server exited during startup" >&2
        tail -n 100 "$artifact_dir/server.log" >&2
        exit 1
    fi
    if (( second % 5 == 0 )); then
        echo "[$label] startup elapsed=${second}s" >&2
    fi
    sleep 1
done
if (( ! ready )); then
    echo "[$label] health timeout" >&2
    exit 1
fi

server_pid=$(pgrep -n -f "^$server .*--port $port( |$)")
printf '%s\n' "$server_pid" > "$artifact_dir/server.pid"

sample_resources() {
    local elapsed=$1
    local vram=0
    local rss_kib=0
    local hwm_kib=0
    local gpu_row
    gpu_row=$(nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits | awk -F, -v pid="$server_pid" \
        '$1 + 0 == pid {gsub(/ /, "", $2); print $2}')
    if [[ $gpu_row =~ ^[0-9]+$ ]]; then
        vram=$gpu_row
    fi
    if [[ -r /proc/$server_pid/status ]]; then
        rss_kib=$(awk '/^VmRSS:/ {print $2}' "/proc/$server_pid/status")
        hwm_kib=$(awk '/^VmHWM:/ {print $2}' "/proc/$server_pid/status")
    fi
    printf '%s\t%s\t%s\t%s\n' "$elapsed" "$vram" "$rss_kib" "$hwm_kib" \
        >> "$artifact_dir/resources.tsv"
}

printf 'elapsed_seconds\tvram_mib\tvmrss_kib\tvmhwm_kib\n' \
    > "$artifact_dir/resources.tsv"
sample_resources 0
request_start=$(date +%s)
(
    while kill -0 "$server_pid" 2>/dev/null; do
        sleep 5
        elapsed=$(( $(date +%s) - request_start ))
        sample_resources "$elapsed"
        slots=$(curl --silent --max-time 2 "http://127.0.0.1:$port/slots" |
            jq -c '.[0] // {} | {n_ctx,n_prompt_tokens,n_decoded,n_remain}' 2>/dev/null || true)
        echo "[$label] request elapsed=${elapsed}s slots=${slots:-unavailable}" >&2
    done
) &
progress_pid=$!

echo "[$label] healthy server_pid=$server_pid; starting profiled 5K request" >&2
curl --fail --show-error --no-buffer \
    -H 'Content-Type: application/json' \
    --data-binary "@$artifact_dir/request.json" \
    "http://127.0.0.1:$port/completion" \
    -o "$artifact_dir/response.json" \
    -w 'http_code=%{http_code}\nwall_seconds=%{time_total}\n' \
    | tee "$artifact_dir/curl-timing.txt"

sample_resources "$(( $(date +%s) - request_start ))"
kill "$progress_pid" 2>/dev/null || true
wait "$progress_pid" 2>/dev/null || true
progress_pid=

jq '{timings,token_count:(.tokens|length)}' "$artifact_dir/response.json" \
    > "$artifact_dir/result-summary.json"
jq -c '.tokens' "$artifact_dir/response.json" | sha256sum \
    > "$artifact_dir/tokens.sha256"
jq -j '.content // ""' "$artifact_dir/response.json" | sha256sum \
    > "$artifact_dir/content.sha256"

kill -INT "$server_pid"
server_pid=
echo "[$label] request complete; waiting for Nsight report finalization" >&2
wait "$nsys_pid"
nsys_pid=

nsys stats --force-export=true --report cuda_gpu_mem_size_sum --format csv \
    "$artifact_dir/trace.nsys-rep" > "$artifact_dir/cuda-gpu-mem-size-sum.csv"
nsys stats --report cuda_gpu_mem_time_sum --format csv \
    "$artifact_dir/trace.nsys-rep" > "$artifact_dir/cuda-gpu-mem-time-sum.csv"

awk -F '\t' 'NR > 1 {if ($2 > peak_vram) peak_vram=$2; if ($3 > peak_rss) peak_rss=$3; if ($4 > peak_hwm) peak_hwm=$4} END {printf "peak_vram_mib=%d\npeak_vmrss_kib=%d\npeak_vmhwm_kib=%d\n", peak_vram,peak_rss,peak_hwm}' \
    "$artifact_dir/resources.tsv" > "$artifact_dir/resource-summary.txt"
sha256sum "$artifact_dir/trace.nsys-rep" "$artifact_dir/response.json" \
    "$artifact_dir/cuda-gpu-mem-size-sum.csv" "$artifact_dir/cuda-gpu-mem-time-sum.csv" \
    > "$artifact_dir/artifact-hashes.txt"
printf 'completed_utc=%s\n' "$(date --utc --iso-8601=seconds)" \
    >> "$artifact_dir/provenance.txt"

echo "[$label] trace complete" >&2
cat "$artifact_dir/curl-timing.txt" >&2
cat "$artifact_dir/tokens.sha256" >&2
cat "$artifact_dir/content.sha256" >&2
cat "$artifact_dir/resource-summary.txt" >&2
cat "$artifact_dir/cuda-gpu-mem-size-sum.csv" >&2
