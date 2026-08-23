#!/usr/bin/env bash

set -euo pipefail

if (( $# != 7 )); then
    echo "usage: $0 LABEL SERVER PORT CTX MODEL REQUEST_JSON OUTPUT_DIR" >&2
    exit 2
fi

label=$1
server=$2
port=$3
ctx=$4
model=$5
request_json=$6
output_dir=$7

base_url="http://127.0.0.1:${port}"

mkdir -p "$output_dir"

server_command=(
    "$server"
    --model "$model"
    --ctx-size "$ctx"
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
    --n-gpu-layers 999
    --fit off
    --split-mode none
    --main-gpu 0
    --no-kv-offload
    --kv-cpu-pinned
    --flash-attn on
    --cache-ram 0
    --seed 1234
    --slots
    --metrics
    --host 127.0.0.1
    --port "$port"
    --verbosity 4
)

{
    printf 'label=%q\n' "$label"
    printf 'server_sha256='
    sha256sum "$server" | awk '{print $1}'
    printf 'request_sha256='
    sha256sum "$request_json" | awk '{print $1}'
    printf 'command='
    printf '%q ' "${server_command[@]}"
    printf '\n'
} > "$output_dir/manifest.txt"

"${server_command[@]}" > "$output_dir/server.log" 2>&1 &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

sample_once() {
    local phase=$1
    local timestamp_ns
    local vram_mib
    local rss_kib=0
    local hwm_kib=0
    local lck_kib=0

    timestamp_ns=$(date +%s%N)
    vram_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
        awk -F, -v wanted="$server_pid" '$1 + 0 == wanted { gsub(/ /, "", $2); print $2; found=1 } END { if (!found) print 0 }')
    if [[ -r /proc/$server_pid/status ]]; then
        rss_kib=$(awk '$1 == "VmRSS:" { print $2 }' "/proc/$server_pid/status")
        hwm_kib=$(awk '$1 == "VmHWM:" { print $2 }' "/proc/$server_pid/status")
        lck_kib=$(awk '$1 == "VmLck:" { print $2 }' "/proc/$server_pid/status")
    fi
    printf '%s,%s,%s,%s,%s,%s\n' \
        "$timestamp_ns" "$phase" "$vram_mib" "${rss_kib:-0}" "${hwm_kib:-0}" "${lck_kib:-0}"
}

printf 'timestamp_ns,phase,vram_mib,vmrss_kib,vmhwm_kib,vmlck_kib\n' > "$output_dir/resources.csv"

ready=0
for attempt in $(seq 1 240); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server exited before readiness" >&2
        exit 1
    fi
    if curl --silent --fail "$base_url/health" > "$output_dir/health.json" 2>/dev/null; then
        ready=1
        break
    fi
    if (( attempt % 10 == 0 )); then
        echo "$label readiness attempt $attempt/240" >&2
    fi
    sleep 0.25
done
if (( ready == 0 )); then
    echo "server did not become ready" >&2
    exit 1
fi

sample_once startup >> "$output_dir/resources.csv"

(
    while kill -0 "$server_pid" 2>/dev/null; do
        sample_once running >> "$output_dir/resources.csv"
        sleep 0.25
    done
) &
sampler_pid=$!

for turn in 1 2; do
    echo "$label turn $turn starting" >&2
    date +%s%N > "$output_dir/turn-$turn-start-ns.txt"
    curl --silent --show-error --fail --max-time 600 \
        -H 'Content-Type: application/json' \
        --data-binary "@$request_json" \
        "$base_url/completion" > "$output_dir/turn-$turn-response.json" &
    request_pid=$!
    poll=0
    while kill -0 "$request_pid" 2>/dev/null; do
        poll=$((poll + 1))
        curl --silent --show-error "$base_url/slots" \
            > "$output_dir/turn-$turn-slots-$poll.json" || true
        jq -r --arg label "$label" --arg turn "$turn" \
            '.[0] | "\($label) turn \($turn): state=\(.state // "unknown") prompt=\(.next_token.n_tokens // .n_prompt_tokens_processed // 0) generated=\(.n_decoded // .n_tokens_predicted // 0)"' \
            "$output_dir/turn-$turn-slots-$poll.json" 2>/dev/null || true
        sleep 2
    done
    wait "$request_pid"
    date +%s%N > "$output_dir/turn-$turn-end-ns.txt"
    sample_once "post_turn_$turn" >> "$output_dir/resources.csv"
    sha256sum "$output_dir/turn-$turn-response.json" > "$output_dir/turn-$turn-response.sha256"
    jq '{content, tokens_predicted, tokens_evaluated, timings}' \
        "$output_dir/turn-$turn-response.json" > "$output_dir/turn-$turn-summary.json"
    echo "$label turn $turn complete" >&2
done

kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
trap - EXIT
cleanup

awk -F, 'NR == 1 { next }
    { if ($3 > max_vram) max_vram=$3; if ($4 > max_rss) max_rss=$4; if ($5 > max_hwm) max_hwm=$5; if ($6 > max_lck) max_lck=$6 }
    END { printf "peak_vram_mib=%d\npeak_vmrss_kib=%d\npeak_vmhwm_kib=%d\npeak_vmlck_kib=%d\n", max_vram, max_rss, max_hwm, max_lck }' \
    "$output_dir/resources.csv" > "$output_dir/resource-summary.txt"
