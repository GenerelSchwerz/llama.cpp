#!/usr/bin/env bash
set -u
cd /tmp/claude-1000/-home-piggidragon-Services-llama-cpp/83521a4c-01ef-4ab1-b46e-b9ce35655df5/scratchpad/kvoffload-bench
DSPARK=/home/piggidragon/Services/models/llama-cpp/Qwen3.8-27b/Qwen3.8-27B-DSpark-Q8_0.gguf
PORT=8400

CTX16="--ctx-size 16384 --no-kv-offload --kv-cpu-pinned --recurrent-state-offload"
# MTP now MUST inherit target ubatch (512) -- no --spec-draft-ubatch-size.
MTP_BAL="--spec-type draft-mtp --spec-draft-n-max 5 --draft-p-min 0.85 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0"
DSPARK_BASE="--spec-type draft-dspark --model-draft $DSPARK --n-gpu-layers-draft 0 --draft-p-min 0.85 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0"
DSPARK_BAL="$DSPARK_BASE --spec-draft-n-max 6"

next_port() { PORT=$((PORT+1)); echo $PORT; }

echo "########## MATRIX A2 (rebased, corrected MTP ubatch): core comparison across prompts ##########"
for p in short creative coding; do
  case $p in
    short) TOK=64 ;;
    creative) TOK=400 ;;
    coding) TOK=1200 ;;
  esac
  bash run_one.sh "A2-baseline-$p" "$(next_port)" "prompt_${p}.txt" $TOK $CTX16
  bash run_one.sh "A2-mtp-$p"      "$(next_port)" "prompt_${p}.txt" $TOK $CTX16 $MTP_BAL
  bash run_one.sh "A2-dspark-$p"   "$(next_port)" "prompt_${p}.txt" $TOK $CTX16 $DSPARK_BAL
done

echo "########## MATRIX A2-synthetic: clean prefill/decode at 12000 synthetic tokens ##########"
bash run_synth.sh "A2-baseline-synth12k" "$(next_port)" 12000 64 $CTX16
bash run_synth.sh "A2-mtp-synth12k"      "$(next_port)" 12000 64 $CTX16 $MTP_BAL
bash run_synth.sh "A2-dspark-synth12k"   "$(next_port)" 12000 64 $CTX16 $DSPARK_BAL

echo "########## MATRIX G: MTP crash re-check with corrected ubatch geometry ##########"
for i in 1 2 3; do
  bash run_one.sh "G-mtp-coding-crashcheck-$i" "$(next_port)" prompt_coding.txt 1200 $CTX16 --spec-type draft-mtp --spec-draft-n-max 6 --draft-p-min 0.85 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 --spec-mtp-rs-planes 0
done

echo "########## ALL DONE ##########"
