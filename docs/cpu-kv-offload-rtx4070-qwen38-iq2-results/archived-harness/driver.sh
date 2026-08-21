#!/usr/bin/env bash
set -u
cd /tmp/claude-1000/-home-piggidragon-Services-llama-cpp/83521a4c-01ef-4ab1-b46e-b9ce35655df5/scratchpad/kvoffload-bench
DSPARK=/home/piggidragon/Services/models/llama-cpp/Qwen3.8-27b/Qwen3.8-27B-DSpark-Q8_0.gguf
PORT=8200

CTX16="--ctx-size 16384 --no-kv-offload --kv-cpu-pinned --recurrent-state-offload"
MTP_BAL="--spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-ubatch-size 128 --draft-p-min 0.85 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0"
DSPARK_BASE="--spec-type draft-dspark --model-draft $DSPARK --n-gpu-layers-draft 0 --draft-p-min 0.85 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0"
DSPARK_BAL="$DSPARK_BASE --spec-draft-n-max 6"

next_port() { PORT=$((PORT+1)); echo $PORT; }

echo "########## MATRIX A: core comparison across prompts ##########"
for p in short creative coding; do
  case $p in
    short) TOK=64 ;;
    creative) TOK=400 ;;
    coding) TOK=1200 ;;
  esac
  bash run_one.sh "A-baseline-$p" "$(next_port)" "prompt_${p}.txt" $TOK $CTX16
  bash run_one.sh "A-mtp-$p"      "$(next_port)" "prompt_${p}.txt" $TOK $CTX16 $MTP_BAL
  bash run_one.sh "A-dspark-$p"   "$(next_port)" "prompt_${p}.txt" $TOK $CTX16 $DSPARK_BAL
done

echo "########## MATRIX A-synthetic: clean prefill/decode at 12000 synthetic tokens ##########"
bash run_synth.sh "A-baseline-synth12k" "$(next_port)" 12000 64 $CTX16
bash run_synth.sh "A-mtp-synth12k"      "$(next_port)" 12000 64 $CTX16 $MTP_BAL
bash run_synth.sh "A-dspark-synth12k"   "$(next_port)" 12000 64 $CTX16 $DSPARK_BAL

echo "########## MATRIX B: dspark n_max auto-resolution gap ##########"
bash run_one.sh "B-dspark-default-nmax" "$(next_port)" prompt_coding.txt 1200 $CTX16 $DSPARK_BASE
bash run_one.sh "B-dspark-explicit-nmax6" "$(next_port)" prompt_coding.txt 1200 $CTX16 $DSPARK_BAL

echo "########## MATRIX C: draft ubatch sweep ##########"
for ub in 512 128 32; do
  bash run_one.sh "C-mtp-ubatch$ub"    "$(next_port)" prompt_short.txt 64 $CTX16 --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-ubatch-size $ub --draft-p-min 0.85 --cache-type-k-draft q8_0 --cache-type-v-draft q8_0
done
bash run_one.sh "C-dspark-ubatch128-control" "$(next_port)" prompt_short.txt 64 $CTX16 $DSPARK_BASE --spec-draft-n-max 6 --spec-draft-ubatch-size 128

echo "########## MATRIX D: partial GPU KV residency (--kv-gpu-layers) ##########"
for kgl in 0 8 16; do
  bash run_one.sh "D-baseline-kgl$kgl" "$(next_port)" prompt_short.txt 64 $CTX16 --kv-gpu-layers $kgl
  bash run_one.sh "D-mtp-kgl$kgl"      "$(next_port)" prompt_short.txt 64 $CTX16 --kv-gpu-layers $kgl $MTP_BAL
  bash run_one.sh "D-dspark-kgl$kgl"   "$(next_port)" prompt_short.txt 64 $CTX16 --kv-gpu-layers $kgl $DSPARK_BAL
done

echo "########## MATRIX E: MTP recurrent-plane cap vs dspark rejection ##########"
bash run_one.sh "E-mtp-planes-full"    "$(next_port)" prompt_coding.txt 1200 $CTX16 --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-ubatch-size 128 --draft-p-min 0.85 --spec-mtp-rs-planes 0
bash run_one.sh "E-mtp-planes-capped4" "$(next_port)" prompt_coding.txt 1200 $CTX16 --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-ubatch-size 128 --draft-p-min 0.85 --spec-mtp-rs-planes 4
bash run_one.sh "E-dspark-planes-rejected" "$(next_port)" prompt_short.txt 32 $CTX16 $DSPARK_BAL --spec-mtp-rs-planes 4

echo "########## MATRIX F: phase-aware-workspace ##########"
bash run_one.sh "F-mtp-phaseaware-off" "$(next_port)" prompt_coding.txt 1200 $CTX16 $MTP_BAL
bash run_one.sh "F-mtp-phaseaware-on"  "$(next_port)" prompt_coding.txt 1200 $CTX16 $MTP_BAL --phase-aware-workspace
bash run_one.sh "F-dspark-phaseaware-off" "$(next_port)" prompt_coding.txt 1200 $CTX16 $DSPARK_BAL
bash run_one.sh "F-dspark-phaseaware-on"  "$(next_port)" prompt_coding.txt 1200 $CTX16 $DSPARK_BAL --phase-aware-workspace

echo "########## ALL DONE ##########"
