#!/usr/bin/env bash
# Usage: monitor.sh <pid> <out_prefix>
# Polls nvidia-smi process VRAM and /proc/<pid>/status RSS/VmHWM every 0.5s
# until the target process exits. Writes CSV: monitor.sh is stopped by the
# caller killing it (it self-exits once the pid disappears).
set -u
PID="$1"
OUT="$2"
echo "timestamp_s,vram_mib,rss_kib,vmhwm_kib,vmlck_kib" > "${OUT}.csv"
START=$(date +%s.%N)
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s.%N)
  ELAPSED=$(awk -v a="$NOW" -v b="$START" 'BEGIN{printf "%.3f", a-b}')
  VRAM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$PID" '$1==p {print $2}')
  VRAM=${VRAM:-0}
  if [ -r "/proc/$PID/status" ]; then
    RSS=$(grep -m1 '^VmRSS:' "/proc/$PID/status" | awk '{print $2}')
    HWM=$(grep -m1 '^VmHWM:' "/proc/$PID/status" | awk '{print $2}')
    LCK=$(grep -m1 '^VmLck:' "/proc/$PID/status" | awk '{print $2}')
  else
    RSS=0; HWM=0; LCK=0
  fi
  echo "${ELAPSED},${VRAM},${RSS:-0},${HWM:-0},${LCK:-0}" >> "${OUT}.csv"
  sleep 0.5
done
