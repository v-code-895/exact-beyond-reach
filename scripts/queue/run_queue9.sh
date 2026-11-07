#!/usr/bin/env bash
# Small lane from L2 on (2026-11-06 13:42): E1b holds ~6.4 GB, so the toy L-jobs (~0.5-1 GB) start at 1.2 GB free
# (L1 ran beside E1b without trouble). Replaces run_queue7.sh small4.
#   bash run_queue9.sh small5   L2 -> L3 -> L4 -> L5 toy -> S1 (resume) -> S4 -> S2 -> P2b
cd "$(dirname "$0")/../.."; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }
case "$1" in
  small5)
    run small 1200 L2 results/l2/run.log src/l_series.py l2
    run small 1200 L3 results/l3/run.log src/l_series.py l3
    run small 1200 L4 results/l4/run.log src/l_series.py l4
    run small 1200 "L5 toy" results/l5toy/run.log src/l_series.py l5toy
    run small 2500 "S1 resume" results/s1/run_resume.log src/s1_domestication.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 1200 P2b results/p2b/run.log src/p2_p4_contracts.py p2b ;;
esac
