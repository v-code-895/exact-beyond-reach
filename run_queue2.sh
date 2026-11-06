#!/usr/bin/env bash
# Second-lane chain (2026-11-06 05:35), replacing run_queue.sh's "small" and "after_e1" roles: the old small lane was still
# alive and ran S3 next to E1 (fine: E1 slowed only to ~105 s per 100 steps), so the chain no longer waits for E1's end.
# E1 alone holds ~6.4 GB of the 8 GB card, so the light P-series jobs go first (they fit next to it); the TinySR jobs wait
# for 2.5 GB free, which in practice means E1's end. S3 is done and is not repeated. P2b runs after A2 (separate waiter).
#   bash run_queue2.sh chain
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }

case "$1" in
  chain)
    run small 1200 P4 results/p4/run.log src/p2_p4_contracts.py p4
    run small 1200 P5 results/p5/run.log src/p5_p7_contracts.py p5
    run small 1200 P6 results/p6/run.log src/p5_p7_contracts.py p6
    run small 1200 P7 results/p7/run.log src/p5_p7_contracts.py p7
    run small 2500 S1 results/s1/run.log src/s1_domestication.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 2500 A2 results/a2/run.log src/a2_guard_distill.py ;;
esac
