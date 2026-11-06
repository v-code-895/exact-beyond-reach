#!/usr/bin/env bash
# S5b at the end of the large lane (2026-11-06 10:55): after B3 ends, a 20-step smoke run, then the full run if it passed.
#   bash run_queue8.sh s5b
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
case "$1" in
  s5b)
    until grep -qE "\[large\] end B3 \(|\[large\] B3 lr by the rule: none" results/queue.log 2>/dev/null; do sleep 120; done
    mkdir -p results/s5b; wait_slot large 3500
    log large "start S5b smoke"; S5B_SMOKE=1 $PY src/s5b_filter_both.py > results/s5b/smoke.log 2>&1; rc=$?; log large "end S5b smoke (exit $rc)"
    if [ $rc -eq 0 ]; then wait_slot large 3500; log large "start S5b (free $(free_mib) MiB)"; $PY src/s5b_filter_both.py > results/s5b/run.log 2>&1; log large "end S5b (exit $?)"
    else log large "S5b skipped: smoke run failed (see results/s5b/smoke.log)"; fi ;;
esac
