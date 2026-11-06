#!/usr/bin/env bash
# E1b after the large lane (2026-11-06 08:10). Same gate as E1: 4.5 GB free and 3 minutes after the last start.
#   bash run_queue3.sh e1b
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }
case "$1" in
  e1b)
    until grep -qF "[large] end D4 block 2 (" results/queue.log 2>/dev/null; do sleep 120; done
    run large 4500 E1b results/e1b/run.log src/e1_shielded_edit.py e1b ;;
esac
