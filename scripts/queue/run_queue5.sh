#!/usr/bin/env bash
# Large lane after S5 was cut to arm 3 (2026-11-06 ~08:55, the person's decision; PREREG S5 amendment).
# Replaces run_queue4.sh large2. The small lane (run_queue4.sh small2) is unchanged.
#   bash run_queue5.sh large3   S5 arm 3 -> E1b -> S5b (only if registered; skipped at once if results/.skip_s5b exists)
#                               -> D5 -> D4 block 3 -> B3
cd "$(dirname "$0")/../.."; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }
s5b_ready() { [ -f src/s5b_filter.py ] && grep -q "S5b\." PREREG.md; }

case "$1" in
  large3)
    run large 3500 "S5 arm 3" results/s5/run_arm3.log src/s5_filter.py 3
    run large 4500 E1b results/e1b/run.log src/e1_shielded_edit.py e1b
    t=0; until s5b_ready || [ -f results/.skip_s5b ] || [ $t -ge 120 ]; do sleep 60; t=$((t + 1)); done
    if s5b_ready && [ ! -f results/.skip_s5b ]; then run large 3500 S5b results/s5b/run.log src/s5b_filter.py
    else log large "S5b skipped (not registered, or skipped after S5 arm 3)"; fi
    run large 3000 D5 results/d5/run.log src/d5_distil_odd_eye.py
    run large 3500 "D4 block 3" results/d4/run_k3.log src/d4_switch.py 3 2e-4
    LR=$($PY src/choose_gan_lr.py 2>/dev/null | tail -1); log large "B3 lr by the rule: $LR"
    [ "$LR" != "none" ] && run large 3500 B3 results/b3/run.log src/b3_position.py "$LR" ;;
esac
