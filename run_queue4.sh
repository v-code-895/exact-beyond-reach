#!/usr/bin/env bash
# Reordered queue (2026-11-06 ~08:45, the user's go-ahead after the overnight results). Replaces run_queue.sh large,
# run_queue2.sh chain2, run_queue3.sh e1b and the separate P2b waiter. The jobs already running (S5, S1) are left alone:
# each lane first waits for that job's process to exit, then logs its end from its results file.
#   bash run_queue4.sh large2 <S5 python PID>   S5 (running) -> E1b -> S5b (once registered) -> D5 -> D4 block 3 -> B3
#   bash run_queue4.sh small2 <S1 python PID>   S1 (running) -> A2 -> S4 -> S2 -> P2b
# Dropped: D4 block 2 (the same switch at a second depth).
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }
adopt() {   # lane, name, PID of the running job, its results JSON: wait for the process, then log its end
  while tasklist //FI "PID eq $3" //NH 2>/dev/null | grep -q " $3 "; do sleep 60; done
  if $PY -c "import json,sys; sys.exit(0 if 'seconds' in json.load(open('$4')) else 1)" 2>/dev/null; then log "$1" "end $2 (exit 0; adopted, results complete)"
  else log "$1" "end $2 (exit 1; adopted, results incomplete)"; fi; }

case "$1" in
  large2)
    adopt large S5 "$2" results/s5/s5_results.json
    run large 4500 E1b results/e1b/run.log src/e1_shielded_edit.py e1b
    t=0; until { [ -f src/s5b_filter.py ] && grep -q "S5b\." PREREG.md; } || [ $t -ge 120 ]; do sleep 60; t=$((t + 1)); done
    if [ -f src/s5b_filter.py ] && grep -q "S5b\." PREREG.md; then run large 3500 S5b results/s5b/run.log src/s5b_filter.py
    else log large "S5b not registered within 2 hours of E1b's end; skipped"; fi
    run large 3000 D5 results/d5/run.log src/d5_distil_odd_eye.py
    run large 3500 "D4 block 3" results/d4/run_k3.log src/d4_switch.py 3 2e-4
    LR=$($PY src/choose_gan_lr.py 2>/dev/null | tail -1); log large "B3 lr by the rule: $LR"
    [ "$LR" != "none" ] && run large 3500 B3 results/b3/run.log src/b3_position.py "$LR" ;;
  small2)
    adopt small S1 "$2" results/s1/s1_results.json
    run small 2500 A2 results/a2/run.log src/a2_guard_distill.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 1200 P2b results/p2b/run.log src/p2_p4_contracts.py p2b ;;
esac
