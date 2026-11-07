#!/usr/bin/env bash
# Queue with the L-series (the reach law) first, by importance (2026-11-06 ~09:30, the person's instruction: depth over
# breadth). Replaces run_queue5.sh large3 and run_queue6.sh small3; the running jobs (S5 arm 3, A2) are adopted.
#   bash run_queue7.sh small4 <A2 python PID>         A2 -> L1 -> L2 -> L3 -> L4 -> L5 toy -> S1 (resume) -> S4 -> S2 -> P2b
#   bash run_queue7.sh large4 <S5 arm 3 python PID>   S5 arm 3 -> L5 (TinySR) -> E1b -> D5 -> D4 block 3 -> B3
# S5b is no longer queued automatically (registered and added only if S5 arm 3 calls for it).
cd "$(dirname "$0")/../.."; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() { until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }
adopt() {
  while tasklist //FI "PID eq $3" //NH 2>/dev/null | grep -q " $3 "; do sleep 60; done
  if $PY -c "import json,sys; sys.exit(0 if 'seconds' in json.load(open('$4')) else 1)" 2>/dev/null; then log "$1" "end $2 (exit 0; adopted, results complete)"
  else log "$1" "end $2 (exit 1; adopted, results incomplete)"; fi; }

case "$1" in
  small4)
    adopt small A2 "$2" results/a2/a2_results.json
    run small 1500 L1 results/l1/run.log src/l_series.py l1
    run small 2000 L2 results/l2/run.log src/l_series.py l2
    run small 1500 L3 results/l3/run.log src/l_series.py l3
    run small 1500 L4 results/l4/run.log src/l_series.py l4
    run small 1500 "L5 toy" results/l5toy/run.log src/l_series.py l5toy
    run small 2500 "S1 resume" results/s1/run_resume.log src/s1_domestication.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 1200 P2b results/p2b/run.log src/p2_p4_contracts.py p2b ;;
  large4)
    adopt large "S5 arm 3" "$2" results/s5/s5_arm3_results.json
    run large 2500 L5 results/l5/run.log src/l5_exchange.py
    run large 4500 E1b results/e1b/run.log src/e1_shielded_edit.py e1b
    run large 3000 D5 results/d5/run.log src/d5_distil_odd_eye.py
    run large 3500 "D4 block 3" results/d4/run_k3.log src/d4_switch.py 3 2e-4
    LR=$($PY src/choose_gan_lr.py 2>/dev/null | tail -1); log large "B3 lr by the rule: $LR"
    [ "$LR" != "none" ] && run large 3500 B3 results/b3/run.log src/b3_position.py "$LR" ;;
esac
