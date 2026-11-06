#!/usr/bin/env bash
# One ordered GPU queue (2026-11-05), replacing the separate queues that oversubscribed the 8 GB card.
# At most two jobs at once: a "large" lane (LLM / GAN) and a "small" lane (TinySR), each in ranked order, each started
# only when enough GPU memory is free and at least 3 minutes after the other lane last started a job.
#   bash run_queue.sh a3      stop A3 after its 3x-wide seed 0 (person's decision), write its table
#   bash run_queue.sh large   E1, S5, D5, B3, D4 (block 3), D4 (block 2)
#   bash run_queue.sh small   S1, S3, S4, S2, A2
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/starb/.venvs/wsse/Scripts/python
log() { echo "$(date +%m-%d\ %H:%M:%S) [$1] $2" >> results/queue.log; }
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
since_last_start() { [ -f results/.queue_last_start ] && echo $(( $(date +%s) - $(cat results/.queue_last_start) )) || echo 99999; }
wait_slot() {   # $1 lane, $2 MiB needed
  until [ "$(free_mib)" -ge "$2" ] && [ "$(since_last_start)" -ge 180 ]; do sleep 60; done; date +%s > results/.queue_last_start; }
run() {   # lane, MiB, name, log path, script args...
  local lane=$1 mib=$2 name=$3 out=$4; shift 4; mkdir -p "$(dirname "$out")"; wait_slot "$lane" "$mib"
  log "$lane" "start $name (free $(free_mib) MiB)"; $PY "$@" > "$out" 2>&1; log "$lane" "end $name (exit $?)"; }
gates() { until [ -f results/c1b/c1b_results.md ] && [ -f results/a3/a3_results.md ]; do sleep 60; done; }

case "$1" in
  a3)
    until $PY -c "import json,sys; r=json.load(open('results/a3/a3_results.json')); sys.exit(0 if len(r['runs'].get('3x width, joint retraining, ground truth', [])) >= 1 else 1)" 2>/dev/null; do sleep 60; done
    taskkill //PID 62708 //T //F > /dev/null 2>&1; sleep 5; $PY src/a3_write_table.py > results/a3/table.log 2>&1; log a3 "A3 stopped after the 3x-wide seed 0; table written" ;;
  large)
    gates
    run large 4500 E1 results/e1/run.log src/e1_shielded_edit.py
    run large 3500 S5 results/s5/run.log src/s5_filter.py
    run large 3000 D5 results/d5/run.log src/d5_distil_odd_eye.py
    LR=$($PY src/choose_gan_lr.py 2>/dev/null | tail -1); log large "B3 lr by the rule: $LR"
    [ "$LR" != "none" ] && run large 3500 B3 results/b3/run.log src/b3_position.py "$LR"
    run large 3500 "D4 block 3" results/d4/run_k3.log src/d4_switch.py 3 2e-4
    run large 3500 "D4 block 2" results/d4/run_k2.log src/d4_switch.py 2 2e-4 ;;
  after_e1)   # 2026-11-06 03:05: E1 alone needs ~6 GB, so the small lane and the P-series wait for it and then share one chain
    until grep -qF "end E1 (" results/queue.log 2>/dev/null; do sleep 60; done
    run small 2500 S1 results/s1/run.log src/s1_domestication.py
    run small 2000 P4 results/p4/run.log src/p2_p4_contracts.py p4
    run small 2500 S3 results/s3/run.log src/s3_phantom_limb.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2000 P5 results/p5/run.log src/p5_p7_contracts.py p5
    run small 2000 P6 results/p6/run.log src/p5_p7_contracts.py p6
    run small 2000 P7 results/p7/run.log src/p5_p7_contracts.py p7
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 2500 A2 results/a2/run.log src/a2_guard_distill.py ;;
  small)
    gates; until grep -q "start E1" results/queue.log 2>/dev/null; do sleep 60; done
    run small 2500 S1 results/s1/run.log src/s1_domestication.py
    run small 2500 S3 results/s3/run.log src/s3_phantom_limb.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 2500 A2 results/a2/run.log src/a2_guard_distill.py ;;
esac
