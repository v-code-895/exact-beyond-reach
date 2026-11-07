#!/usr/bin/env bash
# Small lane after S1 crashed (2026-11-06 08:47, CUDA "illegal memory access" about a minute after S5 was force-stopped).
# Replaces run_queue4.sh small2. A2 (already running) is adopted; S1 then resumes from its saved runs.
#   bash run_queue6.sh small3 <A2 python PID>   A2 (running) -> S1 (resume) -> S4 -> S2 -> P2b
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
  small3)
    adopt small A2 "$2" results/a2/a2_results.json
    run small 2500 "S1 resume" results/s1/run_resume.log src/s1_domestication.py
    run small 2500 S4 results/s4/run.log src/s4_transplant.py
    run small 2500 S2 results/s2/run.log src/s2_lesion_atlas.py
    run small 1200 P2b results/p2b/run.log src/p2_p4_contracts.py p2b ;;
esac
