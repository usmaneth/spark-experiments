#!/usr/bin/env bash
# Open or close the clean-GPU benchmark window on spark2.
# open:  set the pause flag (ablation chain waits), stop the running ablation container,
#        SIGSTOP the generation client, wait until the gen server slots drain and GPU util is ~0.
# close: SIGCONT the client, remove the flag (the chain restarts the interrupted run).
set -u
V2=/home/usman/Bonsai-demo/dflash-training/v2
case "${1:?open|close}" in
  open)
    touch /tmp/spark2_bench_window; echo "[$(date +%H:%M:%S)] flag set"
    C=$(ssh spark2 'sudo docker ps --format "{{.Names}}" | grep "^eagle3_abl_" | head -1'); [ -n "$C" ] && ssh spark2 "sudo docker stop -t 15 $C" >/dev/null && echo "stopped container $C"
    P=$(ssh spark2 'pgrep -f "python3 gen_client.py 8095"'); echo "gen client pid(s) on spark2: $P"; for p in $P; do ssh spark2 "kill -STOP $p"; done
    for i in $(seq 1 60); do busy=$(ssh spark2 'curl -s -m 5 127.0.0.1:8095/slots' | python3 -c "import sys,json; print(sum(1 for s in json.load(sys.stdin) if s.get('is_processing')))" 2>/dev/null || echo "?"); util=$(ssh spark2 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits'); echo "[$(date +%H:%M:%S)] slots busy=$busy gpu util=$util%"; [ "$busy" = "0" ] && [ "${util:-100}" -le 2 ] && break; sleep 10; done
    ssh spark2 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'; echo "WINDOW OPEN"
    ;;
  close)
    P=$(ssh spark2 'pgrep -f "python3 gen_client.py 8095"'); for p in $P; do ssh spark2 "kill -CONT $p"; done; echo "gen client resumed: $P"
    rm -f /tmp/spark2_bench_window; echo "[$(date +%H:%M:%S)] flag removed; WINDOW CLOSED"
    ;;
esac
