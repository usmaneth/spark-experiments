#!/usr/bin/env bash
# Open or close the clean-GPU benchmark window on node_b.
# open:  set the pause flag (ablation chain waits), stop the running ablation container,
#        SIGSTOP the generation client, wait until the gen server slots drain and GPU util is ~0.
# close: SIGCONT the client, remove the flag (the chain restarts the interrupted run).
set -u
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2
case "${1:?open|close}" in
  open)
    touch /tmp/node_b_bench_window; echo "[$(date +%H:%M:%S)] flag set"
    C=$(ssh node_b 'sudo docker ps --format "{{.Names}}" | grep "^eagle3_abl_" | head -1'); [ -n "$C" ] && ssh node_b "sudo docker stop -t 15 $C" >/dev/null && echo "stopped container $C"
    P=$(ssh node_b 'pgrep -f "python3 gen_client.py 8095"'); echo "gen client pid(s) on node_b: $P"; for p in $P; do ssh node_b "kill -STOP $p"; done
    for i in $(seq 1 60); do busy=$(ssh node_b 'curl -s -m 5 127.0.0.1:8095/slots' | python3 -c "import sys,json; print(sum(1 for s in json.load(sys.stdin) if s.get('is_processing')))" 2>/dev/null || echo "?"); util=$(ssh node_b 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits'); echo "[$(date +%H:%M:%S)] slots busy=$busy gpu util=$util%"; [ "$busy" = "0" ] && [ "${util:-100}" -le 2 ] && break; sleep 10; done
    ssh node_b 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'; echo "WINDOW OPEN"
    ;;
  close)
    P=$(ssh node_b 'pgrep -f "python3 gen_client.py 8095"'); for p in $P; do ssh node_b "kill -CONT $p"; done; echo "gen client resumed: $P"
    rm -f /tmp/node_b_bench_window; echo "[$(date +%H:%M:%S)] flag removed; WINDOW CLOSED"
    ;;
esac
