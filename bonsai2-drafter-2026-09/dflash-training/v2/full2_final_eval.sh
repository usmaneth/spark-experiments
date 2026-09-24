#!/usr/bin/env bash
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; MD=/home/REDACTED/Bonsai-demo/models/bonsai2-dspark; Q=$MD/bonsai2-dspark-full2-Q4_K_M.gguf
echo "[$(date +%H:%M:%S)] round-2 eval waiter started (needs complete $Q; prefers idle node_b; cap 14h)"
for i in $(seq 1 10080); do if [ -f "$Q" ]; then sz=$(stat -c%s "$Q"); age=$(( $(date +%s) - $(stat -c%Y "$Q") )); [ "$sz" -ge 1000000000 ] && [ "$age" -ge 45 ] && break; fi; sleep 5; done
[ -f "$Q" ] || { echo "[$(date +%H:%M:%S)] TIMEOUT: no round-2 gguf"; exit 1; }
echo "[$(date +%H:%M:%S)] ROUND2 GGUF present ($(stat -c%s "$Q") bytes); copying to node_b"; rsync -a "$Q" node_b:$MD/ 2>&1 | tail -1
for i in $(seq 1 120); do n=$(ssh -o ConnectTimeout=8 node_b 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l'); [ "${n:-1}" -eq 0 ] && break; echo "[$(date +%H:%M:%S)] node_b busy ($n procs), waiting for idle"; sleep 60; done
n=$(ssh -o ConnectTimeout=8 node_b 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l'); TAG_C="clean"; [ "${n:-0}" -gt 0 ] && TAG_C="CONTENDED(${n} procs; tok/s invalid)"; echo "[$(date +%H:%M:%S)] GPU state: $TAG_C"
echo "[$(date +%H:%M:%S)] LAUNCH round-2 sweep on node_b (K=4,5,7; exact + tau 0.02/0.05)"
sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#bonsai2-dspark-full2-Q4_K_M.gguf#g; s#RESULT full1ep1#RESULT full2#g" /tmp/ep1_sweep.sh > /tmp/full2_sweep.sh
scp -q /tmp/full2_sweep.sh node_b:/tmp/full2_sweep.sh && ssh node_b 'bash /tmp/full2_sweep.sh'
echo "[$(date +%H:%M:%S)] ROUND2 EVAL FINISHED"
