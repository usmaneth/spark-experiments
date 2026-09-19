#!/usr/bin/env bash
V2=/home/usman/Bonsai-demo/dflash-training/v2; MD=/home/usman/Bonsai-demo/models/bonsai2-dspark; Q=$MD/bonsai2-dspark-full1-Q4_K_M.gguf
echo "[$(date +%H:%M:%S)] final-eval waiter started (needs complete $Q; prefers idle spark2; cap 8h)"
for i in $(seq 1 5760); do if [ -f "$Q" ]; then sz=$(stat -c%s "$Q"); age=$(( $(date +%s) - $(stat -c%Y "$Q") )); [ "$sz" -ge 1000000000 ] && [ "$age" -ge 45 ] && break; fi; sleep 5; done
[ -f "$Q" ] || { echo "[$(date +%H:%M:%S)] TIMEOUT: no full1 gguf"; exit 1; }
echo "[$(date +%H:%M:%S)] FULL1 GGUF present ($(stat -c%s "$Q") bytes); copying to spark2"; rsync -a "$Q" spark2:$MD/ 2>&1 | tail -1
for i in $(seq 1 90); do n=$(ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l'); [ "${n:-1}" -eq 0 ] && break; echo "[$(date +%H:%M:%S)] spark2 busy ($n procs), waiting for idle"; sleep 60; done
n=$(ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l'); TAG_C="clean"; [ "${n:-0}" -gt 0 ] && TAG_C="CONTENDED(${n} procs; tok/s invalid)"; echo "[$(date +%H:%M:%S)] GPU state: $TAG_C"
echo "[$(date +%H:%M:%S)] LAUNCH final sweep on spark2 (K=4,5; PQ2_0)"
sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#bonsai2-dspark-full1-Q4_K_M.gguf#; s#RESULT full1ep1#RESULT full1#" /tmp/ep1_sweep.sh > /tmp/full1_sweep.sh
scp -q /tmp/full1_sweep.sh spark2:/tmp/full1_sweep.sh && ssh spark2 'bash /tmp/full1_sweep.sh'
echo "[$(date +%H:%M:%S)] FINAL EVAL FINISHED"
