#!/usr/bin/env bash
# Detached: when the epoch-1 Q4_K_M exists AND node_b's GPU is empty, run the clean sweep there.
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; MD=/home/REDACTED/Bonsai-demo/models/bonsai2-dspark
Q=$MD/bonsai2-dspark-full1ep1-Q4_K_M.gguf; TAG=full1ep1
echo "[$(date +%H:%M:%S)] eval waiter started (needs $Q + idle node_b; cap 6h)"
for i in $(seq 1 4320); do if [ -f "$Q" ]; then sz=$(stat -c%s "$Q"); age=$(( $(date +%s) - $(stat -c%Y "$Q") )); [ "$sz" -ge 1000000000 ] && [ "$age" -ge 45 ] && break; fi; sleep 5; done
[ -f "$Q" ] || { echo "[$(date +%H:%M:%S)] TIMEOUT: no epoch1 gguf"; exit 1; }
echo "[$(date +%H:%M:%S)] EPOCH1 GGUF present ($(stat -c%s "$Q") bytes); copying to node_b"; rsync -a "$Q" node_b:$MD/ 2>&1 | tail -1
for i in $(seq 1 5); do n=$(ssh -o ConnectTimeout=8 node_b 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l'); [ "${n:-1}" -eq 0 ] && break; echo "[$(date +%H:%M:%S)] node_b busy ($n procs), courtesy wait"; sleep 60; done; n=$(ssh -o ConnectTimeout=8 node_b 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l'); TAG_C="clean"; [ "${n:-0}" -gt 0 ] && TAG_C="CONTENDED(${n} other GPU procs; tok/s invalid, acceptance valid)"; echo "[$(date +%H:%M:%S)] GPU state: $TAG_C"
echo "[$(date +%H:%M:%S)] LAUNCH clean sweep on node_b (K=4,5; p_min=0; PQ2_0)"
cat > /tmp/ep1_sweep.sh <<'EOS'
cd /home/REDACTED/Bonsai-demo; export LD_LIBRARY_PATH=/home/REDACTED/Bonsai-demo/bin/cuda
B=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf
CODE='<|im_start|>user\nWrite a Python function that parses a CSV file of transactions (date, amount, category) and returns the total spent per category as a dict, with error handling for malformed rows and a small unit test.<|im_end|>\n<|im_start|>assistant\n'
MATH='<|im_start|>user\nA train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step.<|im_end|>\n<|im_start|>assistant\n'
CODE2='<|im_start|>user\nImplement binary search in Python with type hints, a docstring, and three assert-based tests.<|im_end|>\n<|im_start|>assistant\n'
for K in 4 5; do for W in math code code2; do P="$MATH"; [ $W = code ] && P="$CODE"; [ $W = code2 ] && P="$CODE2"
  o=$(./bin/cuda/llama-speculative-simple -m $B -md $D --spec-type draft-dspark --spec-draft-n-max $K --spec-draft-p-min 0 -fa on -ngl 99 -ngld 999 -c 4096 -n 200 --temp 0 -e -p "$P" 2>&1)
  acc=$(echo "$o" | grep -oE 'accept[[:space:]]+=[[:space:]]+[0-9.]+' | grep -oE '[0-9.]+$'); spd=$(echo "$o" | grep -oE 'speed:[[:space:]]+[0-9.]+' | tail -1 | grep -oE '[0-9.]+$'); na=$(echo "$o" | grep -oE 'n_accept[[:space:]]+=[[:space:]]+[0-9]+' | grep -oE '[0-9]+$'); nd=$(echo "$o" | grep -oE 'n_drafted[[:space:]]+=[[:space:]]+[0-9]+' | grep -oE '[0-9]+$')
  echo "RESULT full1ep1 K=$K $W | accept ${acc:-?}% | ${spd:-?} tok/s | ${na:-?}/${nd:-?}"
done; done; echo "SWEEP DONE"
EOS
scp -q /tmp/ep1_sweep.sh node_b:/tmp/ep1_sweep.sh && ssh node_b 'bash /tmp/ep1_sweep.sh'
echo "[$(date +%H:%M:%S)] EVAL FINISHED"
