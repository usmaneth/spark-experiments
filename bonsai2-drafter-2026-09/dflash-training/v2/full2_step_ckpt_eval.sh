#!/usr/bin/env bash
# Detached: for each new bonsai2_dspark_full2_step*.safetensors, convert (two-step + Q4_K_M) and eval on spark2 (K=5 exact + tau 0.05).
V2=/home/usman/Bonsai-demo/dflash-training/v2; MD=/home/usman/Bonsai-demo/models/bonsai2-dspark; ROOT=/home/usman/Bonsai-demo; DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf; sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
echo "[$(date +%H:%M:%S)] step-checkpoint waiter started (cap 14h)"; done_list=""
for i in $(seq 1 10080); do
  for CK in $(ls $MD/bonsai2_dspark_full2_step*.safetensors 2>/dev/null); do
    case " $done_list " in *" $CK "*) continue;; esac
    age=$(( $(date +%s) - $(stat -c%Y "$CK") )); [ "$(sz "$CK")" -gt 3000000000 ] && [ "$age" -ge 60 ] || continue
    N=$(echo "$CK" | grep -oE 'step[0-9]+'); TAG=full2$N; sudo chown usman:p-usman "$CK"; chmod 0644 "$CK"
    RAW=$MD/bonsai2-dspark-${TAG}-dspark-raw.gguf; CONV=$MD/bonsai2-dspark-${TAG}-conv.gguf; Q=$MD/bonsai2-dspark-${TAG}-Q4_K_M.gguf
    [ -s "$Q" ] && [ "$(sz "$Q")" -gt 1000000000 ] && { echo "[$(date +%H:%M:%S)] CKPT $N: skip: gguf exists"; done_list="$done_list $CK"; continue; }
    echo "[$(date +%H:%M:%S)] CKPT $N: converting"; python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$RAW')" > $V2/logs/conv1_${TAG}.log 2>&1
    python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py "$RAW" "$DONOR" "$CONV" --drop-shared-tensors > $V2/logs/conv2_${TAG}.log 2>&1
    LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q" Q4_K_M > $V2/logs/quant_${TAG}.log 2>&1; rm -f "$RAW" "$CONV"
    [ "$(sz "$Q")" -gt 1000000000 ] || { echo "[$(date +%H:%M:%S)] CKPT $N: CONVERT FAILED"; done_list="$done_list $CK"; continue; }
    rsync -a "$Q" spark2:$MD/ 2>/dev/null; for j in $(seq 1 30); do n=$(ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l'); [ "${n:-1}" -eq 0 ] && break; sleep 60; done
    n=$(ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l'); echo "[$(date +%H:%M:%S)] CKPT $N: eval on spark2 ($([ "${n:-0}" -gt 0 ] && echo CONTENDED || echo clean))"
    sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#bonsai2-dspark-${TAG}-Q4_K_M.gguf#g; s#RESULT full1ep1#RESULT ${TAG}#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#; s#for K in 5 7; do for TAU in 0.02 0.05#for K in 5; do for TAU in 0.05#" /tmp/ep1_sweep.sh > /tmp/${TAG}_sweep.sh
    scp -q /tmp/${TAG}_sweep.sh spark2:/tmp/${TAG}_sweep.sh && ssh spark2 "bash /tmp/${TAG}_sweep.sh" | grep -E '^RESULT|DONE'
    done_list="$done_list $CK"
  done; sleep 30
done
