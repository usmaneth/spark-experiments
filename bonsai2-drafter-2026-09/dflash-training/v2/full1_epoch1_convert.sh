#!/usr/bin/env bash
# Detached: when the full1 epoch-1 checkpoint appears, convert it (two-step) + quantize so it can be evaluated mid-run.
ROOT=/home/usman/Bonsai-demo; MD=$ROOT/models/bonsai2-dspark; V2=$ROOT/dflash-training/v2
CK=$MD/bonsai2_dspark_full1_epoch1.safetensors; TAG=full1ep1
RAW=$MD/bonsai2-dspark-${TAG}-dspark-raw.gguf; CONV=$MD/bonsai2-dspark-${TAG}-conv.gguf; Q=$MD/bonsai2-dspark-${TAG}-Q4_K_M.gguf
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
echo "[$(date +%H:%M:%S)] epoch1 waiter started (cap 5h)"
for i in $(seq 1 3600); do
  if [ -f "$CK" ]; then age=$(( $(date +%s) - $(stat -c%Y "$CK") )); [ "$age" -ge 60 ] && [ "$(sz "$CK")" -gt 3000000000 ] && break; fi
  sleep 5
done
[ -f "$CK" ] || { echo "[$(date +%H:%M:%S)] TIMEOUT: no epoch1 checkpoint"; exit 1; }
echo "[$(date +%H:%M:%S)] EPOCH1 CHECKPOINT: $(sz "$CK") bytes"; sudo chown usman:p-usman "$CK"; chmod 0644 "$CK"
rm -f "$RAW" "$CONV" "$Q"
python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$RAW')" > $V2/logs/conv1_${TAG}.log 2>&1; echo "step1 exit $? size $(sz "$RAW")"
python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py "$RAW" "$DONOR" "$CONV" --drop-shared-tensors > $V2/logs/conv2_${TAG}.log 2>&1; echo "step2 exit $? size $(sz "$CONV")"
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q" Q4_K_M > $V2/logs/quant_${TAG}.log 2>&1; echo "quantize exit $? size $(sz "$Q")"
[ "$(sz "$Q")" -gt 500000000 ] && echo "[$(date +%H:%M:%S)] EPOCH1 GGUF READY: $Q" || echo "[$(date +%H:%M:%S)] EPOCH1 CONVERT FAILED"
