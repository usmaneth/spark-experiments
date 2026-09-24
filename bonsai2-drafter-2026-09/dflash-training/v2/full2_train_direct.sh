#!/usr/bin/env bash
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; MD=/home/REDACTED/Bonsai-demo/models/bonsai2-dspark; ROOT=/home/REDACTED/Bonsai-demo; ALL=$V2/feats_all; TAG=full2
CK=$MD/bonsai2_dspark_${TAG}.safetensors; RAW=$MD/bonsai2-dspark-${TAG}-dspark-raw.gguf; CONV=$MD/bonsai2-dspark-${TAG}-conv.gguf; Q=$MD/bonsai2-dspark-${TAG}-Q4_K_M.gguf
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf; sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
echo "[$(date +%H:%M:%S)] LAUNCH round-2 (restart): continue from full1, 1 epoch, anchors 256, lr 6e-5, save every 300 steps"; ls -la "$ALL"
sudo docker run --rm --gpus all -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest python3 -u $V2/train_dspark_v2.py \
  --feats-dir "$ALL" --teacher-dir $V2/teacher --warm-start $MD/bonsai2_dspark_full1.safetensors --out "$CK" \
  --epochs 1 --batch-size 2 --lr 6e-5 --num-anchors 256 --chunk-blocks 64 --log-every 10 --save-every 300 > $V2/logs/train_${TAG}.log 2>&1; echo "train exit: $?"
sudo chown usman:p-usman "$MD"/bonsai2_dspark_${TAG}*.safetensors 2>/dev/null; chmod 0644 "$MD"/bonsai2_dspark_${TAG}*.safetensors 2>/dev/null
[ "$(sz "$CK")" -gt 1000000000 ] || { echo "[$(date +%H:%M:%S)] FAIL: no round-2 final checkpoint"; exit 1; }
python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$RAW')" > $V2/logs/conv1_${TAG}.log 2>&1; echo "step1 exit $? size $(sz "$RAW")"
python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py "$RAW" "$DONOR" "$CONV" --drop-shared-tensors > $V2/logs/conv2_${TAG}.log 2>&1; echo "step2 exit $? size $(sz "$CONV")"
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q" Q4_K_M > $V2/logs/quant_${TAG}.log 2>&1; echo "quantize exit $? size $(sz "$Q")"
[ "$(sz "$Q")" -gt 1000000000 ] && echo "[$(date +%H:%M:%S)] ROUND2 GGUF READY: $Q" || echo "[$(date +%H:%M:%S)] ROUND2 CONVERT FAILED"; echo "[$(date +%H:%M:%S)] ROUND2 PIPELINE FINISHED"
