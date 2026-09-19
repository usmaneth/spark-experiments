#!/usr/bin/env bash
# LR probe, part 2: convert on spark1 and evaluate on spark2.
# The probe script converted on spark2, but spark2's host python has no numpy, so the
# conversion produced no GGUF and the sweep printed "?" rows. This script does the
# conversion on spark1 (known-good path), pushes the Q4_K_M to spark2, and runs the
# same K=5 sweep there. It writes the real RESULT lines to logs/lr_probe.log.
set -u
TAG=lrprobe
ROOT=/home/usman/Bonsai-demo
V2=$ROOT/dflash-training/v2
MD=$ROOT/models/bonsai2-dspark
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
CK=$MD/bonsai2_dspark_${TAG}.safetensors
RAW=$MD/bonsai2-dspark-${TAG}-dspark-raw.gguf
CONV=$MD/bonsai2-dspark-${TAG}-conv.gguf
Q=$MD/bonsai2-dspark-${TAG}-Q4_K_M.gguf
LOG=$V2/logs/lr_probe.log
sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ say "LR PROBE FIX FAILED: $*"; exit 1; }

# Remove the "?" rows from the failed sweep so the dashboard parser sees only real results.
sed -i '/accept ?%/d; /| ? tok\/s/d; /MEAN exact K=5 over 6: 0.0/d' "$LOG"
say "LR PROBE FIX: convert on spark1 (spark2 python has no numpy), eval on spark2"

# 1. pull the checkpoint from spark2
rsync -a --inplace spark2:$CK $MD/ || fail "rsync checkpoint from spark2"
[ "$(sz "$CK")" -gt 3000000000 ] || fail "checkpoint too small: $(sz "$CK")"
say "ckpt on spark1: $(sz "$CK") bytes"

# 2. two-step convert + quantize (same commands as full2_step_ckpt_eval.sh)
rm -f "$RAW" "$CONV" "$Q"
python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$RAW')" > $V2/logs/conv1_${TAG}.log 2>&1
[ "$(sz "$RAW")" -gt 1000000000 ] || fail "step1 dspark raw gguf ($(tail -1 $V2/logs/conv1_${TAG}.log))"
python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py "$RAW" "$DONOR" "$CONV" --drop-shared-tensors > $V2/logs/conv2_${TAG}.log 2>&1
[ "$(sz "$CONV")" -gt 1000000000 ] || fail "step2 dflash conv gguf ($(tail -1 $V2/logs/conv2_${TAG}.log))"
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q" Q4_K_M > $V2/logs/quant_${TAG}.log 2>&1
[ "$(sz "$Q")" -gt 500000000 ] || fail "quantize ($(tail -1 $V2/logs/quant_${TAG}.log))"
rm -f "$RAW" "$CONV"
say "gguf: $(sz "$Q") $Q"

# 3. push to spark2, wait until nothing but llama-server holds the GPU
rsync -a "$Q" spark2:$MD/ || fail "rsync gguf to spark2"
for j in $(seq 1 60); do
  n=$(ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l')
  [ "${n:-1}" -eq 0 ] && break; sleep 30
done
say "eval K=5 exact (200-tok x3 + tau 0.05 x3 + long-form x6) on spark2; demo llama-server co-resident (idle unless chatted)"

# 4. same sweep as the probe script
sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#bonsai2-dspark-${TAG}-Q4_K_M.gguf#g; s#RESULT full1ep1#RESULT ${TAG}#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#; s#for K in 5 7; do for TAU in 0.02 0.05#for K in 5; do for TAU in 0.05#" /tmp/ep1_sweep.sh > /tmp/${TAG}_sweep.sh
scp -q /tmp/${TAG}_sweep.sh spark2:/tmp/${TAG}_sweep.sh && ssh spark2 "bash /tmp/${TAG}_sweep.sh" | grep -E '^RESULT|DONE' | tee -a "$LOG"
say "LR PROBE FIX FINISHED"
