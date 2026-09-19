#!/usr/bin/env bash
# spark1: does the drafter's quantization cap DSpark v2's acceptance? Convert the step-600 checkpoint
# once, quantize to Q5_K_M, Q6_K, Q8_0 and F16, and run the 3-prompt K=5 exact probe on each plus the
# shipped Q4_K_M. Idle GPU (the :8085 demo server is resident), so tok/s is comparable across variants.
set -u
ROOT=/home/usman/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-dspark; QP=$MD/quantprobe
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
CK=$MD/bonsai2_dspark_full2_step600.safetensors; LOG=$V2/logs/dspark_quant_probe.log
sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p $QP; cd $ROOT
say "QUANT PROBE: convert step600 to f32 dflash gguf"
python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$QP/raw.gguf')" > $V2/logs/quantprobe_conv1.log 2>&1
python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py $QP/raw.gguf $DONOR $QP/v2-F32.gguf --drop-shared-tensors > $V2/logs/quantprobe_conv2.log 2>&1
[ "$(sz $QP/v2-F32.gguf)" -gt 1000000000 ] || { say "QUANT PROBE FAILED: convert ($(tail -1 $V2/logs/quantprobe_conv2.log))"; exit 1; }
rm -f $QP/raw.gguf; say "f32 gguf: $(sz $QP/v2-F32.gguf) bytes"
for Q in Q5_K_M Q6_K Q8_0 F16; do
  LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize $QP/v2-F32.gguf $QP/v2-$Q.gguf $Q > $V2/logs/quantprobe_q_$Q.log 2>&1
  say "$Q: $(sz $QP/v2-$Q.gguf) bytes"
done
ln -f $MD/bonsai2-dspark-full2step600-Q4_K_M.gguf $QP/v2-Q4_K_M.gguf
for Q in Q4_K_M Q5_K_M Q6_K Q8_0 F16; do
  [ "$(sz $QP/v2-$Q.gguf)" -gt 500000000 ] || { say "skip $Q (no file)"; continue; }
  sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#quantprobe/v2-$Q.gguf#g; s#RESULT full1ep1#RESULT v2-$Q#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#" /tmp/ep1_sweep.sh | head -n "$(grep -n 'SWEEP DONE' /tmp/ep1_sweep.sh | head -1 | cut -d: -f1)" > /tmp/quantprobe_$Q.sh
  bash /tmp/quantprobe_$Q.sh | grep -E '^RESULT' | tee -a "$LOG"
done
say "QUANT PROBE FINISHED"
