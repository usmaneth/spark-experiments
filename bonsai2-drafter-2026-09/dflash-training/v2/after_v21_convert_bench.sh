#!/usr/bin/env bash
# spark1: wait for the DSpark v2.1 fine-tune on spark2, convert it here (spark2 has no numpy),
# push the Q4_K_M back, run the 3-prompt K=5 probe, then the clean-window benchmark
# (baseline + DSpark v2 + DSpark v2.1, one slot, all categories). Runs unattended.
set -u
TAG=v21
ROOT=/home/usman/Bonsai-demo; V2=$ROOT/dflash-training/v2; EV=$ROOT/dflash-training/eval; MD=$ROOT/models/bonsai2-dspark
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
CK=$MD/bonsai2_dspark_v21.safetensors
RAW=$MD/bonsai2-dspark-${TAG}-dspark-raw.gguf; CONV=$MD/bonsai2-dspark-${TAG}-conv.gguf; Q=$MD/bonsai2-dspark-${TAG}-Q4_K_M.gguf
LOG=$V2/logs/dspark_v21_spark1.log
sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ say "V21 FOLLOW-ON FAILED: $*"; exit 1; }

until ssh -o ConnectTimeout=8 spark2 "grep -q 'DSPARK V21 TRAINED' $V2/logs/dspark_v21.log 2>/dev/null"; do sleep 300; done
say "spark2 reports DSPARK V21 TRAINED: $(ssh spark2 "grep -E 'train exit|epoch|loss' $V2/logs/train_v21.log | tail -2 | cut -c1-120" 2>/dev/null | tr '\n' ' ')"

rsync -a --inplace spark2:$CK $MD/ || fail "rsync checkpoint from spark2"
[ "$(sz "$CK")" -gt 3000000000 ] || fail "checkpoint too small: $(sz "$CK")"
say "ckpt on spark1: $(sz "$CK") bytes"

rm -f "$RAW" "$CONV" "$Q"
python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$RAW')" > $V2/logs/conv1_${TAG}.log 2>&1
[ "$(sz "$RAW")" -gt 1000000000 ] || fail "step1 dspark raw gguf ($(tail -1 $V2/logs/conv1_${TAG}.log))"
python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py "$RAW" "$DONOR" "$CONV" --drop-shared-tensors > $V2/logs/conv2_${TAG}.log 2>&1
[ "$(sz "$CONV")" -gt 1000000000 ] || fail "step2 dflash conv gguf ($(tail -1 $V2/logs/conv2_${TAG}.log))"
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q" Q4_K_M > $V2/logs/quant_${TAG}.log 2>&1
[ "$(sz "$Q")" -gt 500000000 ] || fail "quantize ($(tail -1 $V2/logs/quant_${TAG}.log))"
rm -f "$RAW" "$CONV"
say "gguf: $(sz "$Q") $Q"
rsync -a "$Q" spark2:$MD/ || fail "rsync gguf to spark2"

for j in $(seq 1 120); do
  n=$(ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l')
  [ "${n:-1}" -eq 0 ] && break; sleep 30
done
say "probe K=5 exact (3 prompts, 200 tokens) on spark2"
sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#bonsai2-dspark-${TAG}-Q4_K_M.gguf#g; s#RESULT full1ep1#RESULT ${TAG}#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#; s#for K in 5 7; do for TAU in 0.02 0.05#for K in 5; do for TAU in 0.05#" /tmp/ep1_sweep.sh > /tmp/${TAG}_sweep.sh
scp -q /tmp/${TAG}_sweep.sh spark2:/tmp/${TAG}_sweep.sh && ssh spark2 "bash /tmp/${TAG}_sweep.sh" | grep -E '^RESULT|DONE' | tee -a "$LOG"

say "clean-window benchmark: baseline + DSpark v2 + DSpark v2.1, one slot, all categories"
export BENCH_ARGS="--binary-dir bin/cuda --model models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --drafter-v2 models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --add-config dspark-v21=models/bonsai2-dspark/bonsai2-dspark-${TAG}-Q4_K_M.gguf:5 --configs baseline,dspark-v2,dspark-v21 --slots 1 --repeats 1 --allow-busy-gpu"
$EV/run_clean_bench_spark2.sh gb10-v21-1 2>&1 | tail -5 | tee -a "$LOG"
say "V21 FOLLOW-ON FINISHED: $EV/spec_bench_full/results/gb10-v21-1/summary.md"
