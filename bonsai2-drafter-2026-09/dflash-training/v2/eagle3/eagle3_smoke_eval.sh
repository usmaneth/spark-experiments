#!/usr/bin/env bash
# EAGLE-3 smoke checkpoint: wait for the node_b training container to exit, convert on node_a,
# quantize Q8_0 and Q4_K_M, push to node_b, run the K=5 sweep on a clean node_b GPU.
# Usage: eagle3_smoke_eval.sh [tag=smoke] [container=feed9161d260]
set -u
TAG="${1:-smoke}"; CONT="${2:-feed9161d260}"
ROOT=/home/REDACTED/Bonsai-demo
V2=$ROOT/dflash-training/v2
MD=$ROOT/models/bonsai2-eagle3
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
CK=$MD/bonsai2_eagle3_${TAG}.safetensors
CONV=$MD/bonsai2-eagle3-${TAG}-conv.gguf
Q8=$MD/bonsai2-eagle3-${TAG}-Q8_0.gguf
Q4=$MD/bonsai2-eagle3-${TAG}-Q4_K_M.gguf
LOG=$V2/logs/eagle3_${TAG}_eval.log
sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ say "EAGLE3 EVAL FAILED: $*"; exit 1; }
gpu_busy(){ ssh -o ConnectTimeout=8 node_b 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l'; }

say "EAGLE3 $TAG eval: waiting for container $CONT to exit"
for i in $(seq 1 240); do
  ssh -o ConnectTimeout=8 node_b "sudo docker ps -q --no-trunc | grep -q '^$CONT'" || break; sleep 30
done
for i in $(seq 1 20); do
  s=$(ssh -o ConnectTimeout=8 node_b "stat -c%s $CK 2>/dev/null || echo 0"); [ "$s" -gt 3000000000 ] && break; sleep 15
done
[ "${s:-0}" -gt 3000000000 ] || fail "checkpoint missing or small on node_b ($s)"
sleep 30
rsync -a --inplace node_b:$CK $MD/ || fail "rsync checkpoint"
say "ckpt: $(sz "$CK") $CK"

rm -f "$CONV" "$Q8" "$Q4"
python3 $V2/eagle3/eagle3_to_gguf.py "$CK" "$DONOR" "$CONV" > $V2/logs/conv_eagle3_${TAG}.log 2>&1
[ "$(sz "$CONV")" -gt 3000000000 ] || fail "convert ($(tail -1 $V2/logs/conv_eagle3_${TAG}.log))"
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q8" Q8_0 > $V2/logs/quant_eagle3_${TAG}_q8.log 2>&1
[ "$(sz "$Q8")" -gt 1500000000 ] || fail "quantize Q8_0"
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q4" Q4_K_M > $V2/logs/quant_eagle3_${TAG}_q4.log 2>&1
[ "$(sz "$Q4")" -gt 800000000 ] || fail "quantize Q4_K_M"
rm -f "$CONV"
say "gguf: Q8_0 $(sz "$Q8") bytes, Q4_K_M $(sz "$Q4") bytes"
rsync -a "$Q8" "$Q4" node_b:$MD/ || fail "rsync gguf to node_b"

for j in $(seq 1 60); do n=$(gpu_busy); [ "${n:-1}" -eq 0 ] && break; sleep 30; done
n=$(gpu_busy); state="clean"; [ "${n:-1}" -eq 0 ] || state="CONTENDED"
say "EAGLE3 $TAG: eval on node_b ($state), demo llama-server co-resident"

mk(){ # mk <out> <gguf-relpath> <tag> <pmin>
  sed "s#models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf#$2#g; s#draft-dspark#draft-eagle3#g; s#RESULT full1ep1#RESULT $3#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#; s#for K in 5 7; do for TAU in 0.02 0.05#for K in 5; do for TAU in 0.05#; s#--spec-draft-p-min 0 #--spec-draft-p-min $4 #g" /tmp/ep1_sweep.sh > "$1"
}
SHORT=$(grep -n 'SWEEP+TYP DONE' /tmp/ep1_sweep.sh | head -1 | cut -d: -f1)
mk /tmp/e3_${TAG}_q8.sh    models/bonsai2-eagle3/bonsai2-eagle3-${TAG}-Q8_0.gguf   eagle3${TAG}Q8        0
mk /tmp/e3_${TAG}_q4.sh    models/bonsai2-eagle3/bonsai2-eagle3-${TAG}-Q4_K_M.gguf eagle3${TAG}Q4        0
mk /tmp/e3_${TAG}_q8p5.sh  models/bonsai2-eagle3/bonsai2-eagle3-${TAG}-Q8_0.gguf   eagle3${TAG}Q8pmin0.5 0.5
head -n "$SHORT" /tmp/e3_${TAG}_q4.sh   > /tmp/e3_${TAG}_q4s.sh
head -n "$SHORT" /tmp/e3_${TAG}_q8p5.sh > /tmp/e3_${TAG}_q8p5s.sh
for s in q8 q4s q8p5s; do
  scp -q /tmp/e3_${TAG}_$s.sh node_b:/tmp/e3_${TAG}_$s.sh && ssh node_b "bash /tmp/e3_${TAG}_$s.sh" | grep -E '^RESULT|DONE' | tee -a "$LOG"
done
say "EAGLE3 $TAG EVAL FINISHED"
