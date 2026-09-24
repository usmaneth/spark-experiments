#!/usr/bin/env bash
# Quick runtime acceptance probe of an EAGLE-3 checkpoint: convert on node_a (CPU), quantize Q4_K_M,
# push to node_b, run the three 200-token prompts at K=5 exact match. Acceptance is load-independent;
# tok/s from a contended node_b is not.  Usage: eagle3_ckpt_probe.sh <ckpt.safetensors> <tag>
set -u
CK="${1:?ckpt}"; TAG="${2:?tag}"
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-eagle3
DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
CONV=$MD/bonsai2-eagle3-${TAG}-conv.gguf; Q4=$MD/bonsai2-eagle3-${TAG}-Q4_K_M.gguf; LOG=$V2/logs/eagle3_probe_${TAG}.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }
say "PROBE $TAG: $CK"
python3 $V2/eagle3/eagle3_to_gguf.py "$CK" "$DONOR" "$CONV" > $V2/logs/conv_eagle3_${TAG}.log 2>&1; [ "$(sz "$CONV")" -gt 3000000000 ] || { say "PROBE FAILED: convert"; exit 1; }
LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q4" Q4_K_M > $V2/logs/quant_eagle3_${TAG}.log 2>&1; rm -f "$CONV"; [ "$(sz "$Q4")" -gt 800000000 ] || { say "PROBE FAILED: quantize"; exit 1; }
sed "s#models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf#models/bonsai2-eagle3/bonsai2-eagle3-${TAG}-Q4_K_M.gguf#g; s#draft-dspark#draft-eagle3#g; s#RESULT full1ep1#RESULT eagle3-${TAG}#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#" /tmp/ep1_sweep.sh | head -n "$(grep -n 'SWEEP DONE' /tmp/ep1_sweep.sh | head -1 | cut -d: -f1)" > /tmp/e3probe_${TAG}.sh
# Run on the node whose GPU holds no training or generation job (llama-server demo instances are allowed).
busy(){ ssh -o ConnectTimeout=8 ${1} 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l'; }
if [ "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v llama-server | wc -l)" -eq 0 ]; then
  say "probe host: node_a (local GPU free)"; bash /tmp/e3probe_${TAG}.sh | grep -E '^RESULT|DONE' | tee -a "$LOG"
else
  for j in $(seq 1 240); do [ "$(busy node_b)" -eq 0 ] && break; sleep 300; done
  rsync -a "$Q4" node_b:$MD/ || { say "PROBE FAILED: rsync"; exit 1; }
  say "probe host: node_b (waited for a free GPU)"; scp -q /tmp/e3probe_${TAG}.sh node_b:/tmp/ && ssh node_b "bash /tmp/e3probe_${TAG}.sh" | grep -E '^RESULT|DONE' | tee -a "$LOG"
fi
say "PROBE $TAG FINISHED (acceptance valid; tok/s only if the host was idle)"
