#!/usr/bin/env bash
# node_a (idle GPU while v2.1 trains on node_b): extract the 20k broad2 generations in the five-tap
# BON2 format for a DSpark v3 run. Half a first; half b only if at least 1.05 TB stays free after it
# (each half is about 7M tokens x 123 KB = ~0.9 TB). Output: feats_broad2_bon2/broad2_{a,b}.bin
set -u
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; OUT=$V2/feats_broad2_bon2; LOG=$V2/logs/extract_broad2_bon2.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
free_b(){ df --output=avail -B1 /home/REDACTED | tail -1 | tr -d ' '; }
mkdir -p $OUT; cd $V2
rsync -a node_b:$V2/prompts_gen_broad2_b.jsonl $V2/ 2>/dev/null; say "records: a=$(wc -l < prompts_gen_broad2_a.jsonl) b=$(wc -l < prompts_gen_broad2_b.jsonl 2>/dev/null || echo 0); free $(free_b) bytes"
for h in a b; do
  if [ $h = b ] && [ "$(free_b)" -lt 1050000000000 ]; then say "half b skipped: only $(free_b) bytes free"; break; fi
  say "BON2 extraction of half $h (n_ctx 2048)"
  LD_LIBRARY_PATH=$ROOT/bin/cuda ./extract_feats_tf_ctx $ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf prompts_gen_broad2_$h.jsonl $OUT/broad2_$h.bin 100000 2048 > logs/extract_broad2_${h}_bon2.log 2>&1
  say "half $h exit $? size $(stat -c%s $OUT/broad2_$h.bin 2>/dev/null || echo 0); $(grep -E 'done ===' logs/extract_broad2_${h}_bon2.log | tail -1 | cut -c1-120); free $(free_b)"
done
say "BROAD2 BON2 EXTRACTION DONE"
