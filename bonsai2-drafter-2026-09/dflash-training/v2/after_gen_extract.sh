#!/usr/bin/env bash
# Wait for one generation half to finish on THIS node, stop its llama-server, extract BON3 features.
# Usage: after_gen_extract.sh <a|b>
set -u
H="${1:?a or b}"
ROOT=/home/usman/Bonsai-demo; V2=$ROOT/dflash-training/v2
GLOG=$V2/logs/gen_broad2_$H.log; LOG=$V2/logs/after_gen_extract_$H.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "waiting for GEN HALF $H DONE in $GLOG"
until grep -q "GEN HALF $H DONE" "$GLOG" 2>/dev/null; do sleep 60; done
SP=$(grep -oE 'server pid [0-9]+' "$GLOG" | head -1 | awk '{print $3}')
say "gen done: $(wc -l < $V2/prompts_gen_broad2_$H.jsonl) records; stopping gen server pid ${SP:-?}"
[ -n "${SP:-}" ] && kill "$SP" 2>/dev/null; sleep 10
cd $V2 && python3 tag_gen_records.py join prompts_broad2_ids.jsonl prompts_gen_broad2_$H.jsonl prompts_gen_broad2_${H}_tags.jsonl >> "$LOG" 2>&1
say "extracting BON3 -> feats_e3/broad2_$H.bin"
LD_LIBRARY_PATH=$ROOT/bin/cuda ./extract_feats_e3 $ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf prompts_gen_broad2_$H.jsonl feats_e3/broad2_$H.bin 100000 > logs/extract_broad2_$H.log 2>&1
say "extract exit $? size $(stat -c%s feats_e3/broad2_$H.bin 2>/dev/null || echo 0)"
python3 read_feats_e3.py feats_e3/broad2_$H.bin 2>&1 | tail -3 | tee -a "$LOG"
say "EXTRACT HALF $H DONE"
