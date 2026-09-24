#!/usr/bin/env bash
# Clean-window benchmark on node_b: snapshot the tool, open the window, run on node_b, pull results, close.
# Usage: run_clean_bench_node_b.sh <run-tag>
set -u
TAG="${1:?run tag}"
ROOT=/home/REDACTED/Bonsai-demo; EV=$ROOT/dflash-training/eval; SNAP=$EV/spec_bench_run_$TAG; OUT=$EV/spec_bench_full/results/$TAG
LOG=$EV/clean_bench_$TAG.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
rm -rf "$SNAP"; cp -r $EV/spec_bench_full "$SNAP"; rm -rf "$SNAP/results" "$SNAP/dryrun"; say "tool snapshot: $SNAP"
rsync -a --delete "$SNAP/" node_b:"$SNAP/"; ssh node_b "mkdir -p $OUT"
say "opening the window"; $ROOT/dflash-training/v2/eagle3/node_b_bench_window.sh open 2>&1 | tee -a "$LOG"
util=$(ssh node_b 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits'); say "gpu util before start: ${util}%"
say "launching spec_bench on node_b (detached)"
# BENCH_ARGS overrides the harness arguments; the default is the gb10-clean-1 command line.
ARGS="${BENCH_ARGS:---binary-dir bin/cuda --model models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --drafter-v2 models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --add-config dspark-v1=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf:4 --slots 1,2,4 --repeats 1 --allow-busy-gpu}"
say "harness args: $ARGS"
ssh node_b "cd $ROOT && nohup setsid python3 $SNAP/spec_bench.py $ARGS --out $OUT > $OUT/run.log 2>&1 < /dev/null & echo \$!" | tee -a "$LOG"
say "poll: tail -f node_b:$OUT/run.log ; finish marker: 'done in' line"
until ssh node_b "grep -qE '^done in|Traceback|spec_bench: error' $OUT/run.log 2>/dev/null"; do sleep 60; done
ssh node_b "tail -3 $OUT/run.log" | tee -a "$LOG"
rsync -a node_b:"$OUT/" "$OUT/" && say "results pulled to $OUT"
say "closing the window"; $ROOT/dflash-training/v2/eagle3/node_b_bench_window.sh close 2>&1 | tee -a "$LOG"
say "CLEAN BENCH $TAG FINISHED"
