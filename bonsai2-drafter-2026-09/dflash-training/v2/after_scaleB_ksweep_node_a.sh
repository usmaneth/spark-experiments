#!/usr/bin/env bash
# node_a: after EAGLE-3 run B and its probe release the GPU, sweep the DSpark v2 draft length
# (K=3,4,5,6,7) over the workload matrix with the benchmark harness. The demo llama-server on
# :8085 stays resident (idle unless chatted). Output: eval/spec_bench_full/results/gb10-ksweep-1
set -u
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; EV=$ROOT/dflash-training/eval
OUT=$EV/spec_bench_full/results/gb10-ksweep-1; LOG=$V2/logs/ksweep_node_a.log
D=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
until grep -q 'PROBE scaleB-final FINISHED' $V2/logs/eagle3_probe_scaleB-final.log 2>/dev/null; do sleep 300; done
for j in $(seq 1 120); do n=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v llama-server | wc -l); [ "$n" -eq 0 ] && break; sleep 30; done
say "GPU free; K sweep of DSpark v2 (K=3..7), 48 matrix prompts, one slot"
mkdir -p "$OUT"; cd $ROOT
python3 $EV/spec_bench_full/spec_bench.py --binary-dir bin/cuda --model models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --drafter-v2 $D --add-config dspark-v2-k3=$D:3 --add-config dspark-v2-k4=$D:4 --add-config dspark-v2-k6=$D:6 --add-config dspark-v2-k7=$D:7 \
  --configs dspark-v2,dspark-v2-k3,dspark-v2-k4,dspark-v2-k6,dspark-v2-k7 --categories code,math,reasoning,chat,tool,agent \
  --slots 1 --repeats 1 --no-single --allow-busy-gpu --out "$OUT" > "$OUT/run.log" 2>&1
say "harness exit $?: $(grep -E '^done in' "$OUT/run.log" | tail -1)"
say "K SWEEP FINISHED: $OUT/summary.md"
