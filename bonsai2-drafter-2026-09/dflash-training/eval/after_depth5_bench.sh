#!/usr/bin/env bash
# Start the clean benchmark as soon as the depth5 ablation reports done (the pause flag keeps the chain waiting).
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; EV=/home/REDACTED/Bonsai-demo/dflash-training/eval
until grep -q 'ABL depth5 done' $V2/logs/eagle3_ablate.log 2>/dev/null; do sleep 30; done
sleep 20
exec $EV/run_clean_bench_node_b.sh gb10-clean-1
