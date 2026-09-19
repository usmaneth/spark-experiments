#!/usr/bin/env bash
# Start the ablation series once generation half b is running on spark2 (after the smoke eval).
V2=/home/usman/Bonsai-demo/dflash-training/v2
until grep -q 'GEN B LAUNCHED' $V2/logs/after_eval_gen_b.log 2>/dev/null; do sleep 60; done
sleep 120
exec $V2/eagle3/ablate_spark2.sh
