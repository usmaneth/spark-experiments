#!/usr/bin/env bash
# Start the warm-start ablation series (ablate2_spark2.sh) after the first series writes ABLATIONS DONE.
V2=/home/usman/Bonsai-demo/dflash-training/v2
until grep -q 'ABLATIONS DONE' $V2/logs/eagle3_ablate.log 2>/dev/null; do sleep 60; done
sleep 60
exec $V2/eagle3/ablate2_spark2.sh
