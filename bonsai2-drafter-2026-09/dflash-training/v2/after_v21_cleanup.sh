#!/usr/bin/env bash
# node_a: once the v2.1 follow-on (convert + probe + clean bench) is finished, remove the v2.1 training
# set and step checkpoints on node_b. Keeps the final v2.1 safetensors and its Q4_K_M on both nodes.
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; MD=/home/REDACTED/Bonsai-demo/models/bonsai2-dspark; LOG=$V2/logs/after_v21_cleanup.log
until grep -q 'V21 FOLLOW-ON FINISHED' $V2/logs/dspark_v21_node_a.log 2>/dev/null; do sleep 600; done
ssh node_b "rm -rf $V2/feats_v21 /models/REDACTED/bonsai2/feats/feats_tools_bon2; rm -f $V2/feats_tools_bon2 $MD/bonsai2_dspark_v21_step*.safetensors; df -B1G / | tail -1 | awk '{print \"node_b avail\", \$4, \"GB\"}'" 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] V21 CLEANUP DONE" | tee -a $LOG
