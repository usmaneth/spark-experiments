#!/usr/bin/env bash
# On spark2: after half b is generated and extracted, generate the tool/agent prompt set and extract it.
V2=/home/usman/Bonsai-demo/dflash-training/v2
until grep -q 'EXTRACT HALF b DONE' $V2/logs/after_gen_extract_b.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] half b extracted; starting tool/agent generation" >> $V2/logs/after_b_tools.log
cd $V2 && bash gen_tools_node.sh >> $V2/logs/after_b_tools.log 2>&1
echo "[$(date +%H:%M:%S)] TOOLS GEN+EXTRACT DONE" >> $V2/logs/after_b_tools.log
