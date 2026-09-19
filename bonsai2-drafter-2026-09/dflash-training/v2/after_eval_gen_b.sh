#!/usr/bin/env bash
# When the EAGLE-3 smoke eval on spark2 has finished, start generation half b on spark2 and its extractor waiter.
set -u
V2=/home/usman/Bonsai-demo/dflash-training/v2; LOG=$V2/logs/after_eval_gen_b.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
until grep -qE 'EAGLE3 smoke EVAL FINISHED|EAGLE3 EVAL FAILED' $V2/logs/eagle3_smoke_eval.log 2>/dev/null; do sleep 60; done
say "smoke eval finished; launching gen half b on spark2"
ssh spark2 "cd $V2 && nohup setsid ./gen_broad2_node.sh b > logs/gen_broad2_b.log 2>&1 < /dev/null & sleep 1; nohup setsid ./after_gen_extract.sh b > logs/after_gen_extract_b_launcher.log 2>&1 < /dev/null & sleep 2; pgrep -af 'gen_broad2_node.sh b|after_gen_extract.sh b' | grep -v pgrep | cut -c1-70"
say "GEN B LAUNCHED"
