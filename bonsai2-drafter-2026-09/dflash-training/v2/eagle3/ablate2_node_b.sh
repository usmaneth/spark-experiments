#!/usr/bin/env bash
# EAGLE-3 warm-start ablations on node_b (the second series, after ablate_node_b.sh).
# Each run: 1 epoch on the existing 1.57M tokens, 100 held-out samples, val every 300 steps.
# The decoder starts from one DSpark layer (build_warm_init.py). The metric is the final `val` line.
set -u
ROOT=/home/REDACTED/Bonsai-demo
V2=$ROOT/dflash-training/v2
MD=$ROOT/models/bonsai2-eagle3
LOG=$V2/logs/eagle3_ablate2.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
COMMON="--feats-dir $V2/feats --teacher-dir $V2/teacher --draft-vocab $V2/eagle3/draft_vocab.npz --epochs 1 --batch-size 2 --max-seq-len 1024 --warmup 100 --num-workers 2 --log-every 50 --save-every 100000 --val-samples 100 --eval-every 300"
WARM_L0=$MD/bonsai2_eagle3_warmL0_init.safetensors
WARM_L4=$MD/bonsai2_eagle3_warmL4_init.safetensors
sync_init(){ # sync_init <file>: copy the init file to node_b when it is absent or has a different size
  local f="$1"
  local want; want=$(stat -c %s "$f")
  local have; have=$(ssh node_b "stat -c %s $f 2>/dev/null" || true)
  if [ "$want" != "$have" ]; then
    say "COPY $(basename "$f") -> node_b (local $want bytes, remote ${have:-absent})"
    scp -q "$f" "node_b:$f" || { say "COPY FAILED $f"; return 1; }
  fi
  ssh node_b "chmod 0644 $f 2>/dev/null; stat -c '%s %n' $f"
}
run(){ # run <name> <extra flags>
  local name="$1"; shift
  local out=$MD/bonsai2_eagle3_abl_${name}.safetensors
  say "ABL $name start: $*"
  ssh node_b "cd $ROOT && sudo docker run --rm --gpus all --ipc=host --name eagle3_abl_$name -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest \
    python3 -u $V2/eagle3/train_eagle3.py $COMMON --out $out $* 2>&1 | tee $V2/logs/eagle3_abl_${name}.log" > $V2/logs/eagle3_abl_${name}.log 2>&1
  ssh node_b "sudo chown usman:p-usman $MD/bonsai2_eagle3_abl_${name}*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_eagle3_abl_${name}*.safetensors 2>/dev/null"
  local v; v=$(grep -E '^val ' $V2/logs/eagle3_abl_${name}.log | tail -1)
  say "ABL $name done: ${v:-NO VAL LINE} | $(grep -cE '^ep ' $V2/logs/eagle3_abl_${name}.log) train lines"
}
say "ABLATIONS2 START (node_b, warm-start from DSpark layers 0 and 4)"
sync_init "$WARM_L0" || exit 1
sync_init "$WARM_L4" || exit 1
run warmL0      --ttt-depth 4 --lr 1e-4 --warm-start $WARM_L0
run warmL4      --ttt-depth 4 --lr 1e-4 --warm-start $WARM_L4
run warmL0lr5e5 --ttt-depth 4 --lr 5e-5 --warm-start $WARM_L0
say "ABLATIONS2 DONE"
grep -E 'ABL .* done' "$LOG"
