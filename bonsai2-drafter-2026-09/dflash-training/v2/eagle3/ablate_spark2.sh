#!/usr/bin/env bash
# EAGLE-3 recipe ablations on spark2 (runs on spark2 alongside generation half b).
# Each run: 1 epoch on the existing 1.57M tokens, 100 held-out samples, val every 300 steps.
# The metric is the final `val` line (per-depth agreement). tok/s needs a clean GPU later.
set -u
ROOT=/home/usman/Bonsai-demo
V2=$ROOT/dflash-training/v2
MD=$ROOT/models/bonsai2-eagle3
LOG=$V2/logs/eagle3_ablate.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
COMMON="--feats-dir $V2/feats --teacher-dir $V2/teacher --draft-vocab $V2/eagle3/draft_vocab.npz --epochs 1 --batch-size 2 --max-seq-len 1024 --warmup 100 --num-workers 2 --log-every 50 --save-every 100000 --val-samples 100 --eval-every 300"
run(){ # run <name> <extra flags>
  local name="$1"; shift
  local out=$MD/bonsai2_eagle3_abl_${name}.safetensors
  say "ABL $name start: $*"
  ssh spark2 "cd $ROOT && sudo docker run --rm --gpus all --ipc=host --name eagle3_abl_$name -v /home/usman:/home/usman bonsai/dflash-trainer:latest \
    python3 -u $V2/eagle3/train_eagle3.py $COMMON --out $out $* 2>&1 | tee $V2/logs/eagle3_abl_${name}.log" > $V2/logs/eagle3_abl_${name}.log 2>&1
  ssh spark2 "sudo chown usman:p-usman $MD/bonsai2_eagle3_abl_${name}*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_eagle3_abl_${name}*.safetensors 2>/dev/null"
  local v; v=$(grep -E '^val ' $V2/logs/eagle3_abl_${name}.log | tail -1)
  say "ABL $name done: ${v:-NO VAL LINE} | $(grep -cE '^ep ' $V2/logs/eagle3_abl_${name}.log) train lines"
}
say "ABLATIONS START (spark2, contended by gen half b)"
run base      --ttt-depth 4 --lr 1e-4
run depth5    --ttt-depth 5 --lr 1e-4
run taps134   --ttt-depth 4 --lr 1e-4 --taps 1,3,4
run hard05    --ttt-depth 4 --lr 1e-4 --hard-alpha 0.5
run lr3e4b4   --ttt-depth 4 --lr 3e-4 --accum 2
run ep3ema    --ttt-depth 4 --lr 1e-4 --epochs 3 --ema 0.999
say "ABLATIONS DONE"
grep -E 'ABL .* done' "$LOG"
