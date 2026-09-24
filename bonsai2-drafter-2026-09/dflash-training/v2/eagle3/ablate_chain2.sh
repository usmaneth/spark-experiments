#!/usr/bin/env bash
# Reduced ablation chain on node_b (runs alongside generation half b).
# Waits for the running `base` container, then: depth5, warmL0, warmL4.
# A bench window flag (/tmp/node_b_bench_window on node_a) pauses the chain between runs.
set -u
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-eagle3
LOG=$V2/logs/eagle3_ablate.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
COMMON="--feats-dir $V2/feats --teacher-dir $V2/teacher --draft-vocab $V2/eagle3/draft_vocab.npz --epochs 1 --batch-size 2 --max-seq-len 1024 --warmup 100 --num-workers 2 --log-every 50 --save-every 100000 --val-samples 100 --eval-every 300"
finish(){ # finish <name>
  ssh node_b "sudo chown usman:p-usman $MD/bonsai2_eagle3_abl_$1*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_eagle3_abl_$1*.safetensors 2>/dev/null"
  local v; v=$(grep -E '^val ' $V2/logs/eagle3_abl_$1.log | tail -1)
  say "ABL $1 done: ${v:-NO VAL LINE} | $(grep -cE '^ep ' $V2/logs/eagle3_abl_$1.log) train lines"
}
run(){ # run <name> <extra flags>
  local name="$1"; shift
  while [ -f /tmp/node_b_bench_window ]; do sleep 60; done
  say "ABL $name start: $*"
  ssh node_b "cd $ROOT && sudo docker run --rm --gpus all --ipc=host --name eagle3_abl_$name -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest \
    python3 -u $V2/eagle3/train_eagle3.py $COMMON --out $MD/bonsai2_eagle3_abl_${name}.safetensors $* 2>&1 | tee $V2/logs/eagle3_abl_${name}.log" > $V2/logs/eagle3_abl_${name}.log 2>&1
  finish "$name"
}
say "CHAIN2: waiting for the base container to exit"
while ssh -o ConnectTimeout=8 node_b 'sudo docker ps --format "{{.Names}}" | grep -q "^eagle3_abl_base$"'; do sleep 60; done
finish base
run depth5 --ttt-depth 5 --lr 1e-4
run warmL0 --ttt-depth 4 --lr 1e-4 --warm-start $MD/bonsai2_eagle3_warmL0_init.safetensors
run warmL4 --ttt-depth 4 --lr 1e-4 --warm-start $MD/bonsai2_eagle3_warmL4_init.safetensors
say "ABLATIONS DONE"
grep -E 'ABL .* done' "$LOG"
