#!/usr/bin/env bash
# EAGLE-3 smoke training on spark2 (docker bonsai/dflash-trainer:latest), launched from spark1.
# The trainer stdout goes to $LOG on spark2 (tee) and to the same path on spark1 (ssh pipe).
#
# Launch detached from spark1:
#   nohup setsid bash run_smoke_spark2.sh > /home/usman/Bonsai-demo/dflash-training/v2/logs/eagle3_smoke_launcher.log 2>&1 &
set -u
ROOT=/home/usman/Bonsai-demo
V2=$ROOT/dflash-training/v2
E3=$V2/eagle3
MD=$ROOT/models/bonsai2-eagle3
FEATS=$V2/feats
TAG=smoke
CK=$MD/bonsai2_eagle3_${TAG}.safetensors
LOG=$V2/logs/eagle3_${TAG}.log
NAME=eagle3_${TAG}
mkdir -p "$V2/logs"

# preflight on spark2: the code, the vocab, the features and the teacher files must exist
echo "[$(date +%H:%M:%S)] preflight on spark2"
ssh spark2 "mkdir -p $MD $V2/logs; ls -la $FEATS/; ls -la $V2/teacher/; ls -la $E3/train_eagle3.py $E3/draft_vocab.npz $V2/train_dspark_v2.py" || { echo "preflight failed"; exit 1; }

# wait until the GPU on spark2 runs no compute app other than llama-server (the demo).
# An eval sweep has short gaps between its runs: require 3 clean polls 20 s apart.
CLEAN=0
while [ "$CLEAN" -lt 3 ]; do
  OTHER=$(ssh spark2 "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader" | grep -v llama-server | grep -v '^\s*$' || true)
  if [ -z "$OTHER" ]; then
    CLEAN=$((CLEAN + 1))
  else
    CLEAN=0
    echo "[$(date +%H:%M:%S)] GPU busy on spark2: $OTHER ; wait"
  fi
  sleep 20
done
echo "[$(date +%H:%M:%S)] GPU on spark2 is clean (only llama-server)"

echo "[$(date +%H:%M:%S)] LAUNCH eagle3 ${TAG}: 1 epoch, ttt-depth 4, batch 2, seq 1024, lr 1e-4, save every 300 (spark2)"
ssh spark2 "cd $ROOT && sudo docker run --rm --gpus all --ipc=host --name $NAME -v /home/usman:/home/usman bonsai/dflash-trainer:latest \
  python3 -u $E3/train_eagle3.py \
  --feats-dir $FEATS --teacher-dir $V2/teacher --draft-vocab $E3/draft_vocab.npz \
  --out $CK --epochs 1 --ttt-depth 4 --batch-size 2 --max-seq-len 1024 --lr 1e-4 \
  --warmup 100 --num-workers 2 --log-every 10 --save-every 300 2>&1 | tee $LOG" > "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] train exit: $?"

# docker writes as root: fix the owner and the mode of every checkpoint
ssh spark2 "sudo chown usman:p-usman $MD/bonsai2_eagle3_${TAG}*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_eagle3_${TAG}*.safetensors 2>/dev/null; ls -la $MD/"
grep -E '^ep .*step' "$LOG" | tail -3
echo "[$(date +%H:%M:%S)] SMOKE FINISHED: $CK"
