#!/usr/bin/env bash
# EAGLE-3 scale-up run B on node_a: continue from the scale-up A EMA weights on all data
# (BON2 feats_all + BON3 broad2_a + broad2_b, about 12.5M tokens), 1 epoch, lower LR, EMA.
set -u
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-eagle3
LOG=$V2/logs/eagle3_scaleB.log; L=$V2/logs/eagle3_scaleB_launcher.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
say "rsync half-B features from node_b"
rsync -a --inplace node_b:$V2/feats_e3/broad2_b.bin $V2/feats_e3/ || { say "rsync failed"; exit 1; }
say "half-B features: $(stat -c%s $V2/feats_e3/broad2_b.bin) bytes"
mkdir -p $V2/feats_scaleB; ln -f $V2/feats_all/*.bin $V2/feats_scaleB/ 2>/dev/null; ln -f $V2/feats_e3/broad2_a.bin $V2/feats_scaleB/; ln -f $V2/feats_e3/broad2_b.bin $V2/feats_scaleB/
say "training set: $(ls $V2/feats_scaleB | tr '\n' ' ')"
sudo docker run --rm --gpus all --ipc=host --name eagle3_scaleB -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest \
  python3 -u $V2/eagle3/train_eagle3.py --feats-dir $V2/feats_scaleB --teacher-dir $V2/teacher --draft-vocab $V2/eagle3/draft_vocab.npz \
  --warm-start $MD/bonsai2_eagle3_scaleA_ema.safetensors --out $MD/bonsai2_eagle3_scaleB.safetensors \
  --epochs 1 --ttt-depth 4 --batch-size 2 --max-seq-len 1024 --lr 5e-5 --warmup 200 --num-workers 2 \
  --log-every 50 --save-every 2000 --val-samples 200 --eval-every 1000 --ema 0.999 > "$LOG" 2>&1
say "train exit $?"; sudo chown usman:p-usman $MD/bonsai2_eagle3_scaleB*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_eagle3_scaleB*.safetensors 2>/dev/null
say "SCALE-UP B DONE: $(grep -E '^val ' "$LOG" | tail -1 | cut -c1-100)"
CK=$MD/bonsai2_eagle3_scaleB_ema.safetensors; [ -f "$CK" ] || CK=$MD/bonsai2_eagle3_scaleB.safetensors
exec $V2/eagle3/eagle3_ckpt_probe.sh "$CK" scaleB-final
