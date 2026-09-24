#!/usr/bin/env bash
# EAGLE-3 scale-up run A on node_a: waits for the half-A BON3 extraction, then trains on
# BON2 feats_all (1.57M tokens) + BON3 broad2_a (about 4.5M tokens). Control recipe
# (depth 4, lr 1e-4, random init), 2 epochs, EMA, 200 held-out samples.
set -u
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-eagle3
LOG=$V2/logs/eagle3_scaleA.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$V2/logs/eagle3_scaleA_launcher.log"; }
until grep -q 'EXTRACT HALF a DONE' $V2/logs/after_gen_extract_a.log 2>/dev/null; do sleep 120; done
say "half-A features ready: $(stat -c%s $V2/feats_e3/broad2_a.bin 2>/dev/null) bytes"
for i in $(seq 1 60); do n=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v llama-server | wc -l); [ "$n" -eq 0 ] && break; sleep 30; done
mkdir -p $V2/feats_scaleA; ln -f $V2/feats_all/*.bin $V2/feats_scaleA/ 2>/dev/null; ln -f $V2/feats_e3/broad2_a.bin $V2/feats_scaleA/broad2_a.bin
say "training set: $(ls $V2/feats_scaleA | tr '\n' ' ')"
say "launch scale-up A (node_a)"
sudo docker run --rm --gpus all --ipc=host --name eagle3_scaleA -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest \
  python3 -u $V2/eagle3/train_eagle3.py --feats-dir $V2/feats_scaleA --teacher-dir $V2/teacher --draft-vocab $V2/eagle3/draft_vocab.npz \
  --out $MD/bonsai2_eagle3_scaleA.safetensors --epochs 2 --ttt-depth 4 --batch-size 2 --max-seq-len 1024 --lr 1e-4 --warmup 200 \
  --num-workers 2 --log-every 50 --save-every 1000 --val-samples 200 --eval-every 500 --ema 0.999 > "$LOG" 2>&1
say "train exit $?"; sudo chown usman:p-usman $MD/bonsai2_eagle3_scaleA*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_eagle3_scaleA*.safetensors 2>/dev/null
say "SCALE-UP A DONE: $(grep -E '^val ' "$LOG" | tail -1 | cut -c1-100)"
