#!/usr/bin/env bash
# node_a: DSpark v3 = the step-600 (v2) weights trained one epoch on feats_all + the 20k broad2
# generations in BON2 (about 12M tokens, ~8x the v2 data). Starts when the broad2 extraction is done
# and the local GPU holds no other training job. Checkpoints every 500 steps for the probe chain.
set -u
ROOT=/home/REDACTED/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-dspark; LOG=$V2/logs/dspark_v3.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
gpu_busy(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE 'llama-server|gnome-remote-desktop' | wc -l; }
until grep -q 'BROAD2 BON2 EXTRACTION DONE' $V2/logs/extract_broad2_bon2.log 2>/dev/null; do sleep 300; done
for i in $(seq 1 120); do [ "$(gpu_busy)" -eq 0 ] && break; sleep 30; done
mkdir -p $V2/feats_v3; ln -f $V2/feats_all/*.bin $V2/feats_v3/; ln -f $V2/feats_broad2_bon2/*.bin $V2/feats_v3/
say "DSpark v3 training set: $(ls $V2/feats_v3 | tr '\n' ' ') ($(du -shL $V2/feats_v3 | cut -f1))"
cd $ROOT && sudo docker run --rm --gpus all --ipc=host --name dspark_v3 -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest \
  python3 -u $V2/train_dspark_v2.py --feats-dir $V2/feats_v3 --teacher-dir $V2/teacher \
  --warm-start $MD/bonsai2_dspark_full2_step600.safetensors --out $MD/bonsai2_dspark_v3.safetensors \
  --epochs 1 --batch-size 2 --lr 5e-5 --num-anchors 256 --chunk-blocks 64 --max-seq-len 4096 --log-every 10 --save-every 500 > $V2/logs/train_v3.log 2>&1
say "train exit $?"; sudo chown usman:p-usman $MD/bonsai2_dspark_v3*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_dspark_v3*.safetensors 2>/dev/null
say "DSPARK V3 TRAINED"
