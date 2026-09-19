#!/usr/bin/env bash
# spark2: after the tool/agent set is generated, extract it in the five-tap BON2 format and
# fine-tune the DSpark step-600 weights on it (DSpark v2.1). Runs unattended.
set -u
ROOT=/home/usman/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-dspark
LOG=$V2/logs/dspark_v21.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
until grep -q 'TOOLS GEN+EXTRACT DONE' $V2/logs/after_b_tools.log 2>/dev/null; do sleep 180; done
say "tool set ready: $(wc -l < $V2/prompts_gen_tools.jsonl) records"
for i in $(seq 1 60); do n=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v llama-server | wc -l); [ "$n" -eq 0 ] && break; sleep 30; done
mkdir -p $V2/feats_tools_bon2
say "BON2 extraction of the tool set (n_ctx 4608)"
LD_LIBRARY_PATH=$ROOT/bin/cuda $V2/extract_feats_tf_ctx $ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf $V2/prompts_gen_tools.jsonl $V2/feats_tools_bon2/tools.bin 100000 4608 > $V2/logs/extract_tools_bon2.log 2>&1
say "extract exit $? size $(stat -c%s $V2/feats_tools_bon2/tools.bin 2>/dev/null || echo 0)"
mkdir -p $V2/feats_v21; ln -f $V2/feats/batch1.bin $V2/feats/batch2.bin $V2/feats_v21/ 2>/dev/null; ln -f $V2/feats_tools_bon2/tools.bin $V2/feats_v21/tools.bin
say "train DSpark v2.1 from step-600 on $(ls $V2/feats_v21 | tr '\n' ' ')"
cd $ROOT && sudo docker run --rm --gpus all --ipc=host --name dspark_v21 -v /home/usman:/home/usman bonsai/dflash-trainer:latest \
  python3 -u $V2/train_dspark_v2.py --feats-dir $V2/feats_v21 --teacher-dir $V2/teacher \
  --warm-start $MD/bonsai2_dspark_full2_step600.safetensors --out $MD/bonsai2_dspark_v21.safetensors \
  --epochs 1 --batch-size 2 --lr 3e-5 --num-anchors 256 --chunk-blocks 64 --max-seq-len 4096 --log-every 10 --save-every 300 > $V2/logs/train_v21.log 2>&1
say "train exit $?"; sudo chown usman:p-usman $MD/bonsai2_dspark_v21*.safetensors 2>/dev/null; chmod 0644 $MD/bonsai2_dspark_v21*.safetensors 2>/dev/null
say "DSPARK V21 TRAINED"
