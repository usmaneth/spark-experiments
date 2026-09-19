#!/usr/bin/env bash
# spark1, detached: for each DSpark v3 checkpoint (step*.safetensors and the final file), convert on the CPU
# (two-step + Q4_K_M) and run the 3-prompt K=5 exact probe on a GPU that holds no training or benchmark job:
# spark1 when v3 is not training, else spark2 after the v2.1 follow-on (probe + clean bench) is finished.
set -u
ROOT=/home/usman/Bonsai-demo; V2=$ROOT/dflash-training/v2; MD=$ROOT/models/bonsai2-dspark; DONOR=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
LOG=$V2/logs/dspark_v3_ckpt_eval.log; sz(){ stat -c%s "$1" 2>/dev/null || echo 0; }; say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
busy_local(){ nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE 'llama-server|gnome-remote-desktop' | wc -l; }
busy_spark2(){ ssh -o ConnectTimeout=8 spark2 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vE "llama-server|gnome-remote-desktop" | wc -l' 2>/dev/null || echo 1; }
say "v3 checkpoint waiter started (cap 60 h)"; done_list=""
for i in $(seq 1 7200); do
  for CK in $(ls $MD/bonsai2_dspark_v3_step*.safetensors $MD/bonsai2_dspark_v3.safetensors 2>/dev/null); do
    case " $done_list " in *" $CK "*) continue;; esac
    age=$(( $(date +%s) - $(stat -c%Y "$CK") )); [ "$(sz "$CK")" -gt 3000000000 ] && [ "$age" -ge 60 ] || continue
    N=$(echo "$CK" | grep -oE 'step[0-9]+' || echo final); TAG=v3$N; sudo chown usman:p-usman "$CK" 2>/dev/null; chmod 0644 "$CK"
    RAW=$MD/bonsai2-dspark-${TAG}-dspark-raw.gguf; CONV=$MD/bonsai2-dspark-${TAG}-conv.gguf; Q=$MD/bonsai2-dspark-${TAG}-Q4_K_M.gguf
    if [ "$(sz "$Q")" -lt 1000000000 ]; then
      say "CKPT $TAG: converting"
      python3 -c "import sys; sys.path.insert(0,'$ROOT/dflash-training'); from convert_safetensors_to_dspark import convert_to_dspark; convert_to_dspark('$CK','$RAW')" > $V2/logs/conv1_${TAG}.log 2>&1
      python3 $ROOT/llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py "$RAW" "$DONOR" "$CONV" --drop-shared-tensors > $V2/logs/conv2_${TAG}.log 2>&1
      LD_LIBRARY_PATH=$ROOT/bin/cuda $ROOT/bin/cuda/llama-quantize "$CONV" "$Q" Q4_K_M > $V2/logs/quant_${TAG}.log 2>&1; rm -f "$RAW" "$CONV"
      [ "$(sz "$Q")" -gt 1000000000 ] || { say "CKPT $TAG: CONVERT FAILED"; done_list="$done_list $CK"; continue; }
    fi
    sed "s#bonsai2-dspark-full1ep1-Q4_K_M.gguf#bonsai2-dspark-${TAG}-Q4_K_M.gguf#g; s#RESULT full1ep1#RESULT ${TAG}#g; s#for K in 4 5; do for W in math code code2#for K in 5; do for W in math code code2#" /tmp/ep1_sweep.sh | head -n "$(grep -n 'SWEEP DONE' /tmp/ep1_sweep.sh | head -1 | cut -d: -f1)" > /tmp/${TAG}_probe.sh
    # pick a GPU: local when nothing trains here; else spark2 once the v2.1 follow-on is finished and its GPU is free
    host=""; while [ -z "$host" ]; do
      if [ "$(busy_local)" -eq 0 ]; then host=spark1
      elif grep -q 'V21 FOLLOW-ON FINISHED' $V2/logs/dspark_v21_spark1.log 2>/dev/null && [ "$(busy_spark2)" -eq 0 ]; then host=spark2
      else sleep 300; fi
    done
    if [ $host = spark1 ]; then say "CKPT $TAG: probe on spark1"; bash /tmp/${TAG}_probe.sh | grep -E '^RESULT' | tee -a "$LOG"
    else rsync -a "$Q" spark2:$MD/ && scp -q /tmp/${TAG}_probe.sh spark2:/tmp/ && say "CKPT $TAG: probe on spark2" && ssh spark2 "bash /tmp/${TAG}_probe.sh" | grep -E '^RESULT' | tee -a "$LOG"; fi
    done_list="$done_list $CK"
  done
  grep -q 'DSPARK V3 TRAINED' $V2/logs/dspark_v3.log 2>/dev/null && [ -f $MD/bonsai2_dspark_v3.safetensors ] && case " $done_list " in *" $MD/bonsai2_dspark_v3.safetensors "*) say "all v3 checkpoints probed"; break;; esac
  sleep 30
done
say "V3 CKPT EVAL FINISHED"
