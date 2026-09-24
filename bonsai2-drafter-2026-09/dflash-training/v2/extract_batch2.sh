#!/usr/bin/env bash
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; cd $V2; export LD_LIBRARY_PATH=/home/REDACTED/Bonsai-demo/bin/cuda
M=/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
echo "[$(date +%H:%M:%S)] extract start on $(wc -l < prompts_gen_batch2.jsonl) completions"; rm -f feats/batch2.bin.tmp
./extract_feats_tf "$M" prompts_gen_batch2.jsonl feats/batch2.bin.tmp 100000 2>&1 | grep -vE 'llama_model_loader|load_tensors|print_info|ggml_cuda|^\s*$' | tail -20
sz=$(stat -c%s feats/batch2.bin.tmp 2>/dev/null || echo 0); echo "[$(date +%H:%M:%S)] tmp size $sz"
if [ "$sz" -gt 100000000 ]; then mv feats/batch2.bin.tmp feats/batch2.bin && echo "[$(date +%H:%M:%S)] BATCH2 WRITTEN $(stat -c%s feats/batch2.bin) bytes"; else echo "[$(date +%H:%M:%S)] EXTRACT FAILED size $sz"; fi
