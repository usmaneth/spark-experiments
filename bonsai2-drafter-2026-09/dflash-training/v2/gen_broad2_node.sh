#!/usr/bin/env bash
# Generate self-distilled completions for one half of prompts_broad2 on this node.
# Usage: gen_broad2_node.sh <a|b>   (run on the node whose GPU is free; 24 slots, greedy, n_predict 512)
set -u
H="${1:?a or b}"
ROOT=/home/usman/Bonsai-demo
V2=$ROOT/dflash-training/v2
PORT=8095
cd $ROOT
LD_LIBRARY_PATH=bin/cuda nohup bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on -np 24 -c 36864 --host 127.0.0.1 --port $PORT > $V2/logs/server8095_broad2_$H.log 2>&1 &
echo "server pid $!" 
for i in $(seq 1 120); do curl -sf 127.0.0.1:$PORT/health >/dev/null && break; sleep 5; done
echo "[$(date +%H:%M:%S)] server up on $PORT (half $H, $(wc -l < $V2/prompts_broad2_$H.jsonl) prompts)"
cd $V2 && python3 gen_client.py $PORT 24 512 prompts_gen_broad2_$H.jsonl 100000 0 prompts_broad2_$H.jsonl > logs/gen_client_broad2_$H.log 2>&1
echo "[$(date +%H:%M:%S)] gen_client exit $? ; records: $(wc -l < prompts_gen_broad2_$H.jsonl)"
echo "GEN HALF $H DONE"
