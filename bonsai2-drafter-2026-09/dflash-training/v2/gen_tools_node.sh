#!/usr/bin/env bash
# Generate self-distilled completions for the tool/agent prompt set (prompts_tools.jsonl) on this node,
# then extract the BON3 features. Same shape as gen_broad2_node.sh, with gen_client_raw.py and a
# larger per-slot context: 24 slots x 4,160 = 99,840 (3,600 prompt + 512 gen + margin per slot).
# Usage: gen_tools_node.sh          (run on the node whose GPU is free; do not run while another
#        llama-server uses port 8095 on this node)
set -u
ROOT=/home/usman/Bonsai-demo
V2=$ROOT/dflash-training/v2
PORT=8095
cd $ROOT
LD_LIBRARY_PATH=bin/cuda nohup bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on -np 24 -c 99840 --host 127.0.0.1 --port $PORT > $V2/logs/server8095_tools.log 2>&1 &
SERVER_PID=$!; echo "server pid $SERVER_PID"
for i in $(seq 1 120); do curl -sf 127.0.0.1:$PORT/health >/dev/null && break; sleep 5; done
echo "[$(date +%H:%M:%S)] server up on $PORT ($(wc -l < $V2/prompts_tools.jsonl) prompts)"
cd $V2 && python3 gen_client_raw.py --max-prompt-tokens 3600 $PORT 24 512 prompts_gen_tools.jsonl 100000 0 prompts_tools.jsonl > logs/gen_client_tools.log 2>&1
echo "[$(date +%H:%M:%S)] gen_client_raw exit $? ; records: $(wc -l < prompts_gen_tools.jsonl)"
SP=$SERVER_PID
[ -n "${SP:-}" ] && kill "$SP" 2>/dev/null; sleep 10
python3 tag_gen_records.py join prompts_tools_ids.jsonl prompts_gen_tools.jsonl prompts_gen_tools_tags.jsonl > logs/tag_gen_tools.log 2>&1
# extract_feats_e3 has n_ctx 2048 hard coded and skips longer samples; extract_feats_e3_ctx takes n_ctx as the 5th argument.
LD_LIBRARY_PATH=$ROOT/bin/cuda ./extract_feats_e3_ctx $ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf prompts_gen_tools.jsonl feats_e3/tools.bin 100000 4608 > logs/extract_tools.log 2>&1
echo "[$(date +%H:%M:%S)] extract exit $? size $(stat -c%s feats_e3/tools.bin 2>/dev/null || echo 0)"
python3 read_feats_e3.py feats_e3/tools.bin 2>&1 | tail -3
echo "GEN TOOLS DONE"
