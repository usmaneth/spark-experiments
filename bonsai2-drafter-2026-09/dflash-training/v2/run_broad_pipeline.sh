#!/usr/bin/env bash
# Broad-mix Medusa retrain pipeline (node_b). Stages, in order:
#   waitgen  - wait for gen_client_broad to finish, then stop the llama-server on :8095
#   extract  - teacher-force feature extraction -> feats/batch3_broad.bin (+ category tags)
#   rsync    - copy batch3_broad.bin (+ prompt files) to node_a over the direct link (background)
#   train    - Medusa heads on feats/*.bin -> medusa/medusa_heads_v2.safetensors
#   evalcat  - per-category held-out eval (v2 vs v1 on the same split)
#   merge    - merge v2 heads into a copy of the PQ2_0 GGUF
#   bench    - llama-speculative-simple math/code at n_max 2 and 4 (v2 heads, plus v1 n_max=2 reference)
# Usage: run_broad_pipeline.sh [stage ...]   (default: all stages in order)
set -uo pipefail
ROOT=/home/REDACTED/Bonsai-demo
V2=$ROOT/dflash-training/v2
MODEL=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
MERGED=$ROOT/models/bonsai2-medusa/Ternary-Bonsai-2-27B-PQ2_0-medusa-v2.gguf
V1=$ROOT/models/bonsai2-medusa/Ternary-Bonsai-2-27B-PQ2_0-medusa.gguf
DOCKER="sudo docker run --rm --gpus all -v /home/REDACTED:/home/REDACTED bonsai/dflash-trainer:latest"
LOGS=$V2/logs
mkdir -p "$LOGS" "$ROOT/logs/medusa"
ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; }

wait_gpu_free() {
    # wait while any compute process that is not ours sits on the GPU
    while true; do
        procs=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v "^$" || true)
        [ -z "$procs" ] && return 0
        say "GPU busy, waiting: $procs"
        sleep 30
    done
}

stage_waitgen() {
    say "waitgen: waiting for gen_client_broad"
    until grep -q "gen done" "$LOGS/gen_client_broad.log" || ! pgrep -f "gen_client.py 8095" >/dev/null; do sleep 30; done
    tail -2 "$LOGS/gen_client_broad.log"
    say "records: $(wc -l < "$V2/prompts_gen_broad.jsonl")"
    pid=$(pgrep -u "$(id -un)" -f "llama-server -m .*PQ2_0.gguf --port 8095" || true)
    if [ -n "$pid" ]; then say "stopping my llama-server pid $pid"; kill $pid; sleep 5; fi
}

stage_extract() {
    say "extract: start"
    wait_gpu_free
    cd "$V2"
    LD_LIBRARY_PATH=$ROOT/bin/cuda ./extract_feats_tf "$MODEL" prompts_gen_broad.jsonl feats/batch3_broad.bin.tmp 100000 > "$LOGS/extract_batch3_broad.log" 2>&1
    rc=$?
    grep -E "=== done|feat [0-9]+00 " "$LOGS/extract_batch3_broad.log" | tail -3
    if [ $rc -ne 0 ] || ! grep -q "=== done" "$LOGS/extract_batch3_broad.log"; then say "extract FAILED rc=$rc"; return 1; fi
    mv feats/batch3_broad.bin.tmp feats/batch3_broad.bin
    say "extract: $(stat -c%s feats/batch3_broad.bin) bytes -> feats/batch3_broad.bin"
    python3 tag_gen_records.py join prompts_broad_ids.jsonl prompts_gen_broad.jsonl prompts_gen_broad_tags.jsonl 2> "$LOGS/tags_broad.log"
    cat "$LOGS/tags_broad.log"
}

stage_rsync() {
    say "rsync: batch3_broad.bin -> node_a (background, direct link 203.0.113.11)"
    (
        rsync -W --inplace --partial -e "ssh -o StrictHostKeyChecking=accept-new" \
            "$V2/feats/batch3_broad.bin" user@203.0.113.11:"$V2/feats/batch3_broad.bin.tmp" \
        && ssh user@203.0.113.11 "mv $V2/feats/batch3_broad.bin.tmp $V2/feats/batch3_broad.bin && stat -c%s $V2/feats/batch3_broad.bin" \
        && rsync -e "ssh -o StrictHostKeyChecking=accept-new" "$V2"/prompts_broad.jsonl "$V2"/prompts_broad.jsonl.counts.json \
            "$V2"/prompts_broad_ids.jsonl "$V2"/prompts_gen_broad.jsonl "$V2"/prompts_gen_broad_tags.jsonl \
            "$V2"/build_prompts_broad.py "$V2"/tag_gen_records.py "$V2"/run_broad_pipeline.sh user@203.0.113.11:"$V2"/ \
        && echo "RSYNC_DONE $(date +%H:%M:%S)"
        echo "rsync rc=$?"
    ) > "$LOGS/rsync_batch3_broad.log" 2>&1 &
    say "rsync pid $!"
}

stage_train() {
    say "train: start (feats: $(ls "$V2"/feats/*.bin | xargs -n1 basename | tr '\n' ' '))"
    wait_gpu_free
    "$V2/medusa/run_train.sh" --out "$V2/medusa/medusa_heads_v2.safetensors" > "$LOGS/train_medusa_v2.log" 2>&1
    rc=$?
    grep -E "^\[data\]|^\[epoch|FINAL|head [0-9]|base LM|loss first|save" "$LOGS/train_medusa_v2.log" | tail -30
    [ -s "$V2/medusa/medusa_heads_v2.safetensors" ] || { say "train FAILED rc=$rc"; return 1; }
    say "train: done rc=$rc"
}

stage_evalcat() {
    say "evalcat: start"
    wait_gpu_free
    $DOCKER python3 "$V2/medusa/eval_medusa_by_cat.py" --heads "$V2/medusa/medusa_heads_v2.safetensors" \
        --heads-b "$V2/medusa/medusa_heads.safetensors" --out "$V2/medusa/eval_by_cat_v2.json" 2>&1 \
        | grep -v "CUDA Graph\|SHMEM\|NVIDIA\|docker run\|^$\|cuda-compat\|ulimit\|GOVERNING\|found at\|Product-Specific\|Various files\|vLLM\|=====" \
        > "$LOGS/eval_by_cat_v2.log"
    sudo chown "$(id -u):$(id -g)" "$V2/medusa/eval_by_cat_v2.json" 2>/dev/null
    grep -A40 "=== per-head" "$LOGS/eval_by_cat_v2.log"
}

stage_merge() {
    say "merge: start"
    $DOCKER python3 "$V2/medusa/merge_medusa_gguf.py" "$MODEL" "$V2/medusa/medusa_heads_v2.safetensors" "$MERGED" 2>&1 \
        | grep -v "CUDA Graph\|SHMEM\|NVIDIA\|docker run\|^$\|cuda-compat\|ulimit\|GOVERNING\|found at\|Product-Specific\|Various files\|vLLM\|=====" \
        > "$LOGS/merge_medusa_v2.log"
    sudo chown "$(id -u):$(id -g)" "$MERGED" 2>/dev/null; chmod 0644 "$MERGED" 2>/dev/null
    tail -5 "$LOGS/merge_medusa_v2.log"
    say "merge: $(stat -c%s "$MERGED" 2>/dev/null) bytes -> $MERGED"
}

bench_one() {  # <tag> <model> <prompt-name> <n_max>
    local tag=$1 model=$2 which=$3 nmax=$4
    local MATH='A train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step.'
    local CODE='Write a Python function that parses a CSV file of transactions (date, amount, category) and returns the total spent per category as a dict, with error handling for malformed rows and a small unit test.'
    local Q; [ "$which" = math ] && Q=$MATH || Q=$CODE
    local PROMPT=$'<|im_start|>user\n'"$Q"$'<|im_end|>\n<|im_start|>assistant\n'
    local LOG=$ROOT/logs/medusa/$tag-$which-n$nmax.log
    LD_LIBRARY_PATH=$ROOT/bin/medusa $ROOT/bin/medusa/llama-speculative-simple -m "$model" --spec-type draft-medusa \
        --spec-draft-n-max "$nmax" -fa on -ngl 99 -c 4096 -n 200 --temp 0 -e -p "$PROMPT" > "$LOG" 2>&1
    echo "== $tag $which n_max=$nmax rc=$? ($LOG)"
    grep -E "decoded|n_drafted|n_accept|accept  |tokens/step" "$LOG" | sed 's/^[0-9.]* I //'
}

stage_bench() {
    say "bench: start"
    wait_gpu_free
    {
        for which in math code; do
            for nmax in 2 4; do bench_one v2 "$MERGED" $which $nmax; done
            bench_one v1 "$V1" $which 2
        done
    } 2>&1 | tee "$LOGS/bench_medusa_v2.log"
    say "bench: done"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(waitgen extract rsync train evalcat merge bench)
for s in "${stages[@]}"; do
    "stage_$s" || { say "STAGE $s FAILED - stopping"; exit 1; }
done
say "PIPELINE DONE"
