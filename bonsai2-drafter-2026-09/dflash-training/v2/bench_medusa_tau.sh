#!/usr/bin/env bash
# Medusa v2 benchmark with target-side typical acceptance (SPEC_TYPICAL_TAU) on spark2.
# Runs math/code x n_max {2,4} x tau {0.05, 0.02} (the tau-unset rows come from run_broad_pipeline.sh bench).
# Usage: bench_medusa_tau.sh [merged.gguf] [taus...]
set -uo pipefail
ROOT=/home/usman/Bonsai-demo
MODEL=${1:-$ROOT/models/bonsai2-medusa/Ternary-Bonsai-2-27B-PQ2_0-medusa-v2.gguf}
shift || true
TAUS=("$@"); [ ${#TAUS[@]} -eq 0 ] && TAUS=(0.05 0.02)
mkdir -p "$ROOT/logs/medusa"
ts() { date +%H:%M:%S; }

wait_gpu_free() {
    while true; do
        procs=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v "^$" || true)
        [ -z "$procs" ] && return 0
        echo "[$(ts)] GPU busy, waiting: $procs"; sleep 30
    done
}

bench_one() {  # <tag> <model> <prompt-name> <n_max> <tau>
    local tag=$1 model=$2 which=$3 nmax=$4 tau=$5
    local MATH='A train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step.'
    local CODE='Write a Python function that parses a CSV file of transactions (date, amount, category) and returns the total spent per category as a dict, with error handling for malformed rows and a small unit test.'
    local Q; [ "$which" = math ] && Q=$MATH || Q=$CODE
    local PROMPT=$'<|im_start|>user\n'"$Q"$'<|im_end|>\n<|im_start|>assistant\n'
    local LOG=$ROOT/logs/medusa/$tag-$which-n$nmax-tau$tau.log
    SPEC_TYPICAL_TAU=$tau LD_LIBRARY_PATH=$ROOT/bin/medusa $ROOT/bin/medusa/llama-speculative-simple -m "$model" --spec-type draft-medusa \
        --spec-draft-n-max "$nmax" -fa on -ngl 99 -c 4096 -n 200 --temp 0 -e -p "$PROMPT" > "$LOG" 2>&1
    echo "== $tag $which n_max=$nmax tau=$tau rc=$? ($LOG)"
    grep -E "decoded|n_drafted|n_accept|accept  |tokens/step|typical" "$LOG" | sed 's/^[0-9.]* I //'
}

echo "[$(ts)] tau bench: start model=$MODEL taus=${TAUS[*]}"
wait_gpu_free
for which in math code; do
    for nmax in 2 4; do
        for tau in "${TAUS[@]}"; do bench_one v2 "$MODEL" $which $nmax $tau; done
    done
done
echo "[$(ts)] TAU BENCH DONE"
