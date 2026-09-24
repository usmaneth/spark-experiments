#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import json

BASE_DIR = "/home/REDACTED/Bonsai-demo"
CONV_GGUF = f"{BASE_DIR}/models/bonsai2-dspark/bonsai2-dspark-trained-conv.gguf"
TARGET_GGUF = f"{BASE_DIR}/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
QUANT_BIN = f"{BASE_DIR}/bin/cuda/llama-quantize"
SPEC_BIN = f"{BASE_DIR}/bin/cuda/llama-speculative-simple"
MODELS_DIR = f"{BASE_DIR}/models/bonsai2-dspark"

QUANTS_TO_TEST = [
    ("F16", "f16"),
    ("Q8_0", "q8_0"),
    ("Q6_K", "q6_k"),
    ("Q5_K_M", "q5_k_m"),
    ("Q4_K_M", "q4_k_m"),
    ("Q4_0", "q4_0")
]

PROMPT = "<|im_start|>user\nWrite a quick python function to reverse a string and explain it.<|im_end|>\n<|im_start|>assistant\n"

def run_cmd(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=merged_env, text=True)
    return p.stdout + p.stderr

def main():
    if not os.path.exists(CONV_GGUF):
        print(f"Error: Converted GGUF not found at {CONV_GGUF}")
        return 1

    print("==================================================================")
    print("   Bonsai 2 Speculative Drafter: Multi-Quantization Benchmark    ")
    print("==================================================================")

    results = []

    for quant_name, quant_suffix in QUANTS_TO_TEST:
        out_gguf = f"{MODELS_DIR}/bonsai2-dspark-trained-{quant_name}.gguf"
        print(f"\n[1/2] Quantizing to {quant_name} -> {out_gguf}...")
        
        t0 = time.time()
        q_cmd = f"{QUANT_BIN} {CONV_GGUF} {out_gguf} {quant_name}"
        q_out = run_cmd(q_cmd)
        q_time = time.time() - t0
        
        if not os.path.exists(out_gguf):
            print(f"Quantization failed for {quant_name}:\n{q_out}")
            continue
            
        size_mb = os.path.getsize(out_gguf) / (1024 * 1024)
        print(f"-> {quant_name} complete: {size_mb:.2f} MB in {q_time:.2f}s")

        # Benchmark speculative decoding: K=5, p_min=0.60
        print(f"[2/2] Benchmarking {quant_name} (K=5, p_min=0.60)...")
        spec_cmd = (
            f"LD_LIBRARY_PATH={BASE_DIR}/bin/cuda {SPEC_BIN} "
            f"-m {TARGET_GGUF} "
            f"-md {out_gguf} "
            f"--spec-type draft-dspark "
            f"--spec-draft-n-max 5 "
            f"--spec-draft-p-min 0.60 "
            f"-fa on -ngl 99 -ngld 999 -c 4096 -n 60 --temp 0 -e "
            f"-p \"{PROMPT}\""
        )
        s_out = run_cmd(spec_cmd)

        # Parse metrics
        gen_speed = 0.0
        accept_pct = 0.0
        n_drafted = 0
        n_accept = 0

        for line in s_out.splitlines():
            if "decoded" in line and "speed:" in line:
                try:
                    parts = line.split("speed:")
                    gen_speed = float(parts[1].replace("t/s", "").strip())
                except: pass
            if "accept    =" in line:
                try:
                    accept_pct = float(line.split("=")[1].replace("%", "").strip())
                except: pass
            if "n_drafted =" in line:
                try: n_drafted = int(line.split("=")[1].strip())
                except: pass
            if "n_accept  =" in line:
                try: n_accept = int(line.split("=")[1].strip())
                except: pass

        print(f"   => Gen Speed: {gen_speed:.2f} tok/s | Acceptance: {accept_pct:.2f}% ({n_accept}/{n_drafted})")

        results.append({
            "quant": quant_name,
            "size_mb": round(size_mb, 2),
            "gen_speed": gen_speed,
            "acceptance_pct": accept_pct,
            "n_accept": n_accept,
            "n_drafted": n_drafted
        })

    print("\n==================================================================")
    print("                    FINAL BENCHMARK MATRIX                        ")
    print("==================================================================")
    print(f"| Quantization | Size (MB) | Gen Speed (tok/s) | Acceptance (%) | Drafted/Accepted |")
    print(f"|--------------|-----------|-------------------|----------------|------------------|")
    for r in results:
        print(f"| {r['quant']:12s} | {r['size_mb']:9.2f} | {r['gen_speed']:17.2f} | {r['acceptance_pct']:13.2f}% | {r['n_accept']}/{r['n_drafted']} |")

    # Save JSON results
    out_json = f"{MODELS_DIR}/multi_quant_benchmarks.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")

if __name__ == "__main__":
    main()
