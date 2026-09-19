---
license: other
license_name: mixed-see-license-section
base_model: prism-ml/Ternary-Bonsai-2-27B-gguf
language:
- en
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- dflash
- llama.cpp
- gguf
- bonsai
---

# Ternary-Bonsai-2-27B DSpark drafters

Two speculative-decoding drafters for [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (the `Ternary-Bonsai-2-27B-PQ2_0.gguf` target). Both files use the DSpark drafter architecture in the `dflash` GGUF layout that the PrismML fork of llama.cpp loads with `--spec-type draft-dspark`. Neither file holds a token embedding or an output head: the runtime borrows both from the target.

## Highlights

- 1.5x to 2.5x decode speed on an NVIDIA GB10 (DGX Spark) against plain decoding, per workload. The baseline is 29.7 tokens per second; math reaches 74.7 and code reaches 65.7.
- Exact verification at temperature 0. The drafter changes the speed, not the sampling rule. See "Exactness" for the precise claim.
- The two files serve different workloads. DSpark v2 leads on code and 2,000-token long-form. DSpark v1 leads on reasoning, tool calling and agent turns. Blended, they tie.
- Both files need a runtime fix that the released PrismML binaries do not carry. See "Requirement" before you download.

## Files

| file | bytes | size | quant | sha256 |
| --- | ---: | ---: | --- | --- |
| `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf` | 1,104,605,632 | 1.03 GiB | Q4_K_M | `eb80f88fc94f267b6b612cba4deac31ecb0dfff0f587f2786fc3de855ceaf84a` |
| `Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf` | 631,713,664 | 0.59 GiB | Q4_0 | `4ebd761b4f510a94982c1ae10a167ef36360f4e8bedbdb343143ee65dbe6142a` |

`SHA256SUMS` holds the same two hashes in `sha256sum -c` format.

Q4_K_M is the fastest file for the v2 drafter. On the three 200-token probes at K=5, Q5_K_M, Q6_K, Q8_0 and F16 copies of the same weights accept the same tokens within one token per prompt and decode 3% to 21% fewer tokens per second, because the draft pass is bound by memory bandwidth.

## Model overview

| property | DSpark v2 | DSpark v1 |
| --- | --- | --- |
| GGUF architecture | `dflash` | `dflash` |
| `general.name` in the header | `Qwen3.8-27B-DSpark` (inherited, see Limitations) | `Bonsai-27B-dspark` |
| draft layers (`dflash.block_count`) | 5 | 6 |
| block size (`dflash.block_size`) | 7 | 4 |
| draft length for the published numbers (`--spec-draft-n-max`) | 5 | 4 |
| target layers tapped (`dflash.target_layers`) | 6, 20, 34, 48, 62 | 2, 17, 32, 47, 62 |
| attention heads / KV heads | 32 / 8 | 40 / 4 |
| feed-forward length | 17408 | 5120 |
| Markov rank | 256 | 256 |
| confidence head | yes | yes |
| token embedding and output head | borrowed from the target | borrowed from the target |
| trained for | Ternary-Bonsai-2-27B (this target) | Ternary-Bonsai-27B (the earlier model); transfers to this target |
| target file | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` |

## Requirement: a patched runtime

> **Important.** The released PrismML llama.cpp binaries cannot run either file on Ternary-Bonsai-2-27B. On the release tag that `setup.sh` of the Bonsai demo installs (`prism-b10683-d8f26ee`) and on a clean build of PrismML-Eng/llama.cpp `prism` at `1a07bfa5f` (build 10706), both drafters get 0.4% to 2.0% acceptance. The server starts, prints no error, and decodes slower than a server without a drafter.
>
> You need PrismML-Eng/llama.cpp `prism` at `1a07bfa5f` plus the fix "dflash: apply the target's Hadamard transforms to borrowed embeddings and head" (pull request [PrismML-Eng/llama.cpp#210](https://github.com/PrismML-Eng/llama.cpp/pull/210), commit `288859a96` on branch `fix/dflash-borrowed-hadamard`). Check out the merge commit of that pull request when it exists.

Cause: the `dflash` runtime borrows the target's `token_embd` and `output` tensors through the target context. The `PQ2_0` target stores both tensors in a Hadamard-rotated basis and lists them in its `prism.hadamard` metadata. The draft graph took its Hadamard maps from the draft model, which has none, so it applied no inverse transform to the borrowed embedding and no forward transform to the borrowed output head. The fix merges the target's Hadamard maps into the draft context and applies them at the four borrow sites.

`llama-server --version` prints the same build id (`build 10706, commit 1a07bfa5f`) for a clean binary and a patched binary. Check the acceptance instead (below).

Build the runtime for a GB10:

```bash
git clone https://github.com/PrismML-Eng/llama.cpp.git
git -C llama.cpp fetch https://github.com/usmaneth/llama.cpp fix/dflash-borrowed-hadamard
git -C llama.cpp checkout 288859a96   # prism 1a07bfa5f plus the fix of PR #210; use the merge commit once the PR lands
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cuda -j 18 --target llama-speculative-simple llama-server
export LD_LIBRARY_PATH=$PWD/llama.cpp/build-cuda/bin
```

Check that the fix is in the binary. The counts below are exact at temperature 0 on a GB10:

```bash
llama.cpp/build-cuda/bin/llama-speculative-simple \
  -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -md Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fa on -ngl 99 -ngld 999 -c 4096 -n 200 --temp 0 -e \
  -p '<|im_start|>user\nA train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step.<|im_end|>\n<|im_start|>assistant\n'
# patched runtime: n_drafted = 284, n_accept = 148, accept = 52.113%
# clean runtime:   accept about 1.6%
```

The same prompt with the v1 file and `--spec-draft-n-max 4` gives `n_drafted = 250, n_accept = 140` (56.0%) on the patched runtime and about 2.0% on the clean runtime.

## Usage

### llama-server

DSpark v2:

```bash
llama-server -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa on -c 16384 --jinja \
  -md Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 -ngld 999
```

DSpark v1:

```bash
llama-server -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa on -c 16384 --jinja \
  -md Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999
```

Notes:

- `--spec-draft-n-max` is the draft length K. The runtime accepts a value up to the block size of the drafter and clamps a larger value with a warning. For DSpark v2, K=5 gave the best decode rate in our sweeps (K=7 loses, because the verify pass grows faster than the accepted tokens). For DSpark v1, K=4 is the block size.
- The published single-stream numbers add `-np 1`. The server also runs with `-np 2` and `-np 4` (see "Multi-slot").
- Use a large context (`-c 16384` or more). The model thinks for 1,500 to 2,000 tokens before the visible answer.
- The benchmark used temperature 0 and seed 42. Speculative decoding on this runtime uses exact-match verification, so the output at temperature 0 follows the rule in "Exactness".

### Read the acceptance

Every response carries a `timings` object with `draft_n` (tokens drafted) and `draft_n_accepted` (tokens accepted). Acceptance is `draft_n_accepted / draft_n`. The server puts these counters in the response only when a drafter is loaded. If `draft_n` is absent or zero, speculation is not active.

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d '{
  "prompt": "<|im_start|>user\nImplement binary search in Python with type hints, a docstring, and three assert-based tests.<|im_end|>\n<|im_start|>assistant\n",
  "n_predict": 512, "temperature": 0, "seed": 42, "cache_prompt": false
}' | python3 -c 'import sys, json; t = json.load(sys.stdin)["timings"]; print("accepted", t["draft_n_accepted"], "of", t["draft_n"], "drafted;", round(t["predicted_per_second"], 1), "tok/s")'
```

`/v1/chat/completions` carries the same object.

### The Bonsai demo launcher

The [Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo) starts the drafter with `BONSAI_SPECULATIVE=1`:

```bash
# put the patched binaries and shared libraries in bin/cuda/ first
BONSAI_SPECULATIVE=1 BONSAI_SPEC_NMAX=5 ./scripts/start_llama_server.sh
```

- Put the drafter in `models/bonsai2-gguf/27B/`. The launcher takes the first `*dspark-dflash*.gguf` in that directory in sorted order. `...-dspark-dflash-v1-Q4_0.gguf` sorts before `...-dspark-dflash-v2-Q4_K_M.gguf`. When both files are present, the launcher picks DSpark v1. Keep one drafter in the directory, or start `llama-server` by hand with `-md` as above.
- Set `BONSAI_SPEC_NMAX=5` for DSpark v2. The launcher reads the draft length from the GGUF key `dspark.dspark.block_size`. A converted file carries `dflash.block_size` instead, so the launcher falls back to 4.
- The launcher forces `-np 1` and raises the context to 16384.

## Benchmark

Conditions: one NVIDIA GB10 (DGX Spark), driver 580.178.04, CUDA 13.0, idle before the run. Target `Ternary-Bonsai-2-27B-PQ2_0.gguf`. Server flags `-ngl 999 -fa on -c 16384 --jinja` plus the drafter flags above; `-np 1` for the single-stream rows. Temperature 0, seed 42, `cache_prompt` false, one pass per prompt. The runtime was PrismML `prism` build `5d80cff0b` with the same fix as the pull request above. Rates are the arithmetic mean of the decode-only `predicted_per_second` of the server. Acceptance is accepted tokens over drafted tokens, aggregated. Speedup is the ratio of the two means over the same prompts. `tok/step` is generated tokens per verify step, `predicted_n / (predicted_n - accepted)`; a server without a drafter is at 1.00.

Prompts: 40 matrix prompts x 512 tokens (8 each of code, math, reasoning, chat and long-form) through `/completion` with a client-side ChatML template and no system message; 8 tool prompts and 8 agent prompts x 512 tokens through `/v1/chat/completions` with a tools list and the server template (`--jinja`, template default reasoning effort); 6 long prompts x 2,000 tokens. The agent prompts carry a mean prefix of 3,079 tokens (system prompt, tools, task, one tool result).

### Workload matrix

| workload | no drafter | DSpark v1 (K=4) | accept | tok/step | speedup | DSpark v2 (K=5) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 29.67 | 62.92 | 55.2% (2702/4895) | 3.19 | 2.12x | 65.72 | 54.8% (2875/5249) | 3.71 | 2.22x |
| math | 29.66 | 74.69 | 71.5% (2793/3904) | 3.85 | 2.52x | 74.66 | 65.9% (2893/4391) | 4.29 | 2.52x |
| reasoning | 29.67 | 60.90 | 51.5% (2643/5129) | 3.04 | 2.05x | 57.85 | 44.5% (2707/6088) | 3.20 | 1.95x |
| chat | 29.67 | 44.39 | 29.8% (1871/6281) | 2.18 | 1.50x | 44.78 | 29.8% (2062/6910) | 2.49 | 1.51x |
| long-form (512 tokens) | 29.67 | 45.46 | 32.8% (2315/7055) | 2.30 | 1.53x | 43.86 | 29.5% (2431/8242) | 2.46 | 1.48x |
| tool calling | 29.51 | 58.18 | 45.7% (1297/2839) | 2.83 | 1.97x | 49.60 | 37.3% (1306/3497) | 2.87 | 1.68x |
| agent turn | 28.85 | 53.33 | 51.9% (1165/2244) | 3.09 | 1.85x | 46.65 | 40.8% (1157/2838) | 3.05 | 1.62x |
| blended (40 prompts) | 29.67 | 57.67 | 45.2% (12324/27264) | 2.79 | 1.94x | 57.37 | 42.0% (12968/30880) | 3.08 | 1.93x |
| long-form (6 prompts, 2,000 tokens) | 29.53 | 65.24 | 58.7% (7621/12981) | 3.34 | 2.21x | 66.77 | 57.0% (8087/14183) | 3.85 | 2.26x |

All rates are tokens per second.

### Single prompt

The quicksort prompt of the PrismML GB10 benchmark document, 256 tokens, 3 passes on one slot. Tokens per second per pass.

| config | pass 1 | pass 2 | pass 3 | mean | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 29.72 | 29.61 | 29.52 | 29.62 | n/a | 1.00 | n/a |
| DSpark v1 (K=4) | 51.18 | 51.19 | 51.29 | 51.22 | 40.7% (471/1158) | 2.59 | 1.73x |
| DSpark v2 (K=5) | 52.10 | 51.97 | 51.95 | 52.01 | 39.6% (507/1281) | 2.94 | 1.76x |

### Tool calls

A tool prompt is valid when the answer holds at least one well-formed call (a name from the tools list and a JSON object of arguments). All three configurations return 6 of 8 valid calls, 6 of 8 expected calls, and the same finish reasons (2 `length`, 6 `tool_calls`). The two prompts that end with `length` hit the 512-token cap inside the reasoning block on every configuration. The drafter does not change the tool-call validity.

### Multi-slot

Each run sends the same 8 prompts (code-01, code-02, reasoning-01, reasoning-02, chat-01, chat-02, tool-01, tool-02) in waves of `slots` concurrent requests. `per stream` is the mean decode rate that each request saw. `aggregate` is generated tokens over the wall time of all waves, prefill included. The `slots 1` row uses the same prompts from the single-slot run. The server accepted `-np 2` and `-np 4` with a drafter without refusal.

| config | slots | per stream tok/s | aggregate tok/s | accept |
| --- | ---: | ---: | ---: | ---: |
| no drafter | 1 | 29.67 | 28.79 | n/a |
| no drafter | 2 | 25.28 | 45.68 | n/a |
| no drafter | 4 | 20.34 | 56.86 | n/a |
| DSpark v1 (K=4) | 1 | 62.39 | 53.49 | 52.2% |
| DSpark v1 (K=4) | 2 | 43.23 | 67.15 | 53.3% |
| DSpark v1 (K=4) | 4 | 31.65 | 74.84 | 51.8% |
| DSpark v2 (K=5) | 1 | 58.27 | 52.61 | 47.7% |
| DSpark v2 (K=5) | 2 | 40.27 | 66.25 | 48.5% |
| DSpark v2 (K=5) | 4 | 30.09 | 74.43 | 47.2% |

### Power

`nvidia-smi` sampled at 1 Hz. Idle is the mean draw over 10 s with the model loaded and no request in flight. Mean and max cover the generation phase. `gen tok/s` is generated tokens over the wall time of that phase, prefill included. mJ/token is mean W / gen tok/s x 1000.

| config | slots | idle W | mean W | max W | util % | tokens | gen tok/s | mJ/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 1 | 14.5 | 60.9 | 86.5 | 94 | 34332 | 28.52 | 2135.1 |
| no drafter | 2 | 15.4 | 59.9 | 79.5 | 93 | 3053 | 45.68 | 1312.2 |
| no drafter | 4 | 15.9 | 65.4 | 75.6 | 94 | 3053 | 56.86 | 1150.2 |
| DSpark v1 (K=4) | 1 | 16.0 | 77.0 | 85.1 | 92 | 34560 | 52.85 | 1457.5 |
| DSpark v1 (K=4) | 2 | 16.2 | 60.4 | 77.2 | 84 | 3053 | 67.15 | 899.1 |
| DSpark v1 (K=4) | 4 | 15.3 | 59.0 | 77.0 | 87 | 3053 | 74.84 | 788.1 |
| DSpark v2 (K=5) | 1 | 15.8 | 78.1 | 85.0 | 92 | 34614 | 52.24 | 1494.6 |
| DSpark v2 (K=5) | 2 | 16.3 | 59.4 | 82.2 | 86 | 3053 | 66.25 | 896.8 |
| DSpark v2 (K=5) | 4 | 15.3 | 61.3 | 82.2 | 80 | 3053 | 74.43 | 823.1 |

## Exactness

The server accepts a drafted token only when it equals the token the target picks at temperature 0 from the batched verify logits. We compared every drafter output with the no-drafter output for the same prompt and pass: 65 outputs (62 prompts, with three passes of the quicksort prompt). A `/completion` output is identical when its token id sequence equals the no-drafter sequence. A chat output (reasoning, answer, tool calls) is compared as text, because that endpoint returns no token ids.

Result: 37 of 65 outputs are identical for DSpark v1, and 37 of 65 for DSpark v2. The 28 outputs that differ are the same 28 for both drafters, and the first difference is at the same position for both (for example reasoning-06 at token 1, long-form-03 at token 9, code-06 at token 59, math-01 at token 255, tool-05 at character 1773). Two different drafters do not produce the same 28 divergence points if the draft causes them. The cause is the verify pass: it scores several positions in one batch, the batched kernels round differently from single-row decode, and at a near-tie token the argmax flips. The no-drafter server gives identical output across repeated passes.

So the claim is: speculative decoding on this runtime is exact given the batched logits, and identical to plain greedy decoding on 37 of 65 outputs in this benchmark. It is not a claim of byte-identical output on every prompt.

## Which file to use

| workload | pick | reason |
| --- | --- | --- |
| code, 2,000-token long-form | DSpark v2 | +4% on code, +2% on long 2,000; more tokens per step (K=5) |
| reasoning, tool calling, agent turns | DSpark v1 | +5% on reasoning, +17% on tool calling, +14% on agent turns |
| mixed traffic | either | blended 57.7 vs 57.4 tok/s; DSpark v1 is the smaller file |

The DSpark v2 training data holds no tool-call turns, which shows in the tool row.

## Provenance

### DSpark v2

DSpark v2 was trained for this target with the self-distillation recipe in the Bonsai demo (`tools/dspark-retrain/`, pull request `https://github.com/PrismML-Eng/Bonsai-demo/pull/189`). One machine (a GB10) ran every step.

- Warm start: the DSpark drafter [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) (5 draft layers, block size 7), trained for the Qwen3.8-27B target that Bonsai 2 is built on. The trainer accepts every tensor of that checkpoint.
- Teacher: the LM head and the token embedding were dequantized from the `PQ2_0` target with the Hadamard fold applied. A validation gate recomputes the logits from the dequantized head and compares them with the runtime: 100% argmax match, top-1 values within 1%, on every position of the validation prompt.
- Objective: the DSpark block-parallel objective as in the SpecForge reference. Next-token cross-entropy through the borrowed LM head, an L1 match to the target's token distribution, and a confidence-head loss, token-normalized over the supervised block positions.
- Data: self-distilled. The target answered 3,401 prompts (code, math, reasoning, chat, long-form; from public prompt sets) at temperature 0 through `llama-server`, 1.57 M tokens. The drafter trains on the target's hidden states (taps at the inputs of layers 6, 20, 34, 48 and 62) and the target's own tokens. The data holds no tool-call turns.
- Schedule: round 1 on the code prompts from the warm start, one epoch (452 steps). Round 2 on all data from the round-1 weights, batch size 2, learning rate 6e-5, 256 anchors, one epoch of 1,701 steps. The checkpoints at steps 300, 600 and 900 give the same acceptance within noise. The step-600 checkpoint is this file. A probe that continued from step 600 at a higher learning rate lowered acceptance by 5 to 8 points on every workload.
- Conversion: safetensors to a raw `dspark` GGUF, then `gguf_dspark_to_dflash.py --drop-shared-tensors` with the target as the tokenizer donor, then `llama-quantize` to `Q4_K_M`. The converter shifts `target_layers` by one (the runtime taps the input of a layer) and drops `token_embd` and `output`.

### DSpark v1

DSpark v1 is the DSpark drafter that PrismML trained and published for the earlier Ternary-Bonsai-27B model: `Ternary-Bonsai-27B-dspark-bf16.gguf` in [prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) (Apache 2.0). PrismML publishes no drafter for Ternary-Bonsai-2-27B. This file is that drafter re-converted against the Bonsai 2 `PQ2_0` target and quantized to `Q4_0`:

```bash
python3 llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py --drop-shared-tensors \
  Ternary-Bonsai-27B-dspark-bf16.gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-dspark-conv.gguf
llama-quantize Ternary-Bonsai-2-27B-dspark-conv.gguf \
  Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf Q4_0
```

The conversion injects the tokenizer from the Bonsai 2 target and drops the embedding and the output head, so the file borrows both from the Bonsai 2 target at run time. No weight of the drafter was retrained. With the runtime fix, it transfers to the Bonsai 2 target at the acceptance in the tables. Credit for these weights belongs to PrismML.

## Reproduce the numbers

The benchmark tool and the full result document live in the Bonsai demo (`scripts/spec_bench/` and the GB10 Bonsai 2 entry in `community-benchmarks/`, pull request `https://github.com/PrismML-Eng/Bonsai-demo/pull/188`). The retraining recipe is `tools/dspark-retrain/` (pull request `https://github.com/PrismML-Eng/Bonsai-demo/pull/189`).

## Limitations

- Target: `Ternary-Bonsai-2-27B-PQ2_0.gguf` only. A drafter is tied to the exact target it borrows the embedding and the output head from. The `PTQ1_0` target decodes faster alone, but its batched verify pass is slow, and it lost to `PQ2_0` under speculation at every draft length in our sweeps. Do not use these files with the earlier Ternary-Bonsai-27B model.
- Runtime: both files need the patched runtime in "Requirement". On the released binaries they run at 0.4% to 2.0% acceptance without an error message.
- Hardware: measured on one GB10 with CUDA. Not measured on other GPUs or on Apple Silicon.
- The `general.name` header of the v2 file still reads `Qwen3.8-27B-DSpark`, inherited from the warm-start checkpoint. The weights are the retrained Bonsai 2 drafter.
- The chat template of the target sets `reasoning_effort` to `xhigh` by default. The tool and agent rows were measured through that template, so the model reasons at length before a call, and two of the 8 tool prompts end inside the reasoning block at 512 tokens.
- The v2 training data holds no tool-call turns. DSpark v1 leads on tool calling and agent turns.
- The Bonsai demo launcher picks DSpark v1 when both files are in the model directory, and falls back to draft length 4 without `BONSAI_SPEC_NMAX=5`.
- `--spec-draft-n-max 5` with the v1 file clamps to its block size 4 with a warning.

## Credits

- [PrismML](https://huggingface.co/prism-ml): the Ternary-Bonsai-2-27B target, the Ternary-Bonsai-27B DSpark drafter that DSpark v1 is converted from, the `dflash` runtime and the `gguf_dspark_to_dflash.py` converter in the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), and the [Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo).
- [RadixArk](https://huggingface.co/RadixArk): the Qwen3.8-27B-DSpark drafter that DSpark v2 warm-starts from, and the DSpark method in [SpecForge](https://github.com/sgl-project/SpecForge).
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the `draft-dspark` speculative decoding path ([ggml-org/llama.cpp#25173](https://github.com/ggml-org/llama.cpp/pull/25173)).

## License

- `Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf` is a conversion of PrismML's Ternary-Bonsai-27B drafter, released under Apache 2.0. It is redistributed here under Apache 2.0 with attribution to PrismML.
- `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf` warm-started from the RadixArk/Qwen3.8-27B-DSpark checkpoint. That repository declares `license: other` and publishes no license text; its base models (RadixArk/Qwen3.8-27B-NVFP4 and Qwen/Qwen3.8-27B) are Apache 2.0. Until RadixArk states the terms of the drafter checkpoint, the v2 file is offered under the same undeclared terms as its source, and the model card cannot promise more. The training data (self-distilled from Bonsai 2), the recipe and the benchmark tool are Apache 2.0.
