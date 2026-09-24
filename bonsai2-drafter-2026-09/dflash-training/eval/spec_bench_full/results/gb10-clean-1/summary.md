# Speculative decoding benchmark

Generated 2026-09-18T18:50:39Z on node_b (NVIDIA GB10, driver 580.178.04, CUDA 13.0). llama-server version: 0.2.0-dev (build 10687, commit 5d80cff0b), build b10687-5d80cff0b.

Prompts: 40 matrix prompts x 512 tokens, 6 long prompts x 2000 tokens, 8 tool prompts x 512 tokens, 8 agent prompts x 512 tokens. Passes per prompt: 1. Slots: 1, 2, 4. Rates are arithmetic means of the server decode-only `predicted_per_second`; acceptance is aggregated accepted/drafted tokens; speedup is the ratio of the two means over the same prompts. The tool and agent rows come from `/v1/chat/completions` with a tools list and the server template (reasoning effort: template); the other rows come from `/completion` with a client-side ChatML template. The single quicksort prompt runs 3 passes at 256 tokens on every single-slot server.

## Workload matrix: dspark-v1

| workload | no drafter | + dspark-v1 (K=4) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 29.67 | 62.92 | 0.552 (2702/4895) | 3.19 | **2.12x** |
| math | 29.66 | 74.69 | 0.715 (2793/3904) | 3.85 | **2.52x** |
| reasoning | 29.67 | 60.90 | 0.515 (2643/5129) | 3.04 | **2.05x** |
| chat | 29.67 | 44.39 | 0.298 (1871/6281) | 2.18 | **1.50x** |
| long-form | 29.67 | 45.46 | 0.328 (2315/7055) | 2.30 | **1.53x** |
| tool | 29.51 | 58.18 | 0.457 (1297/2839) | 2.83 | **1.97x** |
| agent | 28.85 | 53.33 | 0.519 (1165/2244) | 3.09 | **1.85x** |
| blended (40 prompts) | 29.67 | 57.67 | 0.452 (12324/27264) | 2.79 | **1.94x** |
| long-form (6 prompts, 2000 tokens) | 29.53 | 65.24 | 0.587 (7621/12981) | 3.34 | **2.21x** |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

## Workload matrix: dspark-v2

| workload | no drafter | DSpark v2 (K=5) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 29.67 | 65.72 | 0.548 (2875/5249) | 3.71 | **2.22x** |
| math | 29.66 | 74.66 | 0.659 (2893/4391) | 4.29 | **2.52x** |
| reasoning | 29.67 | 57.85 | 0.445 (2707/6088) | 3.20 | **1.95x** |
| chat | 29.67 | 44.78 | 0.298 (2062/6910) | 2.49 | **1.51x** |
| long-form | 29.67 | 43.86 | 0.295 (2431/8242) | 2.46 | **1.48x** |
| tool | 29.51 | 49.60 | 0.373 (1306/3497) | 2.87 | **1.68x** |
| agent | 28.85 | 46.65 | 0.408 (1157/2838) | 3.05 | **1.62x** |
| blended (40 prompts) | 29.67 | 57.37 | 0.420 (12968/30880) | 3.08 | **1.93x** |
| long-form (6 prompts, 2000 tokens) | 29.53 | 66.77 | 0.570 (8087/14183) | 3.85 | **2.26x** |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

`tok/step` is generated tokens per verify step, predicted_n / (predicted_n - accepted); a server without a drafter is at 1.00.

## Long-form prompts

Per prompt, pass 1, tokens per second.

| prompt | no drafter | + dspark-v1 (K=4) | accept | tok/step | speedup | DSpark v2 (K=5) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long-01 | 29.58 | 59.80 | 0.522 | 3.08 | **2.02x** | 59.63 | 0.490 | 3.45 | **2.02x** |
| long-02 | 29.48 | 59.22 | 0.516 | 3.06 | **2.01x** | 60.28 | 0.499 | 3.49 | **2.04x** |
| long-03 | 29.52 | 62.57 | 0.561 | 3.24 | **2.12x** | 67.81 | 0.586 | 3.92 | **2.30x** |
| long-04 | 29.52 | 62.76 | 0.563 | 3.25 | **2.13x** | 66.19 | 0.568 | 3.83 | **2.24x** |
| long-05 | 29.53 | 74.86 | 0.721 | 3.88 | **2.54x** | 75.12 | 0.672 | 4.35 | **2.54x** |
| long-06 | 29.57 | 72.23 | 0.686 | 3.74 | **2.44x** | 71.58 | 0.631 | 4.16 | **2.42x** |

## Single prompt

The quicksort prompt of the PrismML GB10 document (256 tokens, 3 passes), tokens per second per pass.

| config | pass 1 | pass 2 | pass 3 | mean | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 29.72 | 29.61 | 29.52 | 29.62 | n/a | 1.00 | n/a |
| dspark-v1 | 51.18 | 51.19 | 51.29 | 51.22 | 0.407 (471/1158) | 2.59 | **1.73x** |
| dspark-v2 | 52.10 | 51.97 | 51.95 | 52.01 | 0.396 (507/1281) | 2.94 | **1.76x** |

## Exactness

Each drafter output is compared with the baseline output for the same prompt and pass. `/completion` outputs are identical when their token id sequences are equal; chat outputs (reasoning, answer, tool calls) are compared as text because that endpoint returns no token ids. For a different output the table lists the 0-based index of the first differing token or character; when one output is a prefix of the other, the index equals the shorter length.

| config | identical | first difference |
| --- | ---: | --- |
| dspark-v1 | 37/65 | agent-03: char 862; agent-06: char 26; chat-01: token 380; chat-04: token 32; chat-06: token 29; code-04: token 136; code-06: token 59; long-01: token 372; long-02: token 113; long-03: token 87; long-04: token 831; long-05: token 40; long-form-01: token 41; long-form-02: token 112; long-form-03: token 9; long-form-05: token 483; long-form-06: token 219; long-form-07: token 141; long-form-08: token 227; math-01: token 255; math-05: token 139; math-08: token 372; reasoning-01: token 181; reasoning-02: token 59; reasoning-05: token 61; reasoning-06: token 1; reasoning-08: token 77; tool-05: char 1773 |
| dspark-v2 | 37/65 | agent-03: char 862; agent-06: char 26; chat-01: token 380; chat-04: token 32; chat-06: token 29; code-04: token 136; code-06: token 59; long-01: token 372; long-02: token 113; long-03: token 87; long-04: token 831; long-05: token 40; long-form-01: token 41; long-form-02: token 112; long-form-03: token 9; long-form-05: token 483; long-form-06: token 219; long-form-07: token 141; long-form-08: token 227; math-01: token 255; math-05: token 139; math-08: token 372; reasoning-01: token 181; reasoning-02: token 59; reasoning-05: token 61; reasoning-06: token 1; reasoning-08: token 77; tool-05: char 1773 |

Baseline outputs across passes: identical.

## Tool calls

A tool prompt is valid when the answer holds at least one well-formed call (a name from the tools list and a JSON object of arguments). `expected` counts the answers that call the tool the prompt asks for. `structured` means the server returned `tool_calls`; `text-*` means the call was parsed from the text.

| config | valid | expected | finish reasons | source |
| --- | ---: | ---: | --- | --- |
| baseline | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |
| dspark-v1 | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |
| dspark-v2 | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |

## Multi-slot

Each run fires the same 8 prompts (code-01, code-02, reasoning-01, reasoning-02, chat-01, chat-02, tool-01, tool-02) in waves of `slots` concurrent requests. `per stream` is the mean decode rate that each request saw; `aggregate` is generated tokens over the wall time of all waves, prefill included. The `slots 1` row uses the same prompts from the single-slot run.

| config | slots | prompts | per stream tok/s | aggregate tok/s | accept | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 1 | 8 | 29.67 | 28.79 | n/a | ok |
| baseline | 2 | 8 | 25.28 | 45.68 | n/a | ok |
| baseline | 4 | 8 | 20.34 | 56.86 | n/a | ok |
| dspark-v1 | 1 | 8 | 62.39 | 53.49 | 0.522 | ok |
| dspark-v1 | 2 | 8 | 43.23 | 67.15 | 0.533 | ok |
| dspark-v1 | 4 | 8 | 31.65 | 74.84 | 0.518 | ok |
| dspark-v2 | 1 | 8 | 58.27 | 52.61 | 0.477 | ok |
| dspark-v2 | 2 | 8 | 40.27 | 66.25 | 0.485 | ok |
| dspark-v2 | 4 | 8 | 30.09 | 74.43 | 0.472 | ok |

## Power

`nvidia-smi` sampled at 1 Hz on node_b. Idle is the mean draw over 10 s with the model loaded and no request in flight. Mean and max cover the generation phase. `gen tok/s` is generated tokens over the wall time of that phase, prefill included, and mJ/token is mean W / gen tok/s x 1000.

| config | slots | idle W | mean W | max W | util % | tokens | gen tok/s | mJ/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 14.5 | 60.9 | 86.5 | 94 | 34332 | 28.52 | 2135.1 |
| baseline | 2 | 15.4 | 59.9 | 79.5 | 93 | 3053 | 45.68 | 1312.2 |
| baseline | 4 | 15.9 | 65.4 | 75.6 | 94 | 3053 | 56.86 | 1150.2 |
| dspark-v1 | 1 | 16.0 | 77.0 | 85.1 | 92 | 34560 | 52.85 | 1457.5 |
| dspark-v1 | 2 | 16.2 | 60.4 | 77.2 | 84 | 3053 | 67.15 | 899.1 |
| dspark-v1 | 4 | 15.3 | 59.0 | 77.0 | 87 | 3053 | 74.84 | 788.1 |
| dspark-v2 | 1 | 15.8 | 78.1 | 85.0 | 92 | 34614 | 52.24 | 1494.6 |
| dspark-v2 | 2 | 16.3 | 59.4 | 82.2 | 86 | 3053 | 66.25 | 896.8 |
| dspark-v2 | 4 | 15.3 | 61.3 | 82.2 | 80 | 3053 | 74.43 | 823.1 |

## Configuration

- `baseline` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 3.0 s
- `baseline` slots 2: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 2 --jinja`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 2, n_ctx 8192, load 3.0 s
- `baseline` slots 4: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 4 --jinja`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 4, n_ctx 4096, load 3.0 s
- `dspark-v1` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 4.0 s
  drafter metadata: {"dflash.block_count": 6, "dflash.block_size": 4, "general.architecture": "dflash", "general.name": "Bonsai-27B-dspark"}
- `dspark-v1` slots 2: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 2 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 2, n_ctx 8192, load 4.0 s
  drafter metadata: {"dflash.block_count": 6, "dflash.block_size": 4, "general.architecture": "dflash", "general.name": "Bonsai-27B-dspark"}
- `dspark-v1` slots 4: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 4 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 4, n_ctx 4096, load 4.0 s
  drafter metadata: {"dflash.block_count": 6, "dflash.block_size": 4, "general.architecture": "dflash", "general.name": "Bonsai-27B-dspark"}
- `dspark-v2` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 5 -ngld 999`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 3.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- `dspark-v2` slots 2: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 2 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 5 -ngld 999`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 2, n_ctx 8192, load 4.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- `dspark-v2` slots 4: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 4 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 5 -ngld 999`
  build b10687-5d80cff0b, model `/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 4, n_ctx 4096, load 4.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- GPU: NVIDIA GB10, 580.178.04
- GPU processes at start: ./bin/cuda/llama-server (11006 MiB), bin/cuda/llama-server (13310 MiB)
- Matrix and long prompts: POST /completion, temperature 0, seed 42, cache_prompt false, ChatML template applied by the client, no system message
- Tool and agent prompts: POST /v1/chat/completions, temperature 0, seed 42, cache_prompt false, tools list with tool_choice auto, server template (--jinja)
- Draft counters: `timings.draft_n` and `timings.draft_n_accepted` from each response
