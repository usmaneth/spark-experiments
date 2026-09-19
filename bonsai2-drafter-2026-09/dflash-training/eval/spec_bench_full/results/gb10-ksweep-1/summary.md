# Speculative decoding benchmark

Generated 2026-09-19T10:58:00Z on spark1 (NVIDIA GB10, driver 580.178.04, CUDA 13.0). llama-server version: 0.2.0-dev (build 10687, commit 5d80cff0b), build b10687-5d80cff0b.

Prompts: 32 matrix prompts x 512 tokens, 8 tool prompts x 512 tokens, 8 agent prompts x 512 tokens. Passes per prompt: 1. Slots: 1. Rates are arithmetic means of the server decode-only `predicted_per_second`; acceptance is aggregated accepted/drafted tokens; speedup is the ratio of the two means over the same prompts. The tool and agent rows come from `/v1/chat/completions` with a tools list and the server template (reasoning effort: template); the other rows come from `/completion` with a client-side ChatML template.

## Workload matrix: dspark-v2

| workload | no drafter | DSpark v2 (K=5) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | n/a | 64.75 | 0.548 (2875/5249) | 3.71 | n/a |
| math | n/a | 73.41 | 0.659 (2893/4391) | 4.29 | n/a |
| reasoning | n/a | 56.81 | 0.445 (2707/6088) | 3.20 | n/a |
| chat | n/a | 43.70 | 0.298 (2062/6910) | 2.49 | n/a |
| tool | n/a | 48.08 | 0.373 (1306/3497) | 2.87 | n/a |
| agent | n/a | 46.14 | 0.408 (1157/2838) | 3.05 | n/a |
| blended (32 prompts) | n/a | 59.67 | 0.465 (10537/22638) | 3.31 | n/a |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

## Workload matrix: dspark-v2-k3

| workload | no drafter | + dspark-v2-k3 (K=3) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | n/a | 58.54 | 0.665 (2615/3932) | 2.98 | n/a |
| math | n/a | 64.38 | 0.773 (2635/3411) | 3.31 | n/a |
| reasoning | n/a | 53.55 | 0.597 (2259/3785) | 2.78 | n/a |
| chat | n/a | 44.70 | 0.421 (1924/4569) | 2.26 | n/a |
| tool | n/a | 49.24 | 0.514 (1256/2442) | 2.55 | n/a |
| agent | n/a | 45.32 | 0.557 (1032/1854) | 2.69 | n/a |
| blended (32 prompts) | n/a | 55.29 | 0.601 (9433/15697) | 2.79 | n/a |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

## Workload matrix: dspark-v2-k4

| workload | no drafter | + dspark-v2-k4 (K=4) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | n/a | 62.92 | 0.610 (2784/4566) | 3.41 | n/a |
| math | n/a | 70.13 | 0.714 (2791/3909) | 3.85 | n/a |
| reasoning | n/a | 56.47 | 0.509 (2632/5169) | 3.02 | n/a |
| chat | n/a | 44.86 | 0.353 (2016/5713) | 2.41 | n/a |
| tool | n/a | 48.78 | 0.429 (1269/2955) | 2.73 | n/a |
| agent | n/a | 46.61 | 0.473 (1128/2386) | 2.90 | n/a |
| blended (32 prompts) | n/a | 58.60 | 0.528 (10223/19357) | 3.10 | n/a |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

## Workload matrix: dspark-v2-k6

| workload | no drafter | + dspark-v2-k6 (K=6) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | n/a | 63.21 | 0.489 (2927/5986) | 3.90 | n/a |
| math | n/a | 72.88 | 0.597 (2948/4940) | 4.58 | n/a |
| reasoning | n/a | 55.58 | 0.394 (2758/6998) | 3.34 | n/a |
| chat | n/a | 41.80 | 0.256 (2087/8137) | 2.53 | n/a |
| tool | n/a | 46.62 | 0.323 (1324/4096) | 2.95 | n/a |
| agent | n/a | 43.63 | 0.357 (1243/3486) | 3.16 | n/a |
| blended (32 prompts) | n/a | 58.37 | 0.411 (10720/26061) | 3.45 | n/a |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

## Workload matrix: dspark-v2-k7

| workload | no drafter | + dspark-v2-k7 (K=7) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | n/a | 60.55 | 0.442 (2965/6706) | 4.05 | n/a |
| math | n/a | 70.61 | 0.549 (2992/5451) | 4.84 | n/a |
| reasoning | n/a | 52.38 | 0.351 (2788/7945) | 3.43 | n/a |
| chat | n/a | 39.32 | 0.226 (2108/9317) | 2.57 | n/a |
| tool | n/a | 42.94 | 0.280 (1328/4748) | 2.96 | n/a |
| agent | n/a | 40.51 | 0.311 (1250/4017) | 3.20 | n/a |
| blended (32 prompts) | n/a | 55.72 | 0.369 (10853/29419) | 3.56 | n/a |

The agent prompts carry a mean prefix of 3079 tokens (system prompt, tools, task, one tool result) before generation starts.

`tok/step` is generated tokens per verify step, predicted_n / (predicted_n - accepted); a server without a drafter is at 1.00.

## Tool calls

A tool prompt is valid when the answer holds at least one well-formed call (a name from the tools list and a JSON object of arguments). `expected` counts the answers that call the tool the prompt asks for. `structured` means the server returned `tool_calls`; `text-*` means the call was parsed from the text.

| config | valid | expected | finish reasons | source |
| --- | ---: | ---: | --- | --- |
| dspark-v2 | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |
| dspark-v2-k3 | 6/8 | 7/8 | length 2, tool_calls 6 | structured 7 |
| dspark-v2-k4 | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |
| dspark-v2-k6 | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |
| dspark-v2-k7 | 6/8 | 6/8 | length 2, tool_calls 6 | structured 6 |

## Power

`nvidia-smi` sampled at 1 Hz on spark1. Idle is the mean draw over 10 s with the model loaded and no request in flight. Mean and max cover the generation phase. `gen tok/s` is generated tokens over the wall time of that phase, prefill included, and mJ/token is mean W / gen tok/s x 1000.

| config | slots | idle W | mean W | max W | util % | tokens | gen tok/s | mJ/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dspark-v2 | 1 | 13.6 | 80.3 | 86.8 | 91 | 18822 | 48.16 | 1667.5 |
| dspark-v2-k3 | 1 | 16.5 | 75.5 | 86.2 | 91 | 18407 | 46.65 | 1618.5 |
| dspark-v2-k4 | 1 | 16.5 | 78.5 | 85.5 | 92 | 18822 | 48.06 | 1633.3 |
| dspark-v2-k6 | 1 | 16.7 | 80.1 | 85.7 | 92 | 18918 | 46.94 | 1706.3 |
| dspark-v2-k7 | 1 | 16.7 | 80.9 | 86.0 | 92 | 18918 | 44.55 | 1817.2 |

## Configuration

- `dspark-v2` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 5 -ngld 999`
  build b10687-5d80cff0b, model `/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 3.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- `dspark-v2-k3` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 3 -ngld 999`
  build b10687-5d80cff0b, model `/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 3.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- `dspark-v2-k4` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999`
  build b10687-5d80cff0b, model `/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 3.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- `dspark-v2-k6` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 6 -ngld 999`
  build b10687-5d80cff0b, model `/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 4.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- `dspark-v2-k7` slots 1: `bin/cuda/llama-server -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf --host 127.0.0.1 --port 8099 -ngl 999 -fa on -c 16384 -np 1 --jinja -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --spec-type draft-dspark --spec-draft-n-max 7 -ngld 999`
  build b10687-5d80cff0b, model `/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 16384, load 4.0 s
  drafter metadata: {"dflash.block_count": 5, "dflash.block_size": 7, "general.architecture": "dflash", "general.name": "Qwen3.8-27B-DSpark"}
- GPU: NVIDIA GB10, 580.178.04
- GPU processes at start: /home/usman/Bonsai-demo/bin/cuda/llama-server (9970 MiB)
- Matrix and long prompts: POST /completion, temperature 0, seed 42, cache_prompt false, ChatML template applied by the client, no system message
- Tool and agent prompts: POST /v1/chat/completions, temperature 0, seed 42, cache_prompt false, tools list with tool_choice auto, server template (--jinja)
- Draft counters: `timings.draft_n` and `timings.draft_n_accepted` from each response
