# Speculative decoding benchmark

Note: dry run, not publishable: contended GPUs, external servers (spark1 :8085 plain PQ2_0 with 4 slots, spark2 :8096 bonsai2-27b-spec with the DSpark v2 drafter, 1 slot); no v1 server; power sampled on the client host spark1 only

Generated 2026-09-18T18:26:13Z on spark1 (NVIDIA GB10, driver 580.178.04, CUDA 13.0). llama-server version: 0.2.0-dev (build 10687, commit 5d80cff0b), build b10687-5d80cff0b.

Prompts: 1 matrix prompts x 128 tokens, 1 tool prompts x 128 tokens, 1 agent prompts x 128 tokens. Passes per prompt: 1. Slots: 1. Rates are arithmetic means of the server decode-only `predicted_per_second`; acceptance is aggregated accepted/drafted tokens; speedup is the ratio of the two means over the same prompts. The tool and agent rows come from `/v1/chat/completions` with a tools list and the server template (reasoning effort: template); the other rows come from `/completion` with a client-side ChatML template. The single quicksort prompt runs 3 passes at 64 tokens on every single-slot server.

## Workload matrix: dspark-v2

| workload | no drafter | DSpark v2 (K=5) | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 10.58 | 12.26 | 0.463 (88/190) | 3.20 | **1.16x** |
| tool | 10.26 | 9.94 | 0.336 (37/110) | 2.76 | **0.97x** |
| agent | 11.36 | 11.31 | 0.409 (85/208) | 2.98 | **1.00x** |
| blended (1 prompts) | 10.58 | 12.26 | 0.463 (88/190) | 3.20 | **1.16x** |

The agent prompts carry a mean prefix of 3002 tokens (system prompt, tools, task, one tool result) before generation starts.

`tok/step` is generated tokens per verify step, predicted_n / (predicted_n - accepted); a server without a drafter is at 1.00.

## Single prompt

The quicksort prompt of the PrismML GB10 document (64 tokens, 3 passes), tokens per second per pass.

| config | pass 1 | pass 2 | pass 3 | mean | accept | tok/step | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 10.85 | 13.21 | 10.42 | 11.50 | n/a | 1.00 | n/a |
| dspark-v2 | 10.55 | 11.34 | 11.93 | 11.27 | 0.433 (126/291) | 2.91 | **0.98x** |

## Exactness

Each drafter output is compared with the baseline output for the same prompt and pass. `/completion` outputs are identical when their token id sequences are equal; chat outputs (reasoning, answer, tool calls) are compared as text because that endpoint returns no token ids. For a different output the table lists the 0-based index of the first differing token or character; when one output is a prefix of the other, the index equals the shorter length.

| config | identical | first difference |
| --- | ---: | --- |
| dspark-v2 | 6/6 |  |

Baseline outputs across passes: identical.

## Tool calls

A tool prompt is valid when the answer holds at least one well-formed call (a name from the tools list and a JSON object of arguments). `expected` counts the answers that call the tool the prompt asks for. `structured` means the server returned `tool_calls`; `text-*` means the call was parsed from the text.

| config | valid | expected | finish reasons | source |
| --- | ---: | ---: | --- | --- |
| baseline | 1/1 | 1/1 | tool_calls 1 | structured 1 |
| dspark-v2 | 1/1 | 1/1 | tool_calls 1 | structured 1 |

## Power

`nvidia-smi` sampled at 1 Hz on spark1. Idle is the mean draw over 10 s with the model loaded and no request in flight. Mean and max cover the generation phase. `gen tok/s` is generated tokens over the wall time of that phase, prefill included, and mJ/token is mean W / gen tok/s x 1000. In server-url mode the samples come from the client host, which may not run the server.

| config | slots | idle W | mean W | max W | util % | tokens | gen tok/s | mJ/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 55.3 | 57.8 | 74.1 | 96 | 506 | 8.92 | 6475.9 |
| dspark-v2 | 1 | 55.0 | 55.9 | 63.9 | 86 | 506 | 7.94 | 7040.5 |

## Configuration

- `baseline` slots 1: remote server http://10.99.0.1:8085 (not launched by the tool; its flags are its own)
  build b10687-5d80cff0b, model `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 4, n_ctx 32768
- `dspark-v2` slots 1: remote server http://10.99.0.2:8096 (not launched by the tool; its flags are its own)
  build b10687-5d80cff0b, model `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, slots 1, n_ctx 8192
- GPU: NVIDIA GB10, 580.178.04
- GPU processes at start: /home/usman/Bonsai-demo/bin/cuda/llama-server (9970 MiB), bin/cuda/llama-server (13310 MiB)
- Matrix and long prompts: POST /completion, temperature 0, seed 42, cache_prompt false, ChatML template applied by the client, no system message
- Tool and agent prompts: POST /v1/chat/completions, temperature 0, seed 42, cache_prompt false, tools list with tool_choice auto, server template (--jinja)
- Draft counters: `timings.draft_n` and `timings.draft_n_accepted` from each response
