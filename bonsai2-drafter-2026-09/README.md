# Bonsai 2 27B speculative-decoding drafter experiments (2026-09-17 to 2026-09-19)

This directory is the backup of the experiment tree. It holds the code, the scripts, the logs, the
prompt sets, the generated datasets and the benchmark results. It does not hold tensors: the feature
sets (up to 1.9 TB) and the checkpoints (3.7 GB each) were deleted on 2026-09-19, and the two
drafter GGUF files (1.1 GB and 0.6 GB) go to Hugging Face (see hf-release/UPLOAD.md).

## Contents

| path | what |
| --- | --- |
| dflash-training/v2/ | DSpark v2, v2.1, v3 and EAGLE-3 training, extraction and probe scripts; logs/ |
| dflash-training/v2/eagle3/ | EAGLE-3 design, trainer, converter and ablation scripts |
| dflash-training/v2/prompts_*.jsonl | prompt sets and generated outputs (the dataset recipe) |
| dflash-training/eval/spec_bench_full/ | benchmark harness and results/ (gb10-clean-1, gb10-ksweep-1, ...) |
| dflash-training/dashboard/ | the training dashboard server (port 8090) |
| runtime-fix/ | the llama.cpp Hadamard fix as a patch and the reproduction report (PrismML-Eng/llama.cpp#210) |
| hf-release/ | the model card, SHA256SUMS and upload notes for the two drafters |
| notes/ | the full session note, the blog post, the publish facts, the cluster data-layout rule |

## Results in one table

| variant | math | code | code2 | verdict |
| --- | ---: | ---: | ---: | --- |
| DSpark v2 (K=5, Q4_K_M) | 52.1% / 61.5 tok/s | 42.9% / 53.9 | 49.5% / 59.8 | shipped |
| DSpark v1 (K=4, re-converted) | 56.0% | 46.4% | 56.6% | shipped as the second file |
| EAGLE-3 run B (12.5M tokens) | 43.6% / 52.3 | 30.1% / 41.4 | 34.8% / 44.6 | closed |
| K = 3, 4, 6, 7 | within +-2% of K=5 blended | | | K=5 stays |
| Q5_K_M, Q6_K, Q8_0, F16 drafter | same acceptance, 3-21% fewer tok/s | | | Q4_K_M stays |

Baseline without a drafter: 29.7 tok/s on one GB10. Full workload matrix, power and multi-slot numbers:
dflash-training/eval/spec_bench_full/results/gb10-clean-1/summary.md.

## Not vendored

SpecForge (used only for the EAGLE-3 mask layout test): https://github.com/sgl-project/SpecForge.git at ed64d27.

## Pull requests

- PrismML-Eng/Bonsai-demo#188 (benchmark entry), #189 (retrain pipeline), #190 (benchmark suite)
- PrismML-Eng/llama.cpp#210 (the runtime fix)
