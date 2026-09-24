# Publish track: definitive facts for the finishing pass (2026-09-18)

## Naming (final, maintainer's call)
- Drafter file: `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf`. Prose: "DSpark v2". No codename. The earlier "shohin" name is retracted everywhere.
- Local file: `models/bonsai2-dspark/bonsai2-dspark-full2step600-Q4_K_M.gguf`; a copy under the published name exists in `models/bonsai2-gguf/27B/` on both nodes.
- GGUF header `general.name` is still `Qwen3.8-27B-DSpark` (inherited); note it, do not claim it is fixed.

## The other drafter file
- `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf` is PrismML's Ternary-Bonsai-27B (old model) drafter re-converted against Bonsai 2 by the earlier session. Header: `general.name = Bonsai-27B-dspark`, `dflash.target_layers = [2,17,32,47,62]`, `block_size 4`.
- PrismML ships NO drafter for Bonsai 2 (`scripts/download_models.sh:144`; the HF repos hold none). Never call this file "the shipped Bonsai 2 drafter".
- With the runtime fix it transfers to the Bonsai 2 target: K=4 math 56.0% (140/250), code 46.4% (130/280), code2 56.6% (142/251); about 3.3 tokens per step vs 3.85 for DSpark v2 (K=5). Label in tables: "DSpark v1 (Ternary-Bonsai-27B drafter, K=4)". Keep it as a comparison row.

## Runtime
- Clean PrismML-Eng/llama.cpp prism `1a07bfa5f` (build 10706): 0.4-2.0% acceptance for both drafters, silent. Same defect in the release tag `prism-b10683-d8f26ee` that setup.sh installs.
- Cause: dflash borrows the target's `token_embd`/`output` via ctx_other; the draft graph never applies the target's Hadamard maps (inverse for token_embd, forward for output).
- Fix: branch `fix/dflash-borrowed-hadamard` in `/home/REDACTED/llama.cpp-upstream` (+48/-20; `dflash-borrowed-hadamard.patch`); restores 148/284, 137/319, 143/289 bit for bit. PR title: "dflash: apply the target's Hadamard transforms to borrowed token_embd and output". Report: `/home/REDACTED/llama.cpp-upstream/REPRO-bonsai2-drafter.md`.
- Docs must cite `1a07bfa5f` + the patch (or its merge commit), never `1a07bfa5f` alone. `--version` prints the same build id for clean and patched binaries.
- Our verified numbers were measured with `bin/cuda` = `5d80cff0b` + the local patch (same fix, earlier form).

## Launcher facts
- `scripts/start_llama_server.sh:138` globs `*dspark-dflash*.gguf`; with both files present `...-dspark-dflash-Q4_0.gguf` sorts first. Pass `-md` directly or move the other file.
- `scripts/common.sh bonsai_dspark_block_size` reads `dspark.dspark.block_size`; converted files carry `dflash.block_size`, so the launcher falls back to n-max 4. Our numbers need `BONSAI_SPEC_NMAX=5`.

## Server flags for published numbers
`-ngl 99 -fa on -c 16384 -np 1 --jinja` plus `-md <drafter> --spec-type draft-dspark --spec-draft-n-max 5 -ngld 999` (K=4 for the v1 drafter). Temperature 0, seed 42.

## Verified numbers (clean node_b, base 29.8-29.9 flat at 64/1024/2000 tokens)
- 200 tokens, K=5 exact: math 52.1% / 61.5 tok/s, code 42.9% / 53.9, code2 49.5% / 59.8.
- Long-form 2,000-token budget, 6 prompts: mean 64.4 tok/s (2.16x), math 71.8 (2.41x).
- Workload matrix, tool calling, agent turns, multi-slot and power come from the extended bench (`dflash-training/eval/spec_bench_full`) run in the clean window.

## PR set and hygiene
1. Bonsai-demo `docs/gb10-bonsai2-drafter-benchmark`: supersedes #168 (say so). Files: the GB10 Bonsai 2 doc + one index row in each README.
2. Bonsai-demo `feat/dspark-retraining-pipeline`: `tools/dspark-retrain/` (keep `DSPARK_TRAINER` env var: the docker base image is not public; keep `build.sh` as the three g++ lines with -fopenmp; `BONSAI_ROOT` allowed).
3. Bonsai-demo `bench/gb10-bonsai2-drafter-suite`: `scripts/spec_bench/` + tests + CI step. No dry-run sample output in the PR; `.gitignore` for `scripts/spec_bench/results/`.
4. PrismML-Eng/llama.cpp: the Hadamard fix, from a branch on the fork.
- No `/home/REDACTED` paths in repo files. ASCII only in scripts. No agent/workflow/model references. No Co-Authored-By. Titles in the repo's short scoped style; bodies open with "## Summary".
- HF upload of the GGUF is the maintainer's manual action.
