# EAGLE-3 drafter for Bonsai 2 27B (design)

Date: 2026-09-18. Author: claude-code. Status: approved for a smoke build.

## Why

The DSpark drafter reached a ceiling at about 52% (math) / 43% (code) exact-match acceptance at K=5.
Round-2 (broad data), the LR probe and the epoch-2 run all show the same plateau. The DSpark decoder
predicts all 7 block positions in parallel from one anchor. Position t+k does not see the draft
tokens at t+1..t+k-1. Acceptance falls with depth for that reason. An EAGLE-3 drafter is
autoregressive: each draft token conditions on the previous draft token and on its own hidden state.

The runtime already exists. `llama.cpp` on the `prism` branch has `--spec-type draft-eagle3`
(`src/models/eagle3.cpp`, `common/speculative.cpp`). This design only adds a trainer and a GGUF writer.

## Runtime contract (from `src/models/eagle3.cpp` and `common/speculative.cpp`)

Encoder (runs once per verify batch, one row per target position P):

    x_P  = concat(h_low[P], h_mid[P], h_high[P])        # 3 x 5120 = 15360, target layer INPUTS
    x_P  = rmsnorm(x_P) * w_enc                         # only when eagle3.norm_before_fc = true
    g_P  = W_fc x_P                                     # 15360 -> 5120, no bias

Decoder (one layer). At memory position P the input pair is (token[P+1], g_P). The output
predicts token[P+2].

    e    = tok_embd[token]                              # own token_embd, 5120, frozen copy
    e_n  = rmsnorm(e) * w_attn_norm
    g_n  = rmsnorm(g) * w_attn_norm_2
    x    = concat(e_n, g_n)                             # 10240; e first, then g
    q    = W_q x   (32 heads x 128)     k = W_k x  (8 kv heads x 128)     v = W_v x
    RoPE on q, k: LLAMA_ROPE_TYPE_NORM (adjacent pairs (2i, 2i+1)), theta 1e7, n_rot 128, no q/k norm
    a    = W_o softmax(q k^T / sqrt(128)) v            # causal over the KV cache, GQA
    r    = a + g                                        # raw g in the residual (norm_before_residual = false)
    h    = r + W_down( silu(W_gate rmsnorm(r) * w_ffn_norm) * W_up rmsnorm(r) * w_ffn_norm )
    logits_draft = W_out rmsnorm(h) * w_output_norm     # over the draft vocab
    logits[d2t[i]] = logits_draft[i], others = -inf     # d2t: int64 [n_draft_vocab]

The pre-norm state h of a decoder row is the g of the next draft step. The draft chain is:

    seed: (t_last, g_P) at pos P      -> draft d1, hidden h_P
    step: (d1, h_P)     at pos P+1    -> draft d2, hidden h_{P+1}
    step: (d2, h_{P+1}) at pos P+2    -> draft d3 ...

The chain is greedy (top-1) and stops at `--spec-draft-n-max` or when top-1 p < `--spec-draft-p-min`.

## Training (mirror of the chain, teacher-forced tokens, own hidden states)

Data: BON2 files in `v2/feats_all` (batch1, batch2, batch3_broad; 3,401 samples, 1.57M tokens).
Taps are target layer inputs at layers [6, 20, 34, 48, 62]. Use tap indices 0, 2, 4 = layers [6, 34, 62].
`final_hidden[i]` is the target's last hidden state at position i; `softmax(W_lm final_hidden[i])` is
the target's distribution for token i+1. `loss_mask[i] = 1` marks model-generated tokens.

Step 1 over a sequence of length S: row P = (t[P+1], g_real[P]) at RoPE pos P; target t[P+2].
Step j (2 <= j <= D): row P = (t[P+j], H_{j-1}[P]) at RoPE pos P+j-1; target t[P+j+1].
H_{j-1}[P] is the pre-norm output of the step j-1 row with index P.

Attention for a step-j query with index P sees:
- step-1 keys with index <= P (the verified prefix, real g), and
- step-m keys with index == P for 2 <= m <= j (its own chain, including itself).

Implement this as one SDPA call per step over the concatenated K/V of steps 1..j with a boolean mask.
This is the SpecForge "training-time test" layout. Depth D = 4 for the smoke run (flag `--ttt-depth`).

Loss per step j, per position: soft cross-entropy between the draft logits (draft vocab) and the target
distribution restricted to the draft vocab and renormalized. Apply only where `loss_mask[t index] = 1`.
Total = mean over steps (uniform weights). Log per-step top-1 agreement with the target argmax
(the acceptance proxy) and the hard CE.

Draft vocab: the 32,768 most frequent target token ids over all BON2 token streams, plus every
special token (eos, im_end, pad, mask). `d2t` int64 [32768] sorted ascending by target id.
`t2d` int64 [248320] with -1 for absent ids. Report the coverage of generated tokens.

Init: `token_embd` = `v2/teacher/tok_embd.bin` (Hadamard-corrected, frozen, not trained).
`W_out` rows = `v2/teacher/W_lm.bin[d2t]` (trainable). `W_fc` = [I/3 | I/3 | I/3] (identity average).
Decoder linears N(0, 0.02). Norm gains 1.

Optimizer: AdamW, lr 1e-4 (flag), betas (0.9, 0.95), wd 0.01, warmup 100 steps, cosine to 1e-6,
bf16 autocast, grad clip 1.0, batch 2 sequences, `--max-seq-len 1024`. Save every 300 steps.
Checkpoint = safetensors with GGML tensor names (see below) plus `d2t`.

## GGUF writer (`eagle3_to_gguf.py`)

Arch `eagle3`. Keys: `general.architecture`, `general.name`, `eagle3.block_count = 1`,
`eagle3.context_length = 8192`, `eagle3.embedding_length = 5120`, `eagle3.feed_forward_length = 17408`,
`eagle3.attention.head_count = 32`, `eagle3.attention.head_count_kv = 8`,
`eagle3.attention.key_length = 128`, `eagle3.attention.value_length = 128`,
`eagle3.rope.dimension_count = 128`, `eagle3.rope.freq_base = 1e7`,
`eagle3.attention.layer_norm_rms_epsilon = 1e-6`, `eagle3.vocab_size = 248320`,
`eagle3.target_layers = [6, 34, 62]` (i32 array), `eagle3.target_hidden_size = 5120`,
`eagle3.norm_before_fc = true`, `eagle3.norm_before_residual = false`.
All `tokenizer.*` keys come from the donor `Ternary-Bonsai-2-27B-PQ2_0.gguf`, the same way
`gguf_dspark_to_dflash.py` copies them.

Tensors (numpy shape = torch shape; gguf-py reverses the dims):
`fc.weight` [5120, 15360], `enc.output_norm.weight` [15360], `blk.0.attn_norm.weight` [5120],
`blk.0.attn_norm_2.weight` [5120], `blk.0.attn_q.weight` [4096, 10240], `blk.0.attn_k.weight` [1024, 10240],
`blk.0.attn_v.weight` [1024, 10240], `blk.0.attn_output.weight` [5120, 4096], `blk.0.ffn_norm.weight` [5120],
`blk.0.ffn_gate.weight` [17408, 5120], `blk.0.ffn_up.weight` [17408, 5120], `blk.0.ffn_down.weight` [5120, 17408],
`output_norm.weight` [5120], `output.weight` [32768, 5120], `d2t` int64 [32768],
`token_embd.weight` [248320, 5120] (f16).
Then `llama-quantize <conv> <out> Q8_0` (test Q4_K_M after Q8_0 works; check that the I64 `d2t` passes).

## Eval

Same harness as every other drafter (`/tmp/ep1_sweep.sh` via sed): `llama-speculative-simple`
with `--spec-type draft-eagle3 -md <eagle3.gguf> --spec-draft-n-max 5`, K=5 exact match, the three
200-token prompts and the six long-form prompts, clean spark2. The comparison row is DSpark step-600:
math 52.1% / 61.5 tok/s, code 42.9% / 53.9, long-form mean 64.4 (baseline 29.8).

Cost model to beat: an EAGLE-3 step is 1 batched encoder+decoder pass plus 5 chain passes
(about 15-18 ms at Q8_0, 12 ms at Q4_K_M) against DSpark's 7.8 ms. Break-even needs about
3.9 accepted+1 tokens per step; a win needs 4.5+.

## Files

`v2/eagle3/build_draft_vocab.py`, `v2/eagle3/train_eagle3.py`, `v2/eagle3/eagle3_to_gguf.py`,
`v2/eagle3/run_smoke_spark2.sh`, logs in `v2/logs/eagle3_*.log`, checkpoints in
`models/bonsai2-eagle3/`.
