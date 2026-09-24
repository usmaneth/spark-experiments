# Bonsai 2 DSpark drafter: reproduction on clean upstream llama.cpp

This file is untracked working material for the benchmark doc and the blog post.
It records what a clean build of `PrismML-Eng/llama.cpp` branch `prism` at
`1a07bfa5f` does with the Bonsai 2 DSpark drafters, why, and the exact commands
that reproduce the numbers. Built and measured on node_a on 2026-09-18.

Upstream commit: `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`
(2026-09-17 23:58:45 -0700, "Merge pull request #179 from PrismML-Eng/feat/dspark-shared-head-runtime").

## Verdict

- Reproduces on clean upstream: **no**. The clean build compiles, loads both
  drafters and generates the correct text, but draft acceptance collapses to
  0.4-2.0% (expected 43-57%). Decode gets slower than no drafter. The failure is
  silent: exact-match verification keeps the output identical, so only the
  acceptance counter and the speed show it. No error, no warning.
- Cause: a `dflash` drafter that ships no `token_embd.weight` / `output.weight`
  borrows the target's through `ctx_other`. Bonsai 2 is Hadamard-folded
  (`prism.hadamard.*`), so the borrowed `output.weight` needs the forward
  rotation on its input and the borrowed `token_embd.weight` needs the inverse
  after the lookup. At `1a07bfa5f` the draft context only consults its own
  (empty) transform maps, and the coverage check that exists for this defect
  is skipped for the same reason. Section 4.
- PR #179 does not cover this case: it shares the head transform only for the
  new `dspark` architecture, only with `LLAMA_DSPARK_SHARED_HEAD=1`, and that
  architecture requires the drafter to carry its own `token_embd.weight`. Both
  drafters here are `general.architecture = dflash` and carry neither tensor.
- A 5-file patch (48 insertions, 20 deletions) on branch
  `fix/dflash-borrowed-hadamard` in this worktree restores the numbers on clean
  `1a07bfa5f`, count for count and byte for byte: 148/284, 137/319, 143/289.
  Section 5 has the diff, section 6 the PR text. Nothing is committed or pushed.
- What to cite: `PrismML-Eng/llama.cpp` commit `1a07bfa5f` plus the PR that
  carries this patch (or the PR's merge commit once it exists). Do not cite
  `1a07bfa5f` alone, and do not cite the local 18-file tree behind
  `/home/REDACTED/Bonsai-demo/bin/cuda` (build 10687, commit `5d80cff0b`).
- The demo's pinned release binaries (`RELEASE_TAG="prism-b10683-d8f26ee"` in
  `scripts/download_binaries.sh:15`) predate `1a07bfa5f` by 23 commits and have
  the same gap. `BONSAI_SPECULATIVE=1` on Bonsai 2 does not accelerate with the
  binaries that `setup.sh` installs. A reader must build from source with the
  patch until a release carries it.
- `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf` is not
  a PrismML Bonsai 2 drafter. It is the older `Ternary-Bonsai-27B-dspark-bf16.gguf`
  (trained for Ternary-Bonsai-27B v1) run through `gguf-dspark-to-dflash
  --drop-shared-tensors` with the Bonsai 2 PQ2_0 file as tokenizer donor, then
  quantized to Q4_0. Its layer tensors are byte-identical to the v1 file
  (`blk.0.attn_q.weight`, `blk.5.ffn_down.weight`, `output_norm.weight` checked),
  `general.name` is `Bonsai-27B-dspark`, and the Hugging Face repos
  `prism-ml/Ternary-Bonsai-2-27B-gguf` and `-gguf-dev` hold no drafter file.
  Its rows below are labelled "v1 drafter re-converted", not a vendor baseline.
  A reader cannot download it; section 7 says how to rebuild it.
- Step 4 (a self-contained drafter converted without `--drop-shared-tensors`)
  does not apply: the drafter checkpoint has no embedding or head to keep.
  Section 8.

## 1. Clean build

```bash
git clone https://github.com/PrismML-Eng/llama.cpp.git /home/REDACTED/llama.cpp-upstream
git -C /home/REDACTED/llama.cpp-upstream checkout 1a07bfa5f
cmake -S /home/REDACTED/llama.cpp-upstream -B /home/REDACTED/llama.cpp-upstream/build-cuda \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/REDACTED/llama.cpp-upstream/build-cuda -j 18 \
  --target llama-speculative-simple llama-server llama-bench llama-quantize llama-gguf
```

Result: 4 min 23 s wall time on the GB10 (10:29:41 to 10:34:04, 20 cores, no
ccache), 0 errors, 2 compiler warnings, both pre-existing (`dflash.cpp:500`
unused parameter, one in ggml). Every binary prints the same id:

```text
$ LD_LIBRARY_PATH=build-cuda/bin build-cuda/bin/llama-speculative-simple --version
version: 0.2.0-dev (build 10706, commit 1a07bfa5f)
built with GNU 13.3.0 for Linux aarch64
```

The same flags produced the local patched binaries
(`/home/REDACTED/Bonsai-demo/llama.cpp/build-cuda/CMakeCache.txt`: GGML_CUDA=ON,
CMAKE_CUDA_ARCHITECTURES=121a, GGML_CUDA_FA=ON, GGML_CUDA_FA_ALL_QUANTS=OFF,
GGML_NATIVE=ON, Release), so the comparison is flag for flag.

After the patch, `cmake --build` with the same target list relinks in under two
minutes. The binaries still print `build 10706, commit 1a07bfa5f`
(`cmake/build-info.cmake` adds no dirty marker), so tell the two apart by path.
The unpatched binaries are kept in `build-cuda/bin.clean-1a07bfa5f/`;
`build-cuda/bin/` now holds the patched ones.

## 2. Acceptance

Host: node_a (DGX Spark GB10, aarch64, CUDA 13.0, driver 580.178.04). The GPU
was at 88-91% utilization from a separate generation server for every run, so
acceptance counts are valid (greedy, deterministic) and tokens/s are not. Each
cell is one 200-token run of `llama-speculative-simple`:

```bash
export LD_LIBRARY_PATH=<bin dir>
<bin dir>/llama-speculative-simple -m $TARGET -md $DRAFTER --spec-type draft-dspark --spec-draft-n-max $K \
  --spec-draft-p-min 0 -fa on -ngl 99 -ngld 999 -c 4096 -n 200 --temp 0 -e -p "$PROMPT"
```

`accept` is `n_accept / n_drafted` from the tool's own summary.
`--spec-draft-p-min 0` equals this fork's default (`common/common.h`:
`p_min = 0.0f`). Target: `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`.
Prompts, verbatim (`-e` expands `\n`):

```text
MATH:  <|im_start|>user\nA train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step.<|im_end|>\n<|im_start|>assistant\n
CODE:  <|im_start|>user\nWrite a Python function that parses a CSV file of transactions (date, amount, category) and returns the total spent per category as a dict, with error handling for malformed rows and a small unit test.<|im_end|>\n<|im_start|>assistant\n
CODE2: <|im_start|>user\nImplement binary search in Python with type hints, a docstring, and three assert-based tests.<|im_end|>\n<|im_start|>assistant\n
```

Binaries:

- "clean 1a07bfa5f": `build-cuda/bin.clean-1a07bfa5f/` (section 1).
- "local patched tree": `/home/REDACTED/Bonsai-demo/bin/cuda` (build 10687,
  commit `5d80cff0b` + 18 modified files), the binaries behind the numbers
  verified earlier on an idle node_b.
- "1a07bfa5f + this patch": `build-cuda/bin/` after section 5.

Drafters:

- "DSpark v2 (ours)": `models/bonsai2-dspark/bonsai2-dspark-full2step600-Q4_K_M.gguf`
  (5 layers, block 7, no log-SNR, shared tensors dropped, 1.1 GB). Published
  name: `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf`. n-max 5 is a
  deliberate choice below its block size.
- "v1 drafter re-converted": `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf`
  (6 layers, block 4, log-SNR, shared tensors dropped, 632 MB). See the verdict.
  `--spec-draft-n-max 5` clamps to its block size 4 with a warning, which is
  why the n-max 4 and 5 rows are identical.

| drafter | n-max | prompt | clean 1a07bfa5f | local patched tree (5d80cff0b + 18 files) | 1a07bfa5f + this patch |
|---|---:|---|---:|---:|---:|
| DSpark v2 (ours, Q4_K_M) | 5 | MATH | 1.6% (15/924) | 52.1% (148/284) | 52.1% (148/284) |
| DSpark v2 (ours, Q4_K_M) | 5 | CODE | 0.4% (4/978) | 42.9% (137/319) | 42.9% (137/319) |
| DSpark v2 (ours, Q4_K_M) | 5 | CODE2 | 1.2% (11/940) | 49.5% (143/289) | 49.5% (143/289) |
| v1 drafter re-converted (Q4_0) | 4 | MATH | 2.0% (15/741) | not run | 56.0% (140/250) |
| v1 drafter re-converted (Q4_0) | 4 | CODE | 0.9% (7/772) | not run | 46.4% (130/280) |
| v1 drafter re-converted (Q4_0) | 4 | CODE2 | 1.7% (13/746) | not run | 56.6% (142/251) |
| v1 drafter re-converted (Q4_0) | 5 | MATH | 2.0% (15/741) | not run | 56.0% (140/250) |
| v1 drafter re-converted (Q4_0) | 5 | CODE | 0.9% (7/772) | not run | 46.4% (130/280) |
| v1 drafter re-converted (Q4_0) | 5 | CODE2 | 1.7% (13/746) | not run | 56.6% (142/251) |

Decode speed observed on the contended GPU, tok/s (NOT publishable, listed only to show the direction):

| run | MATH | CODE | CODE2 |
|---|---:|---:|---:|
| clean, ours n-max 5 | 5.786 | 5.479 | 5.775 |
| patched tree, ours n-max 5 | 20.764 | 15.375 | 17.427 |
| fixed, ours n-max 5 | 12.162 | 15.145 | 16.771 |
| clean, v1 drafter n-max 4 | 6.100 | 5.903 | 6.163 |
| fixed, v1 drafter n-max 4 | 16.314 | 14.017 | 15.852 |

## 3. Output text

Every run was compared with the patched tree's output for the same prompt
(the target's greedy text). Trailing newlines are stripped. "yes" means the
shorter text is a prefix of the longer one; lengths differ only because a run
that accepts a whole block can overshoot the 200-token budget by a few tokens
(201-205 decoded).

| run | chars | reference chars | identical over the common prefix | decoded tokens |
|---|---:|---:|---|---:|
| clean-ours-k5-code | 1025 | 1033 | yes | 201 |
| clean-ours-k5-code2 | 945 | 945 | yes | 201 |
| clean-ours-k5-math | 913 | 921 | yes | 201 |
| clean-ship-k4-code | 1025 | 1033 | yes | 201 |
| clean-ship-k4-code2 | 945 | 945 | yes | 201 |
| clean-ship-k4-math | 913 | 921 | yes | 201 |
| clean-ship-k5-code | 1025 | 1033 | yes | 201 |
| clean-ship-k5-code2 | 945 | 945 | yes | 201 |
| clean-ship-k5-math | 913 | 921 | yes | 201 |
| fixed-ours-k5-code | 1033 | 1033 | yes | 202 |
| fixed-ours-k5-code2 | 945 | 945 | yes | 201 |
| fixed-ours-k5-math | 921 | 921 | yes | 205 |
| fixed-ship-k4-code | 1025 | 1033 | yes | 201 |
| fixed-ship-k4-code2 | 962 | 945 | yes | 205 |
| fixed-ship-k4-math | 917 | 921 | yes | 203 |
| fixed-ship-k5-code | 1025 | 1033 | yes | 201 |
| fixed-ship-k5-code2 | 962 | 945 | yes | 205 |
| fixed-ship-k5-math | 917 | 921 | yes | 203 |
| patched-ours-k5-code | 1033 | 1033 | yes | 202 |
| patched-ours-k5-code2 | 945 | 945 | yes | 201 |
| patched-ours-k5-math | 921 | 921 | yes | 205 |

Server check on the patched clean build, one chat request, the math question,
`temperature 0`, `max_tokens 200`, flags as in section 7:

```text
timings: prompt_n = 117, predicted_n = 200, draft_n = 226, draft_n_accepted = 153 (67.7%), predicted_per_second = 19.04 (contended GPU, not publishable)
finish_reason = length; the 200 tokens are the template's thinking block (see section 7)
```

The chat template enables thinking, so the server generated a different token
stream than the raw-prompt probe (a reasoning block instead of the direct
answer) and its counters are the server's own; the ratio is not comparable to
the table above. The server was started on a spare port, served this one
request, and was stopped by PID.

## 4. Diagnosis

What the target carries (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, same for PTQ1_0):

```text
prism.hadamard.block_size            = 1024
prism.hadamard.weight_names          = 401 names, including output.weight
prism.hadamard.inverse_weight_names  = ['token_embd.weight']
prism.hadamard.sign_mode             = explicit
token_embd.weight                    PQ2_0 [5120, 248320]
output.weight                        PQ2_0 [5120, 248320]
```

What both drafters carry: no `token_embd.weight`, no `output.weight`
(`gguf-dspark-to-dflash --drop-shared-tensors`, the conversion documented in
`SPECULATIVE.md`). `dflash.cpp` therefore takes
`tok_embd = model_other->tok_embd` and `output = model_other->output` from the
target through `cparams.ctx_other`.

What `1a07bfa5f` does with them:

1. `src/llama-context.cpp` `graph_params()` passes `&model.hadamard_rotations`
   and `&model.hadamard_inverses`, the draft model's own maps. They are empty.
2. `src/models/dflash.cpp` reads the embedding with a bare
   `ggml_get_rows(ctx0, tok_embd, inp->tokens)` (decoder input, DSV4 backbone,
   DFly predecessor chain). The target's own graph reads the same table through
   `build_inp_embd`, which applies the inverse (`h = s * (H z)`).
3. `build_lora_mm(output, cur)` looks `output` up in the (empty) map and skips
   the forward rotation, so the packed head multiplies an unrotated activation.
4. `graph_reserve` runs `llama_verify_hadamard_graph` only when the model's
   own maps are non-empty, so the check that exists for exactly this defect
   never ran for the draft context.

The `dspark` architecture that #179 added is a different loader
(`src/models/dspark.cpp`): it requires its own `token_embd.weight` and, with
`LLAMA_DSPARK_SHARED_HEAD=1`, copies the target head's rotation into its own
map at load time. Both Bonsai 2 drafters are `general.architecture = dflash`,
so they never reach that code.

Why nobody saw it before Bonsai 2: the published Ternary-Bonsai-27B v1 GGUFs
(`prism-ml/Ternary-Bonsai-27B-gguf`, PQ2_0 and Q2_g64) carry no
`prism.hadamard` metadata at all (checked from the published file headers:
0 folded weights, no inverse table). Borrowing their head and embeddings needed
no transform. Bonsai 2 is the first target with a rotated embedding table and a
folded head, and no vendor drafter has been paired with it yet.

The local 18-file tree fixes this with two inline changes
(`git -C /home/REDACTED/Bonsai-demo/llama.cpp diff -- src/models/dflash.cpp
src/llama-context.cpp`, base `5d80cff0b`): `graph_params()` and
`graph_reserve()` fall back to `ctx_other->model`'s maps when the draft's are
empty (6 lines), and `dflash.cpp` repeats the 9-line inverse block after two of
the three lookups. The other local changes in that tree (Medusa heads, tree
verification, candidate capture in `common/speculative.cpp`, the `seq_cp`
rollback index in `llama-memory-recurrent.cpp`) are not involved: the fix
branch carries none of them and reproduces the counts exactly.

The patch below does the same job in the codebase's shape: one lookup helper
used by the target path and all three drafter lookups (including the DFly chain
from #172, which the local patch predates), and context-owned maps that merge
both models' entries and also feed the coverage check, so a future graph that
borrows a folded tensor without its transform fails at context creation
instead of decoding garbage.

## 5. Patch

Branch `fix/dflash-borrowed-hadamard` in `/home/REDACTED/llama.cpp-upstream`
(based on `1a07bfa5f`, commit `288859a96`, open as PrismML-Eng/llama.cpp PR #210
from `usmaneth/llama.cpp`). The same diff is saved as
`/home/REDACTED/llama.cpp-upstream/dflash-borrowed-hadamard.patch`.
`git diff --check` is clean and the added lines are ASCII.

```text
 src/llama-context.cpp | 18 ++++++++++++++----
 src/llama-context.h   |  6 ++++++
 src/llama-graph.cpp   | 32 +++++++++++++++++++-------------
 src/llama-graph.h     |  6 ++++++
 src/models/dflash.cpp |  6 +++---
 5 files changed, 48 insertions(+), 20 deletions(-)
```

```diff
diff --git a/src/llama-context.cpp b/src/llama-context.cpp
index cc8c004eb..f688ea7ef 100644
--- a/src/llama-context.cpp
+++ b/src/llama-context.cpp
@@ -237,6 +237,16 @@ llama_context::llama_context(
         }
     }
 
+    hadamard_rotations = model.hadamard_rotations;
+    hadamard_inverses  = model.hadamard_inverses;
+    if (cparams.ctx_other) {
+        // tensors borrowed from the target are looked up by pointer, so the
+        // target's entries never collide with this model's
+        const auto & other = cparams.ctx_other->model;
+        hadamard_rotations.insert(other.hadamard_rotations.begin(), other.hadamard_rotations.end());
+        hadamard_inverses .insert(other.hadamard_inverses .begin(), other.hadamard_inverses .end());
+    }
+
     auto rope_scaling_type = params.rope_scaling_type;
     if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
         rope_scaling_type = hparams.rope_scaling_type_train;
@@ -2689,8 +2699,8 @@ ggml_cgraph * llama_context::graph_reserve(
 
     // verify transform coverage on the pristine graph: after scheduling,
     // cross-backend copies break the producer chain the check follows
-    if (!hadamard_verified && gf && (!model.hadamard_rotations.empty() || !model.hadamard_inverses.empty())) {
-        llama_verify_hadamard_graph(gf, model.hadamard_rotations, model.hadamard_inverses);
+    if (!hadamard_verified && gf && (!hadamard_rotations.empty() || !hadamard_inverses.empty())) {
+        llama_verify_hadamard_graph(gf, hadamard_rotations, hadamard_inverses);
         hadamard_verified = true;
     }
 
@@ -2733,8 +2743,8 @@ llm_graph_params llama_context::graph_params(
         /*.dspark_has_context =*/!dspark_ctx.v_ctx_feat.empty(),
         /*.dspark_ctx_rows =*/dspark_ctx.n_ctx_rows,
         /*.dspark_ctx_width =*/dspark_ctx.n_embd_cap,
-        /*.hadamard_rotations =*/&model.hadamard_rotations,
-        /*.hadamard_inverses  =*/&model.hadamard_inverses,
+        /*.hadamard_rotations =*/&hadamard_rotations,
+        /*.hadamard_inverses  =*/&hadamard_inverses,
         /*.samplers    =*/sampling.samplers,
         /*.n_outputs   =*/n_outputs,
         /*.cb          =*/graph_get_cb(),
diff --git a/src/llama-context.h b/src/llama-context.h
index ef0c3a50d..6057da7f4 100644
--- a/src/llama-context.h
+++ b/src/llama-context.h
@@ -394,6 +394,12 @@ private:
     llm_graph_result_ptr gf_res_prev;
     llm_graph_result_ptr gf_res_reserve;
 
+    // the Hadamard transforms this context's graphs consult: the model's own,
+    // plus the target's when the model borrows its token embeddings or output
+    // head through ctx_other (those tensors keep the target's folding)
+    llama_hadamard_rotations hadamard_rotations;
+    llama_hadamard_rotations hadamard_inverses;
+
     // one-time Hadamard transform-coverage check on the first built graph
     bool hadamard_verified = false;
 
diff --git a/src/llama-graph.cpp b/src/llama-graph.cpp
index 99f263e32..9ad7fa925 100644
--- a/src/llama-graph.cpp
+++ b/src/llama-graph.cpp
@@ -2395,6 +2395,24 @@ ggml_tensor * llm_graph_context::build_moe_ffn(
     return moe_out;
 }
 
+ggml_tensor * llm_graph_context::build_embd_rows(ggml_tensor * tok_embd, ggml_tensor * ids) const {
+    ggml_tensor * cur = ggml_get_rows(ctx0, tok_embd, ids);
+
+    // a Hadamard-latent embedding table stores rotated rows; restore the
+    // primal basis right after the lookup: h = s * (H z)
+    if (hadamard_inverses) {
+        const auto it = hadamard_inverses->find(tok_embd);
+        if (it != hadamard_inverses->end()) {
+            cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
+            if (it->second.signs) {
+                cur = ggml_mul(ctx0, cur, it->second.signs);
+            }
+        }
+    }
+
+    return cur;
+}
+
 // input embeddings with optional lora
 ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
     const int64_t n_embd_inp = hparams.n_embd_inp();
@@ -2421,19 +2439,7 @@ ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
     {
         auto & cur = inps[0];
 
-        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);
-
-        // a Hadamard-latent embedding table stores rotated rows; restore the
-        // primal basis right after the lookup: h = s * (H z)
-        if (hadamard_inverses) {
-            const auto it = hadamard_inverses->find(tok_embd);
-            if (it != hadamard_inverses->end()) {
-                cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
-                if (it->second.signs) {
-                    cur = ggml_mul(ctx0, cur, it->second.signs);
-                }
-            }
-        }
+        cur = build_embd_rows(tok_embd, inp->tokens);
 
         // apply lora for embedding tokens if needed
         for (const auto & lora : *loras) {
diff --git a/src/llama-graph.h b/src/llama-graph.h
index e85c1aa20..6c2db6420 100644
--- a/src/llama-graph.h
+++ b/src/llama-graph.h
@@ -1167,6 +1167,12 @@ struct llm_graph_context {
               ggml_tensor * ids,
               ggml_tensor * w_s = nullptr) const;
 
+    // read rows of a token-embedding table; a Hadamard-latent table gets the
+    // inverse transform so the result is in the primal basis
+    ggml_tensor * build_embd_rows(
+              ggml_tensor * tok_embd,
+              ggml_tensor * ids) const;
+
     ggml_tensor * build_norm(
              ggml_tensor * cur,
              ggml_tensor * mw,
diff --git a/src/models/dflash.cpp b/src/models/dflash.cpp
index 155e119f7..92a26c825 100644
--- a/src/models/dflash.cpp
+++ b/src/models/dflash.cpp
@@ -584,7 +584,7 @@ static void build_dfly_correction_head(llm_graph_context & g, const llama_model
         //                         z = [rms(h_i); rms(embd(prev))]
         if (prev) {
             // chained positions condition on the previously drafted token: an argmax, always in range
-            prev_embd = ggml_get_rows(ctx0, tok_embd, prev); // [n_embd, n_blocks]
+            prev_embd = g.build_embd_rows(tok_embd, prev); // [n_embd, n_blocks]
         }
 
         ggml_tensor * h_i = ggml_cont(ctx0, ggml_view_2d(ctx0, hidden, n_embd, n_blocks,
@@ -740,7 +740,7 @@ llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_gra
 
     ggml_tensor * inp_tokens = inp->tokens;
 
-    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
+    ggml_tensor * inpL = build_embd_rows(tok_embd, inp->tokens);
     cb(inpL, "inp_noise_embd", -1);
 
     // the DFly chain conditions position 1 on the block anchor's embedding; reuse the rows
@@ -981,7 +981,7 @@ llama_model_dflash::graph_dsv4::graph_dsv4(const llama_model & model, const llm_
 
     ggml_tensor * inp_tokens = inp->tokens;
 
-    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
+    ggml_tensor * inpL = build_embd_rows(tok_embd, inp->tokens);
     cb(inpL, "inp_noise_embd", -1);
     // the DSV4 hyper-connection backbone replicates inpL across hc lanes below, so
     // the log-SNR term would need to be added per lane. No such checkpoint exists
```

## 6. PR text

For `PrismML-Eng/llama.cpp`, base `prism`. The body follows the fork's
`.github/pull_request_template.md`; the submitter fills the disclosure line.

Title:

```text
dflash: apply the target's Hadamard transforms to borrowed embeddings and head
```

Body:

```markdown
## Overview

A `dflash` drafter that ships no `token_embd.weight` or `output.weight` reads the target's through `ctx_other`. When the target is Hadamard-folded (`prism.hadamard.*`), those two tensors carry the target's transforms: `output.weight` expects a rotated activation, and `token_embd.weight` stores rotated rows that need the inverse after the lookup. The draft context only consulted its own transform maps, which are empty for such a drafter, so it fed the packed head an unrotated activation and used the rotated embedding rows as if they were primal. The graph built and ran, and the draft logits were noise. DSpark drafts against Ternary-Bonsai-2-27B-PQ2_0 accepted 1-2% of drafted tokens instead of 43-57%, with no error and no warning.

Ternary-Bonsai-2-27B (PQ2_0 and PTQ1_0) folds 401 weights including `output.weight` and lists `token_embd.weight` as an inverse-lookup table. A drafter converted with `gguf-dspark-to-dflash --drop-shared-tensors` (the documented path) borrows both tensors, so every Bonsai 2 DSpark pairing takes this route. The earlier Ternary-Bonsai-27B GGUFs (PQ2_0 and Q2_g64) carry no `prism.hadamard` metadata at all (0 folded weights, no inverse tables in the published headers), so borrowing their head and embeddings never needed a transform and the gap stayed invisible until Bonsai 2.

Fix, five files, +48 / -20:

- `llama_context` keeps `hadamard_rotations` and `hadamard_inverses`: the model's own entries plus the target's when `ctx_other` is set. Entries are keyed by tensor pointer, so the two models' entries never collide. `graph_params` and the one-time `llama_verify_hadamard_graph` coverage check use these maps, so a draft graph that bypasses a transform on a borrowed tensor now fails at reserve time with the existing "consumed without its activation transform" error instead of computing wrong logits.
- `llm_graph_context::build_embd_rows(tok_embd, ids)` is `ggml_get_rows` followed by the inverse transform when the table is latent. `build_inp_embd` now calls it (same ops as before), and `dflash.cpp` uses it for the decoder input rows (DFlash/DSpark and the DSV4 backbone) and for the DFly chain's predecessor embedding.
- `build_lora_mm` already applies the forward transform to the borrowed `output.weight` once the map contains it, so the head needs no new code.
- The `dspark` arch from #179 is untouched: it copies the head's rotation into its own map at load time when `dspark_head_source` is set, and requires its own `token_embd.weight`. A drafter that carries its own tables is unaffected: its tensors are not in the target's maps.

Eagle3 and Gemma4-assistant contexts also borrow through `ctx_other`. Neither is changed by this patch: the eagle3 decoder reads `tok_embd` with a bare `ggml_get_rows` (`src/models/eagle3.cpp`), so on a Hadamard-folded target it now fails at the coverage check instead of computing wrong logits; a follow-up can route it through `build_embd_rows`. Gemma4-assistant reads `model_other->tok_embd` with a bare `ggml_get_rows`; against a folded target it will now fail the coverage check at reserve time rather than silently read rotated rows. I have no folded Gemma4 pair to test, so that path is left as is.

## Additional information

Measured on a DGX Spark (GB10, CUDA 13.0) at `1a07bfa5f`, `llama-speculative-simple`, target `Ternary-Bonsai-2-27B-PQ2_0.gguf`, greedy, 200 tokens, `--spec-type draft-dspark --spec-draft-p-min 0 -fa on -ngl 99 -ngld 999 -c 4096`, three prompts (a two-train word problem, a CSV parser with tests, binary search with tests). Acceptance is `n_accept/n_drafted`. The GPU was shared with another workload during these runs, so tokens/s is not reported.

| drafter | n-max | prompt | before | after |
|---|---:|---|---:|---:|
| Bonsai 2 DSpark drafter (5 layers, block 7, Q4_K_M, shared tensors dropped) | 5 | math | 1.6% (15/924) | 52.1% (148/284) |
| same | 5 | code | 0.4% (4/978) | 42.9% (137/319) |
| same | 5 | code2 | 1.2% (11/940) | 49.5% (143/289) |
| Ternary-Bonsai-27B drafter re-converted against the Bonsai 2 tokenizer (6 layers, block 4, Q4_0) | 4 | math | 2.0% (15/741) | 56.0% (140/250) |
| same | 4 | code | 0.9% (7/772) | 46.4% (130/280) |
| same | 4 | code2 | 1.7% (13/746) | 56.6% (142/251) |

The "after" numbers for the first drafter match, count for count, a tree that carried an ad hoc version of this change on top of `5d80cff0b` (148/284, 137/319, 143/289). Generated text is identical to the unpatched run over the common prefix on every prompt, as exact-match verification requires; the only difference is where the 200-token budget truncates a block.

Validation:

- CUDA build of `llama-speculative-simple`, `llama-server`, `llama-bench`, `llama-quantize`, `llama-gguf` at `1a07bfa5f` with and without the patch (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON`); no new warnings.
- The table above: 9 runs before, 9 runs after, plus the same 3 prompts on the reference tree.
- `llama-server` with `-md <drafter> --spec-type draft-dspark --spec-draft-n-max 5 -ngl 99 -ngld 999 -fa on -c 16384 -np 1 --jinja`: one 200-token chat request on the math prompt reports `draft_n_accepted = 153`, `draft_n = 226`. Through the chat template the model thinks first, so this is a different token stream than the raw-prompt table and the ratio is not comparable to it.
- `git diff --check` clean; ASCII only in the added lines.

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: <fill in before opening the PR>
```

## 7. End-to-end reproduction

Run from the Bonsai-demo checkout on a CUDA machine. Step 1 is the demo's own
setup; it installs release binaries that lack the fix, so step 3 builds the
runtime from source. The v2 drafter's Hugging Face location is not published
yet; replace `<HF-REPO>` when it is.

```bash
# 1. target model (PQ2_0, 7.2 GB), vision projector, and the demo layout
BONSAI_FAMILY=bonsai2 BONSAI_MODEL=27B ./setup.sh

# 2. drafter (1.1 GB); keep it as the only *dspark-dflash*.gguf in that directory (see the notes)
hf download <HF-REPO> Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --local-dir models/bonsai2-gguf/27B

# 3. llama.cpp at the cited commit, with the fix of PR #210, CUDA build for the GB10
git clone https://github.com/PrismML-Eng/llama.cpp.git
git -C llama.cpp checkout 1a07bfa5f
git -C llama.cpp apply /path/to/dflash-borrowed-hadamard.patch   # until the PR merges; then check out its merge commit instead
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cuda -j 18 --target llama-speculative-simple llama-server llama-bench
export LD_LIBRARY_PATH=$PWD/llama.cpp/build-cuda/bin

# 4. acceptance check, greedy, 200 tokens (the row "DSpark v2, n-max 5, MATH")
llama.cpp/build-cuda/bin/llama-speculative-simple \
  -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0 \
  -fa on -ngl 99 -ngld 999 -c 4096 -n 200 --temp 0 -e \
  -p '<|im_start|>user\nA train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step.<|im_end|>\n<|im_start|>assistant\n'
# expect in the log: n_drafted = 284, n_accept = 148, accept = 52.113%

# 5. server (the flags the benchmark used)
llama.cpp/build-cuda/bin/llama-server \
  -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 \
  -ngl 99 -ngld 999 -fa on -c 16384 -np 1 --jinja \
  --host 127.0.0.1 --port 8080

# 6. client; the response's "timings" object carries draft_n and draft_n_accepted
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "temperature": 0, "max_tokens": 200,
  "messages": [{"role": "user", "content": "A train leaves city A at 9:00 traveling 80 km/h toward city B, 300 km away. A second train leaves B at 9:30 traveling 100 km/h toward A. At what time and where do they meet? Show your reasoning step by step."}]
}' | python3 -c 'import sys, json; r = json.load(sys.stdin); m = r["choices"][0]["message"]; t = r["timings"]; print(m.get("reasoning_content") or ""); print(m.get("content") or ""); print("draft", t["draft_n_accepted"], "/", t["draft_n"], "accepted;", round(t["predicted_per_second"], 1), "tok/s")'
```

The Bonsai 2 chat template thinks first under `--jinja`, so the server puts the
first tokens in `message.reasoning_content` and a 200-token request ends inside
the thinking (`finish_reason: length`). Raise `max_tokens` for a full answer, or
pass `--reasoning off` to the server for a direct one. The acceptance numbers in
section 2 come from `llama-speculative-simple` with the raw prompt, which has no
thinking block.

To run through the demo launcher instead of step 5, copy the patched binaries
over the release ones and set the draft depth by hand:

```bash
cp llama.cpp/build-cuda/bin/llama-server llama.cpp/build-cuda/bin/llama-speculative-simple \
   llama.cpp/build-cuda/bin/llama-bench llama.cpp/build-cuda/bin/*.so* bin/cuda/
LD_LIBRARY_PATH=bin/cuda bin/cuda/llama-server --version      # expect commit 1a07bfa5f
BONSAI_SPECULATIVE=1 BONSAI_SPEC_NMAX=5 ./scripts/start_llama_server.sh
```

To rebuild the "v1 drafter re-converted" file for the second table block (not
re-run in this session; the commands are the ones recorded when the file was
made, and the byte comparison in the verdict confirms the result):

```bash
BONSAI_FAMILY=ternary BONSAI_MODEL=27B ./scripts/download_models.sh   # fetches Ternary-Bonsai-27B-dspark-bf16.gguf
python3 llama.cpp/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py --drop-shared-tensors \
  models/ternary-gguf/27B/Ternary-Bonsai-27B-dspark-bf16.gguf \
  models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  /tmp/Ternary-Bonsai-2-27B-dspark-conv.gguf
llama.cpp/build-cuda/bin/llama-quantize /tmp/Ternary-Bonsai-2-27B-dspark-conv.gguf \
  models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf Q4_0
```

Notes for the docs workstream (the scripts are read-only for this workstream
and were not changed):

- Cite the fix. `1a07bfa5f` alone gives 1-2% acceptance for both drafters.
  Cite `1a07bfa5f` + the PR (title above) or the PR's merge commit once it
  exists. Do not cite the release tag `prism-b10683-d8f26ee`.
- Launcher drafter pick: `scripts/start_llama_server.sh` with
  `BONSAI_SPECULATIVE=1` takes the first `*dspark-dflash*.gguf` in glob order.
  `...dspark-dflash-Q4_0.gguf` sorts before `...dspark-dflash-v2-Q4_K_M.gguf`,
  so with both files present the launcher picks the v1 re-conversion. Keep one
  drafter in the directory, or use the explicit server flags. The published
  name keeps `dspark-dflash` contiguous so the glob still matches.
- Launcher draft depth: `bonsai_dspark_block_size` in `scripts/common.sh` reads
  the key `dspark.dspark.block_size`, which a converted `dflash` file does not
  carry (it has `dflash.block_size`), so the script falls back to
  `--spec-draft-n-max 4`. Set `BONSAI_SPEC_NMAX=5` for the v2 drafter. For the
  v1 re-conversion the fallback equals its block size by coincidence.
- Build id: `--version` prints `build 10706, commit 1a07bfa5f` for clean and
  patched binaries alike. Record the patch (or the PR merge commit) with every
  number.
- Speed: every tok/s in this file was measured on a shared GPU and is invalid.
  The publishable speeds are the earlier idle-node_b numbers on the local
  patched tree, which the fixed clean build now matches count for count:
  200-token math 61.5 tok/s, code 53.9, code2 59.8, base 29.8-29.9; long-form
  mean 64.4 tok/s. They were not re-measured in this session.

## 8. Self-contained drafter (step 4)

Not applicable. The drafter checkpoint
(`models/bonsai2-dspark/bonsai2_dspark_full2_step600.safetensors`, 62 tensors)
has no `embed_tokens` and no `lm_head`; its non-layer tensors are `fc`,
`hidden_norm`, `norm`, the Markov head pair and the confidence head.
`dflash-training/convert_safetensors_to_dspark.py` maps only those plus the
layer tensors, so the raw dspark GGUF never contains `token_embd.weight` or
`output.weight`, and `gguf-dspark-to-dflash` without `--drop-shared-tensors`
copies the data section verbatim: the output is the same 62 tensors. The flag
is a no-op for this checkpoint (verified on
`bonsai2-dspark-full1ep1-dspark-raw.gguf` and its `-conv.gguf`: 62 tensors
each, no shared tensors). The v1 drafter is different: its published bf16 file
does carry `token_embd.weight` and `output.weight` (BF16), which is why the
demo doc mentions dropping them.

A self-contained Bonsai 2 drafter would need the target's `token_embd.weight`
and `output.weight` copied in and de-rotated offline (about 1.3 B parameters
each, roughly 0.7 GB per tensor at Q4_K_M, so the drafter would grow from
1.1 GB to about 2.5 GB), and `gguf-py` has no PQ2_0 dequantize path to do it.
Kept in the PQ2_0 basis, the copies would still need the runtime transforms.
The runtime fix is the smaller change, keeps the borrowed path that
`SPECULATIVE.md` documents, and also repairs the v1 re-conversion. Note that
the `dspark` architecture from #179 goes the other way and requires the drafter
to carry `token_embd.weight`; a future move of Bonsai 2 drafters to that
architecture needs an unrotated embedding source.

## 9. Environment

```text
Host:      node_a, NVIDIA DGX Spark (GB10), aarch64, 20 cores
OS:        Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia
GPU:       NVIDIA GB10, driver 580.178.04, CUDA 13.0 (nvcc V13.0.88)
Compiler:  gcc 13.3.0, cmake 3.28.3
llama.cpp: PrismML-Eng/llama.cpp prism @ 1a07bfa5f (build 10706), clean checkout
Load:      GPU shared with a generation server at 88-91% utilization during all runs
```
