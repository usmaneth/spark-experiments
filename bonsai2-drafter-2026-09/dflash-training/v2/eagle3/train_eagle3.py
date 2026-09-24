#!/usr/bin/env python3
"""
train_eagle3.py - EAGLE-3 drafter trainer for Bonsai 2 27B.

The model and the training mirror the llama.cpp runtime (src/models/eagle3.cpp
and common/speculative.cpp). See DESIGN.md in this directory for the contract.

Encoder (one row per target position P):
    x_P = concat(h_low[P], h_mid[P], h_high[P])      3 x 5120, target layer inputs
    g_P = W_fc (rmsnorm(x_P) * w_enc)                  norm_before_fc = true

Decoder (one layer). At position P the input pair is (token[P+1], g_P).
    e_n = rmsnorm(tok_embd[token]) * w_attn_norm
    g_n = rmsnorm(g) * w_attn_norm_2
    x   = concat(e_n, g_n)
    q, k, v = W_q x, W_k x, W_v x       RoPE NORM (adjacent pairs), theta 1e7
    r   = W_o attn(q, k, v) + g         raw g in the residual
    h   = r + W_down(silu(W_gate n) * W_up n),  n = rmsnorm(r) * w_ffn_norm
    logits_draft = W_out (rmsnorm(h) * w_output_norm)

Training-time test (TTT) with depth D:
    step 1: row P = (t[P+1], g_real[P]) at pos P,      target t[P+2]
    step j: row P = (t[P+j], H_{j-1}[P]) at pos P+j-1, target t[P+j+1]
A step-j query with index P sees the step-1 keys with index <= P and the
step-m keys with index == P for 2 <= m <= j.

Data: BON2 files (f32, 5 taps, "--taps" selects 3 of them) and BON3 files
(f16, 3 taps that map 1:1 to the model taps). The last "--val-samples" samples
of the index are the held-out validation set.

Check the implementation on the CPU (tiny random weights, synthetic files):
    python3 train_eagle3.py --check
Save an untrained full-size checkpoint for the GGUF writer test:
    python3 train_eagle3.py --check --save-init /tmp/eagle3_init.safetensors
Train:
    python3 train_eagle3.py --feats-dir .../v2/feats --out .../bonsai2_eagle3_smoke.safetensors \
        --epochs 1 --ttt-depth 4 --batch-size 2 --max-seq-len 1024 --lr 1e-4 --save-every 300
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train_dspark_v2 import RMSNorm, bon2_collate, read_teacher_matrix  # noqa: E402
from read_feats_e3 import BON3File  # noqa: E402

DEFAULT_V2 = "/home/REDACTED/Bonsai-demo/dflash-training/v2"
DEFAULT_TAPS = (0, 2, 4)  # BON2 tap indices of layers [6, 34, 62] in [6, 20, 34, 48, 62]
DEFAULT_TARGET_LAYERS = (6, 34, 62)

BON2_MAGIC = b"BON2"
BON3_MAGIC = b"BON3"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Eagle3Config:
    hidden_size: int = 5120
    n_enc_taps: int = 3
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 17408
    target_vocab: int = 248320
    draft_vocab: int = 32768
    rope_theta: float = 1.0e7
    rms_eps: float = 1e-6

    @property
    def enc_input(self) -> int:
        return self.n_enc_taps * self.hidden_size


# GGML tensor names of the checkpoint (the GGUF writer uses the same names).
GGML_NAMES: Dict[str, str] = {
    "fc.weight": "fc.weight",
    "enc_norm.weight": "enc.output_norm.weight",
    "attn_norm.weight": "blk.0.attn_norm.weight",
    "attn_norm_2.weight": "blk.0.attn_norm_2.weight",
    "wq.weight": "blk.0.attn_q.weight",
    "wk.weight": "blk.0.attn_k.weight",
    "wv.weight": "blk.0.attn_v.weight",
    "wo.weight": "blk.0.attn_output.weight",
    "ffn_norm.weight": "blk.0.ffn_norm.weight",
    "ffn_gate.weight": "blk.0.ffn_gate.weight",
    "ffn_up.weight": "blk.0.ffn_up.weight",
    "ffn_down.weight": "blk.0.ffn_down.weight",
    "output_norm.weight": "output_norm.weight",
    "output.weight": "output.weight",
}
GGML_TO_MODULE = {v: k for k, v in GGML_NAMES.items()}


# ---------------------------------------------------------------------------
# RoPE, LLAMA_ROPE_TYPE_NORM: the pair (2i, 2i+1) rotates by pos * theta^(-2i/d)
# ---------------------------------------------------------------------------
def rope_cos_sin(n_pos: int, head_dim: int, theta: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cos, sin of shape [n_pos, head_dim/2] in f32."""
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = theta ** (-i / head_dim)
    pos = torch.arange(n_pos, dtype=torch.float32, device=device)
    ang = pos[:, None] * inv_freq[None, :]
    return ang.cos(), ang.sin()


def apply_rope_norm(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, S, head_dim]; cos, sin: [S, head_dim/2]. Adjacent-pair rotation."""
    dtype = x.dtype
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.to(dtype)


# ---------------------------------------------------------------------------
# TTT attention mask
# ---------------------------------------------------------------------------
def build_ttt_mask(S: int, j: int, device) -> torch.Tensor:
    """Boolean mask [S, j*S], True = attend, for the step-j queries.

    Key block m (0-based) holds the step m+1 rows. Query P sees block 0 keys
    with index <= P and block m >= 1 keys with index == P.
    """
    q = torch.arange(S, device=device)[:, None]
    k = torch.arange(S, device=device)[None, :]
    causal = k <= q
    diag = k == q
    blocks = [causal] + [diag] * (j - 1)
    return torch.cat(blocks, dim=1)


def build_ttt_mask_spec(S: int, j: int) -> torch.Tensor:
    """The same mask by explicit loops. Used by --check only."""
    m = torch.zeros(S, j * S, dtype=torch.bool)
    for P in range(S):
        for blk in range(j):
            for Pk in range(S):
                if blk == 0:
                    ok = Pk <= P
                else:
                    ok = Pk == P
                m[P, blk * S + Pk] = ok
    return m


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Eagle3Drafter(nn.Module):
    """EAGLE-3 drafter. The tensor layout matches the GGUF contract."""

    def __init__(self, cfg: Eagle3Config):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        # encoder
        self.enc_norm = RMSNorm(cfg.enc_input, cfg.rms_eps)
        self.fc = nn.Linear(cfg.enc_input, H, bias=False)
        # decoder layer
        self.attn_norm = RMSNorm(H, cfg.rms_eps)
        self.attn_norm_2 = RMSNorm(H, cfg.rms_eps)
        self.wq = nn.Linear(2 * H, cfg.num_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(2 * H, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(2 * H, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.num_heads * cfg.head_dim, H, bias=False)
        self.ffn_norm = RMSNorm(H, cfg.rms_eps)
        self.ffn_gate = nn.Linear(H, cfg.intermediate_size, bias=False)
        self.ffn_up = nn.Linear(H, cfg.intermediate_size, bias=False)
        self.ffn_down = nn.Linear(cfg.intermediate_size, H, bias=False)
        # head
        self.output_norm = RMSNorm(H, cfg.rms_eps)
        self.output = nn.Linear(H, cfg.draft_vocab, bias=False)
        # frozen token embedding (own copy, not a parameter)
        self.register_buffer("tok_embd", torch.zeros(cfg.target_vocab, H, dtype=torch.float16), persistent=False)
        self.register_buffer("d2t", torch.zeros(cfg.draft_vocab, dtype=torch.int64), persistent=False)

    # -- init ---------------------------------------------------------------
    def init_weights(self, tok_embd: torch.Tensor, w_out: Optional[torch.Tensor], d2t: torch.Tensor, std: float = 0.02):
        """Init per DESIGN.md. tok_embd f16 [target_vocab, H]; w_out [draft_vocab, H] or None."""
        H = self.cfg.hidden_size
        with torch.no_grad():
            eye = torch.eye(H) / float(self.cfg.n_enc_taps)
            self.fc.weight.copy_(torch.cat([eye] * self.cfg.n_enc_taps, dim=1))
            for lin in (self.wq, self.wk, self.wv, self.wo, self.ffn_gate, self.ffn_up, self.ffn_down):
                nn.init.normal_(lin.weight, mean=0.0, std=std)
            for norm in (self.enc_norm, self.attn_norm, self.attn_norm_2, self.ffn_norm, self.output_norm):
                norm.weight.fill_(1.0)
            if w_out is None:
                nn.init.normal_(self.output.weight, mean=0.0, std=std)
            else:
                self.output.weight.copy_(w_out.to(self.output.weight.dtype))
            self.tok_embd = tok_embd.to(torch.float16).contiguous()
            self.d2t = d2t.to(torch.int64).contiguous()

    # -- pieces -------------------------------------------------------------
    def encode(self, taps: torch.Tensor) -> torch.Tensor:
        """taps [B, S, 3*H] -> g [B, S, H]."""
        return self.fc(self.enc_norm(taps))

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.tok_embd).float()

    def qkv(self, tokens: torch.Tensor, g: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """One decoder row batch. tokens [B, S], g [B, S, H], cos/sin [S, hd/2]."""
        cfg = self.cfg
        B, S = tokens.shape
        e_n = self.attn_norm(self.embed(tokens))
        g_n = self.attn_norm_2(g)
        x = torch.cat([e_n, g_n], dim=-1)
        q = self.wq(x).view(B, S, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope_norm(q, cos, sin)
        k = apply_rope_norm(k, cos, sin)
        return q, k, v

    def attend(self, q, k, v, mask: torch.Tensor) -> torch.Tensor:
        """q [B, nh, S, hd]; k, v [B, nkv, L, hd]; mask [S, L] bool. Returns [B, S, nh*hd]."""
        cfg = self.cfg
        rep = cfg.num_heads // cfg.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=cfg.head_dim ** -0.5)
        B, nh, S, hd = out.shape
        return out.transpose(1, 2).reshape(B, S, nh * hd)

    def post_attn(self, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """r = W_o a + g; h = r + ffn(norm(r)). Returns the pre-norm hidden h."""
        r = self.wo(a) + g
        n = self.ffn_norm(r)
        return r + self.ffn_down(F.silu(self.ffn_gate(n)) * self.ffn_up(n))

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return self.output(self.output_norm(h))

    # -- TTT forward --------------------------------------------------------
    def forward_ttt(self, tokens: torch.Tensor, g_real: torch.Tensor, depth: int):
        """Run the TTT chain.

        tokens [B, S] (t[0..S-1]); g_real [B, S, H] (encoder output at index P).
        Returns lists over steps j=1..depth: logits_j [B, S, draft_vocab] and
        hidden_j [B, S, H] (pre-norm), both indexed by the row index P.
        Row P of step j has input (t[P+j], H_{j-1}[P]) at RoPE pos P+j-1. Input
        tokens past the end of the sequence clamp to the last index; those rows
        never receive loss and no valid row attends to them.
        """
        B, S = tokens.shape
        device = tokens.device
        cos_all, sin_all = rope_cos_sin(S + depth, self.cfg.head_dim, self.cfg.rope_theta, device)
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        logits_steps: List[torch.Tensor] = []
        hidden_steps: List[torch.Tensor] = []
        g = g_real
        for j in range(1, depth + 1):
            idx = torch.arange(S, device=device) + j
            tok_j = tokens.gather(1, idx.clamp(max=S - 1)[None, :].expand(B, -1))
            cos = cos_all[j - 1 : j - 1 + S]
            sin = sin_all[j - 1 : j - 1 + S]
            q, k, v = self.qkv(tok_j, g, cos, sin)
            ks.append(k)
            vs.append(v)
            mask = build_ttt_mask(S, j, device)
            a = self.attend(q, torch.cat(ks, dim=2), torch.cat(vs, dim=2), mask)
            h = self.post_attn(a, g)
            logits_steps.append(self.head(h))
            hidden_steps.append(h)
            g = h
        return logits_steps, hidden_steps


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def shift_left(x: torch.Tensor, k: int) -> torch.Tensor:
    """out[:, P] = x[:, P+k]; zeros past the end."""
    if k == 0:
        return x
    pad = torch.zeros_like(x[:, :k])
    return torch.cat([x[:, k:], pad], dim=1)


def ttt_loss(
    logits_steps: List[torch.Tensor],   # [B, S, V_d] per step
    teacher_probs: torch.Tensor,        # [B, S, V_d]  probs for the token at index i+1, from final_hidden[i]
    teacher_argmax: torch.Tensor,       # [B, S] full-vocab argmax of the target at index i (token i+1)
    tokens: torch.Tensor,               # [B, S]
    loss_mask: torch.Tensor,            # [B, S] float
    t2d: torch.Tensor,                  # [target_vocab] int64
    d2t: torch.Tensor,                  # [draft_vocab] int64
    hard_alpha: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Soft CE per step (+ hard_alpha * hard CE), mean over steps. Returns (loss, stats).

    stats hold per-step sums, so a caller can pool them over batches:
      step_soft_sum[j], step_correct[j], step_den[j] (loss_mask rows), hard_sum, hard_den.
    """
    B, S = tokens.shape
    n_steps = len(logits_steps)
    stats = {"step_loss": [], "step_acc": [], "step_soft_sum": [], "step_correct": [], "step_den": [],
             "step_hard": [], "den": 0.0}
    total = 0.0
    hard_total = 0.0
    for j, logits in enumerate(logits_steps, start=1):
        # target token index i = P + j + 1; teacher row index i - 1 = P + j
        w = shift_left(loss_mask, j + 1)                      # [B, S]
        q = shift_left(teacher_probs, j)                      # [B, S, V_d]
        tgt_arg = shift_left(teacher_argmax, j)               # [B, S]
        hard_tok = shift_left(t2d[tokens], j + 1)             # [B, S] draft id or -1
        logp = F.log_softmax(logits.float(), dim=-1)
        soft = -(q * logp).sum(-1)                            # [B, S]
        den = w.sum().clamp_min(1.0)
        soft_sum = (soft * w).sum()
        step_loss = soft_sum / den
        total = total + step_loss
        hw = w * (hard_tok >= 0).float()
        hce = F.nll_loss(logp.reshape(-1, logp.shape[-1]), hard_tok.clamp(min=0).reshape(-1), reduction="none").reshape(B, S)
        step_hard = (hce * hw).sum() / hw.sum().clamp_min(1.0)   # per-step mean hard CE
        if hard_alpha > 0.0:
            hard_total = hard_total + step_hard
        with torch.no_grad():
            pred = logits.argmax(-1)                          # draft ids
            correct = ((d2t[pred] == tgt_arg).float() * w).sum()
            acc = correct / den
        stats["step_loss"].append(step_loss.item())
        stats["step_acc"].append(acc.item())
        stats["step_soft_sum"].append(soft_sum.item())
        stats["step_correct"].append(correct.item())
        stats["step_den"].append(w.sum().item())
        stats["step_hard"].append(step_hard.item())
        stats["den"] += den.item()
    loss = total / n_steps
    if hard_alpha > 0.0:
        loss = loss + hard_alpha * hard_total / n_steps
    stats["loss"] = loss.item()
    # mean over steps of the per-step hard CE: the same quantity that hard_alpha scales
    stats["hard_ce"] = sum(stats["step_hard"]) / n_steps
    return loss, stats


@torch.no_grad()
def teacher_targets(final_hidden: torch.Tensor, W_lm: torch.Tensor, W_lm_draft: torch.Tensor, chunk: int = 512):
    """Return (probs over the draft vocab [B, S, V_d] f32, full-vocab argmax [B, S])."""
    B, S, H = final_hidden.shape
    fh = final_hidden.reshape(B * S, H)
    probs = torch.softmax(F.linear(fh, W_lm_draft).float(), dim=-1)
    arg = torch.empty(B * S, dtype=torch.int64, device=fh.device)
    for i in range(0, B * S, chunk):
        arg[i : i + chunk] = F.linear(fh[i : i + chunk], W_lm).argmax(-1)
    return probs.view(B, S, -1), arg.view(B, S)


def compute_loss(model: Eagle3Drafter, W_lm, W_lm_draft, t2d, d2t, tokens, loss_mask, taps, final_hidden,
                 depth: int, hard_alpha: float, use_amp: bool):
    """One forward of the TTT objective on one batch (already on the device)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        teacher_probs, teacher_arg = teacher_targets(final_hidden.to(W_lm.dtype), W_lm, W_lm_draft)
        g_real = model.encode(taps)
        logits_steps, _ = model.forward_ttt(tokens, g_real, depth)
        return ttt_loss(logits_steps, teacher_probs, teacher_arg, tokens, loss_mask, t2d, d2t, hard_alpha)


# ---------------------------------------------------------------------------
# Data: BON2 (f32, 5 taps) and BON3 (f16, 3 taps) feature files
# ---------------------------------------------------------------------------
def bon_magic(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read(4)


class E3Dataset(Dataset):
    """Index over BON2 and BON3 files. Returns 3-tap features.

    BON2 sample: n u32 | tokens i32[n] | loss_mask u8[n] | taps f32[n*5*embd] | final_hidden f32[n*embd]
                 "taps" (BON2 tap indices) selects the 3 model taps.
    BON3 sample: n u32 | tokens i32[n] | loss_mask u8[n] | taps f16[n*3*embd] | final_hidden f16[n*embd]
                 the 3 taps map 1:1 to the model taps.
    The index order is the file order, then the sample order in the file.
    target_layers = the target layer ids of the 3 model taps; every file must agree.
    """

    def __init__(self, paths: Sequence[str], max_seq_len: int = 1024, taps: Sequence[int] = DEFAULT_TAPS):
        self.paths = list(paths)
        self.max_seq_len = max_seq_len
        self.taps = tuple(int(t) for t in taps)
        if len(self.taps) != 3:
            raise ValueError(f"--taps needs 3 indices, got {self.taps}")
        self.kind: List[str] = []           # per file: "BON2" or "BON3"
        self.file_meta: List[dict] = []     # per file: embd, n_taps, tap_layers
        self.bon3: List[Optional[BON3File]] = []
        self.index: List[Tuple[int, int, int]] = []  # (file_idx, byte_offset, n_tokens)
        self.embd: Optional[int] = None
        self.target_layers: Optional[Tuple[int, ...]] = None
        for fi, p in enumerate(self.paths):
            magic = bon_magic(p)
            if magic == BON2_MAGIC:
                meta = self._index_bon2(fi, p)
                layers = tuple(meta["tap_layers"][i] for i in self.taps)
                self.bon3.append(None)
            elif magic == BON3_MAGIC:
                b3 = BON3File(p)
                meta = {"embd": b3.embd, "n_taps": b3.n_taps, "tap_layers": list(b3.tap_layers)}
                if b3.n_taps != 3:
                    raise ValueError(f"{p}: BON3 n_taps={b3.n_taps}, expected 3")
                for off, n in b3.index:
                    self.index.append((fi, off, n))
                layers = tuple(b3.tap_layers)
                self.bon3.append(b3)
            else:
                raise ValueError(f"{p}: bad magic {magic!r}, expected BON2 or BON3")
            self.kind.append(magic.decode())
            self.file_meta.append(meta)
            if self.embd is None:
                self.embd = meta["embd"]
            elif self.embd != meta["embd"]:
                raise ValueError(f"{p}: embd {meta['embd']} != {self.embd}")
            if self.target_layers is None:
                self.target_layers = layers
            elif self.target_layers != layers:
                raise ValueError(f"{p}: target layers {layers} != {self.target_layers} of the first file")
        n2 = sum(1 for k in self.kind if k == "BON2")
        n3 = len(self.kind) - n2
        print(f"[data] indexed {len(self.index)} samples across {len(self.paths)} file(s) "
              f"(BON2: {n2}, BON3: {n3}); embd={self.embd} taps={list(self.taps)} target_layers={list(self.target_layers)}")

    def _index_bon2(self, fi: int, p: str) -> dict:
        with open(p, "rb") as f:
            f.read(4)
            embd, n_taps = struct.unpack("<II", f.read(8))
            tap_layers = list(struct.unpack(f"<{n_taps}I", f.read(4 * n_taps)))
            if max(self.taps) >= n_taps:
                raise ValueError(f"{p}: --taps {self.taps} out of range for n_taps={n_taps}")
            tap_stride = n_taps * embd
            while True:
                off = f.tell()
                b = f.read(4)
                if len(b) < 4:
                    break
                n = struct.unpack("<I", b)[0]
                f.seek(n * 4 + n + n * tap_stride * 4 + n * embd * 4, os.SEEK_CUR)
                self.index.append((fi, off, n))
        return {"embd": embd, "n_taps": n_taps, "tap_layers": tap_layers}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, off, n = self.index[i]
        embd = self.embd
        if self.kind[fi] == "BON2":
            n_taps = self.file_meta[fi]["n_taps"]
            with open(self.paths[fi], "rb") as f:
                f.seek(off + 4)
                tokens = np.frombuffer(f.read(n * 4), dtype=np.int32)
                loss_mask = np.frombuffer(f.read(n), dtype=np.uint8)
                taps = np.frombuffer(f.read(n * n_taps * embd * 4), dtype=np.float32).reshape(n, n_taps, embd)
                final_hidden = np.frombuffer(f.read(n * embd * 4), dtype=np.float32).reshape(n, embd)
            taps = taps[:, list(self.taps), :]
        else:
            with open(self.paths[fi], "rb") as f:
                tokens, loss_mask, taps, final_hidden = self.bon3[fi].read_at(f, off, n)
        L = min(n, self.max_seq_len)
        tokens = tokens[:L]
        loss_mask = loss_mask[:L]
        taps = np.ascontiguousarray(taps[:L]).reshape(L, 3 * embd)
        final_hidden = final_hidden[:L]
        return (
            torch.from_numpy(tokens.copy()).long(),
            torch.from_numpy(loss_mask.copy()).float(),
            torch.from_numpy(taps.copy()).float(),
            torch.from_numpy(final_hidden.copy()).float(),
        )


def resolve_feature_files(feats_dir: str, feats: Optional[Sequence[str]]) -> List[str]:
    files: List[str] = []
    if feats_dir:
        files += sorted(glob.glob(os.path.join(feats_dir, "*.bin")))
    if feats:
        files += [f for f in feats if f not in files]
    return files


def split_train_val(dataset: Dataset, val_samples: int) -> Tuple[Subset, Subset]:
    """The last val_samples samples of the index are the validation set."""
    n = len(dataset)
    n_val = min(max(val_samples, 0), n - 1) if n > 1 else 0
    train_idx = list(range(0, n - n_val))
    val_idx = list(range(n - n_val, n))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def fix_owner(path: str):
    """Docker runs as root. Give the file the owner and the mode of its directory."""
    try:
        st = os.stat(os.path.dirname(os.path.abspath(path)))
        os.chown(path, st.st_uid, st.st_gid)
        os.chmod(path, 0o644)
    except Exception as e:  # noqa: BLE001
        print(f"[save] owner fix skipped: {e}")


def save_checkpoint(model: Eagle3Drafter, path: str, with_embd: bool = True, meta: Optional[dict] = None):
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = {}
    for k, v in model.state_dict().items():
        if k not in GGML_NAMES:
            continue
        sd[GGML_NAMES[k]] = v.detach().to(torch.bfloat16).contiguous().cpu()
    sd["d2t"] = model.d2t.detach().to(torch.int64).contiguous().cpu()
    if with_embd:
        sd["token_embd.weight"] = model.tok_embd.detach().to(torch.float16).contiguous().cpu()
    metadata = {str(k): str(v) for k, v in (meta or {}).items()}
    save_file(sd, path, metadata=metadata or None)
    fix_owner(path)
    print(f"[save] wrote {len(sd)} tensors -> {path}", flush=True)


def read_checkpoint_meta(path: str) -> dict:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        return dict(f.metadata() or {})


def load_checkpoint(model: Eagle3Drafter, path: str) -> dict:
    from safetensors import safe_open

    print(f"[warm-start] loading {path}")
    sd = {}
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            if k in GGML_TO_MODULE:
                sd[GGML_TO_MODULE[k]] = f.get_tensor(k).float()
            elif k == "d2t":
                model.d2t = f.get_tensor(k).to(torch.int64)
            elif k == "token_embd.weight":
                model.tok_embd = f.get_tensor(k).to(torch.float16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[warm-start] loaded {len(sd)} tensors. missing={len(missing)} unexpected={len(unexpected)} meta={meta}")
    return meta


# ---------------------------------------------------------------------------
# EMA of the trainable parameters
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average with a warmup: decay_t = min(decay, (1+t)/(10+t))."""

    def __init__(self, params: Sequence[torch.Tensor], decay: float):
        self.decay = decay
        self.n = 0
        self.shadow = [p.detach().clone() for p in params]
        self.backup: Optional[List[torch.Tensor]] = None

    @torch.no_grad()
    def update(self, params: Sequence[torch.Tensor]):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for s, p in zip(self.shadow, params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, params: Sequence[torch.Tensor]):
        self.backup = [p.detach().clone() for p in params]
        for s, p in zip(self.shadow, params):
            p.copy_(s)

    @torch.no_grad()
    def swap_out(self, params: Sequence[torch.Tensor]):
        assert self.backup is not None
        for b, p in zip(self.backup, params):
            p.copy_(b)
        self.backup = None


# ---------------------------------------------------------------------------
# Init from the teacher files
# ---------------------------------------------------------------------------
def load_draft_vocab(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    z = np.load(path)
    return torch.from_numpy(z["d2t"].astype(np.int64)), torch.from_numpy(z["t2d"].astype(np.int64))


def build_full_model(cfg: Eagle3Config, teacher_dir: str, vocab_path: str, warm_start: Optional[str] = None):
    """Build the full-size model, init from the teacher files. Returns (model, W_lm f16, t2d)."""
    d2t, t2d = load_draft_vocab(vocab_path)
    assert d2t.numel() == cfg.draft_vocab, (d2t.numel(), cfg.draft_vocab)
    print("[init] reading teacher matrices (2 x 2.5 GB f16) ...", flush=True)
    W_lm = read_teacher_matrix(os.path.join(teacher_dir, "W_lm.bin"))
    tok_embd = read_teacher_matrix(os.path.join(teacher_dir, "tok_embd.bin"))
    assert W_lm.shape == (cfg.target_vocab, cfg.hidden_size), W_lm.shape
    assert tok_embd.shape == (cfg.target_vocab, cfg.hidden_size), tok_embd.shape
    model = Eagle3Drafter(cfg)
    model.init_weights(tok_embd, W_lm[d2t].float(), d2t)
    if warm_start:
        load_checkpoint(model, warm_start)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] trainable params = {n_train:,}; token_embd frozen {tuple(tok_embd.shape)}", flush=True)
    return model, W_lm, t2d


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, lr: float, total_steps: int, warmup: int, eta_min: float = 1e-6):
    floor = eta_min / lr

    def f(step):
        if step < warmup:
            return (step + 1) / float(warmup)
        prog = (step - warmup) / float(max(1, total_steps - warmup))
        prog = min(prog, 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def fmt_list(xs: List[float], nd: int = 3) -> str:
    return "/".join(f"{x:.{nd}f}" for x in xs)


def parse_taps(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


@torch.no_grad()
def evaluate(model: Eagle3Drafter, loader: DataLoader, W_lm, W_lm_draft, t2d, d2t, depth: int,
             device, use_amp: bool, ema: Optional[EMA] = None, params: Optional[Sequence[torch.Tensor]] = None) -> dict:
    """Held-out eval with the training forward. Pools per-depth sums over the batches."""
    was_training = model.training
    model.eval()
    if ema is not None:
        ema.swap_in(params)
    soft = [0.0] * depth
    correct = [0.0] * depth
    den = [0.0] * depth
    n_batches = 0
    for tokens, loss_mask, taps, final_hidden in loader:
        tokens = tokens.to(device)
        loss_mask = loss_mask.to(device)
        taps = taps.to(device)
        final_hidden = final_hidden.to(device)
        _, st = compute_loss(model, W_lm, W_lm_draft, t2d, d2t, tokens, loss_mask, taps, final_hidden,
                             depth, 0.0, use_amp)
        for j in range(depth):
            soft[j] += st["step_soft_sum"][j]
            correct[j] += st["step_correct"][j]
            den[j] += st["step_den"][j]
        n_batches += 1
    if ema is not None:
        ema.swap_out(params)
    if was_training:
        model.train()
    return {
        "acc": [correct[j] / max(den[j], 1.0) for j in range(depth)],
        "ce": [soft[j] / max(den[j], 1.0) for j in range(depth)],
        "n_tok": int(den[0]),
        "n_batches": n_batches,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    print(f"[train] device={device} bf16-autocast={use_amp} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    files = resolve_feature_files(args.feats_dir, args.feats)
    if not files:
        print(f"[train] ERROR: no feature files (--feats-dir {args.feats_dir}, --feats {args.feats})", file=sys.stderr)
        sys.exit(2)
    for p in (os.path.join(args.teacher_dir, "W_lm.bin"), os.path.join(args.teacher_dir, "tok_embd.bin"), args.draft_vocab):
        if not os.path.exists(p):
            print(f"[train] ERROR: missing {p}", file=sys.stderr)
            sys.exit(2)

    cfg = Eagle3Config()
    model, W_lm, t2d = build_full_model(cfg, args.teacher_dir, args.draft_vocab, args.warm_start)
    model = model.to(device)
    W_lm = W_lm.to(device=device, dtype=torch.bfloat16 if use_amp else torch.float32)
    W_lm_draft = W_lm[model.d2t.to(device)].contiguous()
    t2d = t2d.to(device)
    d2t = model.d2t.to(device)

    taps = parse_taps(args.taps)
    dataset = E3Dataset(files, max_seq_len=args.max_seq_len, taps=taps)
    assert dataset.embd == cfg.hidden_size, dataset.embd
    target_layers = list(dataset.target_layers)
    train_set, val_set = split_train_val(dataset, args.val_samples)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=bon2_collate,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0, generator=gen)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=bon2_collate,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0) if len(val_set) else None
    print(f"[data] train {len(train_set)} samples, val {len(val_set)} samples (last {len(val_set)} of the index)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    accum = max(1, args.accum)
    n_micro = len(loader)
    steps_per_epoch = (n_micro + accum - 1) // accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, args.lr, total_steps, args.warmup)
    ema = EMA(params, args.ema) if args.ema > 0.0 else None
    meta_base = {"target_layers": ",".join(str(x) for x in target_layers), "taps": ",".join(str(x) for x in taps),
                 "ttt_depth": args.ttt_depth, "max_seq_len": args.max_seq_len, "lr": args.lr,
                 "batch_size": args.batch_size, "accum": accum, "hard_alpha": args.hard_alpha,
                 "val_samples": len(val_set), "ema": args.ema}
    print(f"[train] {total_steps} steps, {args.epochs} epochs, {len(train_set)} samples, "
          f"ttt_depth={args.ttt_depth} batch={args.batch_size} accum={accum} max_seq_len={args.max_seq_len} "
          f"lr={args.lr} warmup={args.warmup} ema={args.ema} hard_alpha={args.hard_alpha} "
          f"taps={list(taps)} target_layers={target_layers}", flush=True)

    def run_eval(step: int, tag_ema: bool = True):
        if val_loader is None:
            return
        t_e = time.time()
        r = evaluate(model, val_loader, W_lm, W_lm_draft, t2d, d2t, args.ttt_depth, device, use_amp,
                     ema if tag_ema else None, params)
        print(f"val step {step}/{total_steps} acc {fmt_list(r['acc'], 3)} ce {fmt_list(r['ce'], 3)} "
              f"n_tok {r['n_tok']}{' (ema)' if (ema is not None and tag_ema) else ''} {time.time() - t_e:.0f}s", flush=True)

    def do_save(path: str, step: int, epoch: int):
        meta = dict(meta_base, step=step, epoch=epoch)
        save_checkpoint(model, path, with_embd=bool(args.save_embd), meta=meta)
        if ema is not None:
            ema.swap_in(params)
            save_checkpoint(model, path.replace(".safetensors", "_ema.safetensors"), with_embd=bool(args.save_embd),
                            meta=dict(meta, ema_weights=1))
            ema.swap_out(params)

    model.train()
    step = 0
    last_eval_step = -1
    t0 = time.time()
    t_last = t0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        acc_stats = {"loss": 0.0, "hard": 0.0, "step_loss": [0.0] * args.ttt_depth, "step_acc": [0.0] * args.ttt_depth, "n": 0}
        optimizer.zero_grad(set_to_none=True)
        for mi, (tokens, loss_mask, taps_b, final_hidden) in enumerate(loader):
            group = accum if (n_micro - (mi // accum) * accum) >= accum else n_micro - (mi // accum) * accum
            tokens = tokens.to(device)
            loss_mask = loss_mask.to(device)
            taps_b = taps_b.to(device)
            final_hidden = final_hidden.to(device)
            loss, stats = compute_loss(model, W_lm, W_lm_draft, t2d, d2t, tokens, loss_mask, taps_b, final_hidden,
                                       args.ttt_depth, args.hard_alpha, use_amp)
            (loss / group).backward()
            acc_stats["loss"] += stats["loss"] / group
            acc_stats["hard"] += stats["hard_ce"] / group
            for j in range(args.ttt_depth):
                acc_stats["step_loss"][j] += stats["step_loss"][j] / group
                acc_stats["step_acc"][j] += stats["step_acc"][j] / group
            acc_stats["n"] += 1
            if acc_stats["n"] < group:
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(params)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                now = time.time()
                el = now - t0
                sps = (now - t_last) / args.log_every
                t_last = now
                vram = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
                print(f"ep {epoch+1}/{args.epochs} step {step}/{total_steps} "
                      f"loss {acc_stats['loss']:.4f} (hard {acc_stats['hard']:.4f} "
                      f"steps {fmt_list(acc_stats['step_loss'], 3)}) "
                      f"acc {fmt_list(acc_stats['step_acc'], 3)} "
                      f"gnorm {float(gnorm):.2f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"vram {vram:.1f}G {sps:.2f}s/step {el/60:.1f}m", flush=True)
            acc_stats = {"loss": 0.0, "hard": 0.0, "step_loss": [0.0] * args.ttt_depth, "step_acc": [0.0] * args.ttt_depth, "n": 0}
            if args.max_steps and step >= args.max_steps:
                print(f"[train] max_steps {args.max_steps} reached; stopping early", flush=True)
                stop = True
                break
            save_now = bool(args.save_every) and step % args.save_every == 0 and step < total_steps
            eval_now = bool(args.eval_every) and step % args.eval_every == 0
            if (save_now or eval_now) and last_eval_step != step:
                run_eval(step)
                last_eval_step = step
            if save_now:
                do_save(args.out.replace(".safetensors", f"_step{step}.safetensors"), step, epoch + 1)
        if last_eval_step != step:
            run_eval(step)
            last_eval_step = step
        do_save(args.out, step, epoch + 1)
        do_save(args.out.replace(".safetensors", f"_epoch{epoch+1}.safetensors"), step, epoch + 1)
    print("[train] done.", flush=True)


# ---------------------------------------------------------------------------
# VRAM estimate (dry calculation, no GPU)
# ---------------------------------------------------------------------------
def vram_estimate(cfg: Eagle3Config, depth: int, seq: int, batch: int, ema: bool = False,
                  act_per_row_mb: float = 0.80, math_sdpa: bool = False) -> dict:
    """Rough peak-memory estimate in GiB for the training step.

    act_per_row_mb: saved activations of the decoder + head per row per depth
    (calibrated on the smoke run: 20.8 GiB peak at D=4, S~1024, B=2).
    """
    n_train = 597_724_160 if cfg.hidden_size == 5120 else sum(1 for _ in [])
    gib = 1024 ** 3
    static = n_train * 4 * 4 / gib                       # params, grads, adam m, adam v (f32)
    static += cfg.target_vocab * cfg.hidden_size * 2 * 2 / gib   # token_embd f16 + W_lm bf16
    static += cfg.draft_vocab * cfg.hidden_size * 2 / gib        # W_lm_draft bf16
    if ema:
        static += n_train * 4 / gib
    rows = batch * seq
    act = rows * depth * act_per_row_mb * 1024 ** 2 / gib
    # repeated K and V per step j: [B, nh, j*S, hd] bf16, both saved for backward
    kv_rep = sum(batch * cfg.num_heads * j * seq * cfg.head_dim * 2 * 2 for j in range(1, depth + 1)) / gib
    teacher = rows * cfg.draft_vocab * 4 / gib           # teacher probs f32 [B, S, V_d]
    sdpa = 0.0
    if math_sdpa:
        sdpa = sum(batch * cfg.num_heads * seq * j * seq * 4 for j in range(1, depth + 1)) / gib
    total = static + act + kv_rep + teacher + sdpa
    return {"static": static, "act": act, "kv_rep": kv_rep, "teacher": teacher, "sdpa_math": sdpa, "total": total}


def print_vram_table(ema: bool = False):
    cfg = Eagle3Config()
    print(f"[vram] estimate, batch 2, bf16 autocast, f32 params (GiB){' with EMA' if ema else ''}; "
          f"'+math' = extra if SDPA falls back to the math kernel")
    print(f"[vram] {'depth':>5} {'seq':>5} {'static':>7} {'act':>6} {'kv_rep':>7} {'teacher':>8} {'total':>6} {'+math':>7}")
    for depth in (4, 5, 7):
        for seq in (1024, 2048):
            r = vram_estimate(cfg, depth, seq, 2, ema=ema)
            m = vram_estimate(cfg, depth, seq, 2, ema=ema, math_sdpa=True)
            print(f"[vram] {depth:>5} {seq:>5} {r['static']:>7.1f} {r['act']:>6.1f} {r['kv_rep']:>7.2f} "
                  f"{r['teacher']:>8.2f} {r['total']:>6.1f} {m['sdpa_math']:>7.1f}")


# ---------------------------------------------------------------------------
# --check: tiny CPU checks against the contract
# ---------------------------------------------------------------------------
def naive_decoder_d1(model: Eagle3Drafter, tokens: torch.Tensor, g: torch.Tensor):
    """Step-1 decoder for one sequence by explicit loops. tokens [S], g [S, H]."""
    cfg = model.cfg
    S = tokens.shape[0]
    nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    rep = nh // nkv

    def rms(x, w):
        return x / torch.sqrt((x * x).mean() + cfg.rms_eps) * w

    def rope(vec, pos):
        out = vec.clone()
        for i in range(hd // 2):
            ang = pos * (cfg.rope_theta ** (-(2 * i) / hd))
            c, s = math.cos(ang), math.sin(ang)
            a, b = vec[2 * i].item(), vec[2 * i + 1].item()
            out[2 * i] = a * c - b * s
            out[2 * i + 1] = a * s + b * c
        return out

    ks, vs = [], []
    hs, logits = [], []
    for P in range(S):
        # row P takes the pair (t[P+1], g_P); the last row clamps the token index
        e = model.tok_embd[tokens[min(P + 1, S - 1)]].float()
        e_n = rms(e, model.attn_norm.weight)
        g_n = rms(g[P], model.attn_norm_2.weight)
        x = torch.cat([e_n, g_n])
        q = (model.wq.weight @ x).view(nh, hd)
        k = (model.wk.weight @ x).view(nkv, hd)
        v = (model.wv.weight @ x).view(nkv, hd)
        q = torch.stack([rope(q[h], P) for h in range(nh)])
        k = torch.stack([rope(k[h], P) for h in range(nkv)])
        ks.append(k)
        vs.append(v)
        heads = []
        for h in range(nh):
            kh = h // rep
            scores = torch.tensor([float(q[h] @ ks[Pk][kh]) / math.sqrt(hd) for Pk in range(P + 1)])
            p = torch.softmax(scores, dim=0)
            heads.append(sum(p[Pk] * vs[Pk][kh] for Pk in range(P + 1)))
        a = torch.cat(heads)
        r = model.wo.weight @ a + g[P]
        n = rms(r, model.ffn_norm.weight)
        h_out = r + model.ffn_down.weight @ (F.silu(model.ffn_gate.weight @ n) * (model.ffn_up.weight @ n))
        hs.append(h_out)
        logits.append(model.output.weight @ rms(h_out, model.output_norm.weight))
    return torch.stack(logits), torch.stack(hs)


def rotate_half_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NEOX-style rotation for the negative check."""
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    c = torch.cat([cos, cos], dim=-1)[None, None]
    s = torch.cat([sin, sin], dim=-1)[None, None]
    rot = torch.cat([-x2, x1], dim=-1)
    return xf * c + rot * s


def write_synthetic_bon(path: str, magic: bytes, embd: int, n_taps: int, tap_layers: Sequence[int],
                        samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    """Write a BON2 (f32) or BON3 (f16) file. samples: (tokens i32[n], mask u8[n], taps [n, n_taps, embd], fh [n, embd])."""
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<II", embd, n_taps))
        f.write(struct.pack(f"<{n_taps}I", *tap_layers))
        for tokens, mask, taps, fh in samples:
            n = len(tokens)
            f.write(struct.pack("<I", n))
            f.write(np.asarray(tokens, dtype=np.int32).tobytes())
            f.write(np.asarray(mask, dtype=np.uint8).tobytes())
            dt = np.float32 if magic == BON2_MAGIC else np.float16
            f.write(np.ascontiguousarray(taps, dtype=dt).tobytes())
            f.write(np.ascontiguousarray(fh, dtype=dt).tobytes())


def run_check(args) -> bool:
    import tempfile

    torch.manual_seed(0)
    cfg = Eagle3Config(hidden_size=64, n_enc_taps=3, num_heads=4, num_kv_heads=2, head_dim=16,
                       intermediate_size=128, target_vocab=100, draft_vocab=40)
    S, D, B = 8, 3, 2
    model = Eagle3Drafter(cfg)
    d2t = torch.arange(0, 2 * cfg.draft_vocab, 2, dtype=torch.int64)  # even target ids
    t2d = torch.full((cfg.target_vocab,), -1, dtype=torch.int64)
    t2d[d2t] = torch.arange(cfg.draft_vocab)
    model.init_weights(torch.randn(cfg.target_vocab, cfg.hidden_size).half(), torch.randn(cfg.draft_vocab, cfg.hidden_size) * 0.02, d2t, std=0.2)
    model.eval()
    ok = True

    def report(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[check] {'PASS' if cond else 'FAIL'} {name} {detail}")

    # 1. shapes
    tokens = torch.randint(0, cfg.target_vocab, (B, S))
    taps = torch.randn(B, S, cfg.enc_input)
    with torch.no_grad():
        g = model.encode(taps)
        logits_steps, hidden_steps = model.forward_ttt(tokens, g, D)
    report("encoder shape", tuple(g.shape) == (B, S, cfg.hidden_size), f"{tuple(g.shape)}")
    report("n steps", len(logits_steps) == D and len(hidden_steps) == D, f"{len(logits_steps)}")
    for j in range(D):
        report(f"step {j+1} logits shape", tuple(logits_steps[j].shape) == (B, S, cfg.draft_vocab), f"{tuple(logits_steps[j].shape)}")
        report(f"step {j+1} hidden shape", tuple(hidden_steps[j].shape) == (B, S, cfg.hidden_size), f"{tuple(hidden_steps[j].shape)}")

    # 2. mask equals the spec up to depth 7
    for j in range(1, 8):
        m = build_ttt_mask(S, j, torch.device("cpu"))
        spec = build_ttt_mask_spec(S, j)
        report(f"step {j} mask == spec", tuple(m.shape) == (S, j * S) and torch.equal(m, spec), f"shape {tuple(m.shape)} attend={int(m.sum())}")
    with torch.no_grad():
        l7, h7 = model.forward_ttt(tokens, g, 7)
    report("depth 7 forward", len(l7) == 7 and all(torch.isfinite(x).all() for x in l7), f"{len(l7)} steps, finite")

    # 3. RoPE is adjacent-pair
    x = torch.randn(1, 1, S, cfg.head_dim)
    cos, sin = rope_cos_sin(S, cfg.head_dim, cfg.rope_theta, torch.device("cpu"))
    y = apply_rope_norm(x, cos, sin)
    y_ref = torch.zeros_like(x)
    for P in range(S):
        for i in range(cfg.head_dim // 2):
            ang = P * (cfg.rope_theta ** (-(2 * i) / cfg.head_dim))
            a, b = x[0, 0, P, 2 * i], x[0, 0, P, 2 * i + 1]
            y_ref[0, 0, P, 2 * i] = a * math.cos(ang) - b * math.sin(ang)
            y_ref[0, 0, P, 2 * i + 1] = a * math.sin(ang) + b * math.cos(ang)
    err = (y - y_ref).abs().max().item()
    report("rope adjacent-pair == loop", err < 1e-5, f"max err {err:.2e}")
    y_neox = rotate_half_rope(x, cos, sin)
    report("rope differs from rotate-half", (y - y_neox).abs().max().item() > 1e-3, f"max diff {(y - y_neox).abs().max().item():.3f}")
    report("rope pos 0 is identity", (y[0, 0, 0] - x[0, 0, 0]).abs().max().item() < 1e-6)

    # 4. D=1 decoder equals the naive loop for one sequence
    with torch.no_grad():
        l1, h1 = model.forward_ttt(tokens[:1], g[:1], 1)
        l_ref, h_ref = naive_decoder_d1(model, tokens[0], g[0])
    e_l = (l1[0][0] - l_ref).abs().max().item()
    e_h = (h1[0][0] - h_ref).abs().max().item()
    report("D=1 logits == naive loop", e_l < 1e-4, f"max err {e_l:.2e}")
    report("D=1 hidden == naive loop", e_h < 1e-4, f"max err {e_h:.2e}")

    # 5. causality: a change at index P0 leaves step-1 rows < P0 unchanged, changes row P0
    P0 = 4
    g2 = g.clone()
    g2[:, P0] += 1.0
    with torch.no_grad():
        l2, _ = model.forward_ttt(tokens, g2, D)
    before = (l2[0][:, :P0] - logits_steps[0][:, :P0]).abs().max().item()
    at = (l2[0][:, P0] - logits_steps[0][:, P0]).abs().max().item()
    report("step-1 causal", before < 1e-5 and at > 1e-5, f"rows<P0 diff {before:.1e}, row P0 diff {at:.2e}")
    s2_before = (l2[1][:, :P0] - logits_steps[1][:, :P0]).abs().max().item()
    report("step-2 rows < P0 unchanged", s2_before < 1e-5, f"diff {s2_before:.1e}")

    # 6. loss runs and the masks line up
    loss_mask = torch.ones(B, S)
    fh = torch.randn(B, S, cfg.hidden_size)
    W_lm = torch.randn(cfg.target_vocab, cfg.hidden_size)
    with torch.no_grad():
        probs, arg = teacher_targets(fh, W_lm, W_lm[d2t])
    loss, stats = ttt_loss(logits_steps, probs, arg, tokens, loss_mask, t2d, d2t)
    dens = [S - 2 - j + 1 for j in range(1, D + 1)]  # rows with P + j + 1 <= S - 1
    report("loss finite", torch.isfinite(loss).item(), f"loss {loss.item():.4f} steps {fmt_list(stats['step_loss'])} acc {fmt_list(stats['step_acc'])}")
    report("loss rows per step", abs(stats["den"] - B * sum(dens)) < 1e-6, f"den {stats['den']:.0f} expected {B * sum(dens)}")
    loss_h, stats_h = ttt_loss(logits_steps, probs, arg, tokens, loss_mask, t2d, d2t, hard_alpha=0.5)
    report("hard-alpha adds hard CE", abs(loss_h.item() - (loss.item() + 0.5 * stats_h["hard_ce"])) < 1e-4,
           f"soft {loss.item():.4f} hard {stats_h['hard_ce']:.4f} total {loss_h.item():.4f}")

    # 7. checkpoint round trip with metadata
    tmp = tempfile.mkdtemp(prefix="eagle3_check_")
    ck = os.path.join(tmp, "ckpt.safetensors")
    save_checkpoint(model, ck, meta={"target_layers": "20,48,62", "taps": "1,3,4", "step": 7})
    from safetensors import safe_open
    with safe_open(ck, framework="pt") as f:
        keys = set(f.keys())
    expect = set(GGML_NAMES.values()) | {"d2t", "token_embd.weight"}
    report("checkpoint keys", keys == expect, f"{len(keys)} keys")
    meta = read_checkpoint_meta(ck)
    report("checkpoint metadata", meta.get("target_layers") == "20,48,62" and meta.get("taps") == "1,3,4" and meta.get("step") == "7", f"{meta}")

    # 8. BON2 + BON3 parsing, taps selection, val split
    rng = np.random.default_rng(1)
    embd, n2taps, layers2 = cfg.hidden_size, 5, (6, 20, 34, 48, 62)
    samples2 = []
    for n in (6, 8, 8, 7, 8, 8):
        tk = rng.integers(0, cfg.target_vocab, size=n, dtype=np.int32)
        mk = np.ones(n, dtype=np.uint8)
        mk[:2] = 0
        tp = (rng.standard_normal((n, n2taps, embd)) * 0.5).astype(np.float16).astype(np.float32)  # exact in f16
        fh_ = (rng.standard_normal((n, embd)) * 0.5).astype(np.float16).astype(np.float32)
        samples2.append((tk, mk, tp, fh_))
    p2 = os.path.join(tmp, "a_bon2.bin")
    p3 = os.path.join(tmp, "b_bon3.bin")
    write_synthetic_bon(p2, BON2_MAGIC, embd, n2taps, layers2, samples2)
    samples3 = [(tk, mk, tp[:, [0, 2, 4], :], fh_) for tk, mk, tp, fh_ in samples2]
    write_synthetic_bon(p3, BON3_MAGIC, embd, 3, (6, 34, 62), samples3)
    ds = E3Dataset([p2, p3], max_seq_len=16, taps=(0, 2, 4))
    report("BON2+BON3 index", len(ds) == 12 and ds.kind == ["BON2", "BON3"] and ds.target_layers == (6, 34, 62), f"{len(ds)} samples kinds={ds.kind}")
    same = True
    for i in range(6):
        a = ds[i]
        b = ds[6 + i]
        same = same and all(torch.equal(x, y) for x, y in zip(a, b))
    report("BON3 sample == BON2 sample (taps 0,2,4)", same)
    a0 = ds[0]
    report("sample tensors", tuple(a0[2].shape) == (6, 3 * embd) and tuple(a0[3].shape) == (6, embd) and a0[0].dtype == torch.long, f"taps {tuple(a0[2].shape)} fh {tuple(a0[3].shape)}")
    ds_t = E3Dataset([p2], max_seq_len=16, taps=(1, 3, 4))
    tp_ref = torch.from_numpy(samples2[0][2][:, [1, 3, 4], :].reshape(6, -1).copy())
    report("--taps 1,3,4 selection", torch.equal(ds_t[0][2], tp_ref) and ds_t.target_layers == (20, 48, 62), f"target_layers={ds_t.target_layers}")
    ds_b3 = E3Dataset([p3], max_seq_len=16, taps=(1, 3, 4))
    report("BON3 ignores --taps", torch.equal(ds_b3[0][2], ds[6][2]) and ds_b3.target_layers == (6, 34, 62))
    try:
        E3Dataset([p2, p3], max_seq_len=16, taps=(1, 3, 4))
        mixed_err = False
    except ValueError:
        mixed_err = True
    report("mixed BON2(--taps 1,3,4)+BON3 rejected", mixed_err)
    ds_trunc = E3Dataset([p2], max_seq_len=5, taps=(0, 2, 4))
    report("max_seq_len truncation", ds_trunc[1][0].shape[0] == 5 and ds_trunc[1][2].shape[0] == 5)
    tr, va = split_train_val(ds, 4)
    tr_idx, va_idx = set(tr.indices), set(va.indices)
    report("val split disjoint + last N", len(va) == 4 and va_idx == {8, 9, 10, 11} and not (tr_idx & va_idx) and (tr_idx | va_idx) == set(range(12)),
           f"train {sorted(tr_idx)} val {sorted(va_idx)}")
    tr2, va2 = split_train_val(ds, 4)
    report("val split deterministic", tr2.indices == tr.indices and va2.indices == va.indices)

    # 9. gradient accumulation == bigger batch (D=2, equal-length samples, no padding)
    model.train()
    ds_eq = E3Dataset([p3], max_seq_len=16, taps=(0, 2, 4))
    s_a, s_b = ds_eq[1], ds_eq[2]   # both n=8
    W_lm_t = torch.randn(cfg.target_vocab, cfg.hidden_size)
    W_lm_d = W_lm_t[d2t].contiguous()
    params = [p for p in model.parameters() if p.requires_grad]

    def grads_for(batches, group):
        for p in params:
            p.grad = None
        for (tk, mk, tp, fh_) in batches:
            loss_b, _ = compute_loss(model, W_lm_t, W_lm_d, t2d, d2t, tk, mk, tp, fh_, 2, 0.0, False)
            (loss_b / group).backward()
        return [p.grad.detach().clone() for p in params]

    big = bon2_collate([s_a, s_b])
    g_big = grads_for([big], 1)
    g_acc = grads_for([bon2_collate([s_a]), bon2_collate([s_b])], 2)
    max_rel = 0.0
    for ga, gb in zip(g_big, g_acc):
        max_rel = max(max_rel, ((ga - gb).abs().max() / (ga.abs().max() + 1e-12)).item())
    report("accum G=2 == batch 2 gradient", max_rel < 1e-4, f"max rel diff {max_rel:.2e}")
    for p in params:
        p.grad = None

    # 10. EMA swap round trip
    ema = EMA(params, 0.999)
    before_p = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p in params:
            p.add_(1.0)
    ema.update(params)
    ema.swap_in(params)
    ema_vals = [p.detach().clone() for p in params]
    ema.swap_out(params)
    d_ema = 1.0 - min(0.999, 2.0 / 11.0)
    exp0 = before_p[0] * (1 - d_ema) + (before_p[0] + 1.0) * d_ema
    report("EMA update + swap", all(torch.equal(p, b + 1.0) for p, b in zip(params, before_p)) and (ema_vals[0] - exp0).abs().max().item() < 1e-5,
           f"decay_1={d_ema:.3f}")

    # 11. evaluate() pools the val set
    model.eval()
    va_loader = DataLoader(Subset(ds_eq, [0, 1, 2, 3]), batch_size=4, shuffle=False, collate_fn=bon2_collate)
    r = evaluate(model, va_loader, W_lm_t, W_lm_d, t2d, d2t, 2, torch.device("cpu"), False)
    report("evaluate() runs", len(r["acc"]) == 2 and len(r["ce"]) == 2 and r["n_tok"] > 0 and all(math.isfinite(c) for c in r["ce"]),
           f"acc {fmt_list(r['acc'])} ce {fmt_list(r['ce'])} n_tok {r['n_tok']}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[check] RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


BON2_TAP_LAYERS = (6, 20, 34, 48, 62)


def save_init(args):
    cfg = Eagle3Config()
    taps = parse_taps(args.taps)
    layers = [BON2_TAP_LAYERS[i] for i in taps]
    model, _, _ = build_full_model(cfg, args.teacher_dir, args.draft_vocab)
    save_checkpoint(model, args.save_init, meta={"target_layers": ",".join(str(x) for x in layers),
                                                  "taps": args.taps, "step": 0})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EAGLE-3 drafter trainer (Bonsai 2 27B)")
    ap.add_argument("--check", action="store_true", help="tiny CPU checks of the contract")
    ap.add_argument("--save-init", default="", help="save an untrained full-size checkpoint to this path")
    ap.add_argument("--vram-table", action="store_true", help="print the VRAM estimate table and exit")
    ap.add_argument("--feats-dir", default=os.path.join(DEFAULT_V2, "feats"), help="directory with BON2 and/or BON3 *.bin files")
    ap.add_argument("--feats", nargs="*", default=None, help="explicit feature files (added to --feats-dir)")
    ap.add_argument("--taps", default="0,2,4", help="BON2 tap indices for the 3 model taps (1,3,4 = layers 20/48/62)")
    ap.add_argument("--teacher-dir", default=os.path.join(DEFAULT_V2, "teacher"))
    ap.add_argument("--draft-vocab", default=os.path.join(HERE, "draft_vocab.npz"))
    ap.add_argument("--warm-start", default="")
    ap.add_argument("--out", default="/home/REDACTED/Bonsai-demo/models/bonsai2-eagle3/bonsai2_eagle3.safetensors")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--ttt-depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (effective batch = batch-size x accum)")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off); saves *_ema.safetensors, val uses EMA")
    ap.add_argument("--hard-alpha", type=float, default=0.0, help="weight of the hard CE on the data token added to the soft CE")
    ap.add_argument("--val-samples", type=int, default=100, help="last N samples of the index are held out")
    ap.add_argument("--eval-every", type=int, default=300, help="evaluate on the held-out set every N steps (0 = only before saves and at the end)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0, help="save an intermediate checkpoint every N steps (0=off)")
    ap.add_argument("--save-embd", type=int, default=1, help="1 = include token_embd.weight in every checkpoint (GGUF needs it); 0 = omit (tests)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0=full epochs)")
    args = ap.parse_args()

    if args.vram_table:
        print_vram_table(ema=False)
        print_vram_table(ema=True)
        return
    if args.check:
        ok = run_check(args)
        if args.save_init:
            save_init(args)
        sys.exit(0 if ok else 1)
    if args.save_init:
        save_init(args)
        return
    if args.ttt_depth < 1 or args.ttt_depth > 7:
        sys.exit("--ttt-depth must be in 1..7")
    train(args)


if __name__ == "__main__":
    main()
