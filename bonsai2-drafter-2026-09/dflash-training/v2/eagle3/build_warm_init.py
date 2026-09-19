#!/usr/bin/env python3
"""
build_warm_init.py - build an EAGLE-3 init checkpoint from one DSpark decoder layer.

The output has the key set, the dtypes and the metadata format of a
train_eagle3.py --save-init checkpoint. train_eagle3.py --warm-start loads it.

Kept from the reference init (train_eagle3.build_full_model):
    token_embd.weight        = teacher tok_embd (f16, frozen)
    d2t                      = draft vocab map (int64)
    output.weight            = W_lm[d2t]
    fc.weight                = [I/3 | I/3 | I/3]
    enc.output_norm.weight   = ones
    output_norm.weight       = ones
    blk.0.attn_norm.weight   = ones (the norm of the token embedding half)

Set from DSpark layer L (tensor names of train_dspark_v2.py save_checkpoint):
    blk.0.ffn_gate.weight          <- layers.L.mlp.gate_proj.weight              [17408, 5120]
    blk.0.ffn_up.weight            <- layers.L.mlp.up_proj.weight                [17408, 5120]
    blk.0.ffn_down.weight          <- layers.L.mlp.down_proj.weight              [5120, 17408]
    blk.0.ffn_norm.weight          <- layers.L.post_attention_layernorm.weight   [5120]
    blk.0.attn_output.weight       <- layers.L.self_attn.o_proj.weight           [5120, 4096]
    blk.0.attn_norm_2.weight       <- layers.L.input_layernorm.weight            [5120] (the g norm)
    blk.0.attn_q.weight[:, 5120:]  <- layers.L.self_attn.q_proj.weight           [4096, 5120]
    blk.0.attn_k.weight[:, 5120:]  <- layers.L.self_attn.k_proj.weight           [1024, 5120]
    blk.0.attn_v.weight[:, 5120:]  <- layers.L.self_attn.v_proj.weight           [1024, 5120]
    blk.0.attn_{q,k,v}.weight[:, :5120] = N(0, 0.02) * --embd-scale             (the e half)

The EAGLE-3 decoder input is concat(e_n, g_n). e_n is the normed token
embedding (columns 0:5120). g_n is the normed fused feature (columns
5120:10240). DSpark projects one normed hidden state. The DSpark weights go
to the g half. The e half gets small random weights.

q_norm and k_norm: DSpark applies an RMSNorm with a gain over head_dim to q
and to k. The EAGLE-3 runtime (src/models/eagle3.cpp) has no q/k norm and the
checkpoint has no tensor for it. --fold-qk-norm (default on) multiplies row j
of the 128 rows of each head in W_q by q_norm[j] and in W_k by k_norm[j].
The product equals the DSpark forward when the per-head RMS of q and of k is
1. The 1/rms factor is not folded. The warm start is approximate for that
reason. The fold applies to the full row, so the e half gets the same gain.

Usage:
    build_warm_init.py <dspark.safetensors> <layer_index> <out.safetensors>
        [--embd-scale 0.5] [--no-fold-qk-norm] [--seed 0]
        [--teacher-dir DIR] [--draft-vocab NPZ] [--taps 0,2,4]
        [--ref-init FILE]            compare keys, dtypes, shapes and the kept tensors
        [--verify-feats FILE]        reload the output with --warm-start and run a CPU forward
        [--verify-samples 2] [--verify-seq 256] [--verify-depth 4] [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from typing import Dict, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import train_eagle3 as te  # noqa: E402

H = 5120
FF = 17408
NH, NKV, HD = 32, 8, 128

# module attribute of Eagle3Drafter -> (DSpark key template, expected shape)
DSPARK_MAP: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "ffn_gate.weight": ("layers.{L}.mlp.gate_proj.weight", (FF, H)),
    "ffn_up.weight": ("layers.{L}.mlp.up_proj.weight", (FF, H)),
    "ffn_down.weight": ("layers.{L}.mlp.down_proj.weight", (H, FF)),
    "ffn_norm.weight": ("layers.{L}.post_attention_layernorm.weight", (H,)),
    "wo.weight": ("layers.{L}.self_attn.o_proj.weight", (H, NH * HD)),
    "attn_norm_2.weight": ("layers.{L}.input_layernorm.weight", (H,)),
    "wq.weight": ("layers.{L}.self_attn.q_proj.weight", (NH * HD, H)),
    "wk.weight": ("layers.{L}.self_attn.k_proj.weight", (NKV * HD, H)),
    "wv.weight": ("layers.{L}.self_attn.v_proj.weight", (NKV * HD, H)),
}
QK_NORM_KEYS = {
    "wq.weight": ("layers.{L}.self_attn.q_norm.weight", (HD,)),
    "wk.weight": ("layers.{L}.self_attn.k_norm.weight", (HD,)),
}
# the attention projections take concat(e_n, g_n); the DSpark weights go to the g half
SPLIT_KEYS = ("wq.weight", "wk.weight", "wv.weight")


def read_header(path: str) -> Tuple[Dict[str, dict], dict]:
    """Read the safetensors header only. Returns ({name: {dtype, shape}}, metadata)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None) or {}
    return hdr, meta


def check_dspark_header(path: str, layer: int) -> None:
    hdr, _ = read_header(path)
    n_layers = 1 + max(int(k.split(".")[1]) for k in hdr if k.startswith("layers."))
    if not 0 <= layer < n_layers:
        sys.exit(f"layer_index {layer} out of range: the checkpoint has {n_layers} layers")
    for attr, (tmpl, shape) in list(DSPARK_MAP.items()) + list(QK_NORM_KEYS.items()):
        key = tmpl.format(L=layer)
        if key not in hdr:
            sys.exit(f"missing DSpark tensor {key}")
        got = tuple(hdr[key]["shape"])
        assert got == shape, f"{key}: shape {got} != expected {shape}"
    print(f"[dspark] {os.path.basename(path)}: {n_layers} layers, layer {layer} tensors present with the expected shapes")


def tensor_stats(t: torch.Tensor) -> str:
    t = t.float()
    return f"rms {t.pow(2).mean().sqrt().item():.4f} mean {t.mean().item():+.4f} absmax {t.abs().max().item():.3f}"


def build(args) -> None:
    from safetensors import safe_open

    cfg = te.Eagle3Config()
    layer = args.layer_index
    check_dspark_header(args.dspark, layer)
    torch.manual_seed(args.seed)
    # the reference init: teacher tok_embd, W_lm[d2t], identity-average fc, N(0, 0.02) decoder, unit norms
    model, W_lm, _ = te.build_full_model(cfg, args.teacher_dir, args.draft_vocab)
    del W_lm
    params = dict(model.named_parameters())
    src: Dict[str, str] = {}
    with safe_open(args.dspark, framework="pt") as f, torch.no_grad():
        for attr, (tmpl, shape) in DSPARK_MAP.items():
            key = tmpl.format(L=layer)
            t = f.get_tensor(key).float()
            assert tuple(t.shape) == shape, (key, tuple(t.shape), shape)
            dst = params[attr]
            if attr in SPLIT_KEYS:
                # columns 0:H = e half (random, scaled); columns H:2H = g half (DSpark)
                assert tuple(dst.shape) == (shape[0], 2 * H), (attr, tuple(dst.shape))
                e_half = torch.randn(shape[0], H) * 0.02 * args.embd_scale
                dst[:, :H].copy_(e_half)
                dst[:, H:].copy_(t)
                src[attr] = f"{key} -> cols {H}:{2 * H}; N(0,0.02)*{args.embd_scale} -> cols 0:{H}"
            else:
                assert tuple(dst.shape) == shape, (attr, tuple(dst.shape), shape)
                dst.copy_(t)
                src[attr] = key
            del t
        if args.fold_qk_norm:
            for attr, (tmpl, shape) in QK_NORM_KEYS.items():
                key = tmpl.format(L=layer)
                gain = f.get_tensor(key).float()
                assert tuple(gain.shape) == shape, (key, tuple(gain.shape))
                dst = params[attr]
                n_heads = dst.shape[0] // HD
                # row j of each head's HD rows times gain[j]
                dst.view(n_heads, HD, 2 * H).mul_(gain[None, :, None])
                src[attr] += f"; rows folded with {key} (mean {gain.mean().item():.4f})"
        else:
            print("[fold] --no-fold-qk-norm: q/k rows keep the raw DSpark scale")
        # the e half norm stays at ones; attn_norm_2 (the g half norm) came from input_layernorm
        params["attn_norm.weight"].fill_(1.0)
    print(f"[build] layer {layer}, embd_scale {args.embd_scale}, fold_qk_norm {int(args.fold_qk_norm)}")
    print(f"[build] {'module tensor':22s} {'ggml name':28s} {'shape':16s} source")
    for attr, ggml in te.GGML_NAMES.items():
        p = params[attr]
        print(f"[build] {attr:22s} {ggml:28s} {str(list(p.shape)):16s} {src.get(attr, 'reference init')} | {tensor_stats(p)}")
    meta = {
        "target_layers": ",".join(str(te.BON2_TAP_LAYERS[i]) for i in te.parse_taps(args.taps)),
        "taps": args.taps,
        "step": 0,
        "warm_source": os.path.basename(args.dspark),
        "warm_layer": layer,
        "embd_scale": args.embd_scale,
        "fold_qk_norm": int(args.fold_qk_norm),
        "seed": args.seed,
    }
    te.save_checkpoint(model, args.out, with_embd=True, meta=meta)
    del model, params


KEPT_FROM_REF = ("token_embd.weight", "d2t", "output.weight", "fc.weight", "enc.output_norm.weight",
                 "output_norm.weight", "blk.0.attn_norm.weight")


def compare_with_ref(out: str, ref: str) -> bool:
    """Same keys, dtypes and shapes as the reference; the kept tensors are equal."""
    from safetensors import safe_open

    ok = True
    h_out, m_out = read_header(out)
    h_ref, m_ref = read_header(ref)
    if set(h_out) != set(h_ref):
        ok = False
        print(f"[ref] FAIL key set differs: only out {sorted(set(h_out) - set(h_ref))}, only ref {sorted(set(h_ref) - set(h_out))}")
    else:
        print(f"[ref] PASS key set: {len(h_out)} keys equal to the reference")
    for k in sorted(set(h_out) & set(h_ref)):
        a, b = h_out[k], h_ref[k]
        same = a["dtype"] == b["dtype"] and list(a["shape"]) == list(b["shape"])
        ok = ok and same
        print(f"[ref] {'PASS' if same else 'FAIL'} {k:28s} {a['dtype']:5s} {a['shape']}" + ("" if same else f" != ref {b['dtype']} {b['shape']}"))
    missing_meta = [k for k in m_ref if k not in m_out]
    print(f"[ref] {'PASS' if not missing_meta else 'FAIL'} metadata keys of the reference present: ref {m_ref} out {m_out}")
    ok = ok and not missing_meta
    with safe_open(out, framework="pt") as fo, safe_open(ref, framework="pt") as fr:
        for k in KEPT_FROM_REF:
            same = torch.equal(fo.get_tensor(k), fr.get_tensor(k))
            ok = ok and same
            print(f"[ref] {'PASS' if same else 'FAIL'} kept tensor equal to the reference: {k}")
    return ok


def verify(args) -> bool:
    """Reload the output through train_eagle3.load_checkpoint, run the TTT forward on real data."""
    from safetensors import safe_open

    cfg = te.Eagle3Config()
    ok = True
    files = [args.verify_feats]
    ds = te.E3Dataset(files, max_seq_len=args.verify_seq, taps=te.parse_taps(args.taps))
    n = min(args.verify_samples, len(ds))
    batch = te.bon2_collate([ds[len(ds) - n + i] for i in range(n)])
    tokens, loss_mask, taps, final_hidden = batch
    print(f"[verify] {n} samples from the end of {os.path.basename(args.verify_feats)}, tokens {tuple(tokens.shape)}, "
          f"loss rows {int(loss_mask.sum())}")

    results = {}
    for tag, path in (("warm", args.out), ("ref", args.ref_init)):
        if not path:
            continue
        model, W_lm, t2d = te.build_full_model(cfg, args.teacher_dir, args.draft_vocab, warm_start=path)
        model.eval()
        # prove the DSpark tensors landed: compare two loaded tensors with the file
        with safe_open(path, framework="pt") as f:
            for attr in ("ffn_gate.weight", "wq.weight"):
                same = torch.equal(dict(model.named_parameters())[attr].detach(), f.get_tensor(te.GGML_NAMES[attr]).float())
                ok = ok and same
                print(f"[verify] {'PASS' if same else 'FAIL'} {tag}: loaded {attr} equals the file tensor")
        W_lm = W_lm.float()
        d2t = model.d2t
        W_lm_draft = W_lm[d2t].contiguous()
        t0 = time.time()
        with torch.no_grad():
            loss, st = te.compute_loss(model, W_lm, W_lm_draft, t2d, d2t, tokens, loss_mask, taps, final_hidden,
                                       args.verify_depth, 0.0, False)
            g = model.encode(taps)
            logits_steps, hidden_steps = model.forward_ttt(tokens, g, args.verify_depth)
        finite = all(torch.isfinite(x).all().item() for x in logits_steps + hidden_steps) and torch.isfinite(loss).item()
        ok = ok and finite
        hid_rms = [x.float().pow(2).mean().sqrt().item() for x in hidden_steps]
        print(f"[verify] {'PASS' if finite else 'FAIL'} {tag}: finite logits/hidden/loss; loss {loss.item():.4f} "
              f"soft-ce/step {te.fmt_list(st['step_loss'])} top1-acc/step {te.fmt_list(st['step_acc'])} "
              f"hidden rms/step {te.fmt_list(hid_rms, 2)} ({time.time() - t0:.0f}s)")
        results[tag] = st
        del model, W_lm, W_lm_draft, logits_steps, hidden_steps, g
    if "warm" in results and "ref" in results:
        print(f"[verify] step-1 top1-acc warm {results['warm']['step_acc'][0]:.3f} vs ref {results['ref']['step_acc'][0]:.3f}; "
              f"soft-ce warm {results['warm']['step_loss'][0]:.3f} vs ref {results['ref']['step_loss'][0]:.3f}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="EAGLE-3 warm-start init from one DSpark layer")
    ap.add_argument("dspark", help="DSpark checkpoint (.safetensors, train_dspark_v2.py names)")
    ap.add_argument("layer_index", type=int, help="DSpark decoder layer to copy (0..n_layers-1)")
    ap.add_argument("out", help="output EAGLE-3 init (.safetensors)")
    ap.add_argument("--embd-scale", type=float, default=0.5, help="scale of the N(0, 0.02) e-half columns of W_q/W_k/W_v")
    ap.add_argument("--fold-qk-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="multiply the q/k rows by the DSpark q_norm/k_norm gains (default on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-dir", default=os.path.join(te.DEFAULT_V2, "teacher"))
    ap.add_argument("--draft-vocab", default=os.path.join(HERE, "draft_vocab.npz"))
    ap.add_argument("--taps", default="0,2,4", help="BON2 tap indices (metadata only)")
    ap.add_argument("--ref-init", default="", help="reference --save-init file to compare against")
    ap.add_argument("--verify-feats", default="", help="one BON2/BON3 file for the CPU forward check (off when empty)")
    ap.add_argument("--verify-samples", type=int, default=2)
    ap.add_argument("--verify-seq", type=int, default=256)
    ap.add_argument("--verify-depth", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.time()
    build(args)
    print(f"[build] wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)
    ok = True
    if args.ref_init:
        ok = compare_with_ref(args.out, args.ref_init) and ok
    if args.verify_feats:
        ok = verify(args) and ok
    print(f"[result] {'PASS' if ok else 'FAIL'} {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
