#!/usr/bin/env python3
"""
build_draft_vocab.py - build the EAGLE-3 draft vocabulary for Bonsai 2 27B.

The script reads the token streams of the BON2 feature files. It skips the
feature payload of each sample with a seek. It counts the token ids and keeps
the most frequent ids. It adds every special token of the donor GGUF tokenizer
(token types CONTROL and USER_DEFINED). The draft vocabulary has 32768 ids.

Output: draft_vocab.npz with two arrays.
  d2t  int64 [n_draft]   draft id -> target id, sorted ascending by target id
  t2d  int64 [n_target]  target id -> draft id, -1 when the id is absent

The script prints the coverage of the loss_mask=1 tokens (the tokens that the
target model generated).

Usage:
    python3 build_draft_vocab.py --feats-dir /home/usman/Bonsai-demo/dflash-training/v2/feats_all
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
import time

import numpy as np

LLAMA_CPP = "/home/usman/Bonsai-demo/llama.cpp"
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))

DEFAULT_FEATS = "/home/usman/Bonsai-demo/dflash-training/v2/feats_all"
DEFAULT_DONOR = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab.npz")

# llama.cpp token types
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4


def scan_token_streams(paths, max_seq_len=0):
    """Read the token ids and the loss mask of every sample. Skip the features.

    Returns (counts_all, counts_gen, n_samples, lengths) where the counts are
    int64 arrays over the target vocabulary. When max_seq_len > 0, the count
    uses only the first max_seq_len tokens of each sample (the trainer
    truncation).
    """
    counts_all = None
    counts_gen = None
    n_samples = 0
    lengths = []
    for p in paths:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"BON2":
                raise ValueError(f"{p}: bad magic {magic!r}, expected BON2")
            embd, n_taps = struct.unpack("<II", f.read(8))
            tap_layers = struct.unpack("<5I", f.read(20))
            print(f"[vocab] {os.path.basename(p)}: embd={embd} n_taps={n_taps} tap_layers={list(tap_layers)}")
            tap_stride = n_taps * embd
            n_file = 0
            while True:
                b = f.read(4)
                if len(b) < 4:
                    break
                n = struct.unpack("<I", b)[0]
                toks = np.frombuffer(f.read(n * 4), dtype=np.int32)
                mask = np.frombuffer(f.read(n), dtype=np.uint8)
                # skip taps f32[n*tap_stride] + final_hidden f32[n*embd]
                f.seek(n * tap_stride * 4 + n * embd * 4, os.SEEK_CUR)
                if max_seq_len > 0 and n > max_seq_len:
                    toks = toks[:max_seq_len]
                    mask = mask[:max_seq_len]
                lengths.append(n)
                n_samples += 1
                n_file += 1
                yield toks, mask
            print(f"[vocab] {os.path.basename(p)}: {n_file} samples")


def load_special_ids(donor_path):
    """Return (special_ids, n_vocab, names) from the donor GGUF tokenizer."""
    from gguf import GGUFReader

    r = GGUFReader(donor_path)
    tt = r.fields["tokenizer.ggml.token_type"]
    types = np.array([tt.parts[i][0] for i in tt.data], dtype=np.int64)
    toks = r.fields["tokenizer.ggml.tokens"]
    n_vocab = len(toks.data)
    special = np.where((types == TOKEN_TYPE_CONTROL) | (types == TOKEN_TYPE_USER_DEFINED))[0]
    names = {int(i): bytes(toks.parts[toks.data[i]]).decode("utf-8", "replace") for i in special}
    # explicit ids from the metadata (eos, pad, bos) - already CONTROL, kept for safety
    extra = []
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
                "tokenizer.ggml.bos_token_id", "tokenizer.ggml.mask_token_id"):
        fld = r.fields.get(key)
        if fld is not None:
            extra.append(int(fld.parts[fld.data[0]][0]))
    special = np.unique(np.concatenate([special, np.array(extra, dtype=np.int64)]))
    return special.astype(np.int64), n_vocab, names


def main():
    ap = argparse.ArgumentParser(description="Build the EAGLE-3 draft vocabulary (Bonsai 2)")
    ap.add_argument("--feats-dir", default=DEFAULT_FEATS)
    ap.add_argument("--donor", default=DEFAULT_DONOR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-draft", type=int, default=32768)
    ap.add_argument("--mask-token-id", type=int, default=248070, help="dflash mask token id, kept in the draft vocab")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.feats_dir, "*.bin")))
    if not paths:
        print(f"[vocab] ERROR: no *.bin files in {args.feats_dir}", file=sys.stderr)
        sys.exit(2)

    special, n_vocab, names = load_special_ids(args.donor)
    special = np.unique(np.concatenate([special, np.array([args.mask_token_id], dtype=np.int64)]))
    print(f"[vocab] donor vocab={n_vocab} special ids={len(special)}: "
          f"{[names.get(int(i), str(int(i))) for i in special]}")

    t0 = time.time()
    counts_all = np.zeros(n_vocab, dtype=np.int64)
    counts_gen = np.zeros(n_vocab, dtype=np.int64)
    lengths = []
    n_samples = 0
    for toks, mask in scan_token_streams(paths):
        n_samples += 1
        lengths.append(len(toks))
        counts_all += np.bincount(toks, minlength=n_vocab)
        gen = toks[mask > 0]
        if gen.size:
            counts_gen += np.bincount(gen, minlength=n_vocab)
    lengths = np.array(lengths)
    print(f"[vocab] scanned {n_samples} samples, {int(counts_all.sum())} tokens, "
          f"{int(counts_gen.sum())} loss_mask=1 tokens in {time.time() - t0:.1f}s")
    print(f"[vocab] sample length: min={lengths.min()} mean={lengths.mean():.1f} "
          f"p50={np.percentile(lengths, 50):.0f} p90={np.percentile(lengths, 90):.0f} max={lengths.max()} "
          f"n>1024={int((lengths > 1024).sum())}")

    # rank by total count over all token streams (prompt + generated)
    order = np.argsort(-counts_all, kind="stable")
    chosen = set(int(i) for i in special)
    for tid in order:
        if len(chosen) >= args.n_draft:
            break
        if counts_all[tid] == 0:
            break
        chosen.add(int(tid))
    if len(chosen) < args.n_draft:
        # the streams have fewer distinct ids than n_draft: fill with the lowest absent ids
        for tid in range(n_vocab):
            if len(chosen) >= args.n_draft:
                break
            chosen.add(tid)
    d2t = np.array(sorted(chosen), dtype=np.int64)
    assert len(d2t) == args.n_draft, len(d2t)
    t2d = np.full(n_vocab, -1, dtype=np.int64)
    t2d[d2t] = np.arange(args.n_draft, dtype=np.int64)

    present = t2d >= 0
    cov_gen = counts_gen[present].sum() / max(1, counts_gen.sum())
    cov_all = counts_all[present].sum() / max(1, counts_all.sum())
    n_distinct_all = int((counts_all > 0).sum())
    n_distinct_gen = int((counts_gen > 0).sum())
    # reference: coverage when the ranking uses the generated tokens only
    order_gen = np.argsort(-counts_gen, kind="stable")[: args.n_draft]
    cov_gen_alt = counts_gen[order_gen].sum() / max(1, counts_gen.sum())

    print(f"[vocab] distinct ids: all={n_distinct_all} gen={n_distinct_gen}")
    print(f"[vocab] COVERAGE loss_mask=1 tokens: {cov_gen * 100:.4f}%  (all tokens: {cov_all * 100:.4f}%)")
    print(f"[vocab] reference: ranking by generated counts only would give {cov_gen_alt * 100:.4f}%")
    print(f"[vocab] d2t[0..7]={d2t[:8].tolist()} d2t[-4..]={d2t[-4:].tolist()} "
          f"min={int(d2t.min())} max={int(d2t.max())}")

    np.savez(args.out, d2t=d2t, t2d=t2d)
    print(f"[vocab] wrote {args.out}: d2t {d2t.shape} int64, t2d {t2d.shape} int64")


if __name__ == "__main__":
    main()
