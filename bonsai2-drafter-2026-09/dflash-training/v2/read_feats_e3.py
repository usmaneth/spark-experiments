#!/usr/bin/env python3
"""read_feats_e3.py - reader and verifier for the BON3 feature format (extract_feats_e3).

BON3 layout (little endian):
    header : "BON3"(4) | embd u32 | n_taps u32 = 3 | tap_layers[3] u32 = {6, 34, 62}
    sample : n u32 | tokens i32[n] | loss_mask u8[n] | taps f16[n*3*embd] | final_hidden f16[n*embd]

Library use (numpy only):
    from read_feats_e3 import BON3File
    f = BON3File("feats_e3/batch2.bin")          # builds the sample index (file offsets)
    tokens, loss_mask, taps, final_hidden = f[i]  # taps: [n, 3, embd] float32, final_hidden: [n, embd] float32
    for tokens, loss_mask, taps, final_hidden in f: ...

CLI:
    read_feats_e3.py <bon3.bin> [--scan N]
        Prints the header, scans the samples and reports totals (like read_feats.py for BON2).
    read_feats_e3.py <bon3.bin> --compare-bon2 <bon2.bin> [--n 5]
        Compares the first N samples against the same samples of a BON2 file (f32, 5 taps):
        tokens and loss_mask must be identical; taps[0,2,4] and final_hidden must agree
        within f16 rounding. Prints the max abs / rel errors.
"""
import argparse
import os
import struct
import sys

import numpy as np

MAGIC = b"BON3"
HEADER_SIZE = 4 + 4 + 4 + 12
BON2_MAGIC = b"BON2"
BON2_HEADER_SIZE = 4 + 4 + 4 + 20
# the BON2 tap indices that the BON3 taps correspond to (layers 6, 34, 62 out of [6, 20, 34, 48, 62])
BON2_TAP_IDX = (0, 2, 4)


class BON3File:
    """One BON3 file with an index over the sample offsets."""

    def __init__(self, path):
        self.path = path
        self.index = []  # list of (byte_offset, n_tokens); byte_offset points at the n u32
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
            self.embd, self.n_taps = struct.unpack("<II", f.read(8))
            self.tap_layers = list(struct.unpack(f"<{self.n_taps}I", f.read(4 * self.n_taps)))
            size = os.fstat(f.fileno()).st_size
            off = f.tell()
            while off + 4 <= size:
                f.seek(off)
                (n,) = struct.unpack("<I", f.read(4))
                total = self.sample_bytes(n)
                if off + total > size:
                    print(f"[warn] {path}: truncated sample at offset {off} (n={n}); ignored", file=sys.stderr)
                    break
                self.index.append((off, n))
                off += total

    def sample_bytes(self, n):
        """Total bytes of one sample record with n tokens, including the n u32."""
        return 4 + n * 4 + n * 1 + n * self.n_taps * self.embd * 2 + n * self.embd * 2

    def __len__(self):
        return len(self.index)

    def read_at(self, f, off, n):
        f.seek(off + 4)
        tokens = np.frombuffer(f.read(n * 4), dtype=np.int32)
        loss_mask = np.frombuffer(f.read(n), dtype=np.uint8)
        taps = np.frombuffer(f.read(n * self.n_taps * self.embd * 2), dtype=np.float16)
        taps = taps.reshape(n, self.n_taps, self.embd).astype(np.float32)
        final_hidden = np.frombuffer(f.read(n * self.embd * 2), dtype=np.float16)
        final_hidden = final_hidden.reshape(n, self.embd).astype(np.float32)
        return tokens, loss_mask, taps, final_hidden

    def __getitem__(self, i):
        off, n = self.index[i]
        with open(self.path, "rb") as f:
            return self.read_at(f, off, n)

    def __iter__(self):
        with open(self.path, "rb") as f:
            for off, n in self.index:
                yield self.read_at(f, off, n)


def iter_bon3(paths):
    """Yield (tokens, loss_mask, taps, final_hidden) over several BON3 files."""
    for p in paths:
        yield from BON3File(p)


def iter_bon2_head(path, n_samples):
    """Yield the first n_samples of a BON2 file as (tokens, loss_mask, taps[n,5,embd] f32, final_hidden[n,embd] f32)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != BON2_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {BON2_MAGIC!r}")
        embd, n_taps = struct.unpack("<II", f.read(8))
        tap_layers = list(struct.unpack(f"<{n_taps}I", f.read(4 * n_taps)))
        yield ("header", embd, n_taps, tap_layers)
        for _ in range(n_samples):
            hdr = f.read(4)
            if len(hdr) < 4:
                return
            (n,) = struct.unpack("<I", hdr)
            tokens = np.frombuffer(f.read(n * 4), dtype=np.int32)
            loss_mask = np.frombuffer(f.read(n), dtype=np.uint8)
            taps = np.frombuffer(f.read(n * n_taps * embd * 4), dtype=np.float32).reshape(n, n_taps, embd)
            fh = np.frombuffer(f.read(n * embd * 4), dtype=np.float32).reshape(n, embd)
            yield (tokens, loss_mask, taps, fh)


def err_stats(a, b):
    """max abs error and max relative error (relative to |a| where |a| > 1e-3) of b vs a."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(d.max()) if d.size else 0.0
    denom = np.abs(a.astype(np.float64))
    sel = denom > 1e-3
    max_rel = float((d[sel] / denom[sel]).max()) if sel.any() else 0.0
    return max_abs, max_rel


def cmd_scan(path, scan):
    f = BON3File(path)
    print(f"magic=BON3 embd={f.embd} n_taps={f.n_taps} tap_layers={f.tap_layers} samples={len(f)}")
    tot_tok = sum(n for _, n in f.index)
    tot_gen = 0
    bad = 0
    with open(path, "rb") as fh:
        for i, (off, n) in enumerate(f.index[:scan]):
            tokens, mask, taps, final_hidden = f.read_at(fh, off, n)
            tot_gen += int(mask.sum())
            finite = bool(np.isfinite(taps).all() and np.isfinite(final_hidden).all())
            if not finite:
                bad += 1
            if i < 3:
                print(f"  sample {i}: n={n} prompt={n - int(mask.sum())} gen={int(mask.sum())} finite={finite} "
                      f"tap0_norm={np.linalg.norm(taps[0, 0]):.2f} fh_norm={np.linalg.norm(final_hidden[0]):.2f} "
                      f"tok0..4={tokens[:5].tolist()}")
    print(f"\nsamples={len(f)} total_tokens={tot_tok} scanned={min(scan, len(f))} gen(loss=1)_tokens_scanned={tot_gen} bad={bad}")
    print(f"mean_tokens/sample={tot_tok / max(1, len(f)):.1f} file_bytes={os.path.getsize(path)}")


def cmd_compare(path3, path2, n):
    f3 = BON3File(path3)
    it2 = iter_bon2_head(path2, n)
    _, embd2, n_taps2, layers2 = next(it2)
    print(f"BON3: embd={f3.embd} n_taps={f3.n_taps} tap_layers={f3.tap_layers} samples={len(f3)}")
    print(f"BON2: embd={embd2} n_taps={n_taps2} tap_layers={layers2}")
    assert f3.embd == embd2, "embd mismatch"
    assert [layers2[i] for i in BON2_TAP_IDX] == f3.tap_layers, "tap layer mismatch"
    n_cmp = 0
    worst = {"taps_abs": 0.0, "taps_rel": 0.0, "fh_abs": 0.0, "fh_rel": 0.0}
    ok = True
    for i, s2 in enumerate(it2):
        if i >= len(f3):
            break
        t3, m3, taps3, fh3 = f3[i]
        t2, m2, taps2, fh2 = s2
        same_tok = t3.shape == t2.shape and bool(np.array_equal(t3, t2))
        same_mask = m3.shape == m2.shape and bool(np.array_equal(m3, m2))
        if not (same_tok and same_mask):
            ok = False
            print(f"  sample {i}: MISMATCH tokens_equal={same_tok} loss_mask_equal={same_mask} n3={len(t3)} n2={len(t2)}")
            continue
        taps2_sel = taps2[:, list(BON2_TAP_IDX), :]
        # the exact f16 rounding of the f32 reference must reproduce the BON3 values
        exact = bool(np.array_equal(taps2_sel.astype(np.float16).astype(np.float32), taps3)) and \
            bool(np.array_equal(fh2.astype(np.float16).astype(np.float32), fh3))
        ta, tr = err_stats(taps2_sel, taps3)
        fa, fr = err_stats(fh2, fh3)
        worst["taps_abs"] = max(worst["taps_abs"], ta)
        worst["taps_rel"] = max(worst["taps_rel"], tr)
        worst["fh_abs"] = max(worst["fh_abs"], fa)
        worst["fh_rel"] = max(worst["fh_rel"], fr)
        print(f"  sample {i}: n={len(t3)} tokens=identical loss_mask=identical f16_roundtrip_exact={exact} "
              f"taps max_abs={ta:.4g} max_rel={tr:.4g} | final_hidden max_abs={fa:.4g} max_rel={fr:.4g}")
        n_cmp += 1
    print(f"\ncompared {n_cmp} samples; tokens/loss_mask identical={ok}; "
          f"worst taps max_abs={worst['taps_abs']:.4g} max_rel={worst['taps_rel']:.4g}; "
          f"worst final_hidden max_abs={worst['fh_abs']:.4g} max_rel={worst['fh_rel']:.4g} "
          f"(f16 unit roundoff = {2 ** -11:.4g})")
    return ok and n_cmp > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--scan", type=int, default=10 ** 9)
    ap.add_argument("--compare-bon2", default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    if args.compare_bon2:
        sys.exit(0 if cmd_compare(args.path, args.compare_bon2, args.n) else 1)
    cmd_scan(args.path, args.scan)


if __name__ == "__main__":
    main()
