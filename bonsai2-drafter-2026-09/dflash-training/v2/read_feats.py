#!/usr/bin/env python3
# Reader / verifier for the v2 feature format (v2/feats/*.bin).
# Usage: read_feats.py <feats.bin> [max_samples_to_scan]
import sys, struct, numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "feats/batch1.bin"
    scan = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"BON2", f"bad magic {magic!r}"
        embd, n_taps = struct.unpack("<II", f.read(8))
        tap_layers = struct.unpack("<5I", f.read(20))
        print(f"magic=BON2 embd={embd} n_taps={n_taps} tap_layers={list(tap_layers)}")
        n_samples = 0
        tot_tok = 0
        tot_gen = 0
        bad = 0
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (n,) = struct.unpack("<I", hdr)
            toks = np.frombuffer(f.read(4 * n), dtype=np.int32)
            mask = np.frombuffer(f.read(n), dtype=np.uint8)
            taps = np.frombuffer(f.read(4 * n * n_taps * embd), dtype=np.float32)
            fh = np.frombuffer(f.read(4 * n * embd), dtype=np.float32)
            n_samples += 1
            tot_tok += n
            tot_gen += int(mask.sum())
            if n_samples <= scan:
                if taps.size != n * n_taps * embd or fh.size != n * embd:
                    bad += 1
                    print(f"  sample {n_samples}: TRUNCATED taps/fh")
                    break
                finite = np.isfinite(taps).all() and np.isfinite(fh).all()
                if not finite:
                    bad += 1
                if n_samples <= 3:
                    print(f"  sample {n_samples}: n={n} prompt={n-int(mask.sum())} gen={int(mask.sum())} "
                          f"finite={finite} tap_norm={np.linalg.norm(taps[:embd]):.2f} "
                          f"fh_norm={np.linalg.norm(fh[:embd]):.2f} tok0..4={toks[:5].tolist()}")
        print(f"\nsamples={n_samples} total_tokens={tot_tok} gen(loss=1)_tokens={tot_gen} "
              f"prompt_tokens={tot_tok-tot_gen} bad={bad}")
        print(f"mean_tokens/sample={tot_tok/max(1,n_samples):.1f} mean_gen/sample={tot_gen/max(1,n_samples):.1f}")

if __name__ == "__main__":
    main()
