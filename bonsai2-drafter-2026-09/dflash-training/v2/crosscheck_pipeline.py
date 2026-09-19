#!/usr/bin/env python3
# End-to-end check: extractor final_hidden @ W_lm.T must predict the next token.
# On greedy-generated positions this should match ~100% (generation used argmax of
# logits, and W_lm reproduces those logits). Confirms final_hidden basis + ordering.
# Usage: crosscheck_pipeline.py <feats.bin> <W_lm.bin> [n_samples]
import sys, struct, numpy as np

feats = sys.argv[1] if len(sys.argv) > 1 else "feats/batch1.bin"
wlm_path = sys.argv[2] if len(sys.argv) > 2 else "teacher/W_lm.bin"
n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 3

with open(wlm_path, "rb") as f:
    vocab, embd = struct.unpack("<II", f.read(8))
    W = np.frombuffer(f.read(vocab * embd * 2), dtype=np.float16).reshape(vocab, embd)
print(f"W_lm: vocab={vocab} embd={embd}")

with open(feats, "rb") as f:
    assert f.read(4) == b"BON2"
    e2, n_taps = struct.unpack("<II", f.read(8))
    tap_layers = struct.unpack("<5I", f.read(20))
    assert e2 == embd
    tot_gen = tot_gen_hit = 0
    tot_all = tot_all_hit = 0
    s = 0
    while s < n_check:
        hdr = f.read(4)
        if len(hdr) < 4: break
        (n,) = struct.unpack("<I", hdr)
        toks = np.frombuffer(f.read(4*n), dtype=np.int32).astype(np.int64)
        mask = np.frombuffer(f.read(n), dtype=np.uint8)
        f.read(4*n*n_taps*embd)  # skip taps
        fh = np.frombuffer(f.read(4*n*embd), dtype=np.float32).reshape(n, embd)
        s += 1
        # logits = fh @ W.T ; predict token at i+1 from position i
        logits = fh.astype(np.float32) @ W.T.astype(np.float32)
        pred = logits.argmax(axis=1)
        gen_hit = all_hit = gen_n = all_n = 0
        for i in range(n-1):
            nxt = toks[i+1]
            hit = int(pred[i] == nxt)
            all_hit += hit; all_n += 1
            if mask[i+1] == 1:  # next token is a generated token
                gen_hit += hit; gen_n += 1
        tot_gen += gen_n; tot_gen_hit += gen_hit
        tot_all += all_n; tot_all_hit += all_hit
        print(f"sample {s}: n={n} gen_next={gen_n} gen_match={gen_hit}/{gen_n}"
              f"={100*gen_hit/max(1,gen_n):.1f}% all_next_match={all_hit}/{all_n}={100*all_hit/max(1,all_n):.1f}%")
    print(f"\nTOTAL greedy-gen next-token match: {tot_gen_hit}/{tot_gen}={100*tot_gen_hit/max(1,tot_gen):.2f}%")
    print(f"TOTAL all next-token match:        {tot_all_hit}/{tot_all}={100*tot_all_hit/max(1,tot_all):.2f}%")
