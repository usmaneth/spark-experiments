#!/usr/bin/env python3
# Concurrent self-distillation generation client for llama-server - raw-prompt variant of gen_client.py.
#
# Same contract as gen_client.py, plus records that carry "raw_prompt": that text is the exact chat-templated
# prompt (built by build_prompts_tools.py through /apply-template) and is sent as is. It is tokenized through
# /tokenize with add_special false and with_pieces true; the pieces must join back to the text or the record is
# skipped (roundtrip check). Records without "raw_prompt" fall back to the gen_client.py path: the first user turn
# wrapped in the ChatML template with add_special true.
# Output per line: {"tokens": prompt_ids + gen_ids, "n_prompt": len(prompt_ids), "id": ..., "shape": ...}
# ("id"/"shape" only when the record has them; extract_feats_e3 reads "tokens" and "n_prompt" and ignores the rest).
#
# Usage: gen_client_raw.py [--plan-only] [--plan-n 20] [--max-prompt-tokens 3600] [--host 127.0.0.1]
#                          <port> <workers> <n_predict> <out.jsonl> <max_samples> <skip_lines> <prompts.jsonl> [more.jsonl ...]
# --plan-only tokenizes the first --plan-n records, checks the roundtrip, prints token counts and exits.
# It never calls /completion.
import sys, json, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---- optional flags (removed from argv before the positional parse)
plan_only = False
plan_n = 20
max_prompt_tokens = 3600
host = "127.0.0.1"
argv = []
i = 1
while i < len(sys.argv):
    a = sys.argv[i]
    if a == "--plan-only":
        plan_only = True
    elif a == "--plan-n":
        i += 1; plan_n = int(sys.argv[i])
    elif a == "--max-prompt-tokens":
        i += 1; max_prompt_tokens = int(sys.argv[i])
    elif a == "--host":
        i += 1; host = sys.argv[i]
    else:
        argv.append(a)
    i += 1

port      = int(argv[0])
workers   = int(argv[1])
npred     = int(argv[2])
out_path  = argv[3]
max_samples = int(argv[4])
skip_lines  = int(argv[5])
prompt_files = argv[6:]

BASE = f"http://{host}:{port}"

def post(path, obj, timeout=600):
    req = urllib.request.Request(BASE+path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def first_user(o):
    conv = o.get("conversations") or o.get("messages")
    if isinstance(conv, list):
        for m in conv:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                return m["content"]
    if o.get("prompt"): return o["prompt"]
    return None

def load_item(line):
    """Returns ("raw", text, meta) for raw_prompt records, ("chat", user_text, meta) otherwise, or None."""
    try:
        o = json.loads(line)
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    meta = {k: o[k] for k in ("id", "shape") if k in o}
    if isinstance(o.get("raw_prompt"), str) and o["raw_prompt"]:
        return ("raw", o["raw_prompt"], meta)
    u = first_user(o)
    if not u: return None
    return ("chat", u, meta)

# collect prompts
prompts = []
seen = 0
for pf in prompt_files:
    with open(pf) as f:
        for line in f:
            seen += 1
            if seen <= skip_lines: continue
            it = load_item(line.rstrip("\n"))
            if not it: continue
            prompts.append(it)
            if len(prompts) >= max_samples * 2:  # oversample; some may be filtered/too long
                break
    if len(prompts) >= max_samples * 2: break
n_raw = sum(1 for p in prompts if p[0] == "raw")
print(f"loaded {len(prompts)} candidate prompts ({n_raw} raw_prompt, {len(prompts)-n_raw} chat) (skip_lines={skip_lines}) "
      f"max_prompt_tokens={max_prompt_tokens} server={BASE}", flush=True)

def tokenize_item(it):
    """Returns (prompt_ids, error). Raw prompts: add_special false + roundtrip check. Chat: gen_client.py path."""
    kind, text, meta = it
    if kind == "raw":
        tk = post("/tokenize", {"content": text, "add_special": False, "with_pieces": True})
        ids, pieces = [], []
        for t in tk["tokens"]:
            ids.append(t["id"])
            p = t["piece"]
            pieces.append(p if isinstance(p, str) else bytes(p).decode("utf-8", "replace"))
        if "".join(pieces) != text:
            return ids, "roundtrip"
        return ids, None
    ptext = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    tk = post("/tokenize", {"content": ptext, "add_special": True})
    return tk["tokens"], None

if plan_only:
    props = post("/props", {}) if False else json.loads(urllib.request.urlopen(BASE + "/props", timeout=30).read())
    print(f"plan-only: server model={props.get('model_path')} build={props.get('build_info')} n_ctx(default slot)={props.get('default_generation_settings', {}).get('n_ctx')} slots={props.get('total_slots')}")
    tot = 0; nfail = 0; nlong = 0
    for k, it in enumerate(prompts[:plan_n]):
        ids, err = tokenize_item(it)
        meta = it[2]
        flag = ""
        if err: nfail += 1; flag = f" ROUNDTRIP-FAIL"
        if len(ids) > max_prompt_tokens: nlong += 1; flag += f" TOO-LONG(>{max_prompt_tokens})"
        tot += len(ids)
        print(f"  [{k:3d}] {meta.get('id', '-'):<14} {meta.get('shape', '-'):<6} kind={it[0]:<4} n_prompt={len(ids):5d} ends_with={json.dumps(it[1][-24:])}{flag}")
    n = len(prompts[:plan_n])
    print(f"plan-only done: {n} records tokenized, total prompt tokens={tot}, mean={tot/max(1,n):.1f}, roundtrip_fail={nfail}, too_long={nlong}; "
          f"would request n_predict={npred} per record with {workers} workers -> up to {n*npred} generated tokens for these {n}. No /completion sent.")
    sys.exit(0)

lock = threading.Lock()
fout = open(out_path, "w")
counters = {"done":0, "skip":0, "gen_tokens":0, "roundtrip_fail":0, "too_long":0}
t0 = time.time()

def work(it):
    try:
        pids, err = tokenize_item(it)
        if err == "roundtrip":
            with lock: counters["skip"] += 1; counters["roundtrip_fail"] += 1
            return
        if len(pids) <= 4 or len(pids) > max_prompt_tokens:
            with lock: counters["skip"] += 1; counters["too_long"] += (len(pids) > max_prompt_tokens)
            return
        r = post("/completion", {"prompt": pids, "n_predict": npred, "temperature": 0,
                                 "cache_prompt": True, "return_tokens": True})
        gids = r.get("tokens") or []
        if not gids:
            with lock: counters["skip"] += 1
            return
        rec = {"tokens": pids + gids, "n_prompt": len(pids)}
        rec.update(it[2])
        with lock:
            if counters["done"] >= max_samples:
                return
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            counters["done"] += 1; counters["gen_tokens"] += len(gids)
            d = counters["done"]
            if d % 25 == 0:
                el = time.time() - t0
                print(f"  gen {d} (skip {counters['skip']}) | gen_tok {counters['gen_tokens']} | "
                      f"{counters['gen_tokens']/el:.1f} tok/s aggregate | {el:.1f}s", flush=True)
    except Exception as e:
        with lock: counters["skip"] += 1

with ThreadPoolExecutor(max_workers=workers) as ex:
    futs = []
    for it in prompts:
        if counters["done"] >= max_samples: break
        futs.append(ex.submit(work, it))
    for _ in futs: pass
    ex.shutdown(wait=True)

fout.close()
el = time.time() - t0
print(f"=== gen done === samples={counters['done']} skip={counters['skip']} (roundtrip_fail={counters['roundtrip_fail']} too_long={counters['too_long']}) "
      f"gen_tokens={counters['gen_tokens']} time={el:.1f}s aggregate={counters['gen_tokens']/max(el,1e-9):.1f} tok/s")
print(f"out: {out_path}")
