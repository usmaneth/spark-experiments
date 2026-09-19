#!/usr/bin/env python3
"""tag_gen_records.py - map generated token-id records back to their prompt id / category.

Mode 1 (server up):  tag_gen_records.py tokenize <port> <prompts.jsonl> <prompt_ids.jsonl>
    Tokenizes every prompt with the same chat template as gen_client.py and writes
    {"id", "category", "source", "prompt_ids"} per line.
Mode 2 (offline):    tag_gen_records.py join <prompt_ids.jsonl> <gen.jsonl> <tags.jsonl>
    For gen record i, matches tokens[:n_prompt] against the prompt_ids table and writes
    {"line": i, "id", "category", "source", "n_prompt", "n_gen"} per line, plus a summary
    on stderr.
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def first_user(o):
    conv = o.get("conversations") or o.get("messages") or []
    for m in conv:
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return m["content"]
    return None


def tokenize(port, prompts_path, out_path):
    base = f"http://127.0.0.1:{port}"

    def tok(rec):
        u = first_user(rec)
        ptext = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
        req = urllib.request.Request(base + "/tokenize", data=json.dumps({"content": ptext, "add_special": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        ids = json.loads(urllib.request.urlopen(req, timeout=120).read())["tokens"]
        return {"id": rec.get("id"), "category": rec.get("category"), "source": rec.get("source"), "prompt_ids": ids}

    recs = [json.loads(l) for l in open(prompts_path) if l.strip()]
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(tok, recs))
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"tokenized {len(rows)} prompts -> {out_path}", file=sys.stderr)


def join(ids_path, gen_path, out_path):
    table = {}
    for l in open(ids_path):
        r = json.loads(l)
        table[tuple(r["prompt_ids"])] = r
    n = 0
    miss = 0
    per_cat = {}
    per_cat_gen = {}
    with open(out_path, "w") as f:
        for i, l in enumerate(open(gen_path)):
            g = json.loads(l)
            key = tuple(g["tokens"][: g["n_prompt"]])
            r = table.get(key)
            n_gen = len(g["tokens"]) - g["n_prompt"]
            if r is None:
                miss += 1
                row = {"line": i, "id": None, "category": "unknown", "source": None, "n_prompt": g["n_prompt"], "n_gen": n_gen}
            else:
                row = {"line": i, "id": r["id"], "category": r["category"], "source": r["source"], "n_prompt": g["n_prompt"], "n_gen": n_gen}
            per_cat[row["category"]] = per_cat.get(row["category"], 0) + 1
            per_cat_gen[row["category"]] = per_cat_gen.get(row["category"], 0) + n_gen
            f.write(json.dumps(row) + "\n")
            n += 1
    print(json.dumps({"records": n, "unmatched": miss, "per_category": per_cat,
                      "gen_tokens_per_category": per_cat_gen}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tokenize":
        tokenize(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif mode == "join":
        join(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit("mode: tokenize | join")
