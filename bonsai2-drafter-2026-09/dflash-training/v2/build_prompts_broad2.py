#!/usr/bin/env python3
"""build_prompts_broad2.py - assemble the 20k broad-mix prompt set for the EAGLE-3 data scale-up.

Output: <out.jsonl> with one record per line (same format as prompts_broad.jsonl):
    {"id": "b2_<source>_<n>", "category": "<cat>", "source": "<source>",
     "messages": [{"role": "user", "content": ...}]}
plus <out>.counts.json (per-source / per-category counts, drops, token stats)
plus <out stem>_ids.jsonl with {"id", "category", "source", "prompt_ids"} per line, the same
format that `tag_gen_records.py tokenize` writes (only when the offline tokenizer is available).

Categories and shares (default total 20000), the same mix as build_prompts_broad.py:
    math       25%  gsm8k train + MATH (hendrycks, all levels, L1-3 first) + Open-Platypus math
    reasoning  25%  Open-Platypus (non-math) + ARC-Challenge train/val + ARC-Easy train + LogiQA
                    + StrategyQA train/test + templates (10%)
    chat       20%  no_robots train/test + dolly-15k + ultrachat first turns (train_sft, test_sft)
    code       15%  CodeAlpaca-20k rows >= 5000 + Evol-Instruct-Code
    longform   15%  no_robots Generation + dolly creative/brainstorm + ultrachat long first turns
                    + templates (10%)

Dedupe:
  - by normalized text against every --dedupe file that holds text (messages/conversations),
  - by chat-templated token ids against every --dedupe file that holds {"tokens", "n_prompt"}
    records (the prompts_gen_*.jsonl files), when the offline tokenizer is available,
  - within the new set (text and token ids).
Length: prompts with more than --max-tokens chat-templated tokens are dropped (gen_client.py
skips them). Without the tokenizer, the --max-chars limit is the only length filter.

usage: build_prompts_broad2.py <out.jsonl> [--total 20000] [--seed 1] [--dedupe f1.jsonl ...]
           [--tokenizer-bin ./tokenize_prompts] [--tokenizer-model <gguf>] [--max-tokens 768]
Runs on a CPU-only node. It needs the `datasets` package and network access to the HF Hub
(about 0.6 GB of downloads on a cold cache).
"""
import argparse
import json
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

MIN_CHARS = 20
DEFAULT_MODEL = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
CHAT_TEMPLATE = "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"

# Sources that the task allows only when they are already in the local HF cache.
OPTIONAL_LOCAL_ONLY = ["meta-math/MetaMathQA", "Open-Orca/OpenOrca"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def try_load(name, *args, **kw):
    from datasets import load_dataset
    t0 = time.time()
    try:
        d = load_dataset(name, *args, **kw)
        log(f"[load] {name} {args} {kw.get('split', '')}: ok ({time.time() - t0:.0f}s)")
        return d
    except Exception as e:
        log(f"[load] {name} {args} {kw.get('split', '')}: FAILED: {type(e).__name__}: {str(e)[:200]}")
        return None


def is_cached_locally(repo_id):
    """True when the HF hub cache already holds a snapshot of the dataset repo."""
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    d = os.path.join(home, "hub", "datasets--" + repo_id.replace("/", "--"), "snapshots")
    return os.path.isdir(d) and bool(os.listdir(d))


COT_SUFFIX = [
    "",
    "",
    " Show your reasoning step by step.",
    " Think step by step and give the final answer.",
    " Explain your work.",
    " Solve this step by step.",
]


class Pool:
    """Collects (category, source, text) candidates with global text dedupe."""

    def __init__(self, seen, max_chars):
        self.seen = seen
        self.max_chars = max_chars
        self.by_source = {}
        self.rejected = {"short_or_long": 0, "dup_text": 0, "special_tokens": 0}

    def ok_text(self, s):
        if not isinstance(s, str):
            return False
        s = s.strip()
        if len(s) < MIN_CHARS or len(s) > self.max_chars:
            self.rejected["short_or_long"] += 1
            return False
        if "<|im_start|>" in s or "<|im_end|>" in s:
            self.rejected["special_tokens"] += 1
            return False
        return True

    def add(self, source, cat, text):
        if not self.ok_text(text):
            return False
        text = text.strip()
        k = norm(text)
        if k in self.seen:
            self.rejected["dup_text"] += 1
            return False
        self.seen.add(k)
        self.by_source.setdefault((cat, source), []).append(text)
        return True

    def count(self, cat, source):
        return len(self.by_source.get((cat, source), []))


# --------------------------------------------------------------------------- sources
def src_gsm8k(pool, rng, want):
    d = try_load("openai/gsm8k", "main", split="train")
    if d is None:
        return
    idx = list(range(len(d)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        if n >= want:
            break
        q = d[i]["question"].strip()
        if pool.add("gsm8k", "math", q + rng.choice(COT_SUFFIX)):
            n += 1


def src_math(pool, rng, want):
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    easy, hard = [], []
    for s in subjects:
        d = try_load("EleutherAI/hendrycks_math", s, split="train")
        if d is None:
            continue
        for r in d:
            lvl = str(r.get("level", ""))
            (easy if lvl in ("Level 1", "Level 2", "Level 3") else hard).append(r["problem"])
    if not easy and not hard:
        d = try_load("lighteval/MATH", "all", split="train")
        if d is not None:
            for r in d:
                lvl = str(r.get("level", ""))
                (easy if lvl in ("Level 1", "Level 2", "Level 3") else hard).append(r["problem"])
    rng.shuffle(easy)
    rng.shuffle(hard)
    n = 0
    # the broad set used Level 1-3 only; Level 4-5 is the additional slice, used after L1-3 runs out
    for p in easy + hard:
        if n >= want:
            break
        if pool.add("hendrycks_math", "math", p.strip() + rng.choice(COT_SUFFIX)):
            n += 1


def src_platypus(pool, rng, want_reason, want_math):
    d = try_load("garage-bAInd/Open-Platypus", split="train")
    if d is None:
        return
    idx = list(range(len(d)))
    rng.shuffle(idx)
    n_r = n_m = 0
    hist = {}
    reason_keys = ("reclor", "scienceqa", "scibench", "theoremqa", "arb", "airoboros", "guanaco", "tigerbot")
    for i in idx:
        if n_r >= want_reason and n_m >= want_math:
            break
        r = d[i]
        src = str(r.get("data_source", ""))
        hist[src] = hist.get(src, 0) + 1
        text = (r.get("instruction") or "").strip()
        if r.get("input"):
            text = (r["input"].strip() + "\n\n" + text).strip()
        low = src.lower()
        if "math" in low or "prm800k" in low or "prm-800k" in low:
            if n_m < want_math and pool.add("platypus_math", "math", text):
                n_m += 1
        elif any(k in low for k in reason_keys):
            if n_r < want_reason and pool.add("platypus_reason", "reasoning", text):
                n_r += 1
    log(f"[platypus] data_source histogram (scanned rows): {hist}")


def src_arc(pool, rng, want):
    rows = []
    for cfg, split, tag in (("ARC-Challenge", "train", "arc_challenge"), ("ARC-Challenge", "validation", "arc_challenge"),
                            ("ARC-Easy", "train", "arc_easy")):
        d = try_load("allenai/ai2_arc", cfg, split=split)
        if d is None:
            continue
        for r in d:
            rows.append((tag, r))
    # ARC-Challenge first (the broad-set source), ARC-Easy is the additional slice
    ch = [x for x in rows if x[0] == "arc_challenge"]
    ez = [x for x in rows if x[0] == "arc_easy"]
    rng.shuffle(ch)
    rng.shuffle(ez)
    n = 0
    for tag, r in ch + ez:
        if n >= want:
            break
        c = r["choices"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(c["label"], c["text"]))
        text = f"{r['question'].strip()}\n{opts}\nWhich option is correct? Explain your reasoning, then state the answer."
        if pool.add(tag, "reasoning", text):
            n += 1


def src_logiqa(pool, rng, want):
    d = try_load("lucasmccabe/logiqa", split="train")
    if d is None:
        d = try_load("lucasmccabe/logiqa", split="train", revision="refs/convert/parquet")
    if d is None:
        return
    idx = list(range(len(d)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        if n >= want:
            break
        r = d[i]
        opts = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(r["options"]))
        text = f"{r['context'].strip()}\n\n{r['query'].strip()}\n{opts}\nReason through the options step by step and pick one."
        if pool.add("logiqa", "reasoning", text):
            n += 1


def src_strategyqa(pool, rng, want):
    rows = []
    for split in ("train", "test"):
        d = try_load("ChilleD/StrategyQA", split=split)
        if d is None:
            continue
        rows.extend(r["question"] for r in d if r.get("question"))
    rng.shuffle(rows)
    n = 0
    for q in rows:
        if n >= want:
            break
        text = q.strip() + " Answer yes or no, and explain the chain of facts that leads to your answer."
        if pool.add("strategyqa", "reasoning", text):
            n += 1


def src_no_robots(pool, rng, want_chat, want_long):
    rows = []
    for split in ("train", "test"):
        d = try_load("HuggingFaceH4/no_robots", split=split)
        if d is None:
            continue
        rows.extend(d)
    rng.shuffle(rows)
    n_c = n_l = 0
    for r in rows:
        if n_c >= want_chat and n_l >= want_long:
            break
        msgs = r["messages"]
        if not msgs or msgs[0]["role"] != "user":
            continue
        text = msgs[0]["content"].strip()
        cat = r.get("category", "")
        if cat in ("Generation",) and n_l < want_long:
            if pool.add("no_robots_gen", "longform", text):
                n_l += 1
        elif cat in ("Open QA", "Brainstorm", "Chat", "Rewrite", "Summarize", "Classify", "Closed QA", "Extract") and n_c < want_chat:
            if pool.add("no_robots", "chat", text):
                n_c += 1


def src_dolly(pool, rng, want_chat, want_long):
    d = try_load("databricks/databricks-dolly-15k", split="train")
    if d is None:
        return
    idx = list(range(len(d)))
    rng.shuffle(idx)
    n_c = n_l = 0
    for i in idx:
        if n_c >= want_chat and n_l >= want_long:
            break
        r = d[i]
        text = r["instruction"].strip()
        if r.get("context"):
            text = f"{r['context'].strip()}\n\n{text}"
        cat = r.get("category", "")
        if cat in ("creative_writing", "brainstorming") and n_l < want_long:
            if pool.add("dolly_creative", "longform", text):
                n_l += 1
        elif n_c < want_chat:
            if pool.add("dolly", "chat", text):
                n_c += 1


LONG_RE = re.compile(r"\b(write|essay|article|story|guide|blog|report|describe in detail|explain in detail|create a|compose|draft)\b", re.I)


def src_ultrachat(pool, rng, want_chat, want_long, scan=50000):
    """Streams first user turns of ultrachat_200k. train_sft first, then test_sft (additional split)."""
    from datasets import load_dataset
    n_c = n_l = 0
    for split, budget in (("train_sft", scan), ("test_sft", scan // 2)):
        if n_c >= want_chat and n_l >= want_long:
            break
        try:
            d = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
            d = d.shuffle(seed=rng.randint(0, 10 ** 6), buffer_size=4000)
        except Exception as e:
            log(f"[load] ultrachat_200k {split}: FAILED: {e}")
            continue
        t0 = time.time()
        k = 0
        try:
            for k, r in enumerate(d):
                if k >= budget or (n_c >= want_chat and n_l >= want_long):
                    break
                msgs = r["messages"]
                if not msgs or msgs[0]["role"] != "user":
                    continue
                text = msgs[0]["content"].strip()
                # long-form: writing / essay style requests; chat: the rest
                if len(text) >= 120 and LONG_RE.search(text):
                    if n_l < want_long and pool.add("ultrachat_long", "longform", text):
                        n_l += 1
                elif n_c < want_chat and pool.add("ultrachat", "chat", text):
                    n_c += 1
            log(f"[load] ultrachat_200k {split} streaming: ok scanned={k} chat={n_c} long={n_l} ({time.time() - t0:.0f}s)")
        except Exception as e:
            log(f"[load] ultrachat_200k {split} stream: FAILED mid-way: {e} scanned={k} chat={n_c} long={n_l}")


def src_codealpaca(pool, rng, want, offset=5000):
    d = try_load("sahil2801/CodeAlpaca-20k", split="train")
    if d is None:
        return
    idx = list(range(offset, len(d)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        if n >= want:
            break
        r = d[i]
        text = r["instruction"].strip()
        if r.get("input"):
            text = f"{text}\n{r['input'].strip()}"
        if pool.add("codealpaca_fresh", "code", text):
            n += 1


def src_evol_code(pool, rng, want):
    d = try_load("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    if d is None:
        return
    idx = list(range(len(d)))
    rng.shuffle(idx)
    n = 0
    for i in idx:
        if n >= want:
            break
        if pool.add("evol_code", "code", d[i]["instruction"].strip()):
            n += 1


# --------------------------------------------------------------------------- templates (fallback + a small fixed share)
NAMES = ["Alice", "Ben", "Chloe", "Dan", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kira", "Leo", "Maya", "Noor", "Omar", "Priya",
         "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Yara", "Zane"]
ITEMS = ["apples", "notebooks", "tickets", "cupcakes", "marbles", "stickers", "bottles of water", "pencils", "books", "t-shirts",
         "seeds", "coins", "postcards", "candles", "batteries"]
CITIES = ["Lyon", "Osaka", "Denver", "Lagos", "Porto", "Austin", "Delhi", "Seoul", "Perth", "Quito",
          "Bergen", "Cusco", "Hanoi", "Malmo", "Nairobi", "Tbilisi"]


def tmpl_math(rng):
    t = rng.randrange(8)
    n1, n2 = rng.choice(NAMES), rng.choice(NAMES)
    while n2 == n1:
        n2 = rng.choice(NAMES)
    it = rng.choice(ITEMS)
    if t == 0:
        a, b, c = rng.randint(12, 90), rng.randint(2, 9), rng.randint(3, 40)
        return f"{n1} has {a} {it}. She gives {b} {it} to each of her {c // 4 + 1} friends and then buys {c} more. How many {it} does {n1} have now?"
    if t == 1:
        s1, s2, d = rng.choice([40, 50, 60, 70, 80, 90]), rng.choice([45, 55, 65, 75, 100, 110]), rng.choice([180, 240, 300, 360, 420])
        return f"Two cars start {d} km apart and drive toward each other at {s1} km/h and {s2} km/h. How long until they meet, and how far does each car travel? Show your reasoning step by step."
    if t == 2:
        p, r, y = rng.choice([1000, 2500, 5000, 8000]), rng.choice([3, 4, 5, 6, 8]), rng.choice([2, 3, 5])
        return f"{n1} invests ${p} at {r}% annual interest, compounded yearly. How much money will {n1} have after {y} years? Round to the nearest cent and explain each step."
    if t == 3:
        w, l = rng.randint(3, 25), rng.randint(3, 25)
        return f"A rectangular garden is {w} m wide and {l} m long. A path 1 m wide runs around the outside of the garden. What is the area of the path? Explain your work."
    if t == 4:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return f"A bakery sells muffins in boxes of {a} and cookies in bags of {b}. {n1} needs the same number of muffins and cookies for a party. What is the smallest number of each she must buy, and how many boxes and bags is that? Think step by step."
    if t == 5:
        x, y, z = rng.randint(10, 99), rng.randint(10, 99), rng.randint(2, 9)
        return f"The sum of two numbers is {x + y}. One number is {z} more than the other, and their product is {x * y}. Is that consistent? Find the two numbers if a solution exists, and show the algebra."
    if t == 6:
        h, m = rng.randint(1, 12), rng.choice([0, 15, 20, 30, 40, 45])
        return f"What is the angle between the hour hand and the minute hand of a clock at {h}:{m:02d}? Show the calculation."
    a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 30)
    return f"{n1} can paint a room in {a} hours, {n2} in {b} hours, and a third painter in {c} hours. How long will it take all three working together? Explain your reasoning step by step."


def tmpl_reasoning(rng):
    t = rng.randrange(6)
    a, b, c, d = rng.sample(NAMES, 4)
    if t == 0:
        return f"{a}, {b}, {c} and {d} sit in a row of four seats. {a} is not at either end. {b} sits immediately to the left of {c}. {d} is not next to {a}. List every possible seating order and explain how you eliminated the others."
    if t == 1:
        return f"On an island, knights always tell the truth and knaves always lie. {a} says: '{b} is a knave.' {b} says: 'At least one of us is a knight.' What are {a} and {b}? Reason step by step."
    if t == 2:
        seq = [rng.randint(1, 5)]
        step = rng.randint(2, 4)
        for _ in range(4):
            seq.append(seq[-1] * step + rng.randint(0, 2))
        return f"What is the next number in the sequence {', '.join(map(str, seq))}, ...? Explain the rule you found and check it against every term."
    if t == 3:
        return f"All {rng.choice(['engineers', 'bakers', 'sailors', 'poets'])} in a town are {rng.choice(['early risers', 'left-handed', 'tea drinkers'])}. Some {rng.choice(['early risers', 'left-handed people', 'tea drinkers'])} are {rng.choice(['musicians', 'gardeners', 'chess players'])}. {a} is a {rng.choice(['musician', 'gardener', 'chess player'])}. Can we conclude anything about whether {a} is an {rng.choice(['engineer', 'early riser'])}? Explain the logic carefully."
    if t == 4:
        return f"Three boxes are labeled '{rng.choice(['apples', 'pears', 'plums'])}', '{rng.choice(['oranges', 'lemons', 'limes'])}' and 'mixed', and every label is wrong. {a} may take one fruit from one box without looking inside. Which box should {a} pick, and how can {a} relabel all three correctly? Explain the reasoning."
    return f"{a} is older than {b}. {c} is younger than {b} but older than {d}. {rng.choice(NAMES)} is the same age as {c}. Rank all the people mentioned from oldest to youngest and say which relationships are uncertain."


LONG_TOPICS = [
    "how public-key cryptography works", "the causes of the French Revolution", "how vaccines train the immune system",
    "the water cycle", "how a compiler turns source code into machine code", "why the sky is blue",
    "the history of the printing press", "how neural networks learn from data", "the rules of chess for a beginner",
    "how to start a vegetable garden on a balcony", "the differences between TCP and UDP", "how black holes form",
    "how inflation affects savings", "the plot structure of a classic tragedy", "how a bill becomes a law",
    "how to prepare for a marathon in 16 weeks", "the basics of photography exposure", "how git branching works",
    "what happens during a solar eclipse", "how to negotiate a salary", "the life cycle of a star",
    "how recommendation systems work", "the pros and cons of remote work", "how to write a good cover letter",
    "the theory of plate tectonics", "how DNS resolves a domain name", "how the stock market works",
    "the health effects of sleep deprivation", "how to make sourdough bread", "the history of the internet",
    "how batteries store energy", "the basics of supply and demand", "how airplanes stay in the air",
    "how the immune system fights a cold", "the history of the Silk Road", "how a hash table works",
    "how to plan a two-week trip on a budget", "the greenhouse effect", "how elections are counted",
    "how coffee is grown and roasted", "the basics of music theory", "how a bicycle gear system works",
    "how a search engine ranks pages", "the causes of the 2008 financial crisis", "how tides work",
    "how to learn a new language as an adult", "how antibiotics work and why resistance grows",
    "the basics of personal budgeting", "how a nuclear reactor produces electricity", "how bees make honey",
]


def tmpl_longform(rng):
    t = rng.randrange(8)
    topic = rng.choice(LONG_TOPICS)
    if t == 0:
        return f"Write a detailed, well-structured explanation of {topic} for a curious high-school student. Use headings and give at least three concrete examples."
    if t == 1:
        return f"Write a {rng.choice([400, 500, 600, 800])}-word blog post about {topic}. Open with a hook, cover the key ideas, and end with practical takeaways."
    if t == 2:
        return f"Compare and contrast {topic} with {rng.choice(LONG_TOPICS)}. Discuss at least four points of comparison and summarize which is more important to understand first and why."
    if t == 3:
        return f"Write a short story (about {rng.choice([300, 400, 500])} words) set in {rng.choice(CITIES)} about a character who unexpectedly learns {topic}. Include dialogue and a clear ending."
    if t == 4:
        return f"Create a step-by-step tutorial on {topic}. Number the steps, explain why each step matters, and list common mistakes at the end."
    if t == 5:
        return f"Write a persuasive essay arguing that everyone should understand {topic}. Address at least two counterarguments."
    if t == 6:
        return f"Write a lesson plan for a 45-minute class on {topic}. Include learning goals, a warm-up activity, the main explanation, a group exercise and a short quiz with answers."
    return f"Write an FAQ page about {topic} with at least eight questions. Give each answer two or three sentences and order the questions from basic to advanced."


def fill_templates(pool, rng, cat, gen, want):
    n = 0
    tries = 0
    while n < want and tries < want * 30:
        tries += 1
        if pool.add(f"template_{cat}", cat, gen(rng)):
            n += 1
    return n


# --------------------------------------------------------------------------- tokenizer (offline, exact server match)
def tokenize_texts(texts, tok_bin, model):
    """Tokenizes chat-templated texts with tokenize_prompts. Returns a list of id lists (None on failure)."""
    with tempfile.TemporaryDirectory(prefix="bp2_tok_") as td:
        fin = os.path.join(td, "in.bin")
        fout = os.path.join(td, "out.txt")
        with open(fin, "wb") as f:
            for u in texts:
                b = CHAT_TEMPLATE.format(u=u).encode("utf-8")
                f.write(struct.pack("<I", len(b)))
                f.write(b)
        t0 = time.time()
        r = subprocess.run([tok_bin, model, fin, fout], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"tokenize_prompts failed rc={r.returncode}: {r.stderr[-500:]}")
        out = []
        with open(fout) as f:
            for line in f:
                p = line.split()
                out.append(None if not p or p[0] == "-1" else [int(x) for x in p[1:]])
        if len(out) != len(texts):
            raise RuntimeError(f"tokenize_prompts returned {len(out)} rows for {len(texts)} texts")
        log(f"[tokenize] {len(texts)} prompts in {time.time() - t0:.1f}s")
        return out


# --------------------------------------------------------------------------- main
def load_dedupe(paths):
    seen_text, seen_tok = set(), set()
    n_text = n_tok = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if not isinstance(o, dict):
                    continue
                conv = o.get("conversations") or o.get("messages") or []
                for m in conv:
                    if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                        seen_text.add(norm(m["content"]))
                        n_text += 1
                if o.get("prompt"):
                    seen_text.add(norm(o["prompt"]))
                    n_text += 1
                if isinstance(o.get("tokens"), list) and isinstance(o.get("n_prompt"), int):
                    seen_tok.add(tuple(o["tokens"][: o["n_prompt"]]))
                    n_tok += 1
                if isinstance(o.get("prompt_ids"), list):
                    seen_tok.add(tuple(o["prompt_ids"]))
                    n_tok += 1
        log(f"[dedupe] {p}: text={n_text} tokens={n_tok} (cumulative)")
    return seen_text, seen_tok, n_text, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--total", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dedupe", nargs="*", default=[])
    ap.add_argument("--tokenizer-bin", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenize_prompts"))
    ap.add_argument("--tokenizer-model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-chars", type=int, default=None, help="default 2600 with the tokenizer, 2000 without")
    ap.add_argument("--ultrachat-scan", type=int, default=50000)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    use_tok = os.path.isfile(args.tokenizer_bin) and os.access(args.tokenizer_bin, os.X_OK) and os.path.isfile(args.tokenizer_model)
    max_chars = args.max_chars or (2600 if use_tok else 2000)
    if not use_tok:
        log(f"[warn] tokenizer not available ({args.tokenizer_bin} / {args.tokenizer_model}); only the {max_chars}-char limit applies")

    seen_text, seen_tok, n_dd_text, n_dd_tok = load_dedupe(args.dedupe)
    log(f"[dedupe] {n_dd_text} text prompts and {n_dd_tok} token-id prompts loaded from {args.dedupe}")

    T = args.total
    target = {"math": round(T * 0.25), "reasoning": round(T * 0.25), "chat": round(T * 0.20),
              "code": round(T * 0.15), "longform": T - round(T * 0.25) * 2 - round(T * 0.20) - round(T * 0.15)}
    log(f"[target] {target}")
    pool = Pool(seen_text, max_chars)

    # intended shares per source inside each category; templates are a small fixed share
    shares = {
        "math": {"gsm8k": 0.50, "hendrycks_math": 0.35, "platypus_math": 0.15},
        "reasoning": {"platypus_reason": 0.45, "arc_challenge": 0.13, "arc_easy": 0.07, "logiqa": 0.10,
                      "strategyqa": 0.10, "template_reasoning": 0.10},
        "chat": {"no_robots": 0.40, "dolly": 0.35, "ultrachat": 0.25},
        "code": {"codealpaca_fresh": 0.65, "evol_code": 0.35},
        "longform": {"no_robots_gen": 0.40, "dolly_creative": 0.20, "ultrachat_long": 0.30, "template_longform": 0.10},
    }
    # oversample each HF source (x1.3): the token-length filter and the token dedupe drop some rows
    ov = 1.3

    def want(cat, src):
        return int(target[cat] * shares[cat][src] * ov)

    skipped = {}
    for repo in OPTIONAL_LOCAL_ONLY:
        skipped[repo] = "used only when cached locally; not in the HF cache" if not is_cached_locally(repo) else "cached but not wired"
    log(f"[optional] {skipped}")

    src_gsm8k(pool, rng, want("math", "gsm8k"))
    src_math(pool, rng, want("math", "hendrycks_math"))
    src_platypus(pool, rng, want("reasoning", "platypus_reason"), want("math", "platypus_math"))
    src_arc(pool, rng, want("reasoning", "arc_challenge") + want("reasoning", "arc_easy"))
    src_logiqa(pool, rng, want("reasoning", "logiqa"))
    src_strategyqa(pool, rng, want("reasoning", "strategyqa"))
    src_no_robots(pool, rng, want("chat", "no_robots"), want("longform", "no_robots_gen"))
    src_dolly(pool, rng, want("chat", "dolly"), want("longform", "dolly_creative"))
    src_ultrachat(pool, rng, want("chat", "ultrachat"), want("longform", "ultrachat_long"), scan=args.ultrachat_scan)
    src_codealpaca(pool, rng, want("code", "codealpaca_fresh"))
    src_evol_code(pool, rng, want("code", "evol_code"))
    fill_templates(pool, rng, "reasoning", tmpl_reasoning, int(target["reasoning"] * shares["reasoning"]["template_reasoning"] * 1.1))
    fill_templates(pool, rng, "longform", tmpl_longform, int(target["longform"] * shares["longform"]["template_longform"] * 1.1))
    log("[collected] " + json.dumps({f"{c}/{s}": len(v) for (c, s), v in sorted(pool.by_source.items())}))
    log(f"[collected] rejected: {pool.rejected}")

    # ---- token step: exact chat-templated length + dedupe against token-id records + within the set
    drops = {"too_long": {}, "too_short": {}, "dup_tokens": {}, "tok_fail": {}}
    ids_of = {}  # text -> prompt_ids

    def token_filter(items):
        """items: list of (cat, src, text). Returns the kept items and fills ids_of."""
        if not use_tok:
            return items
        toks = tokenize_texts([t for _, _, t in items], args.tokenizer_bin, args.tokenizer_model)
        kept = []
        for (cat, src, text), ids in zip(items, toks):
            key = f"{cat}/{src}"
            if ids is None:
                drops["tok_fail"][key] = drops["tok_fail"].get(key, 0) + 1
                continue
            if len(ids) > args.max_tokens:
                drops["too_long"][key] = drops["too_long"].get(key, 0) + 1
                continue
            if len(ids) <= 4:
                drops["too_short"][key] = drops["too_short"].get(key, 0) + 1
                continue
            tk = tuple(ids)
            if tk in seen_tok:
                drops["dup_tokens"][key] = drops["dup_tokens"].get(key, 0) + 1
                continue
            seen_tok.add(tk)
            ids_of[text] = ids
            kept.append((cat, src, text))
        return kept

    all_items = [(c, s, t) for (c, s), v in pool.by_source.items() for t in v]
    all_items = token_filter(all_items)
    cands = {}
    for c, s, t in all_items:
        cands.setdefault(c, {}).setdefault(s, []).append(t)
    log("[after-token-filter] " + json.dumps({f"{c}/{s}": len(v) for c in sorted(cands) for s, v in sorted(cands[c].items())}))
    log(f"[after-token-filter] drops: {json.dumps(drops)}")

    # ---- assemble per category: proportional shares first, then round-robin over the leftovers, then templates
    gens = {"math": tmpl_math, "reasoning": tmpl_reasoning, "longform": tmpl_longform}
    out = []
    counts = {}
    for cat, tgt in target.items():
        srcs = {s: list(v) for s, v in cands.get(cat, {}).items()}
        for v in srcs.values():
            rng.shuffle(v)
        chosen = []
        for s, share in shares[cat].items():
            take = min(len(srcs.get(s, [])), int(round(tgt * share)), tgt - len(chosen))
            for _ in range(take):
                chosen.append((s, srcs[s].pop()))
        assert sum(shares[cat].values()) <= 1.0 + 1e-9, f"shares of {cat} exceed 1.0"
        while len(chosen) < tgt and any(srcs.values()):
            for s in list(srcs):
                if srcs[s] and len(chosen) < tgt:
                    chosen.append((s, srcs[s].pop()))
        if len(chosen) < tgt:
            if cat in gens:
                need = tgt - len(chosen)
                log(f"[warn] category {cat}: {len(chosen)}/{tgt} from HF sources; topping up {need} from templates")
                before = pool.count(cat, f"template_{cat}")
                fill_templates(pool, rng, cat, gens[cat], need + need // 5 + 5)
                extra = pool.by_source[(cat, f"template_{cat}")][before:]
                extra = token_filter([(cat, f"template_{cat}", t) for t in extra])
                for _, s, t in extra:
                    if len(chosen) >= tgt:
                        break
                    chosen.append((s, t))
            if len(chosen) < tgt:
                log(f"[warn] category {cat}: only {len(chosen)}/{tgt}")
        for s, t in chosen:
            out.append({"category": cat, "source": s, "text": t})
            counts.setdefault(cat, {}).setdefault(s, 0)
            counts[cat][s] += 1

    rng.shuffle(out)
    stem = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    ids_path = stem + "_ids.jsonl"
    n_tok_total = 0
    max_tok = 0
    with open(args.out, "w") as f, (open(ids_path, "w") if use_tok else open(os.devnull, "w")) as fi:
        for i, r in enumerate(out):
            rid = f"b2_{r['source']}_{i}"
            rec = {"id": rid, "category": r["category"], "source": r["source"],
                   "messages": [{"role": "user", "content": r["text"]}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if use_tok:
                ids = ids_of[r["text"]]
                n_tok_total += len(ids)
                max_tok = max(max_tok, len(ids))
                fi.write(json.dumps({"id": rid, "category": r["category"], "source": r["source"], "prompt_ids": ids}) + "\n")
    summary = {"total": len(out), "target": target, "per_category": {c: sum(v.values()) for c, v in counts.items()},
               "per_source": counts, "dedupe_against": {"files": args.dedupe, "text_prompts": n_dd_text, "token_prompts": n_dd_tok},
               "rejected_at_collect": pool.rejected, "dropped_at_token_filter": drops,
               "avg_chars": sum(len(r["text"]) for r in out) / max(1, len(out)),
               "avg_tokens": (n_tok_total / max(1, len(out))) if use_tok else None,
               "max_tokens": max_tok if use_tok else None, "max_tokens_limit": args.max_tokens if use_tok else None,
               "max_chars_limit": max_chars, "seed": args.seed, "optional_sources_skipped": skipped,
               "ids_file": ids_path if use_tok else None}
    with open(args.out + ".counts.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))
    log(f"[done] wrote {len(out)} prompts -> {args.out}" + (f" and {ids_path}" if use_tok else ""))


if __name__ == "__main__":
    main()
