#!/usr/bin/env python3
"""build_prompts_tools.py - assemble the tool-calling / agent-turn prompt set for the drafter self-distillation.

Output: <out.jsonl> with one record per line:
    {"id": "t_<shape>_<n>", "shape": "tool" | "agent" | "mixed",
     "category": "tool_call" | "agent_call" | "agent_report" | "no_call" | "clarify",
     "source": "<target tool | scenario tag | template id>", "expect": {...},
     "messages": [...], "tools": [...], "reasoning_effort": "template" | "low" | "medium" | "xhigh",
     "raw_prompt": "<the exact text /apply-template returned>", "n_tokens": <templated token count>}
plus <out>.counts.json (counts, histograms, token stats, dedupe numbers)
plus <out stem>_ids.jsonl with {"id", "category", "source", "prompt_ids"} (the tag_gen_records.py join format).

Shapes (default 1,400 / 800 / 800):
    tool   one user request, a tools list of 3-6 definitions from the catalog in tools_catalog.py; the natural
           answer is one tool call of the target tool.
    agent  a coding/ops agent system prompt (agent_prompts.py, 6 variants), a task, one prior tool call and a
           fake result of 300-700 tokens (agent_scenarios_*.py); the natural answer is the next call or a report.
    mixed  tools are present but the natural answer is a normal reply (no call) or a clarifying question.

Every record is rendered through the server's /apply-template with {"messages", "tools"} so raw_prompt is the
exact text llama-server builds for /v1/chat/completions. The token count comes from /tokenize (add_special
false, with_pieces true; the pieces must join back to the text). Records over --max-tokens are dropped.
Dedupe: normalized user text against every --dedupe file and within the set; the benchmark prompt texts in
--bench are excluded (exact and substring). The only network use is /apply-template and /tokenize.

usage: build_prompts_tools.py <out.jsonl> [--server http://10.99.0.2:8096] [--seed 1] [--n-tool 1400]
           [--n-agent 800] [--n-mixed 800] [--max-tokens 3600] [--dedupe f1.jsonl ...] [--bench prompts.json]
           [--workers 6] [--effort-mix template:1.0]
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools_catalog as tc  # noqa: E402
import agent_prompts as ap  # noqa: E402
import agent_scenarios_code as sc  # noqa: E402
import agent_scenarios_ops as so  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCH = os.path.join(HERE, "..", "eval", "spec_bench_full", "prompts.json")
RESULT_MIN, RESULT_MAX = 300, 700
SCENARIOS = {"fern": sc.FERN, "pixel": sc.PIXEL, "anvil": sc.ANVIL, "ridge": so.RIDGE, "quill": so.QUILL, "grain": so.GRAIN}
# starting unit count per generator (calibrated so the first draw lands near 500 tokens)
N0 = {"fern_grep_markers": 24, "fern_grep_symbol": 24, "fern_pytest": 14, "fern_ruff": 14, "fern_mypy": 14, "fern_list_dir": 20,
      "fern_git_diff": 16, "fern_git_log": 14, "fern_logs": 14, "pixel_vitest": 20, "pixel_eslint": 16, "pixel_tsc": 12, "pixel_search": 22,
      "pixel_build": 24, "pixel_list": 30, "anvil_go_test": 30, "anvil_cargo_test": 20, "anvil_go_build": 36, "anvil_clippy": 18, "anvil_rg": 24,
      "anvil_race": 40, "anvil_git_diff": 24, "ridge_get_pods": 8, "ridge_logs": 12, "ridge_describe": 8, "ridge_top": 14, "ridge_rollout": 24,
      "ridge_probe": 12, "ridge_manifest": 24, "quill_git_log": 14, "quill_markdownlint": 14, "quill_linkcheck": 40, "quill_find_files": 30,
      "quill_read_page": 40, "quill_ascii": 20, "quill_git_diff": 40, "grain_job_status": 6, "grain_query": 14, "grain_list_bucket": 7,
      "grain_read_csv": 14, "grain_python": 30, "grain_transform_sql": 40}

SPRINT_NOTES = [
    "The release branch is frozen until Friday. Only bug fixes land on it.",
    "The staging database is a copy from 2026-09-12. Numbers there are stale.",
    "The retry helper changed its signature last week. Old call sites may still pass a positional timeout.",
    "The CI runner has 4 cores. Keep test parallelism at 4.",
    "The team renamed the tenant field to workspace_id in the API. Internal code still says tenant_id.",
    "The on-call engineer this week is reachable through the coordinator only.",
    "Coverage must not drop below 82 percent on the touched packages.",
    "The log format changed to JSON on 2026-09-15. Older log files are plain text.",
    "Feature flags live in the config service. A flag that is unset reads as false.",
    "The export job runs at 03:00 UTC. Avoid heavy queries between 03:00 and 04:00 UTC.",
    "Secrets rotate every 30 days. A 401 from an internal service usually means a stale secret.",
    "Snapshot tests are frozen for the design refresh. Do not update snapshots this sprint.",
    "The memory node pool is at capacity. New pods that request more than 4 Gi stay Pending.",
    "The docs site deploys from main every hour. A broken build blocks the deploy.",
    "The parser crate must stay no_std compatible. Do not add a std only dependency.",
    "The warehouse replica lags by up to 15 minutes. Use analytics for freshness checks.",
]
CLARIFY_PREFIX = ["", "", "", "Hey, ", "Quick one: ", "When you get a second, ", "Please ", "Can you ", "Could you ", "I need you to ", "Next: ", "Also, ", "One more thing - "]
CLARIFY_SUFFIX = ["", "", "", " Thanks.", " asap.", " today if possible.", " - let me know when done.", " (same as last time)", " before the standup.", " and confirm."]
NOCALL_PREFIX = ["", "", "", "Question: ", "Not urgent: ", "Curious - ", "For my notes: ", "Help me understand something. ", "Quick sanity check: "]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


class Server:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.calls = 0

    def post(self, path, obj, timeout=120):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
                self.calls += 1
                return out
            except Exception as e:  # transient network errors: retry with a pause
                if attempt == 3:
                    raise
                time.sleep(0.5 * (attempt + 1))

    def apply_template(self, messages, tools, effort):
        body = {"messages": messages, "tools": tools}
        if effort != "template":
            body["chat_template_kwargs"] = {"reasoning_effort": effort}
        return self.post("/apply-template", body)["prompt"]

    def tokenize(self, text, with_pieces=False):
        r = self.post("/tokenize", {"content": text, "add_special": False, "with_pieces": with_pieces})
        if not with_pieces:
            return r["tokens"], None
        ids, pieces = [], []
        for t in r["tokens"]:
            ids.append(t["id"])
            p = t["piece"]
            pieces.append(p if isinstance(p, str) else bytes(p).decode("utf-8", "replace"))
        return ids, "".join(pieces)

    def ntok(self, text):
        return len(self.tokenize(text)[0])


# --------------------------------------------------------------------------- dedupe sources
def load_dedupe(paths):
    seen = set()
    n = 0
    for p in paths:
        if not os.path.isfile(p):
            log(f"[dedupe] missing file {p}")
            continue
        with open(p) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if not isinstance(o, dict):
                    continue
                for m in o.get("conversations") or o.get("messages") or []:
                    if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                        seen.add(norm(m["content"]))
                        n += 1
                if o.get("prompt"):
                    seen.add(norm(o["prompt"]))
                    n += 1
        log(f"[dedupe] {p}: {n} user texts (cumulative)")
    return seen, n


def load_bench(path):
    """All prompt texts of the benchmark (62), the 16 tool+agent ones flagged, plus the tool result lines and the system prompt."""
    if not os.path.isfile(path):
        log(f"[bench] missing {path}")
        return [], set(), 0
    d = json.load(open(path))
    texts = []
    for p in d.get("prompts", []):
        texts.append(p["prompt"])
        for line in p.get("tool_result") or []:
            texts.append(line)
    if isinstance(d.get("agent_system"), list):
        texts.append("\n".join(d["agent_system"]))
    n_ta = sum(1 for p in d.get("prompts", []) if p.get("set") in ("tool", "agent"))
    return texts, {norm(t) for t in texts}, n_ta


class Dedupe:
    def __init__(self, seen_text, bench_texts, bench_norm):
        self.seen = set(seen_text)
        self.bench_norm = bench_norm
        self.bench_long = [norm(t) for t in bench_texts if len(t) >= 30]
        self.raw_seen = set()
        self.rejected = {"dup_text_external": 0, "dup_text_internal": 0, "bench_exact": 0, "bench_substring": 0, "dup_raw": 0}
        self.internal = set()

    def ok_text(self, text):
        k = norm(text)
        if k in self.bench_norm:
            self.rejected["bench_exact"] += 1
            return False
        if len(k) >= 30 and any(b in k or k in b for b in self.bench_long):
            self.rejected["bench_substring"] += 1
            return False
        if k in self.seen:
            self.rejected["dup_text_external"] += 1
            return False
        if k in self.internal:
            self.rejected["dup_text_internal"] += 1
            return False
        self.internal.add(k)
        return True

    def ok_raw(self, raw):
        k = norm(raw)
        if k in self.raw_seen:
            self.rejected["dup_raw"] += 1
            return False
        self.raw_seen.add(k)
        return True


# --------------------------------------------------------------------------- candidate generators
def pick_effort(rng, mix):
    x = rng.random()
    acc = 0.0
    for name, w in mix:
        acc += w
        if x < acc:
            return name
    return mix[-1][0]


def gen_tool(rng, dd, want):
    """Shape (a): one request per record, target tools round-robin so the usage is flat."""
    out = []
    names = list(tc.ALL_TOOL_NAMES)
    rng.shuffle(names)
    i = 0
    tries = 0
    while len(out) < want and tries < want * 20:
        tries += 1
        target = names[i % len(names)]
        i += 1
        text = rng.choice(tc.REQUESTS[target])(rng)
        if not text.isascii() or not dd.ok_text(text):
            continue
        k = rng.randint(3, 6)
        tools = tc.sample_tools(rng, [target], k)
        msgs = []
        if rng.random() < 0.3:
            msgs.append({"role": "system", "content": rng.choice(tc.SYSTEM_SHORT)})
        msgs.append({"role": "user", "content": text})
        out.append({"shape": "tool", "category": "tool_call", "source": target, "expect": {"kind": "tool_call", "tool": target},
                    "messages": msgs, "tools": tools})
    return out


def gen_mixed(rng, dd, want):
    """Shape (c): half no-call, half clarify."""
    out = []
    tries = 0
    want_nc = want // 2
    want_cl = want - want_nc
    n_nc = n_cl = 0
    while (n_nc < want_nc or n_cl < want_cl) and tries < want * 40:
        tries += 1
        if n_nc < want_nc and (n_cl >= want_cl or rng.random() < 0.5):
            idx = rng.randrange(len(tc.MIXED_NOCALL))
            text = rng.choice(NOCALL_PREFIX) + tc.MIXED_NOCALL[idx](rng)
            if not text.isascii() or not dd.ok_text(text):
                continue
            tools = tc.sample_tools(rng, [], rng.randint(3, 6))
            msgs = []
            if rng.random() < 0.4:
                msgs.append({"role": "system", "content": rng.choice(tc.SYSTEM_SHORT)})
            msgs.append({"role": "user", "content": text})
            out.append({"shape": "mixed", "category": "no_call", "source": f"nocall_{idx}", "expect": {"kind": "no_call"}, "messages": msgs, "tools": tools})
            n_nc += 1
        else:
            idx = rng.randrange(len(tc.MIXED_CLARIFY))
            body = tc.MIXED_CLARIFY[idx](rng)
            pre = rng.choice(CLARIFY_PREFIX)
            if pre and pre[-1] == " " and pre.strip().lower() in ("please", "can you", "could you", "i need you to"):
                body = body[0].lower() + body[1:]
            text = pre + body + rng.choice(CLARIFY_SUFFIX)
            if not text.isascii() or not dd.ok_text(text):
                continue
            must = [rng.choice(tc.CLARIFY_TOOLS[idx])]
            tools = tc.sample_tools(rng, must, rng.randint(3, 6))
            msgs = []
            if rng.random() < 0.4:
                msgs.append({"role": "system", "content": rng.choice(tc.SYSTEM_SHORT)})
            msgs.append({"role": "user", "content": text})
            out.append({"shape": "mixed", "category": "clarify", "source": f"clarify_{idx}", "expect": {"kind": "clarify", "tool": must[0]}, "messages": msgs, "tools": tools})
            n_cl += 1
    return out


def size_result(srv, rng, gen, stats):
    """Draw a scenario whose result is RESULT_MIN..RESULT_MAX tokens. Returns (scenario, n_result_tokens) or None."""
    n = N0.get(gen.__name__, 16)
    for attempt in range(7):
        s = gen(rng, max(3, n))
        t = srv.ntok(s["result"])
        stats["draws"] += 1
        if RESULT_MIN <= t <= RESULT_MAX:
            return s, t
        target = rng.randint(380, 620)
        n = max(3, min(80, int(round(n * target / max(1, t)))))
        if attempt >= 2 and t < RESULT_MIN:
            n = min(80, n + 6)
    stats["size_fail"] += 1
    return None


def gen_agent(srv, rng, dd, want, stats):
    out = []
    variants = list(ap.VARIANTS)
    i = 0
    tries = 0
    while len(out) < want and tries < want * 6:
        tries += 1
        v = variants[i % len(variants)]
        i += 1
        gen = rng.choice(SCENARIOS[v["name"]])
        r = size_result(srv, rng, gen, stats)
        if r is None:
            continue
        s, n_res = r
        if not s["task"].isascii() or not s["result"].isascii() or not dd.ok_text(s["task"]):
            continue
        system = v["system"]
        if rng.random() < 0.4:
            notes = rng.sample(SPRINT_NOTES, rng.randint(2, 5))
            system = system + "\n\n# Notes for this sprint\n\n" + "\n".join(f"- {x}" for x in notes)
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": s["task"]},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                                     "function": {"name": s["call"]["name"], "arguments": json.dumps(s["call"]["arguments"])}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": s["call"]["name"], "content": s["result"]}]
        cat = "agent_call" if s["kind"] == "call" else "agent_report"
        out.append({"shape": "agent", "category": cat, "source": s["tag"], "expect": {"kind": "tool_call" if s["kind"] == "call" else "report", "variant": v["name"], "prior_tool": s["call"]["name"]},
                    "messages": msgs, "tools": v["tools"], "n_result_tokens": n_res})
    return out


# --------------------------------------------------------------------------- render + tokenize
def render_one(srv, rec):
    try:
        prompt = srv.apply_template(rec["messages"], rec["tools"], rec["reasoning_effort"])
        ids, joined = srv.tokenize(prompt, with_pieces=True)
        if joined != prompt:
            return rec, None, None, "roundtrip"
        if not prompt.isascii():
            return rec, prompt, ids, "nonascii"
        return rec, prompt, ids, None
    except Exception as e:
        return rec, None, None, f"error:{type(e).__name__}"


def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("out")
    apr.add_argument("--server", default="http://10.99.0.2:8096")
    apr.add_argument("--seed", type=int, default=1)
    apr.add_argument("--n-tool", type=int, default=1400)
    apr.add_argument("--n-agent", type=int, default=800)
    apr.add_argument("--n-mixed", type=int, default=800)
    apr.add_argument("--max-tokens", type=int, default=3600)
    apr.add_argument("--dedupe", nargs="*", default=[os.path.join(HERE, "prompts_broad2.jsonl")])
    apr.add_argument("--bench", default=DEFAULT_BENCH)
    apr.add_argument("--workers", type=int, default=6)
    apr.add_argument("--effort-mix", default="template:1.0", help="e.g. template:0.7,medium:0.15,low:0.15")
    apr.add_argument("--oversample", type=float, default=1.12)
    args = apr.parse_args()
    rng = random.Random(args.seed)
    srv = Server(args.server)
    t_start = time.time()

    props = srv.post("/props", {}) if False else json.loads(urllib.request.urlopen(args.server.rstrip("/") + "/props", timeout=30).read())
    log(f"[server] {args.server} model={props.get('model_path')} build={props.get('build_info')} template_len={len(props.get('chat_template', ''))}")

    mix = []
    for part in args.effort_mix.split(","):
        name, w = part.split(":")
        assert name in ("template", "low", "medium", "xhigh"), name
        mix.append((name, float(w)))
    assert abs(sum(w for _, w in mix) - 1.0) < 1e-6, "effort mix must sum to 1"

    seen_text, n_dd = load_dedupe(args.dedupe)
    bench_texts, bench_norm, n_bench_ta = load_bench(args.bench)
    log(f"[bench] {len(bench_texts)} texts loaded from {args.bench} ({n_bench_ta} tool+agent prompts)")
    dd = Dedupe(seen_text, bench_texts, bench_norm)

    # ---- candidates (oversampled: the token cap and the raw dedupe drop a few)
    ov = args.oversample
    stats_agent = {"draws": 0, "size_fail": 0}
    t0 = time.time()
    cand_tool = gen_tool(rng, dd, int(args.n_tool * ov))
    log(f"[cand] tool: {len(cand_tool)} ({time.time() - t0:.1f}s)")
    t0 = time.time()
    cand_mixed = gen_mixed(rng, dd, int(args.n_mixed * ov))
    log(f"[cand] mixed: {len(cand_mixed)} ({time.time() - t0:.1f}s)")
    t0 = time.time()
    cand_agent = gen_agent(srv, rng, dd, int(args.n_agent * ov), stats_agent)
    log(f"[cand] agent: {len(cand_agent)} draws={stats_agent['draws']} size_fail={stats_agent['size_fail']} ({time.time() - t0:.1f}s)")
    cands = cand_tool + cand_agent + cand_mixed
    for c in cands:
        c["reasoning_effort"] = pick_effort(rng, mix)

    # ---- render through the server template, tokenize, cap, raw dedupe
    t0 = time.time()
    drops = {"too_long": {}, "roundtrip": 0, "nonascii": 0, "error": 0}
    kept = {"tool": [], "agent": [], "mixed": []}
    with ThreadPoolExecutor(args.workers) as ex:
        for rec, prompt, ids, err in ex.map(lambda r: render_one(srv, r), cands):
            if err == "roundtrip":
                drops["roundtrip"] += 1
                continue
            if err == "nonascii":
                drops["nonascii"] += 1
                continue
            if err:
                drops["error"] += 1
                continue
            if len(ids) > args.max_tokens:
                drops["too_long"][rec["shape"]] = drops["too_long"].get(rec["shape"], 0) + 1
                continue
            if not dd.ok_raw(prompt):
                continue
            rec["raw_prompt"] = prompt
            rec["n_tokens"] = len(ids)
            rec["_ids"] = ids
            kept[rec["shape"]].append(rec)
    log(f"[render] kept tool={len(kept['tool'])} agent={len(kept['agent'])} mixed={len(kept['mixed'])} drops={json.dumps(drops)} "
        f"server_calls={srv.calls} ({time.time() - t0:.1f}s)")

    # ---- assemble: exact shape counts, mixed split kept even
    want = {"tool": args.n_tool, "agent": args.n_agent, "mixed": args.n_mixed}
    out = []
    for shape in ("tool", "agent", "mixed"):
        pool = kept[shape]
        rng.shuffle(pool)
        if shape == "mixed":
            nc = [r for r in pool if r["category"] == "no_call"][: want["mixed"] // 2]
            cl = [r for r in pool if r["category"] == "clarify"][: want["mixed"] - len(nc)]
            chosen = nc + cl
        else:
            chosen = pool[: want[shape]]
        if len(chosen) < want[shape]:
            log(f"[warn] shape {shape}: only {len(chosen)}/{want[shape]}")
        out.extend(chosen)
    rng.shuffle(out)

    stem = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    ids_path = stem + "_ids.jsonl"
    per_shape, per_cat, per_source = {}, {}, {}
    tool_present, tool_target = {}, {}
    tok_by_shape = {}
    res_tokens = []
    effort_hist = {}
    with open(args.out, "w") as f, open(ids_path, "w") as fi:
        for i, r in enumerate(out):
            rid = f"t_{r['shape']}_{i}"
            rec = {"id": rid, "shape": r["shape"], "category": r["category"], "source": r["source"], "expect": r["expect"],
                   "messages": r["messages"], "tools": r["tools"], "reasoning_effort": r["reasoning_effort"],
                   "raw_prompt": r["raw_prompt"], "n_tokens": r["n_tokens"]}
            if "n_result_tokens" in r:
                rec["n_result_tokens"] = r["n_result_tokens"]
                res_tokens.append(r["n_result_tokens"])
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            fi.write(json.dumps({"id": rid, "category": r["category"], "source": r["source"], "prompt_ids": r["_ids"]}) + "\n")
            per_shape[r["shape"]] = per_shape.get(r["shape"], 0) + 1
            per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
            per_source[r["source"]] = per_source.get(r["source"], 0) + 1
            effort_hist[r["reasoning_effort"]] = effort_hist.get(r["reasoning_effort"], 0) + 1
            for t in r["tools"]:
                nm = t["function"]["name"]
                tool_present[nm] = tool_present.get(nm, 0) + 1
            if r["shape"] == "tool":
                tool_target[r["source"]] = tool_target.get(r["source"], 0) + 1
            tok_by_shape.setdefault(r["shape"], []).append(r["n_tokens"])

    def tstats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {"n": len(xs), "min": xs[0], "mean": round(statistics.mean(xs), 1), "median": xs[len(xs) // 2],
                "p90": xs[int(len(xs) * 0.9)], "max": xs[-1]}

    all_tok = [x for v in tok_by_shape.values() for x in v]
    summary = {
        "total": len(out), "per_shape": per_shape, "per_category": per_cat, "per_source": dict(sorted(per_source.items())),
        "tool_target_hist": dict(sorted(tool_target.items(), key=lambda kv: -kv[1])),
        "tool_present_hist": dict(sorted(tool_present.items(), key=lambda kv: -kv[1])),
        "catalog_tools": len(tc.TOOLS), "agent_variants": [v["name"] for v in ap.VARIANTS],
        "agent_system_tokens": None,
        "reasoning_effort_hist": effort_hist,
        "templated_tokens": {"all": tstats(all_tok), **{k: tstats(v) for k, v in tok_by_shape.items()}},
        "agent_result_tokens": tstats(res_tokens),
        "max_tokens_limit": args.max_tokens,
        "dedupe": {"files": args.dedupe, "external_user_texts": n_dd, "bench_file": args.bench, "bench_texts": len(bench_texts),
                   "bench_tool_agent_prompts": n_bench_ta, "rejected": dd.rejected},
        "candidates": {"tool": len(cand_tool), "agent": len(cand_agent), "mixed": len(cand_mixed), "agent_result_draws": stats_agent["draws"],
                       "agent_size_fail": stats_agent["size_fail"]},
        "dropped_at_render": drops, "server": args.server, "server_calls": srv.calls, "seed": args.seed,
        "ids_file": ids_path, "elapsed_s": round(time.time() - t_start, 1),
    }
    summary["agent_system_tokens"] = {v["name"]: srv.ntok(v["system"]) for v in ap.VARIANTS}
    with open(args.out + ".counts.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))
    log(f"[done] wrote {len(out)} prompts -> {args.out} and {ids_path}")


if __name__ == "__main__":
    main()
