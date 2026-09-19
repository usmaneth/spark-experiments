#!/usr/bin/env python3
# Benchmark llama-server aggregate generation throughput with N concurrent /completion requests.
# Usage: bench_server.py <port> <concurrency> <n_predict>
import sys, time, json, urllib.request
from concurrent.futures import ThreadPoolExecutor

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8095
conc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
npred = int(sys.argv[3]) if len(sys.argv) > 3 else 256

prompts = [
    "Write a Python function to reverse a linked list.",
    "Explain how a hash table handles collisions.",
    "Implement binary search in C.",
    "What is the time complexity of quicksort and why?",
    "Write a SQL query to find the second highest salary.",
    "Describe the difference between processes and threads.",
    "Implement a stack using two queues.",
    "What is 47 multiplied by 89? Show your work.",
    "Write a regex to match a valid email address.",
    "Explain the CAP theorem in distributed systems.",
    "Implement merge sort in Python.",
    "How does garbage collection work in modern languages?",
    "Write a function to detect a cycle in a graph.",
    "Explain memoization with an example.",
    "What is a B-tree and where is it used?",
    "Implement fizzbuzz from 1 to 100 in Python.",
]

def one(p):
    body = json.dumps({
        "prompt": f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
        "n_predict": npred, "temperature": 0, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    # tokens_predicted is in timings/top-level
    tp = r.get("tokens_predicted") or r.get("timings", {}).get("predicted_n", 0)
    return tp

sel = [prompts[i % len(prompts)] for i in range(conc)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    toks = list(ex.map(one, sel))
el = time.time() - t0
total = sum(toks)
print(f"concurrency={conc} n_predict={npred} total_gen_tokens={total} wall={el:.1f}s")
print(f"AGGREGATE gen throughput = {total/el:.1f} tok/s   (per-slot {total/el/conc:.1f} tok/s)")
