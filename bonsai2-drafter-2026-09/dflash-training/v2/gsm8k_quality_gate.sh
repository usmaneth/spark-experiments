#!/usr/bin/env bash
# Detached: after the final clean eval finishes on spark2, run a GSM8K exact-match quality gate for typical acceptance.
V2=/home/usman/Bonsai-demo/dflash-training/v2; echo "[$(date +%H:%M:%S)] gsm8k gate waiter started (chained after FINAL EVAL FINISHED; cap 10h)"
for i in $(seq 1 7200); do grep -q 'FINAL EVAL FINISHED' $V2/logs/full1_final_eval.log 2>/dev/null && break; sleep 5; done
grep -q 'FINAL EVAL FINISHED' $V2/logs/full1_final_eval.log 2>/dev/null || { echo "[$(date +%H:%M:%S)] TIMEOUT waiting for final eval"; exit 1; }
echo "[$(date +%H:%M:%S)] final eval done -> preparing gsm8k on spark2"
cat > /tmp/gsm8k_gate_spark2.sh <<'EOS'
set -u; cd /home/usman/Bonsai-demo; V2=/home/usman/Bonsai-demo/dflash-training/v2; mkdir -p $V2/gsm8k
# 1) fetch 50 GSM8K test problems via the container's datasets lib
sudo docker run --rm -v /home/usman:/home/usman bonsai/dflash-trainer:latest python3 -c "
from datasets import load_dataset; import json
ds=load_dataset('gsm8k','main',split='test'); n=50
with open('$V2/gsm8k/problems.jsonl','w') as f:
    for i in range(n):
        q=ds[i]['question']; a=ds[i]['answer'].split('####')[-1].strip().replace(',','')
        f.write(json.dumps({'q':q,'a':a})+'\n')
print('wrote',n)" 2>&1 | tail -1
sudo chown -R usman:p-usman $V2/gsm8k 2>/dev/null
# 2) score exact-match at each tau (K=5, epoch-1 drafter, typ binary)
export LD_LIBRARY_PATH=/home/usman/Bonsai-demo/bin/typ:/home/usman/Bonsai-demo/bin/cuda
B=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf
python3 - <<'PY'
import json,subprocess,re,os,time
V2='/home/usman/Bonsai-demo/dflash-training/v2'; probs=[json.loads(l) for l in open(f'{V2}/gsm8k/problems.jsonl')]
B='models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf'; D='models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf'
def last_num(s):
    m=re.findall(r'-?\d+(?:\.\d+)?', s.replace(',',''))
    return m[-1] if m else None
for tau in ['0','0.02','0.05']:
    ok=0; tot=0; toks=0; secs=0.0; t0=time.time()
    for p in probs:
        prompt=f"<|im_start|>user\n{p['q']}\nSolve step by step and end with the final numeric answer.<|im_end|>\n<|im_start|>assistant\n"
        env=dict(os.environ, SPEC_TYPICAL_TAU=tau)
        r=subprocess.run(['./bin/typ/llama-speculative-simple','-m',B,'-md',D,'--spec-type','draft-dspark','--spec-draft-n-max','5','--spec-draft-p-min','0','-fa','on','-ngl','99','-ngld','999','-c','4096','-n','400','--temp','0','-e','-p',prompt],capture_output=True,text=True,env=env,timeout=600)
        out=r.stdout; ans=out.split('<|im_start|>assistant')[-1]
        m=re.search(r'decoded\s+(\d+) tokens in\s+([\d.]+) seconds',r.stdout+r.stderr)
        if m: toks+=int(m.group(1)); secs+=float(m.group(2))
        pred=last_num(re.sub(r'^[0-9.]+ I .*$','',ans,flags=re.M))
        gold=p['a']
        try: hit = pred is not None and abs(float(pred)-float(gold))<1e-6
        except: hit=False
        ok+=hit; tot+=1
    print(f"GSM8K tau={tau}: exact-match {ok}/{tot} = {100*ok/tot:.1f}%  | decode {toks/secs if secs else 0:.1f} tok/s | wall {time.time()-t0:.0f}s", flush=True)
print("GSM8K GATE DONE")
PY

# --- long-form termination / repetition check (code prompts, 2000 tokens) per tau ---
python3 - <<'PY2'
import subprocess,re,os,collections
B='models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf'; D='models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf'
prompts=[
 "Write a Python function that parses a CSV file of transactions (date, amount, category) and returns the total spent per category as a dict, with error handling for malformed rows and a small unit test. Put all code in one ```python block that runs the tests when executed.",
 "Implement an LRU cache class in Python with get/put in O(1), a docstring, and unit tests that run when the file is executed.",
 "Write a Python script that reads a log file, counts requests per IP, prints the top 10, and includes tests with a temporary file.",
 "Implement Dijkstra's shortest path in Python over an adjacency dict, with a small graph example and assert-based tests.",
 "Write a Python function to validate and normalize email addresses, with edge cases and unit tests.",
]
for tau in ['0','0.02','0.05']:
    term=0; loops=0; toks=0; secs=0.0
    for q in prompts:
        prompt=f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        env=dict(os.environ, SPEC_TYPICAL_TAU=tau)
        r=subprocess.run(['./bin/typ/llama-speculative-simple','-m',B,'-md',D,'--spec-type','draft-dspark','--spec-draft-n-max','5','--spec-draft-p-min','0','-fa','on','-ngl','99','-ngld','999','-c','4096','-n','2000','--temp','0','-e','-p',prompt],capture_output=True,text=True,env=env,timeout=900)
        out=r.stdout+r.stderr; m=re.search(r'decoded\s+(\d+) tokens in\s+([\d.]+) seconds',out)
        n=int(m.group(1)) if m else 0; secs+=float(m.group(2)) if m else 0; toks+=n
        body=re.sub(r'^[0-9.]+ I .*$','',r.stdout.split('<|im_start|>assistant',1)[-1],flags=re.M)
        w=body[-1500:].split(); grams=collections.Counter(' '.join(w[i:i+6]) for i in range(max(0,len(w)-6))); rep=grams.most_common(1)[0][1] if grams else 0
        term += (n < 1990); loops += (rep>=5)
    print(f"LONGCODE tau={tau}: terminated {term}/5 | repetition-loops {loops}/5 | decode {toks/secs if secs else 0:.1f} tok/s", flush=True)
print("LONGCODE GATE DONE")
PY2
EOS
scp -q /tmp/gsm8k_gate_spark2.sh spark2:/tmp/gsm8k_gate_spark2.sh && ssh spark2 'bash /tmp/gsm8k_gate_spark2.sh'
echo "[$(date +%H:%M:%S)] GSM8K GATE FINISHED"
