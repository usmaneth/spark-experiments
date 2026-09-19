#!/usr/bin/env bash
# Detached: after the long-code rows finish, fetch openai/gsm8k test (50) on spark2 and score exact-match at tau 0/0.02/0.05.
V2=/home/usman/Bonsai-demo/dflash-training/v2; echo "[$(date +%H:%M:%S)] gsm8k v2 waiter (chained after LONGCODE GATE DONE; cap 8h)"
for i in $(seq 1 5760); do grep -q 'LONGCODE GATE DONE' $V2/logs/gsm8k_gate.log 2>/dev/null && break; sleep 5; done
grep -q 'LONGCODE GATE DONE' $V2/logs/gsm8k_gate.log 2>/dev/null || { echo "TIMEOUT waiting for long-code rows"; exit 1; }
cat > /tmp/gsm8k_v2_spark2.sh <<'EOS'
set -u; cd /home/usman/Bonsai-demo; V2=/home/usman/Bonsai-demo/dflash-training/v2; mkdir -p $V2/gsm8k
sudo docker run --rm -v /home/usman:/home/usman bonsai/dflash-trainer:latest python3 -c "
import json,sys
try:
    from datasets import load_dataset; ds=load_dataset('openai/gsm8k','main',split='test')
except Exception as e:
    print('FETCH FAILED:',repr(e)[:300]); sys.exit(1)
with open('$V2/gsm8k/problems.jsonl','w') as f:
    for i in range(50):
        f.write(json.dumps({'q':ds[i]['question'],'a':ds[i]['answer'].split('####')[-1].strip().replace(',','')})+'\n')
print('wrote 50 gsm8k test problems')" 2>&1 | tail -2
sudo chown -R usman:p-usman $V2/gsm8k 2>/dev/null; [ -s $V2/gsm8k/problems.jsonl ] || { echo "GSM8K DATA MISSING"; exit 1; }
export LD_LIBRARY_PATH=/home/usman/Bonsai-demo/bin/typ:/home/usman/Bonsai-demo/bin/cuda
python3 - <<'PY'
import json,subprocess,re,os,time
V2='/home/usman/Bonsai-demo/dflash-training/v2'; probs=[json.loads(l) for l in open(f'{V2}/gsm8k/problems.jsonl')]
B='models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf'; D='models/bonsai2-dspark/bonsai2-dspark-full1ep1-Q4_K_M.gguf'
def last_num(s):
    m=re.findall(r'-?\d+(?:\.\d+)?', s.replace(',','')); return m[-1] if m else None
for tau in ['0','0.02','0.05']:
    ok=0; tot=0; t0=time.time()
    for p in probs:
        prompt=f"<|im_start|>user\n{p['q']}\nSolve step by step and end with the final numeric answer.<|im_end|>\n<|im_start|>assistant\n"
        r=subprocess.run(['./bin/typ/llama-speculative-simple','-m',B,'-md',D,'--spec-type','draft-dspark','--spec-draft-n-max','5','--spec-draft-p-min','0','-fa','on','-ngl','99','-ngld','999','-c','4096','-n','500','--temp','0','-e','-p',prompt],capture_output=True,text=True,env=dict(os.environ,SPEC_TYPICAL_TAU=tau),timeout=600)
        ans=re.sub(r'^[0-9.]+ I .*$','',r.stdout.split('<|im_start|>assistant',1)[-1],flags=re.M); ans=ans.split('</think>')[-1] if '</think>' in ans else ans
        pred=last_num(ans); gold=p['a']
        try: hit = pred is not None and abs(float(pred)-float(gold))<1e-6
        except: hit=False
        ok+=hit; tot+=1
    print(f"GSM8K tau={tau}: exact-match {ok}/{tot} = {100*ok/tot:.1f}% | wall {time.time()-t0:.0f}s", flush=True)
print("GSM8K V2 DONE")
PY
EOS
scp -q /tmp/gsm8k_v2_spark2.sh spark2:/tmp/gsm8k_v2_spark2.sh && ssh spark2 'bash /tmp/gsm8k_v2_spark2.sh'; echo "[$(date +%H:%M:%S)] GSM8K V2 FINISHED"
