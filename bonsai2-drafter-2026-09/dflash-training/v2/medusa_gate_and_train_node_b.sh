#!/usr/bin/env bash
# Detached: wait for batch2 gate (same as DSpark), rsync feats to node_b, train Medusa heads there in the container.
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; B2=$V2/feats/batch2.bin
ct(){ python3 - "$1" <<'EOF'
import struct,os,sys
p=sys.argv[1]; sz=os.path.getsize(p); f=open(p,'rb'); f.read(4); embd,ntaps=struct.unpack('<II',f.read(8)); f.read(20)
per=4+1+ntaps*embd*4+embd*4; off=32; n=0; toks=0
while True:
    f.seek(off); b=f.read(4)
    if len(b)<4: break
    nt=struct.unpack('<I',b)[0]; e=off+4+nt*per
    if e>sz: break
    off=e; n+=1; toks+=nt
print(toks, n, off)
EOF
}
echo "[$(date +%H:%M:%S)] medusa/node_b waiter started"
toks=0; n=0; off=0
for i in $(seq 1 2160); do
  if [ -f "$B2" ]; then age=$(( $(date +%s) - $(stat -c%Y "$B2") )); read toks n off < <(ct "$B2"); [ "$age" -ge 90 ] && [ "$toks" -ge 100000 ] && break; fi
  sleep 5
done
[ "$toks" -ge 100000 ] || { echo "[$(date +%H:%M:%S)] TIMEOUT: gate never met"; exit 1; }
echo "[$(date +%H:%M:%S)] GATE MET: $n samples / $toks tokens -> rsync to node_b"
mkdir -p /home/REDACTED/Bonsai-demo/dflash-training/v2/feats_medusa && python3 -c "src,dst,off='$B2','/home/REDACTED/Bonsai-demo/dflash-training/v2/feats_medusa/batch2.bin',$off;open(dst,'wb').write(open(src,'rb').read(off))"
rsync -a /home/REDACTED/Bonsai-demo/dflash-training/v2/feats_medusa/batch2.bin node_b:$V2/feats/batch2.bin && echo "[$(date +%H:%M:%S)] batch2 on node_b: $(ssh node_b stat -c%s $V2/feats/batch2.bin) bytes"
echo "[$(date +%H:%M:%S)] LAUNCH medusa training on node_b"
ssh node_b "cd $V2/medusa && bash run_train.sh" 2>&1 | tail -40
echo "[$(date +%H:%M:%S)] MEDUSA FINISHED exit=$?"
ssh node_b "ls -la $V2/medusa/medusa_heads.safetensors $V2/medusa/medusa_heads.json 2>&1; cat $V2/medusa/medusa_heads.json 2>/dev/null | head -40"
