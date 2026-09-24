#!/usr/bin/env bash
# Detached: wait for batch2 gate (stable>=90s, >=100k tokens), snapshot, run full pipeline.
V2=/home/REDACTED/Bonsai-demo/dflash-training/v2; B2=$V2/feats/batch2.bin; FULL=$V2/feats_full
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
echo "[$(date +%H:%M:%S)] gate waiter started (batch2 stable>=90s AND >=100k tokens; 3h cap)"
toks=0; n=0; off=0
for i in $(seq 1 2160); do
  if [ -f "$B2" ]; then
    age=$(( $(date +%s) - $(stat -c%Y "$B2") )); read toks n off < <(ct "$B2")
    if [ "$age" -ge 90 ] && [ "$toks" -ge 100000 ]; then echo "[$(date +%H:%M:%S)] GATE MET: $n samples / $toks tokens, stable ${age}s"; break; fi
    [ $((i % 36)) -eq 0 ] && echo "[$(date +%H:%M:%S)] batch2: $n samples / $toks tokens (age ${age}s)"
  fi
  sleep 5
done
[ "$toks" -ge 100000 ] || { echo "[$(date +%H:%M:%S)] TIMEOUT: gate never met (tokens=$toks)"; exit 1; }
rm -rf "$FULL"; mkdir -p "$FULL"; cp "$V2/feats/batch1.bin" "$FULL/batch1.bin"
python3 -c "src,dst,off='$B2','$FULL/batch2.bin',$off;open(dst,'wb').write(open(src,'rb').read(off));print('batch2 snapshot bytes',off)"
ls -la "$FULL"
echo "[$(date +%H:%M:%S)] LAUNCH full retrain: 2 epochs on batch1+batch2"
bash "$V2/run_full_pipeline.sh" full1 "$FULL" 2
echo "[$(date +%H:%M:%S)] PIPELINE FINISHED exit=$?"
