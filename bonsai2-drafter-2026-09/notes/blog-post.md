---
title: "Bonsai 2 27B: the runtime fix that makes speculative decoding work, a DGX Spark benchmark, and the DSpark v2 recipe"
slug: "2026-09-19-bonsai2-dspark-v2-runtime-fix"
date: "2026-09-19"
kind: "research"
author: "Local Kernel"
summary: "A missing Hadamard transform in the draft graph silently breaks every speculative drafter on Bonsai 2 27B; we fixed the runtime, benchmarked two drafters on a DGX Spark over 62 prompts at exact match, and published the recipe for the one you can rebuild."
tags: ["bonsai-2", "speculative-decoding", "dspark", "dgx-spark", "gb10", "llama.cpp", "ternary", "benchmarks"]
read: "23 min"
---

Bonsai 2 27B ships without a speculative drafter, and the released llama.cpp binaries cannot run one. Any drafter that borrows the target's token embedding and output head accepts 0.4-2.0% of drafted tokens on this target, silently, and decodes slower than with no drafter at all. The cause is a missing Hadamard transform in the draft graph. We fixed it (+48 / -20 on the PrismML fork), and with the fix in place we benchmarked two drafters on a DGX Spark (GB10) over 62 prompts with exact-match verification: PrismML's Ternary-Bonsai-27B drafter re-converted for Bonsai 2 (DSpark v1), and a drafter we trained for this target from an open, reproducible recipe (DSpark v2). Both land at about 1.9x blended over a 29.7 tok/s server baseline (57.7 and 57.4 tok/s), 2.5x on math and 2.1-2.2x on code. They tie. v2 is ahead on code and on 2,000-token long-form, v1 on reasoning, tool calls and agent turns. The fix is the headline: without it nobody runs any drafter on Bonsai 2. The benchmark and the recipe are the second act. DSpark v2 is the drafter you can rebuild from that recipe, with the better code and long-form profile, not a speed win over v1.

## 1. No drafter ships, and the released runtime cannot run one

PrismML ships no drafter for Bonsai 2: `scripts/download_models.sh` fetches none, and the Bonsai 2 GGUF repositories hold none. The nearest thing is the Ternary-Bonsai-27B drafter, trained for the older model and re-converted against the Bonsai 2 target (header `general.name = Bonsai-27B-dspark`, block size 4): DSpark v1 in section 2. On the released runtime, PrismML-Eng/llama.cpp `prism` at `1a07bfa5f` (build 10706) or the release tag `prism-b10683-d8f26ee` that `setup.sh` installs, either drafter generates the correct text at 0.4-2.0% acceptance, slower than with no drafter. No error, no warning. Only the acceptance counter and the speed show it.

The cause. A `dflash` drafter that ships no `token_embd.weight` or `output.weight` borrows the target's through `ctx_other`. Bonsai 2 is the first target with Hadamard-folded weights: the `PQ2_0` file lists `output.weight` among 401 rotated tensors and `token_embd.weight` as an inverse-lookup table. The target's own graph applies the inverse after each embedding lookup and the forward rotation before the head. The draft graph does neither, because its Hadamard maps come from the drafter's own model, which has none. The draft logits are noise, and the coverage check for this defect is skipped for the same reason. The older Ternary-Bonsai-27B GGUFs carry no `prism.hadamard` metadata, so the gap stayed invisible until Bonsai 2.

The fix is five files, +48 / -20, on branch `fix/dflash-borrowed-hadamard`, open as [PrismML-Eng/llama.cpp#210](https://github.com/PrismML-Eng/llama.cpp/pull/210): "dflash: apply the target's Hadamard transforms to borrowed embeddings and head". The draft context merges the target's Hadamard maps with its own, a `build_embd_rows` helper does lookup plus inverse for every embedding read, and the existing head path applies the forward rotation once the map holds the tensor. With the merged maps, the coverage check now catches a borrowed folded tensor at context creation. Measured on `1a07bfa5f`, `llama-speculative-simple`, 200 tokens, temperature 0:

| drafter | K | prompt | clean `1a07bfa5f` | `1a07bfa5f` + fix |
| --- | ---: | --- | ---: | ---: |
| DSpark v2 | 5 | math | 1.6% (15/924) | 52.1% (148/284) |
| DSpark v2 | 5 | code (CSV parser) | 0.4% (4/978) | 42.9% (137/319) |
| DSpark v2 | 5 | code (binary search) | 1.2% (11/940) | 49.5% (143/289) |
| DSpark v1 (Ternary-Bonsai-27B drafter) | 4 | math | 2.0% (15/741) | 56.0% (140/250) |
| DSpark v1 (Ternary-Bonsai-27B drafter) | 4 | code (CSV parser) | 0.9% (7/772) | 46.4% (130/280) |
| DSpark v1 (Ternary-Bonsai-27B drafter) | 4 | code (binary search) | 1.7% (13/746) | 56.6% (142/251) |

<div class="lk-spec" data-lk-spec aria-label="How speculative decoding works">
<style>
.lk-spec{margin:28px 0 32px;padding:18px 18px 14px;border:1px solid var(--line);border-radius:var(--radius-md);background:var(--raised);font-family:var(--mono);color:var(--ink);overflow:hidden}
.lk-spec__head{display:flex;justify-content:space-between;align-items:flex-end;gap:10px 18px;flex-wrap:wrap;margin:0 0 14px}
.lk-spec__title{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0}
.lk-spec__counter{font-size:12px;color:var(--muted);display:flex;align-items:baseline;gap:8px;white-space:nowrap}
.lk-spec__counter b{font-family:var(--sans);font-size:24px;line-height:1;font-weight:500;color:var(--ink);font-variant-numeric:tabular-nums;min-width:3ch}
.lk-spec__row{display:grid;grid-template-columns:78px 1fr;align-items:center;gap:10px;min-height:40px;padding:3px 0}
.lk-spec__lab{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.lk-spec__stream{display:flex;align-items:center;justify-content:flex-end;gap:6px;overflow:hidden;min-width:0;min-height:32px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 14%);mask-image:linear-gradient(90deg,transparent,#000 14%)}
.lk-spec__step{display:flex;align-items:center;gap:6px;min-width:0;min-height:32px;overflow:hidden;padding:5px 0;margin:-5px 0}
.lk-spec__grp{display:flex;align-items:center;gap:6px;position:relative;flex:none}
.lk-spec__tok{display:inline-block;flex:none;font-size:12.5px;line-height:1;padding:8px 8px;border-radius:var(--radius-sm);border:1px solid var(--line-hover);color:var(--muted);background:var(--surface);max-width:160px;overflow:hidden;white-space:pre;transition:background-color .3s var(--ease-out),color .3s var(--ease-out),border-color .3s var(--ease-out),opacity .4s var(--ease-out),transform .4s var(--ease-out),max-width .4s var(--ease-out),padding .4s var(--ease-out),margin .4s var(--ease-out)}
.lk-spec__tok--in{opacity:0;transform:translateY(-8px)}
.lk-spec__tok--ok{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}
.lk-spec__tok--tgt{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.lk-spec__tok--rej{opacity:.5;text-decoration:line-through;border-style:dashed}
.lk-spec__tok--out{opacity:0;transform:translateY(16px);max-width:0;padding-left:0;padding-right:0;margin-left:-6px;border-color:transparent}
.lk-spec__scan{position:absolute;top:-5px;bottom:-5px;left:0;width:0;border-radius:var(--radius-sm);pointer-events:none;opacity:0;background:linear-gradient(90deg,transparent 0%,color-mix(in oklab,var(--accent) 22%,transparent) 100%);border-right:2px solid var(--accent)}
.lk-spec__scan--run{animation:lk-spec-scan .7s cubic-bezier(.4,0,.2,1) forwards}
@keyframes lk-spec-scan{0%{width:0;opacity:1}85%{opacity:1}100%{width:100%;opacity:0}}
.lk-spec__status{font-size:12px;color:var(--muted);min-height:18px;margin:8px 0 0 88px}
.lk-spec__legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:14px 0 0;font-size:11.5px;color:var(--muted)}
.lk-spec__legend span{display:inline-flex;align-items:center;gap:6px}
.lk-spec__legend i{display:inline-block;width:12px;height:12px;border-radius:3px;border:1px solid var(--line-hover);background:var(--surface)}
.lk-spec__legend i.ok{background:var(--accent);border-color:var(--accent)}
.lk-spec__legend i.tgt{background:var(--ink);border-color:var(--ink)}
.lk-spec__legend i.rej{opacity:.5;border-style:dashed}
.lk-spec__cap{font-family:var(--sans);font-size:13px;line-height:1.5;color:var(--muted);margin:12px 0 0;max-width:var(--measure)}
@media (max-width:640px){.lk-spec{padding:14px 14px 12px}.lk-spec__row{grid-template-columns:1fr;gap:4px;min-height:0}.lk-spec__lab{font-size:10px}.lk-spec__status{margin-left:0}.lk-spec__tok{font-size:11.5px;padding:7px 6px}.lk-spec__counter b{font-size:20px}}
@media (prefers-reduced-motion:reduce){.lk-spec *{transition:none!important;animation:none!important}}
</style>
<div class="lk-spec__head"><p class="lk-spec__title">How speculative decoding works</p><div class="lk-spec__counter"><b data-avg>-</b><span data-avglab>tokens per step</span></div></div>
<div class="lk-spec__row"><span class="lk-spec__lab">committed</span><div class="lk-spec__stream" data-stream></div></div>
<div class="lk-spec__row"><span class="lk-spec__lab">this step</span><div class="lk-spec__step" data-step></div></div>
<div class="lk-spec__status" data-status></div>
<div class="lk-spec__legend"><span><i></i>drafted</span><span><i class="ok"></i>accepted draft</span><span><i class="tgt"></i>target's own token</span><span><i class="rej"></i>rejected</span></div>
<p class="lk-spec__cap">Illustrative. The draft model proposes 5 tokens (K=5); the target scores all of them in one verify pass, keeps the longest matching prefix, rejects the first mismatch and everything after it, and appends its own token. The measured value for DSpark v2 is 3.85 tokens per step on the 2,000-token server set.</p>
<script>
(function(){
var roots=document.querySelectorAll('.lk-spec:not([data-ready])');
for(var r=0;r<roots.length;r++)init(roots[r]);
function init(root){
root.setAttribute('data-ready','1');
var stream=root.querySelector('[data-stream]'),step=root.querySelector('[data-step]'),avgEl=root.querySelector('[data-avg]'),avgLab=root.querySelector('[data-avglab]'),status=root.querySelector('[data-status]');
var SEQ='def search ( arr , x ) : lo , hi = 0 , len ( arr ) - 1 while lo <= hi : mid = ( lo + hi ) // 2 if arr [ mid ] == x : return mid elif arr [ mid ] < x : lo = mid + 1 else : hi = mid - 1 return - 1'.split(' ');
var ALT='i n ] for in range None 0 1 ) j k self not and or : if is == len arr x mid lo hi return print'.split(' ');
var ACC=[3,3,2,4,1,5,2,4,0,5,3,1,4,5,2,0,3,5,2,3];
var K=5,pos=0,stepN=0,total=0,alt=0;
var reduce=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
function tok(text,cls){var s=document.createElement('span');s.className='lk-spec__tok'+(cls?' '+cls:'');s.textContent=text;return s;}
function next(i){return SEQ[(pos+i)%SEQ.length];}
function wrong(i){var t=next(i),c=ALT[alt++%ALT.length];if(c===t)c=ALT[alt++%ALT.length];return c;}
function push(text,cls){stream.appendChild(tok(text,cls));while(stream.children.length>40)stream.removeChild(stream.firstChild);}
function sleep(ms){return new Promise(function(res){setTimeout(res,ms);});}
function setAvg(){avgEl.textContent=stepN?(total/stepN).toFixed(2):'-';avgLab.textContent='tokens per step, mean of '+stepN+' step'+(stepN===1?'':'s');}
if(reduce){
for(var i=0;i<10;i++)push(SEQ[i],i%4===3?'lk-spec__tok--tgt':'lk-spec__tok--ok');
pos=10;
var g0=document.createElement('span');g0.className='lk-spec__grp';
for(var j0=0;j0<K;j0++)g0.appendChild(j0<3?tok(next(j0),'lk-spec__tok--ok'):tok(wrong(j0),'lk-spec__tok--rej'));
step.appendChild(g0);step.appendChild(tok(next(3),'lk-spec__tok--tgt'));
avgEl.textContent='3.85';avgLab.textContent='tokens per step (measured)';
status.textContent='3 drafted tokens accepted, 2 rejected, 1 from the target: 4 tokens in one step';
return;
}
var visible=true;
if('IntersectionObserver' in window){visible=false;new IntersectionObserver(function(es){for(var i=0;i<es.length;i++)visible=es[i].isIntersecting;},{threshold:0.05}).observe(root);}
for(var i=0;i<6;i++)push(SEQ[i],i===2?'lk-spec__tok--tgt':'lk-spec__tok--ok');
pos=6;
async function loop(){
while(root.isConnected){
while(!visible&&root.isConnected)await sleep(300);
var a=ACC[stepN%ACC.length];
step.innerHTML='';
var g=document.createElement('span');g.className='lk-spec__grp';
var pills=[];
for(var j=0;j<K;j++){var p=tok(j<a?next(j):wrong(j),'lk-spec__tok--in');g.appendChild(p);pills.push(p);}
var scan=document.createElement('span');scan.className='lk-spec__scan';g.appendChild(scan);
step.appendChild(g);
status.textContent='draft: proposes '+K+' tokens';
for(var j=0;j<K;j++){void pills[j].offsetWidth;pills[j].classList.remove('lk-spec__tok--in');await sleep(70);}
await sleep(420);
status.textContent='verify: the target scores all '+K+' in one pass';
scan.classList.add('lk-spec__scan--run');
await sleep(760);
status.textContent=a===K?'accept: all '+K+' match; the target appends its own token':'accept: '+a+' match, the draft is wrong at position '+(a+1)+'; '+(K-a)+' fall away';
for(var j=0;j<a;j++){pills[j].classList.add('lk-spec__tok--ok');await sleep(60);}
for(var j=a;j<K;j++)pills[j].classList.add('lk-spec__tok--rej');
await sleep(260);
for(var j=a;j<K;j++)pills[j].classList.add('lk-spec__tok--out');
await sleep(380);
var t=tok(next(a),'lk-spec__tok--tgt lk-spec__tok--in');step.appendChild(t);void t.offsetWidth;t.classList.remove('lk-spec__tok--in');
stepN++;total+=a+1;setAvg();
status.textContent=(a+1)+' token'+(a+1===1?'':'s')+' this step: '+a+' accepted + 1 from the target';
await sleep(720);
for(var j=0;j<a;j++)push(next(j),'lk-spec__tok--ok');
push(next(a),'lk-spec__tok--tgt');
pos=(pos+a+1)%SEQ.length;
step.innerHTML='';
await sleep(320);
}
}
loop();
}
})();
</script>
</div>

With the fix, the old drafter transfers to Bonsai 2: about 3.3 tokens per step at K=4, against 3.85 for DSpark v2 at K=5. More tokens per step does not make v2 the faster drafter overall; section 2 has the matrix. `--version` prints `build 10706, commit 1a07bfa5f` for clean and patched binaries alike; tell them apart by path.

## 2. The result

<div class="lk-bench" data-lk-bench aria-label="DGX Spark benchmark, tok/s by workload">
<style>
.lk-bench{margin:8px 0 32px;padding:18px 18px 16px;border:1px solid var(--line);border-radius:var(--radius-md);background:var(--raised);font-family:var(--mono);color:var(--ink);--lab:96px}
.lk-bench__head{display:flex;justify-content:space-between;align-items:flex-end;gap:8px 18px;flex-wrap:wrap;margin:0 0 6px}
.lk-bench__title{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0}
.lk-bench__sub{font-family:var(--sans);font-size:13px;line-height:1.5;color:var(--muted);margin:0 0 14px;max-width:var(--measure)}
.lk-bench__legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:11.5px;color:var(--muted);margin:0 0 6px}
.lk-bench__legend span,.lk-bench__stat>span:first-child{display:inline-flex;align-items:center;gap:6px}
.lk-bench i{display:inline-block;width:12px;height:12px;border-radius:3px;flex:none}
.lk-bench i.base{background:var(--muted)}
.lk-bench i.v1{background:var(--ink)}
.lk-bench i.v2{background:var(--accent)}
.lk-bench__grp{display:grid;grid-template-columns:136px 1fr;gap:10px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.lk-bench__lab{font-size:12.5px;color:var(--ink);line-height:1.3}
.lk-bench__lab small{display:block;font-size:10.5px;color:var(--faint);letter-spacing:.02em;margin-top:2px}
.lk-bench__bars{display:flex;flex-direction:column;gap:4px;min-width:0}
.lk-bench__row{display:flex;align-items:center;gap:8px;height:12px}
.lk-bench__bar{display:block;flex:none;height:10px;border-radius:0 4px 4px 0;width:calc((100% - var(--lab)) * var(--w));transition:width .9s var(--ease-out) var(--d,0ms)}
.lk-bench__bar--base{background:var(--muted)}
.lk-bench__bar--v1{background:var(--ink)}
.lk-bench__bar--v2{background:var(--accent)}
.lk-bench__val{font-size:11.5px;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums;transition:opacity .4s ease calc(var(--d,0ms) + .45s)}
.lk-bench__val b{font-weight:500;color:var(--ink)}
.lk-bench__io.is-wait .lk-bench__bar{width:0;transition:none}
.lk-bench__io.is-wait .lk-bench__val{opacity:0;transition:none}
.lk-bench__tiles{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0 0}
.lk-bench__tile{border:1px solid var(--line);border-radius:var(--radius-sm);padding:12px 14px;background:var(--surface);min-width:0}
.lk-bench__tile h4{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:400;margin:0 0 8px}
.lk-bench__stat{display:flex;justify-content:space-between;gap:10px;align-items:baseline;padding:7px 0;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
.lk-bench__stat:nth-of-type(1){border-top:0}
.lk-bench__stat b{font-family:var(--sans);font-size:17px;font-weight:500;color:var(--ink);white-space:nowrap}
.lk-bench__stat b+b{margin-left:10px}
.lk-bench__stat b small{font-family:var(--mono);font-size:10.5px;color:var(--muted);font-weight:400;margin-left:3px}
.lk-bench__delta{font-family:var(--sans);font-size:12.5px;line-height:1.45;color:var(--muted);margin:10px 0 0}
.lk-bench__mini{display:grid;grid-template-columns:52px 1fr;gap:6px 8px;align-items:center;--lab:44px}
.lk-bench__minilab{font-size:11.5px;color:var(--ink)}
.lk-bench__mini .lk-bench__row{height:10px}
.lk-bench__mini .lk-bench__bar{height:8px}
@media (max-width:640px){.lk-bench{padding:14px 14px 12px;--lab:88px}.lk-bench__grp{grid-template-columns:86px 1fr;gap:8px}.lk-bench__lab{font-size:11.5px}.lk-bench__tiles{grid-template-columns:1fr}.lk-bench__stat b{font-size:15px}}
@media (prefers-reduced-motion:reduce){.lk-bench *{transition:none!important}.lk-bench__io.is-wait .lk-bench__bar{width:calc((100% - var(--lab)) * var(--w))}.lk-bench__io.is-wait .lk-bench__val{opacity:1}}
</style>
<div class="lk-bench__head"><p class="lk-bench__title">DGX Spark, one slot: tok/s by workload</p></div>
<p class="lk-bench__sub">llama-server on one GB10, temperature 0, exact-match verification, 512 output tokens per prompt (the last row: 2,000). Each speedup is the drafter rate over the no-drafter rate for the same prompts on the same harness. The full table with acceptance rates follows below.</p>
<div class="lk-bench__legend"><span><i class="base"></i>no drafter</span><span><i class="v1"></i>DSpark v1 (K=4)</span><span><i class="v2"></i>DSpark v2 (K=5)</span></div>
<div class="lk-bench__chart lk-bench__io">
<div class="lk-bench__grp"><div class="lk-bench__lab">code<small>8 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.7 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3712;--d:0ms"></span><span class="lk-bench__val"><b>29.7</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 62.9 tok/s, 2.12x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.7863;--d:60ms"></span><span class="lk-bench__val"><b>62.9</b> 2.12x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 65.7 tok/s, 2.22x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.8213;--d:120ms"></span><span class="lk-bench__val"><b>65.7</b> 2.22x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">math<small>8 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.7 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3712;--d:70ms"></span><span class="lk-bench__val"><b>29.7</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 74.7 tok/s, 2.52x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.9338;--d:130ms"></span><span class="lk-bench__val"><b>74.7</b> 2.52x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 74.7 tok/s, 2.52x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.9338;--d:190ms"></span><span class="lk-bench__val"><b>74.7</b> 2.52x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">reasoning<small>8 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.7 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3712;--d:140ms"></span><span class="lk-bench__val"><b>29.7</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 60.9 tok/s, 2.05x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.7612;--d:200ms"></span><span class="lk-bench__val"><b>60.9</b> 2.05x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 57.8 tok/s, 1.95x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.7225;--d:260ms"></span><span class="lk-bench__val"><b>57.8</b> 1.95x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">chat<small>8 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.7 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3712;--d:210ms"></span><span class="lk-bench__val"><b>29.7</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 44.4 tok/s, 1.50x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.5550;--d:270ms"></span><span class="lk-bench__val"><b>44.4</b> 1.50x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 44.8 tok/s, 1.51x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.5600;--d:330ms"></span><span class="lk-bench__val"><b>44.8</b> 1.51x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">long-form 512<small>8 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.7 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3712;--d:280ms"></span><span class="lk-bench__val"><b>29.7</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 45.5 tok/s, 1.53x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.5687;--d:340ms"></span><span class="lk-bench__val"><b>45.5</b> 1.53x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 43.9 tok/s, 1.48x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.5487;--d:400ms"></span><span class="lk-bench__val"><b>43.9</b> 1.48x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">tool calls<small>8 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.5 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3688;--d:350ms"></span><span class="lk-bench__val"><b>29.5</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 58.2 tok/s, 1.97x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.7275;--d:410ms"></span><span class="lk-bench__val"><b>58.2</b> 1.97x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 49.6 tok/s, 1.68x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.6200;--d:470ms"></span><span class="lk-bench__val"><b>49.6</b> 1.68x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">agent turns<small>8 turns</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 28.8 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3600;--d:420ms"></span><span class="lk-bench__val"><b>28.8</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 53.3 tok/s, 1.85x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.6663;--d:480ms"></span><span class="lk-bench__val"><b>53.3</b> 1.85x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 46.6 tok/s, 1.62x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.5825;--d:540ms"></span><span class="lk-bench__val"><b>46.6</b> 1.62x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">blended<small>40 matrix prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.7 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3712;--d:490ms"></span><span class="lk-bench__val"><b>29.7</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 57.7 tok/s, 1.94x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.7213;--d:550ms"></span><span class="lk-bench__val"><b>57.7</b> 1.94x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 57.4 tok/s, 1.93x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.7175;--d:610ms"></span><span class="lk-bench__val"><b>57.4</b> 1.93x</span></div></div></div>
<div class="lk-bench__grp"><div class="lk-bench__lab">long-form 2,000<small>6 prompts</small></div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter: 29.5 tok/s"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3688;--d:560ms"></span><span class="lk-bench__val"><b>29.5</b></span></div><div class="lk-bench__row" title="DSpark v1 (K=4): 65.2 tok/s, 2.21x over no drafter"><span class="lk-bench__bar lk-bench__bar--v1" style="--w:0.8150;--d:620ms"></span><span class="lk-bench__val"><b>65.2</b> 2.21x</span></div><div class="lk-bench__row" title="DSpark v2 (K=5): 66.8 tok/s, 2.26x over no drafter"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.8350;--d:680ms"></span><span class="lk-bench__val"><b>66.8</b> 2.26x</span></div></div></div>
</div>
<div class="lk-bench__tiles"><div class="lk-bench__tile lk-bench__io"><h4>Power, single slot</h4><div class="lk-bench__stat"><span><i class="base"></i>no drafter</span><span><b>60.9<small>W</small></b> <b>2,135<small>mJ/token</small></b></span></div><div class="lk-bench__stat"><span><i class="v2"></i>DSpark v2</span><span><b>78.1<small>W</small></b> <b>1,495<small>mJ/token</small></b></span></div><p class="lk-bench__delta">More watts while it runs, about 30% less energy per generated token.</p></div><div class="lk-bench__tile lk-bench__io"><h4>Aggregate tok/s by slots</h4><div class="lk-bench__mini"><div class="lk-bench__minilab">1 slot</div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter, 1 slot(s): 28.8 tok/s aggregate"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.3600;--d:0ms"></span><span class="lk-bench__val"><b>28.8</b></span></div><div class="lk-bench__row" title="DSpark v2, 1 slot(s): 52.6 tok/s aggregate"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.6575;--d:80ms"></span><span class="lk-bench__val"><b>52.6</b></span></div></div><div class="lk-bench__minilab">2 slots</div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter, 2 slot(s): 45.7 tok/s aggregate"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.5713;--d:0ms"></span><span class="lk-bench__val"><b>45.7</b></span></div><div class="lk-bench__row" title="DSpark v2, 2 slot(s): 66.3 tok/s aggregate"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.8287;--d:80ms"></span><span class="lk-bench__val"><b>66.3</b></span></div></div><div class="lk-bench__minilab">4 slots</div><div class="lk-bench__bars"><div class="lk-bench__row" title="no drafter, 4 slot(s): 56.9 tok/s aggregate"><span class="lk-bench__bar lk-bench__bar--base" style="--w:0.7112;--d:0ms"></span><span class="lk-bench__val"><b>56.9</b></span></div><div class="lk-bench__row" title="DSpark v2, 4 slot(s): 74.4 tok/s aggregate"><span class="lk-bench__bar lk-bench__bar--v2" style="--w:0.9300;--d:80ms"></span><span class="lk-bench__val"><b>74.4</b></span></div></div></div><p class="lk-bench__delta">Generated tokens over the wall time of all waves, prefill included.</p></div></div>
<script>
(function(){
var roots=document.querySelectorAll('.lk-bench:not([data-ready])');
for(var r=0;r<roots.length;r++)init(roots[r]);
function init(root){
root.setAttribute('data-ready','1');
var reduce=!!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);
var parts=root.querySelectorAll('.lk-bench__io');
if(reduce||!('IntersectionObserver' in window))return;
for(var i=0;i<parts.length;i++)parts[i].classList.add('is-wait');
var io=new IntersectionObserver(function(es){
for(var i=0;i<es.length;i++){if(es[i].isIntersecting){var el=es[i].target;void el.offsetWidth;el.classList.remove('is-wait');io.unobserve(el);}}
},{threshold:0.12});
for(var i=0;i<parts.length;i++)io.observe(parts[i]);
}
})();
</script>
</div>

One DGX Spark, GB10, 128 GB unified LPDDR5X, CUDA 13.0. Target: Ternary-Bonsai-2-27B `PQ2_0` (6.70 GiB). Two drafters on the patched runtime, temperature 0, exact-match verification, idle GPU: DSpark v2, `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf` (1.03 GiB) at `--spec-draft-n-max 5` (K=5), and DSpark v1, the re-converted Ternary-Bonsai-27B drafter at K=4. Plain decode is flat with output length (llama-bench 29.68 / 29.93 / 29.77 tok/s at 64 / 1,024 / 2,000 tokens), so 29.8 is the baseline at every length. The table shows DSpark v2; the v1 numbers follow it. The first five rows are a bare loop (`llama-speculative-simple`), which reads higher than a server on the same hardware. The rest are `llama-server` with `-c 16384 -np 1 --jinja`, seed 42, `cache_prompt: false`, one unrecorded 16-token warm-up request per server.

| workload | tokens | no drafter | + DSpark v2 | accept | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| math (two trains), bare loop | 200 | 29.8 | 61.5 | 52.1% (148/284) | 2.06x |
| code (CSV parser), bare loop | 200 | 29.8 | 53.9 | 42.9% (137/319) | 1.81x |
| code (binary search), bare loop | 200 | 29.8 | 59.8 | 49.5% (143/289) | 2.01x |
| long-form, mean of 6 prompts, bare loop | 2,000 | 29.8 | 64.4 | 53.3% (7959/14942) | 2.16x |
| long-form, math prompt, bare loop | 2,000 | 29.8 | 71.8 | 62.7% (1100/1755) | 2.41x |
| server, quicksort, mean of 3 passes | 256 | 29.6 | 52.0 | 39.6% | 1.76x |
| server, blended, 40 matrix prompts | 512 | 29.7 | 57.4 | 42.0% | 1.93x |
| server, code, 8 prompts | 512 | 29.7 | 65.7 | 54.8% | 2.22x |
| server, math, 8 prompts | 512 | 29.7 | 74.7 | 65.9% | 2.52x |
| server, reasoning, 8 prompts | 512 | 29.7 | 57.8 | 44.5% | 1.95x |
| server, chat, 8 prompts | 512 | 29.7 | 44.8 | 29.8% | 1.51x |
| server, long-form, 8 prompts | 512 | 29.7 | 43.9 | 29.5% | 1.48x |
| server, tool calls, 8 prompts | 512 | 29.5 | 49.6 | 37.3% | 1.68x |
| server, agent turns, 8 turns | 512 | 28.8 | 46.6 | 40.8% | 1.62x |
| server, long-form, 6 prompts | 2,000 | 29.5 | 66.8 | 57.0% | 2.26x |
| server, 2 slots, aggregate tok/s | - | 45.7 | 66.3 | 48.5% | 1.45x |
| server, 4 slots, aggregate tok/s | - | 56.9 | 74.4 | 47.2% | 1.31x |
| power, mean W, single slot | - | 60.9 | 78.1 | n/a | - |
| energy, mJ per generated token | - | 2135 | 1495 | n/a | - |

Power readings come from `nvidia-smi --query-gpu=power.draw`, sampled at 1 Hz by a background thread while the server ran: mean W over the generation phase, and mJ/token = mean W / (generated tokens per wall second, prefill included) x 1000. Idle with the model loaded and no request in flight was 14.5-16.3 W across runs. Each speedup is the drafter rate over the no-drafter rate for the same prompts on the same harness. Acceptance is accepted over drafted tokens. Tokens per step is generated tokens over verify steps: DSpark v2 advances 3.85 per step on the 2,000-token server set and 3.08 blended, and that number does not depend on the hardware.

DSpark v1 against DSpark v2, same servers, same prompts, same run. Blended over the 40 matrix prompts: 57.7 tok/s (1.94x) for v1, 57.4 (1.93x) for v2. A tie. v2 leads on code (65.7 against 62.9, +4%) and on the 2,000-token long-form set (66.8 against 65.2, +2%), and it gets there with more tokens per step (3.71 against 3.19 on code, 3.85 against 3.34 long-form), because its block holds seven positions and runs at K=5 where v1 runs at K=4. v1 leads on reasoning (60.9 against 57.8, +5%), tool calls (58.2 against 49.6, +17%) and agent turns (53.3 against 46.6, +14%). Math is a dead heat at 74.7 (2.52x) for both; chat is 44.4 against 44.8. Our self-distilled training data holds no tool-call turns, and the tool and agent rows are where v1 leads most. Multi-slot: neither drafter refused a request at 2 or 4 slots. v1 gave 43.2 / 67.2 tok/s (per stream / aggregate) at 2 slots and 31.7 / 74.8 at 4; v2 gave 40.3 / 66.3 and 30.1 / 74.4; aggregate is generated tokens over the wall time of all waves, prefill included. Power: v1 measured 77.0 W and 1,458 mJ per generated token, v2 78.1 W and 1,495; both use about 30% less energy per token than the no-drafter server at 60.9 W and 2,135 mJ. At 4 slots it is 823 mJ per token for v2 (788 for v1) against 1,150 with no drafter. So the verdict: the broad self-distilled recipe did not beat PrismML's old drafter once the runtime fix let that drafter run on Bonsai 2. What v2 is: the drafter anyone can rebuild from section 4, with the better code and long-form profile.

Tool calls. A call is valid when it names a tool from the list and carries a JSON object of arguments. Every configuration, baseline included, produced 6 valid calls out of 8, each to the expected tool. The other two prompts hit the 512-token budget inside the reasoning block on every configuration, so the drafter does not change validity. One caveat on the tool and agent rows: those requests go through the server's chat template, which injects its default reasoning effort (`xhigh` on Bonsai 2), while the matrix rows use plain ChatML with no system message. They are not like for like with the matrix rows; compare them with each other.

What exact match means. The runtime accepts a drafted token only when it equals the token the target picks at temperature 0 from the batched verify logits. The drafter changes the speed, not the sampling rule. The output can still differ from a plain run, because the verify pass scores several positions in one batch and the batched kernels round differently from single-row decode. We measured it over all 65 single-slot outputs (62 prompts plus three passes of the quicksort prompt), drafter output against no-drafter output at temperature 0 and seed 42, token ids for `/completion` and text for chat: 37 of 65 identical for v1, 37 of 65 for v2. The telling part is that the 28 outputs that differ are the same 28 for both drafters, and the first difference sits at the same position: reasoning-06 at token 1, long-form-03 at token 9, code-06 at token 59, tool-05 at character 1773. Two different drafters do not pick the same 28 divergence points if the draft causes them. The position belongs to the prompt: the batched verify pass rounds differently from single-row decode at a near-tie token, and the argmax flips. Given the batched logits, acceptance is exact, and the baseline's own passes are identical across repeats. So we do not claim byte-identical output. Speculative decoding on this runtime is exact given the batched logits, and identical to plain greedy decoding on 37 of 65 outputs here. A separate claim, and a true one: the three 200-token probes reproduce count for count and byte for byte across the two patched builds, the same fix on two binaries on the same path.

## 3. Why the first training run plateaued

Our first attempt at this drafter stalled at 16-22% acceptance on real prompts, about 30-34 tok/s. We audited it and found five causes.

1. Degenerate objective. The trainer regressed the layer-62 hidden state at the same token position from an input that contained that slice. That is an autoencoder: no token shift, no next-token cross-entropy, no block rollout. Two epochs on 20,022 samples gave a flat loss and no acceptance gain.
2. Corrupted data. The feature extractor never JSON-unescaped message content. 8,710 of 10,000 code rows contained a literal `\n`.
3. Wrong data regime. The data was raw CodeAlpaca gold text, not the target's own generations.
4. Inflated evaluation. A reported 95-100% acceptance came from one short prompt (15 x 12) with a drafter-side p-min filter, which proposes fewer tokens; throughput fell to 25 tok/s, below the 27.8 tok/s baseline. A six-quantization sweep ran on a contended GPU and was not valid.
5. Wrong step arithmetic. The projections assumed 27.5 ms per step; measured single-token decode is 36 ms, so every 100+ tok/s projection was inflated.

The throughput equation is short: tok/s = (1 + accepted tokens per step) / step time. Get the objective wrong and nothing else matters.

## 4. The recipe that worked

Validated teacher. The drafter borrows the target's LM head and token embedding, both in the `PQ2_0` Hadamard basis. We dequantized both on the CPU with the runtime's fold (block Hadamard 1024, explicit signs) and gated on `final_hidden @ W_lm.T` against the runtime's logits: 100% argmax match at every position, top-1 logits within 0.06%. No training until it passed.

Corrected objective. The trainer implements the DSpark block-parallel objective and mirrors the runtime forward (five layer-input taps to context, per-layer K/V, a decoder over `[anchor, mask x 6]`, block size 7). The loss is `0.1 * CE + 0.9 * L1 + 1.0 * confidence BCE` through the borrowed head. The smoke run makes the point: 47 samples, 24 steps, 12 minutes, and at K=4 code acceptance went from 14.3% to 42.6% and math from 21.1% to 48.9% against the old objective's drafter trained on 20,022 samples.

Self-distilled data through llama-server. The target writes its own greedy completions, and a teacher-forced pass records the taps and the final hidden state. Single-stream generation is 7.4 tok/s, so we ran the target as a server with continuous batching: 86.7 tok/s aggregate at 12 slots, 136.7 at 24.

Broad prompt mix. Round 1 was code only (CodeAlpaca). Round 2 added 2,500 prompts at 25% math, 25% reasoning, 20% chat, 15% code, 15% long-form (GSM8K train, MATH, Open-Platypus, ARC-Challenge, no_robots, Dolly, UltraChat, CodeAlpaca, Evol-Instruct-Code). 66.5% of the answers hit the 512-token cap, so the data weights the first 512 tokens of an answer.

Round 2 with checkpoints. Round 1 warm-started from the RadixArk Qwen3.8-27B DSpark checkpoint (5 draft layers, block size 7) and stopped after epoch 1: train accuracy rose, held-out acceptance did not. Round 2 continued on all data at lr 6e-5, 256 anchors, batch 2, a checkpoint every 300 steps. Step 300 lifted math acceptance at K=5 from 43.9% to 52.1% and left code unchanged; steps 600 and 900 gave the same counts within noise, so we stopped at step 910 and kept step 600.

K=5 and Q4_K_M. K=4 and K=5 tie and K=7 loses, at 200 and at 2,000 tokens (table in section 5). The draft step costs 7.8 ms: about 4 ms for the 1 GB Q4_K_M weight read, 1.2 ms for the LM head over the 248k vocabulary. We did not compare drafter quantizations on an idle GPU; Q4_K_M is a size choice that keeps the draft step small, not a measured optimum.

Measured cost, one GB10, one stage on the GPU at a time:

| stage | measured |
| --- | --- |
| smoke run | 47 samples, 13.4k tokens, 24 steps, 12 min |
| round-1 generation | 856 samples, 209k generated tokens, 86.7 tok/s aggregate, 12 slots, 40 min |
| round-2 generation | 2,498 samples, 1.11M generated tokens, 136.7 tok/s aggregate, 24 slots, 2 h 16 min |
| feature extraction | 762-879 tok/s; 123 KB per token; 193 GB for 1.57M tokens |
| round-1 training | 903 samples, 250k tokens; 21.7 s per step, batch 2, 512 anchors; 452 steps (1 epoch); 18.7 GB VRAM |
| round-2 training | 3,401 samples, 1.57M tokens; 22-23 s per step, batch 2, 256 anchors; 910 of 1,701 steps, 355 min |

## 5. What did not work, with numbers

Same GB10, same `PQ2_0` target, temperature 0 throughout. Numbers in the table, mechanism below.

| experiment | condition | result |
| --- | --- | --- |
| K=7 | 200 tokens, math, smoke drafter; 2,000 tokens, six prompts, step-300 drafter | 49.7 tok/s against 53.8 (K=4) / 54.1 (K=5); 60.9 against 63.1 / 64.6 |
| typical acceptance, short-form | 200 tokens, K=5, tau 0.05, step-600 drafter | math 65.3% / 70.8 tok/s, code 53.2% / 62.6 (exact match: 52.1% / 61.5, 42.9% / 53.9) |
| typical acceptance, GSM8K | epoch-1 drafter, 50 test problems, K=5, 500-token budget, last-number match | exact 43/50, tau 0.02 44/50, tau 0.05 43/50 |
| typical acceptance, long-form | epoch-1 drafter, five 2,000-token code prompts, K=5 | repetition loops: exact 0/5, tau 0.05 1/5, tau 0.02 4/5 |
| repetition guard | same drafter and prompts; 6-gram check, 64-token ring, cap on consecutive relaxed accepts | tau 0.02 4/5 -> 2/5, tau 0.05 1/5 -> 1/5; both guarded configs wrong on the 2,000-token math answer, exact match right |
| tree verification | 10 tree shapes, epoch-1 drafter | -29% (best) to -66% (widest) against linear K=4; tokens per step +3-18%; each chain writes 48 layers x 3.1 MB of state per step; decode ~= 40 ms + 2.0 ms x rows |
| PTQ1_0 target, plain | llama-bench tg64; pp512 | 34.69 tok/s against 29.90 for PQ2_0; 432.59 against 919.05 |
| PTQ1_0 target, speculation | math, 200 tokens, K=4 / 5 / 7 | 41.1 / 37.7 / 24.6 tok/s against 53.8 / 54.1 / 49.7 on PQ2_0 |
| drafter on a second Spark | RPC device placement, 400 Gbps link | -5.3 to -5.7%; RDMA about 0.4 ms per step |
| step breakdown | smoke drafter, K=4, idle GPU | target 45.0 + draft 7.8 + process 0.56 + sample 0.8 = 54.2 ms; ceiling for any drafter-hiding scheme +18.3% |
| higher learning rate | from step 600, lr 2e-4 (3.3x), 200 steps, K=5 exact | math 52.1% -> 47.4%, code 42.9% -> 35.2%; long-form mean 64.4 -> 60.0 tok/s |
| more narrow data | 19x more code-only data, K=4 | acceptance +1-3 points |
| Medusa v2, held-out | 4 heads, broad data, 3 epochs | top-1 67.3 / 41.5 / 26.8 / 18.7% for t+2 through t+5 |
| Medusa v2, runtime | 200 tokens, exact match, n-max 2, clean GPU | math 47.1 tok/s (2.01 tokens per step), code 42.8; DSpark on the same GPU 61.5 / 51.8 |
| EAGLE-3 smoke | 1 layer from scratch, 1 epoch on 1.57M tokens (1,701 steps, 24 min); 200 tokens, K=5 exact, Q4_K_M | math 21.1% / 35.4 tok/s, code 16.1% / 30.4, binary search 19.7% / 33.7; 1.05 accepted per step; long-form mean below baseline; 3 tokens per parameter; train-batch agreement 56% at depth 1 against 21% on fresh prompts |
| verify pass kernel profile | nsys, idle GB10, llama-bench -p 1 and -p 8 | one row: 45 ms of kernel time, 79% in PQ2_0 `mul_mat_vec_q`, 6.7 GB in 35.6 ms at about 188 GB/s (69% of the 273 GB/s peak); eight rows: 52.9 ms |

Typical acceptance accepts a drafted token when the target's probability for it is within a band (tau) of its top token. Short-form it is a +24-43% knob at GSM8K parity. Long-form it loops, and a guard does not fix it: one within-band wrong token seeds a repeat, after 2-3 repeats the target's own argmax continues it, and the drafter then matches exactly, so no guard on relaxed accepts can see it. Exact match is what we run.

Trees lose because Bonsai 2's backbone, `qwen35`, is a hybrid of Gated DeltaNet and attention: the recurrent layers hold per-sequence state, so a tree cannot share a prefix across branches, and every extra chain pays a full state write.

PTQ1_0 loses under speculation because its prompt processing is half of PQ2_0's, and the batched verify pass runs at that speed.

A second Spark cannot hide the drafter: the speculative loop is strictly sequential, and the drafter's inputs (the taps and the anchor token) are outputs of the verify pass, so realizable overlap is under 1.5 ms per step.

More steps, more data and a higher learning rate all plateau or regress. Train-set accuracy of 0.73-0.85 against about 50% held-out acceptance points to capacity, not under-training.

Medusa's step is cheaper (39.5 ms at n-max 2), but DSpark's 7-position block predicts deeper (2.9 tokens per step) and wins by 20-30%. Medusa v2 is a solid second.

The EAGLE-3 mirror is right: a trainer/runtime mismatch gives near-zero acceptance, and deeper positions do accept. The chain costs about 14 ms per step against DSpark's 7.8, so it needs about 3.3 accepted per step to win. It is data-starved, at about 200x fewer tokens than published EAGLE recipes.

The kernel profile says the runtime is near the floor: the one-row verify pass already runs at 69% of the GB10's memory bandwidth, and extra rows are cheap up to eight, where the matmul leaves the batched-matvec path (`MMVQ_MAX_BATCH_SIZE = 8`; hence K=7 loses). At most 10-15% is left in the runtime. Accepted tokens per step is the lever: 100 tok/s needs about 6 per step at 55-60 ms, or 90%+ per-token acceptance at K=6-7.

## 6. Where this sits publicly

Same hardware, same model family. PrismML's own GB10 entry for the older Ternary-Bonsai-27B reports DSpark at 70.0 t/s (2.45x) on one quicksort prompt and 2.35x blended over 16 prompts, with that model's own drafter. [Kubesimplify](https://blog.kubesimplify.com/bonsai-27b-rtx-pro-6000-dgx-spark) measured the Spark at 28.5 tok/s tg128, which matches our 29.8, and a DSpark loss (28.2 to 17.6 tok/s). Our 61-72 tok/s at exact match is, as far as we can find, the first positive speculative number for Bonsai 2 on this hardware. PrismML's [SPECULATIVE.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) reports 1.8-2.4x (2.06x blended) on an L40S, the same band we see on the Spark.

Same hardware, different model and stack. An [NVIDIA forum entry](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257) runs Qwen3.8-27B NVFP4 with DSpark in SGLang at 34-38 tok/s on the Spark. [llama.cpp discussion #27080](https://github.com/ggml-org/llama.cpp/discussions/27080) gets Q4_K_M plus an MTP draft to 18 tok/s. Both are 4-bit targets; our ternary target does 29.8 tok/s with no drafter.

Different regime. [EAGLE-3 in vLLM](https://vllm.ai/blog/2026-05-26-eagle-3-1) reports 4.5-5.0 accepted tokens per step, and the [DFlash paper](https://arxiv.org/abs/2602.06036) over 6x on Qwen3-8B with decode-only timing. Both run bf16 targets on data-center GPUs, where the verify pass is nearly free and the ratio is mostly acceptance. On a 273 GB/s GB10 the one-row verify pass already streams 6.7 GB, so the ratio does not transfer. Compare accepted tokens per step: ours 3.85 at K=5, DSpark-NVFP4 on the Spark 3.3-4.7, MTP 2.4, EAGLE-3 class 4.5-5.0.

## 7. Reproduce it

Run from the Bonsai-demo checkout after `BONSAI_FAMILY=bonsai2 BONSAI_MODEL=27B ./setup.sh`. Setup installs release binaries without the fix, so build the runtime from source.

```bash
# drafter; keep it as the only *dspark-dflash*.gguf in the directory
hf download <HF-REPO> Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf --local-dir models/bonsai2-gguf/27B

# runtime: prism 1a07bfa5f plus the fix of PR #210
git clone https://github.com/PrismML-Eng/llama.cpp.git
git -C llama.cpp fetch https://github.com/usmaneth/llama.cpp fix/dflash-borrowed-hadamard
git -C llama.cpp checkout 288859a96   # prism 1a07bfa5f plus the fix of PR #210; use the merge commit once the PR lands
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cuda -j 18 --target llama-speculative-simple llama-server llama-bench
export LD_LIBRARY_PATH=$PWD/llama.cpp/build-cuda/bin

# server with the flags the benchmark used
llama.cpp/build-cuda/bin/llama-server \
  -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -md models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 \
  -ngl 99 -ngld 999 -fa on -c 16384 -np 1 --jinja \
  --host 127.0.0.1 --port 8080
```

For the demo launcher instead, copy the patched binaries over `bin/cuda/` and run `BONSAI_SPECULATIVE=1 BONSAI_SPEC_NMAX=5 ./scripts/start_llama_server.sh`. The launcher takes the first `*dspark-dflash*.gguf` in glob order (`...-Q4_0.gguf` sorts before `...-v2-Q4_K_M.gguf`), so keep one drafter in the directory or pass `-md`. It reads the draft length from the key `dspark.dspark.block_size`, which a converted file lacks (it has `dflash.block_size`), so it falls back to K=4; `BONSAI_SPEC_NMAX=5` sets it.

Read acceptance from the response: `timings.draft_n_accepted / timings.draft_n`, on `/completion` and `/v1/chat/completions`, present only when a drafter is loaded. The quick check is the math row of the 200-token table: `llama-speculative-simple` with the raw prompt, 200 tokens, temperature 0, prints `n_drafted = 284, n_accept = 148`. Under `--jinja` the model thinks first, so a short `max_tokens` ends inside the reasoning block.

In the Bonsai-demo repository: the benchmark doc at `docs/community-benchmarks/ternary-bonsai/cuda-gb10-bonsai2-27b-linux.md`, the benchmark suite at `scripts/spec_bench/`, and the training recipe at `tools/dspark-retrain/README.md`. The drafter's GGUF header still says `general.name = Qwen3.8-27B-DSpark`, inherited from the warm-start checkpoint.

## 8. What is next

A data scale-up is in progress: 20,000 new prompts with three-tap f16 features, about 9M tokens, for both an EAGLE-3 drafter and a DSpark v3. We will run both on the same harness, the same prompts and the same GPU, and report accepted tokens per step next to tok/s. Whether either passes 3.85 tokens per step on this target is an open question.
