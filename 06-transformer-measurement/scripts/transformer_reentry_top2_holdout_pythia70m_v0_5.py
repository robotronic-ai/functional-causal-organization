#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRANSFORMER-REENTRY TOP2-HOLDOUT v0.5."""

from __future__ import annotations
import argparse, json, math, platform, re
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ID="EleutherAI/pythia-70m-deduped"
REVISION="5ff092d907c3d8ba420d9fbef792426789eb2cd2"
SOURCE_LAYER=2
SOURCE_POS=1
RETURN_POS=3
PRE_POS=0
NEGATIVE_MI_TOL=1e-8
TOKEN_RE=re.compile(r"^ ?[A-Za-z]{2,10}$")

EXPECTED_FIRST25=[
    248,249,250,251,253,254,255,257,
    261,262,263,264,265,266,
    267,271,272,273,274,275,
    276,279,280,281,282
]
OLD_TOP2=((0,),(1,),(2,),(3,7),(4,),(5,),(6,))

def H(ps):
    return -sum(p*math.log2(p) for p in ps if p>0)

def cmi(prior,uprior,kernels):
    joint=defaultdict(float)
    for u,pu in uprior.items():
        for s,ps in prior.items():
            for y,p in enumerate(kernels[(u,s)]):
                joint[(s,y,u)] += pu*ps*p
    pu=defaultdict(float); psu=defaultdict(float); pyu=defaultdict(float)
    for (s,y,u),p in joint.items():
        pu[u]+=p; psu[(s,u)]+=p; pyu[(y,u)]+=p
    z=0.0
    for (s,y,u),p in joint.items():
        if p>0:
            z += p*math.log2((p*pu[u])/(psu[(s,u)]*pyu[(y,u)]))
    return z

def eligible(tok,n):
    special=set(tok.all_special_ids); out=[]
    for tid in range(len(tok)):
        if tid in special:
            continue
        txt=tok.decode([tid],clean_up_tokenization_spaces=False)
        if not TOKEN_RE.match(txt):
            continue
        if tok.encode(txt,add_special_tokens=False)!=[tid]:
            continue
        out.append((tid,txt))
        if len(out)>=n:
            break
    if len(out)<n:
        raise RuntimeError("insufficient eligible tokens")
    return out

def geth(o):
    return o[0] if isinstance(o,tuple) else o

def puth(o,h):
    return (h,)+o[1:] if isinstance(o,tuple) else h

def ids(seq,torch,device):
    return torch.tensor([seq],dtype=torch.long,device=device)

def probs(logits,tids,torch):
    x=logits[tids].detach().to(dtype=torch.float64,device="cpu")
    x=x-x.max()
    p=torch.exp(x); p=p/p.sum()
    return [float(v) for v in p.tolist()]

def capture(model,inp,torch):
    got={}; layer=model.gpt_neox.layers[SOURCE_LAYER]
    def hook(_m,_i,o):
        h=geth(o)
        got["v"]=h[0,SOURCE_POS,:].detach().clone()
        return o
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=inp,use_cache=False)
    finally:
        hh.remove()
    return got["v"]

def patch(model,inp,vec,torch):
    layer=model.gpt_neox.layers[SOURCE_LAYER]
    def hook(_m,_i,o):
        h=geth(o); hp=h.clone()
        hp[:,SOURCE_POS,:]=vec.to(device=hp.device,dtype=hp.dtype)
        return puth(o,hp)
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out=model(input_ids=inp,use_cache=False)
    finally:
        hh.remove()
    return out.logits[0]

def rank(p):
    return tuple(sorted(range(len(p)),key=lambda i:(-p[i],i)))

def qmap(signatures):
    uniq=sorted(set(signatures.values()))
    m={s:i for i,s in enumerate(uniq)}
    return {q:m[s] for q,s in signatures.items()}

def blocks(q2c):
    g=defaultdict(list)
    for q,c in q2c.items():
        g[c].append(q)
    return tuple(sorted(tuple(sorted(v)) for v in g.values()))

def prior(q2c):
    cnt=Counter(q2c.values()); n=len(q2c)
    return {c:k/n for c,k in cnt.items()}

def aggregate(q2c,qrows):
    g=defaultdict(list)
    for q,row in qrows.items():
        g[q2c[q]].append(row)
    return {
        c:[sum(r[j] for r in rows)/len(rows) for j in range(len(rows[0]))]
        for c,rows in g.items()
    }

def metrics(q2c,raw,uprior):
    pr=prior(q2c); kernels={}
    for u,qrows in raw.items():
        for c,row in aggregate(q2c,qrows).items():
            kernels[(u,c)]=row
    K=H(pr.values())
    B=cmi(pr,uprior,kernels)
    R=None if K==0 else B/K
    return K,B,R

def relation(new_partition):
    old=[set(x) for x in OLD_TOP2]
    new=[set(x) for x in new_partition]
    new_refines_old=all(any(n<=o for o in old) for n in new)
    old_refines_new=all(any(o<=n for n in new) for o in old)
    if tuple(new_partition)==OLD_TOP2:
        return "TOP2-HOLDOUT-PASS"
    if new_refines_old and not old_refines_new:
        return "TOP2-HOLDOUT-REFINED"
    if old_refines_new and not new_refines_old:
        return "TOP2-HOLDOUT-COARSENED"
    return "TOP2-HOLDOUT-INCOMPATIBLE"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",default="TRANSFORMER_REENTRY_TOP2_HOLDOUT_RESULT_v0.5.json")
    ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args()

    if args.plan_only:
        print(json.dumps({
            "candidate":"ordered top-2 rank signature",
            "new_aux_keys":7,
            "new_return_contexts":4,
            "old_partition":[list(x) for x in OLD_TOP2],
            "no_numeric_target":True
        },indent=2))
        return 0

    import numpy as np
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    e=eligible(tok,37)
    if [x[0] for x in e[:25]]!=EXPECTED_FIRST25:
        raise RuntimeError("historical eligible-token allocation changed")

    candidates=e[:8]
    keys=e[8:14]
    fillers=e[14:20]
    anchor=e[20]
    new_aux=e[25:32]
    aux_filler=e[32]
    new_return_keys=e[33:35]
    new_return_fillers=e[35:37]

    cand=[x[0] for x in candidates]
    keyids=[x[0] for x in keys]
    fills=[x[0] for x in fillers]
    anchorid=anchor[0]

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,revision=REVISION,torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    cfg=model.config
    if cfg.model_type!="gpt_neox" or int(cfg.num_hidden_layers)!=6 or int(cfg.hidden_size)!=512 or int(cfg.num_attention_heads)!=8:
        raise RuntimeError("architecture mismatch")

    donors={}
    for q,vid in enumerate(cand):
        donors[q]=capture(
            model,
            ids([keyids[0],vid,fills[0]],torch,args.device),
            torch
        )

    sig2={}; sig4={}; margins={}
    for q,vec in donors.items():
        s2=[]; s4=[]; mg=[]
        for kid,_txt in new_aux:
            out=patch(
                model,
                ids([kid,anchorid,aux_filler[0]],torch,args.device),
                vec,
                torch
            )
            p=probs(out[SOURCE_POS],cand,torch)
            order=rank(p)
            sp=sorted(p,reverse=True)
            s2.append(order[:2])
            s4.append(order[:4])
            mg.append({
                "top1_top2":sp[0]-sp[1],
                "top2_top3":sp[1]-sp[2],
            })
        sig2[q]=tuple(s2)
        sig4[q]=tuple(s4)
        margins[str(q)]=mg

    q2=qmap(sig2)
    q4=qmap(sig4)
    p2=blocks(q2)
    p4=blocks(q4)
    source_verdict=relation(p2)

    rawret={}; rawpre={}; contexts=[]
    u=0
    for rk,_rt in new_return_keys:
        for sf,_st in new_return_fillers:
            contexts.append((rk,sf))
            inp=ids([rk,anchorid,sf,rk],torch,args.device)
            rr={}; rp={}
            for q,vec in donors.items():
                out=patch(model,inp,vec,torch)
                rr[q]=probs(out[RETURN_POS],cand,torch)
                rp[q]=probs(out[PRE_POS],cand,torch)
            rawret[u]=rr
            rawpre[u]=rp
            u+=1

    uprior={u:1/4 for u in range(4)}
    qprior={q:1/8 for q in donors}
    BQ=cmi(qprior,uprior,{(u,q):r for u,rows in rawret.items() for q,r in rows.items()})
    BQpre=cmi(qprior,uprior,{(u,q):r for u,rows in rawpre.items() for q,r in rows.items()})

    K2,B2,R2=metrics(q2,rawret,uprior)
    _,B2pre,_=metrics(q2,rawpre,uprior)
    eta=None if BQ==0 else B2/BQ
    residual=BQ-B2

    pr=prior(q2)
    dupmap={}
    for q,c in q2.items():
        dupmap[f"{q}a"]=c
        dupmap[f"{q}b"]=c
    prdup=prior(dupmap)
    dupok=all(
        abs(pr.get(c,0)-prdup.get(c,0))<1e-15
        for c in set(pr)|set(prdup)
    )

    causal_ok=(BQpre<NEGATIVE_MI_TOL and B2pre<NEGATIVE_MI_TOL and dupok)
    verdict="TR5-V1" if causal_ok else "TR5-V3"

    result={
        "schema":"TRANSFORMER-REENTRY-TOP2-HOLDOUT-result-v0.5",
        "verdict":verdict,
        "source_verdict":source_verdict,
        "runtime":{
            "python":platform.python_version(),
            "torch":torch.__version__,
            "transformers":transformers.__version__,
            "numpy":np.__version__,
            "device":args.device,
        },
        "model":{"id":MODEL_ID,"revision":REVISION},
        "tokens":{
            "new_aux_keys":[{"id":i,"text":t} for i,t in new_aux],
            "aux_filler":{"id":aux_filler[0],"text":aux_filler[1]},
            "new_return_keys":[{"id":i,"text":t} for i,t in new_return_keys],
            "new_return_fillers":[{"id":i,"text":t} for i,t in new_return_fillers],
        },
        "source":{
            "old_top2_partition":[list(x) for x in OLD_TOP2],
            "new_top2_partition":[list(x) for x in p2],
            "new_top4_partition":[list(x) for x in p4],
            "top2_signatures":{str(q):[list(x) for x in s] for q,s in sig2.items()},
            "margins":margins,
        },
        "holdout":{
            "contexts":[list(x) for x in contexts],
            "B_Q_reentry_bits":BQ,
            "B_Q_pre_bits":BQpre,
            "K_eff_bits":K2,
            "B_reentry_bits":B2,
            "R":R2,
            "eta_vs_Q":eta,
            "residual_bits":residual,
            "B_pre_bits":B2pre,
            "duplication_invariance":dupok,
        },
    }

    Path(args.output).write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8"
    )

    print("VERDICT:",verdict)
    print("source_verdict:",source_verdict)
    print("new_top2_partition:",p2)
    print("new_top4_partition:",p4)
    print("B_Q_reentry_bits:",BQ)
    print("K_eff_bits:",K2)
    print("B_reentry_bits:",B2)
    print("R:",R2)
    print("eta_vs_Q:",eta)
    print("residual_bits:",residual)
    print("B_Q_pre_bits:",BQpre)
    print("B_pre_bits:",B2pre)
    print("duplication_invariance:",dupok)
    print("output:",args.output)

    return 0 if verdict=="TR5-V1" else 2

if __name__=="__main__":
    raise SystemExit(main())
