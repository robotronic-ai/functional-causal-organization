#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRANSFORMER-REENTRY RANK-SIGNATURE v0.4."""

from __future__ import annotations
import argparse,json,math,platform,re
from collections import Counter,defaultdict
from pathlib import Path

MODEL_ID="EleutherAI/pythia-70m-deduped"
REVISION="5ff092d907c3d8ba420d9fbef792426789eb2cd2"
SOURCE_LAYER=2; SOURCE_POS=1; RETURN_POS=3; PRE_POS=0
NEGATIVE_MI_TOL=1e-8
TOL=1e-12
TOKEN_RE=re.compile(r"^ ?[A-Za-z]{2,10}$")
EXPECTED_HISTORICAL_IDS=[
    248,249,250,251,253,254,255,257,
    261,262,263,264,265,266,
    267,271,272,273,274,275,
    276,
]
EXPECTED_EXTRA_IDS=[279,280,281,282]
KS=(1,2,4,8)
HOLDOUT=((4,4),(4,5),(5,4),(5,5))

def entropy(ps):
    return -sum(p*math.log2(p) for p in ps if p>0)

def cmi(prior,uprior,kernels):
    joint=defaultdict(float)
    for u,pu in uprior.items():
        for s,ps in prior.items():
            for y,p in enumerate(kernels[(u,s)]):
                joint[(s,y,u)]+=pu*ps*p
    pu=defaultdict(float);psu=defaultdict(float);pyu=defaultdict(float)
    for (s,y,u),p in joint.items():
        pu[u]+=p;psu[(s,u)]+=p;pyu[(y,u)]+=p
    z=0.0
    for (s,y,u),p in joint.items():
        if p<=0: continue
        z+=p*math.log2((p*pu[u])/(psu[(s,u)]*pyu[(y,u)]))
    return z

def select_eligible(tok,n=25):
    special=set(tok.all_special_ids); out=[]
    for tid in range(len(tok)):
        if tid in special: continue
        txt=tok.decode([tid],clean_up_tokenization_spaces=False)
        if not TOKEN_RE.match(txt): continue
        if tok.encode(txt,add_special_tokens=False)!=[tid]: continue
        out.append((tid,txt))
        if len(out)>=n: break
    if len(out)<n: raise RuntimeError("insufficient eligible tokens")
    return out

def get_h(o): return o[0] if isinstance(o,tuple) else o
def put_h(o,h): return (h,)+o[1:] if isinstance(o,tuple) else h
def ids(seq,torch,dev): return torch.tensor([seq],dtype=torch.long,device=dev)

def probs_subset(logits,tids,torch):
    x=logits[tids].detach().to(dtype=torch.float64,device="cpu")
    x=x-x.max(); p=torch.exp(x);p=p/p.sum()
    return [float(v) for v in p.tolist()]

def capture(model,inp,torch):
    got={}; layer=model.gpt_neox.layers[SOURCE_LAYER]
    def hook(_m,_i,o):
        h=get_h(o);got["v"]=h[0,SOURCE_POS,:].detach().clone();return o
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad(): model(input_ids=inp,use_cache=False)
    finally: hh.remove()
    return got["v"]

def patched(model,inp,vec,torch):
    layer=model.gpt_neox.layers[SOURCE_LAYER]
    def hook(_m,_i,o):
        h=get_h(o);hp=h.clone()
        hp[:,SOURCE_POS,:]=vec.to(device=hp.device,dtype=hp.dtype)
        return put_h(o,hp)
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad(): out=model(input_ids=inp,use_cache=False)
    finally: hh.remove()
    return out.logits[0]

def rank_order(probs):
    return tuple(sorted(range(len(probs)),key=lambda i:(-probs[i],i)))

def quotient(sigs):
    uniq=sorted(set(sigs.values()));m={s:i for i,s in enumerate(uniq)}
    return {q:m[s] for q,s in sigs.items()}

def blocks(q2c):
    g=defaultdict(list)
    for q,c in q2c.items():g[c].append(q)
    return tuple(sorted(tuple(sorted(v)) for v in g.values()))

def cp(q2c):
    n=len(q2c);cnt=Counter(q2c.values())
    return {c:k/n for c,k in cnt.items()}

def aggregate(q2c,qrows):
    g=defaultdict(list)
    for q,row in qrows.items():g[q2c[q]].append(row)
    return {c:[sum(r[j] for r in rows)/len(rows) for j in range(len(rows[0]))]
            for c,rows in g.items()}

def metrics(q2c,raw,uprior):
    prior=cp(q2c); kernels={}
    for u,qrows in raw.items():
        for c,row in aggregate(q2c,qrows).items():kernels[(u,c)]=row
    K=entropy(prior.values());B=cmi(prior,uprior,kernels)
    R=None if K==0 else B/K
    return {"class_count":len(prior),"partition_blocks":[list(b) for b in blocks(q2c)],
            "K_eff_bits":K,"B_reentry_bits":B,"R":R}

def refines(fine,coarse):
    fb=[set(b) for b in blocks(fine)];cb=[set(b) for b in blocks(coarse)]
    return all(any(b<=c for c in cb) for b in fb)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",default="TRANSFORMER_REENTRY_RANK_SIGNATURE_RESULT_v0.4.json")
    ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args()
    if args.plan_only:
        print(json.dumps({"k_values":KS,"holdout":HOLDOUT,"model":MODEL_ID,
                          "revision":REVISION,"no_numeric_target":True},indent=2))
        return 0

    import numpy as np
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM

    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    eligible=select_eligible(tok,25)
    if [x[0] for x in eligible[:21]]!=EXPECTED_HISTORICAL_IDS:
        raise RuntimeError("historical token allocation changed")
    if [x[0] for x in eligible[21:25]]!=EXPECTED_EXTRA_IDS:
        raise RuntimeError("extra key allocation changed")

    candidates=eligible[:8];keys=eligible[8:14];fillers=eligible[14:20]
    anchor=eligible[20];extra=eligible[21:25]
    cand=[x[0] for x in candidates];keyids=[x[0] for x in keys]
    fills=[x[0] for x in fillers];anchorid=anchor[0]
    auxkeys=[keyids[1],keyids[2],keyids[3]]+[x[0] for x in extra]

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,revision=REVISION,torch_dtype=torch.float32
    ).to(args.device);model.eval()
    cfg=model.config
    if cfg.model_type!="gpt_neox" or int(cfg.num_hidden_layers)!=6 or int(cfg.hidden_size)!=512 or int(cfg.num_attention_heads)!=8:
        raise RuntimeError("architecture mismatch")

    donors={}
    for q,vid in enumerate(cand):
        donors[q]=capture(model,ids([keyids[0],vid,fills[0]],torch,args.device),torch)

    orders={};full_probs={}
    min_gap=1.0
    for q,vec in donors.items():
        orders[q]=[];full_probs[str(q)]=[]
        for kid in auxkeys:
            inp=ids([kid,anchorid,fills[1]],torch,args.device)
            logits=patched(model,inp,vec,torch)
            p=probs_subset(logits[SOURCE_POS],cand,torch)
            order=rank_order(p)
            orders[q].append(order);full_probs[str(q)].append(p)
            sp=sorted(p,reverse=True)
            for a,b in zip(sp,sp[1:]): min_gap=min(min_gap,a-b)

    sigs={};q2c={}
    for k in KS:
        s={q:tuple(tuple(order[:k]) for order in orders[q]) for q in donors}
        sigs[k]=s;q2c[k]=quotient(s)

    for a,b in zip(KS,KS[1:]):
        if not refines(q2c[b],q2c[a]):
            raise RuntimeError(f"non-nested rank partitions {a}->{b}")

    rawret={};rawpre={}
    for u,(kj,fk) in enumerate(HOLDOUT):
        inp=ids([keyids[kj],anchorid,fills[fk],keyids[kj]],torch,args.device)
        rr={};rp={}
        for q,vec in donors.items():
            logits=patched(model,inp,vec,torch)
            rr[q]=probs_subset(logits[RETURN_POS],cand,torch)
            rp[q]=probs_subset(logits[PRE_POS],cand,torch)
        rawret[u]=rr;rawpre[u]=rp

    uprior={u:1/len(HOLDOUT) for u in range(len(HOLDOUT))}
    qprior={q:1/len(donors) for q in donors}
    qkret={(u,q):r for u,rows in rawret.items() for q,r in rows.items()}
    qkpre={(u,q):r for u,rows in rawpre.items() for q,r in rows.items()}
    BQ=cmi(qprior,uprior,qkret);BQpre=cmi(qprior,uprior,qkpre)

    ms={};pres={}
    for k in KS:
        m=metrics(q2c[k],rawret,uprior);mp=metrics(q2c[k],rawpre,uprior)
        m["eta_vs_Q"]=None if BQ==0 else m["B_reentry_bits"]/BQ
        m["residual_bits"]=BQ-m["B_reentry_bits"]
        ms[k]=m;pres[k]=mp["B_reentry_bits"]

    Ks=[ms[k]["K_eff_bits"] for k in KS]
    Bs=[ms[k]["B_reentry_bits"] for k in KS]
    monoK=all(b+TOL>=a for a,b in zip(Ks,Ks[1:]))
    monoB=all(b+TOL>=a for a,b in zip(Bs,Bs[1:]))
    causal=(BQpre<NEGATIVE_MI_TOL and all(v<NEGATIVE_MI_TOL for v in pres.values()))

    same=all(blocks(q2c[k])==blocks(q2c[1]) for k in KS[1:])
    identity=(len(blocks(q2c[8]))==len(donors) and all(len(b)==1 for b in blocks(q2c[8])))
    pstatus="RANK-STABLE" if same else "RANK-REFINING"
    verdict="TR4-V1" if monoK and monoB and causal else "TR4-V3"

    result={"schema":"TRANSFORMER-REENTRY-RANK-SIGNATURE-result-v0.4",
            "verdict":verdict,"partition_status":pstatus,"rank8_identity":identity,
            "runtime":{"python":platform.python_version(),"torch":torch.__version__,
                       "transformers":transformers.__version__,"numpy":np.__version__,
                       "device":args.device},
            "model":{"id":MODEL_ID,"revision":REVISION},
            "metrics":{"B_Q_reentry_bits":BQ,"B_Q_pre_bits":BQpre,
                       "by_k":{str(k):ms[k] for k in KS},
                       "B_pre_by_k":{str(k):pres[k] for k in KS},
                       "monotone_K":monoK,"monotone_B":monoB},
            "auxiliary":{"partition_blocks":{str(k):[list(b) for b in blocks(q2c[k])] for k in KS},
                         "signatures":{str(k):{str(q):[[*x] for x in sigs[k][q]] for q in donors} for k in KS},
                         "min_adjacent_rank_gap":min_gap,
                         "full_probs":full_probs},
            "tokens":{"candidates":[{"id":i,"text":t} for i,t in candidates],
                      "aux_keys":[{"id":i,"text":t} for i,t in [keys[1],keys[2],keys[3],*extra]]}}
    out=Path(args.output);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("VERDICT:",verdict)
    print("partition_status:",pstatus)
    print("rank8_identity:",identity)
    print("B_Q_reentry_bits:",BQ)
    print("B_Q_pre_bits:",BQpre)
    for k in KS:
        m=ms[k]
        print("k=",k,"classes=",m["class_count"],"K=",m["K_eff_bits"],
              "B=",m["B_reentry_bits"],"R=",m["R"],
              "eta=",m["eta_vs_Q"],"residual=",m["residual_bits"],
              "B_pre=",pres[k])
    print("min_adjacent_rank_gap:",min_gap)
    print("output:",out)
    return 0 if verdict=="TR4-V1" else 2

if __name__=="__main__":
    raise SystemExit(main())
