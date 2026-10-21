#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRANSFORMER-REENTRY SOURCE-QUOTIENT v0.3."""

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

EXPECTED_HISTORICAL_IDS=[
    248,249,250,251,253,254,255,257,   # candidates
    261,262,263,264,265,266,           # keys K0..K5
    267,271,272,273,274,275,           # fillers F0..F5
    276,                               # anchor
]

HOLDOUT=((4,4),(4,5),(5,4),(5,5))

def entropy(ps):
    return -sum(p*math.log2(p) for p in ps if p>0)

def conditional_mi(prior,uprior,kernels):
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
        if p<=0: continue
        den=psu[(s,u)]*pyu[(y,u)]
        z += p*math.log2((p*pu[u])/den)
    return z

def select_eligible(tokenizer,n=25):
    special=set(tokenizer.all_special_ids)
    out=[]
    for tid in range(len(tokenizer)):
        if tid in special: continue
        txt=tokenizer.decode([tid],clean_up_tokenization_spaces=False)
        if not TOKEN_RE.match(txt): continue
        if tokenizer.encode(txt,add_special_tokens=False)!=[tid]: continue
        out.append((tid,txt))
        if len(out)>=n: break
    if len(out)<n:
        raise RuntimeError(f"Need {n} eligible tokens, found {len(out)}")
    return out

def get_h(out):
    return out[0] if isinstance(out,tuple) else out

def put_h(out,h):
    return (h,)+out[1:] if isinstance(out,tuple) else h

def ids(seq,torch,device):
    return torch.tensor([seq],dtype=torch.long,device=device)

def softmax_subset(logits,tids,torch):
    x=logits[tids].detach().to(dtype=torch.float64,device="cpu")
    x=x-x.max(); p=torch.exp(x); p=p/p.sum()
    return [float(v) for v in p.tolist()]

def capture(model,input_ids,torch):
    got={}
    layer=model.gpt_neox.layers[SOURCE_LAYER]
    def hook(_m,_i,out):
        h=get_h(out); got["v"]=h[0,SOURCE_POS,:].detach().clone(); return out
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad(): model(input_ids=input_ids,use_cache=False)
    finally:
        hh.remove()
    return got["v"]

def patched(model,input_ids,vec,torch):
    layer=model.gpt_neox.layers[SOURCE_LAYER]
    def hook(_m,_i,out):
        h=get_h(out); hp=h.clone()
        hp[:,SOURCE_POS,:]=vec.to(device=hp.device,dtype=hp.dtype)
        return put_h(out,hp)
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad(): out=model(input_ids=input_ids,use_cache=False)
    finally:
        hh.remove()
    return out.logits[0]

def quotient(signatures):
    uniq=sorted(set(signatures.values()))
    m={s:i for i,s in enumerate(uniq)}
    return {q:m[s] for q,s in signatures.items()}

def blocks(q2c):
    g=defaultdict(list)
    for q,c in q2c.items(): g[c].append(q)
    return tuple(sorted(tuple(sorted(v)) for v in g.values()))

def class_probs(q2c):
    cnt=Counter(q2c.values()); n=len(q2c)
    return {c:k/n for c,k in cnt.items()}

def aggregate(q2c,qrows):
    g=defaultdict(list)
    for q,row in qrows.items(): g[q2c[q]].append(row)
    return {
        c:[sum(r[j] for r in rows)/len(rows) for j in range(len(rows[0]))]
        for c,rows in g.items()
    }

def metrics(q2c,raw,uprior):
    cp=class_probs(q2c)
    kernels={}
    for u,qrows in raw.items():
        for c,row in aggregate(q2c,qrows).items():
            kernels[(u,c)]=row
    K=entropy(cp.values())
    B=conditional_mi(cp,uprior,kernels)
    R=None if K==0 else B/K
    if len(cp)>1:
        L=math.log2(len(cp)); kappa=K/L; rcap=B/L
    else:
        kappa=0.0; rcap=0.0
    return {
        "class_count":len(cp),
        "class_probabilities":{str(c):p for c,p in cp.items()},
        "partition_blocks":[list(b) for b in blocks(q2c)],
        "K_eff_bits":K,"B_reentry_bits":B,"R":R,
        "kappa_R":kappa,"R_cap":rcap,
    }

def refines(fine,coarse):
    fb=[set(b) for b in blocks(fine)]
    cb=[set(b) for b in blocks(coarse)]
    return all(any(b<=c for c in cb) for b in fb)

def duplicate_ok(q2c):
    a=class_probs(q2c)
    qdup={}
    for q,c in q2c.items():
        qdup[f"{q}a"]=c; qdup[f"{q}b"]=c
    b=class_probs(qdup)
    return all(abs(a.get(c,0)-b.get(c,0))<1e-15 for c in set(a)|set(b))

def plan():
    return {
        "model_id":MODEL_ID,
        "revision":REVISION,
        "source_layer":SOURCE_LAYER,
        "source_pos":SOURCE_POS,
        "auxiliary":{
            "C3":["K1","K2","K3"],
            "C5":["K1","K2","K3","E0","E1"],
            "C7":["K1","K2","K3","E0","E1","E2","E3"],
            "fixed_future_filler":"F1",
        },
        "holdout":[["K4","F4"],["K4","F5"],["K5","F4"],["K5","F5"]],
        "no_numeric_target":True,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",default="TRANSFORMER_REENTRY_SOURCE_QUOTIENT_RESULT_v0.3.json")
    ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args()

    if args.plan_only:
        print(json.dumps(plan(),indent=2,sort_keys=True))
        return 0

    import numpy as np
    import torch, transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM

    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    eligible=select_eligible(tokenizer,25)

    historical_ids=[x[0] for x in eligible[:21]]
    if historical_ids!=EXPECTED_HISTORICAL_IDS:
        raise RuntimeError(
            "Historical token allocation changed; refusing v0.3 execution.\n"
            f"expected={EXPECTED_HISTORICAL_IDS}\nactual={historical_ids}"
        )

    candidates=eligible[:8]
    keys=eligible[8:14]
    fillers=eligible[14:20]
    anchor=eligible[20]
    extra_keys=eligible[21:25]

    cand_ids=[x[0] for x in candidates]
    key_ids=[x[0] for x in keys]
    filler_ids=[x[0] for x in fillers]
    anchor_id=anchor[0]
    extra_ids=[x[0] for x in extra_keys]

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,revision=REVISION,torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    cfg=model.config
    if cfg.model_type!="gpt_neox" or int(cfg.num_hidden_layers)!=6 or int(cfg.hidden_size)!=512 or int(cfg.num_attention_heads)!=8:
        raise RuntimeError("Frozen architecture check failed")

    donors={}
    for q,vid in enumerate(cand_ids):
        donors[q]=capture(
            model,
            ids([key_ids[0],vid,filler_ids[0]],torch,args.device),
            torch
        )

    aux_keys3=[key_ids[1],key_ids[2],key_ids[3]]
    aux_keys5=aux_keys3+extra_ids[:2]
    aux_keys7=aux_keys5+extra_ids[2:]
    aux_sets={"C3":aux_keys3,"C5":aux_keys5,"C7":aux_keys7}

    signatures={}
    margins={}
    q2c={}
    for name,akeys in aux_sets.items():
        sig={}
        for q,vec in donors.items():
            symbols=[]; qm=[]
            for kid in akeys:
                inp=ids([kid,anchor_id,filler_ids[1]],torch,args.device)
                logits=patched(model,inp,vec,torch)
                probs=softmax_subset(logits[SOURCE_POS],cand_ids,torch)
                order=sorted(probs,reverse=True)
                symbols.append(max(range(len(probs)),key=lambda i:probs[i]))
                qm.append(order[0]-order[1])
            sig[q]=tuple(symbols)
            margins.setdefault(str(q),{})[name]=qm
        signatures[name]=sig
        q2c[name]=quotient(sig)

    raw_ret={}; raw_pre={}
    for u,(kj,fk) in enumerate(HOLDOUT):
        inp=ids([key_ids[kj],anchor_id,filler_ids[fk],key_ids[kj]],torch,args.device)
        qr={}; qp={}
        for q,vec in donors.items():
            logits=patched(model,inp,vec,torch)
            qr[q]=softmax_subset(logits[RETURN_POS],cand_ids,torch)
            qp[q]=softmax_subset(logits[PRE_POS],cand_ids,torch)
        raw_ret[u]=qr; raw_pre[u]=qp

    uprior={u:1/len(HOLDOUT) for u in range(len(HOLDOUT))}
    qprior={q:1/len(donors) for q in donors}
    qk_ret={(u,q):row for u,rows in raw_ret.items() for q,row in rows.items()}
    qk_pre={(u,q):row for u,rows in raw_pre.items() for q,row in rows.items()}
    BQ=conditional_mi(qprior,uprior,qk_ret)
    BQpre=conditional_mi(qprior,uprior,qk_pre)

    outm={}; pre={}
    for name in ("C3","C5","C7"):
        m=metrics(q2c[name],raw_ret,uprior)
        mp=metrics(q2c[name],raw_pre,uprior)
        m["eta_vs_Q"]=None if BQ==0 else m["B_reentry_bits"]/BQ
        m["residual_I_Q_Y_given_C_U_bits"]=BQ-m["B_reentry_bits"]
        m["duplication_invariance"]=duplicate_ok(q2c[name])
        outm[name]=m
        pre[name]=mp["B_reentry_bits"]

    nested=refines(q2c["C5"],q2c["C3"]) and refines(q2c["C7"],q2c["C5"])
    if not nested:
        pstatus="SQ-INCONSISTENT"
    elif blocks(q2c["C5"])==blocks(q2c["C7"]):
        pstatus="SQ-STABLE"
    else:
        pstatus="SQ-REFINING"

    causal_ok=(BQpre<NEGATIVE_MI_TOL
               and all(v<NEGATIVE_MI_TOL for v in pre.values())
               and all(outm[n]["duplication_invariance"] for n in outm))
    verdict="TR3-V1" if causal_ok and nested else "TR3-V3"

    result={
        "schema":"TRANSFORMER-REENTRY-SOURCE-QUOTIENT-result-v0.3",
        "verdict":verdict,
        "partition_status":pstatus,
        "plan":plan(),
        "runtime":{
            "python":platform.python_version(),
            "torch":torch.__version__,
            "transformers":transformers.__version__,
            "numpy":np.__version__,
            "device":args.device,
        },
        "model":{"id":MODEL_ID,"revision":REVISION,"config":cfg.to_dict()},
        "tokens":{
            "candidates":[{"id":i,"text":t} for i,t in candidates],
            "keys":[{"id":i,"text":t} for i,t in keys],
            "fillers":[{"id":i,"text":t} for i,t in fillers],
            "anchor":{"id":anchor[0],"text":anchor[1]},
            "extra_keys":[{"id":i,"text":t} for i,t in extra_keys],
        },
        "auxiliary":{
            "signatures":{
                n:{str(q):list(s) for q,s in signatures[n].items()}
                for n in signatures
            },
            "q_to_class":{
                n:{str(q):c for q,c in q2c[n].items()}
                for n in q2c
            },
            "margins":margins,
            "min_margin":min(v for q in margins.values() for a in q.values() for v in a),
        },
        "holdout":{
            "B_Q_reentry_bits":BQ,
            "B_Q_pre_bits":BQpre,
            "quotients":outm,
            "B_pre_by_quotient_bits":pre,
            "raw_q_return_kernels":{
                str(u):{str(q):row for q,row in rows.items()}
                for u,rows in raw_ret.items()
            },
        },
    }

    out=Path(args.output)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("VERDICT:",verdict)
    print("partition_status:",pstatus)
    print("B_Q_reentry_bits:",BQ)
    print("B_Q_pre_bits:",BQpre)
    for n in ("C3","C5","C7"):
        m=outm[n]
        print(n,
              "classes=",m["class_count"],
              "K_eff_bits=",m["K_eff_bits"],
              "B_reentry_bits=",m["B_reentry_bits"],
              "R=",m["R"],
              "eta_vs_Q=",m["eta_vs_Q"],
              "residual=",m["residual_I_Q_Y_given_C_U_bits"],
              "B_pre=",pre[n])
    print("min_margin:",result["auxiliary"]["min_margin"])
    print("extra_keys:",[(i,t) for i,t in extra_keys])
    print("output:",out)
    return 0 if verdict=="TR3-V1" else 2

if __name__=="__main__":
    raise SystemExit(main())
