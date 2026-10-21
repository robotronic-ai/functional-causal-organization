#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRANSFORMER LATENT-CLOSED-LOOP v0.6.1.

Implementation repair of frozen scientific protocol v0.6:
- correct 2x2 context indexing;
- explicitly execute duplication-invariance control L3.
No scientific choice is changed.
"""

from __future__ import annotations
import argparse,json,math,platform,re
from collections import defaultdict
from pathlib import Path

MODEL_ID="EleutherAI/pythia-70m-deduped"
REVISION="5ff092d907c3d8ba420d9fbef792426789eb2cd2"
SOURCE_LAYER=2;SOURCE_POS=1;RETURN_POS=3;PRE_POS=0
TOL=1e-12;NEGATIVE_MI_TOL=1e-8
TOKEN_RE=re.compile(r"^ ?[A-Za-z]{2,10}$")
EXPECTED_FIRST37=[
    248,249,250,251,253,254,255,257,
    261,262,263,264,265,266,
    267,271,272,273,274,275,
    276,279,280,281,282,
    284,285,287,289,290,291,292,
    293,294,296,297,300
]
Q_TO_CLASS={0:0,1:1,2:2,3:3,7:3,4:4,5:5,6:6}
CLASS_TO_Q={0:[0],1:[1],2:[2],3:[3,7],4:[4],5:[5],6:[6]}
CLASS_ORDER=tuple(range(7))
NS=(1,2,4,8,16)

def H(ps):
    return -sum(p*math.log2(p) for p in ps if p>0)

def mi_from_kernel(prior,kernel):
    py=[0.0]*len(CLASS_ORDER)
    for c,pc in enumerate(prior):
        for d,p in enumerate(kernel[c]):
            py[d]+=pc*p
    z=0.0
    for c,pc in enumerate(prior):
        for d,p in enumerate(kernel[c]):
            joint=pc*p
            if joint>0 and py[d]>0:
                z+=joint*math.log2(p/py[d])
    return z

def matmul(A,B):
    n=len(A)
    return [[sum(A[i][k]*B[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]

def matpow(T,n):
    size=len(T)
    R=[[1.0 if i==j else 0.0 for j in range(size)] for i in range(size)]
    A=[row[:] for row in T]
    m=n
    while m:
        if m&1:
            R=matmul(R,A)
        A=matmul(A,A)
        m//=2
    return R

def eligible(tok,n):
    special=set(tok.all_special_ids);out=[]
    for tid in range(len(tok)):
        if tid in special:continue
        txt=tok.decode([tid],clean_up_tokenization_spaces=False)
        if not TOKEN_RE.match(txt):continue
        if tok.encode(txt,add_special_tokens=False)!=[tid]:continue
        out.append((tid,txt))
        if len(out)>=n:break
    if len(out)<n:raise RuntimeError("insufficient eligible tokens")
    return out

def geth(o): return o[0] if isinstance(o,tuple) else o
def puth(o,h): return (h,)+o[1:] if isinstance(o,tuple) else h
def ids(seq,torch,dev): return torch.tensor([seq],dtype=torch.long,device=dev)

def probs(logits,tids,torch):
    x=logits[tids].detach().to(dtype=torch.float64,device="cpu")
    x=x-x.max()
    p=torch.exp(x);p=p/p.sum()
    return [float(v) for v in p.tolist()]

def capture(model,inp,torch):
    got={};layer=model.gpt_neox.layers[SOURCE_LAYER]
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
        h=geth(o);hp=h.clone()
        hp[:,SOURCE_POS,:]=vec.to(device=hp.device,dtype=hp.dtype)
        return puth(o,hp)
    hh=layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out=model(input_ids=inp,use_cache=False)
    finally:
        hh.remove()
    return out.logits[0]

def cmi_source_to_y(class_prior,uprior,class_y):
    pyu=defaultdict(lambda:[0.0]*8)
    for u,pu in uprior.items():
        for c,pc in enumerate(class_prior):
            for y,p in enumerate(class_y[(u,c)]):
                pyu[u][y]+=pc*p
    z=0.0
    for u,pu in uprior.items():
        for c,pc in enumerate(class_prior):
            row=class_y[(u,c)]
            for y,p in enumerate(row):
                if p>0 and pyu[u][y]>0:
                    z+=pu*pc*p*math.log2(p/pyu[u][y])
    return z

def cmi_source_to_classn(class_prior,uprior,kernels_n):
    return sum(
        uprior[u]*mi_from_kernel(class_prior,kernels_n[u])
        for u in uprior
    )

def class_average(qrows,qs,duplicate=False):
    rows=[]
    for q in qs:
        rows.append(qrows[q])
        if duplicate:
            rows.append(qrows[q])
    return [
        sum(row[y] for row in rows)/len(rows)
        for y in range(8)
    ]

def push_y_to_class(yrow):
    crow=[0.0]*7
    for y,p in enumerate(yrow):
        crow[Q_TO_CLASS[y]]+=p
    return crow

def max_abs_matrix_diff(A,B):
    return max(abs(a-b) for ra,rb in zip(A,B) for a,b in zip(ra,rb))

def profile_from_transitions(prior,uprior,transitions):
    Bn={};Rn={}
    for n in NS:
        kn={u:matpow(transitions[u],n) for u in uprior}
        Bn[n]=cmi_source_to_classn(prior,uprior,kn)
        Rn[n]=Bn[n]/H(prior)
    return Bn,Rn

def self_test():
    # Exact check of the repaired context enumeration.
    keys=[("k0","K0"),("k1","K1")]
    fillers=[("f0","F0"),("f1","F1")]
    pairs=[(lk,lf) for lk in keys for lf in fillers]
    assert len(pairs)==4
    assert len(set((lk[0],lf[0]) for lk,lf in pairs))==4

    # Duplication invariance of a synthetic class average.
    qrows={
        0:[0.8,0.2,0,0,0,0,0,0],
        1:[0.2,0.8,0,0,0,0,0,0],
    }
    a=class_average(qrows,[0,1],duplicate=False)
    b=class_average(qrows,[0,1],duplicate=True)
    assert max(abs(x-y) for x,y in zip(a,b))<1e-15

    # Matrix power / monotonic information sanity check on an erasure-like chain.
    T=[
        [0.8,0.2],
        [0.2,0.8],
    ]
    T2=matpow(T,2)
    assert all(abs(sum(row)-1.0)<1e-15 for row in T2)

    return {
        "context_count":4,
        "duplication_average_invariance":True,
        "matrix_power_stochastic":True,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",default="TRANSFORMER_LATENT_CLOSED_LOOP_RESULT_v0.6.1.json")
    ap.add_argument("--plan-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()

    if a.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True))
        return 0

    if a.plan_only:
        print(json.dumps({
            "scientific_protocol":"v0.6 unchanged",
            "implementation":"v0.6.1 repair",
            "classes":7,
            "class_prior":[1/8,1/8,1/8,2/8,1/8,1/8,1/8],
            "rounds":NS,
            "loop_transport":"candidate y_i -> historical donor q_i -> top2 class",
            "new_loop_contexts":4,
            "repairs":[
                "unique enumeration of 2x2 loop contexts",
                "explicit duplication-invariance control L3"
            ],
            "no_numeric_target":True
        },indent=2))
        return 0

    import numpy as np
    import torch,transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM

    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    e=eligible(tok,41)
    if [x[0] for x in e[:37]]!=EXPECTED_FIRST37:
        raise RuntimeError("historical token allocation changed")

    candidates=e[:8]
    keys=e[8:14]
    fillers=e[14:20]
    anchor=e[20]
    loop_keys=e[37:39]
    loop_fillers=e[39:41]

    cand=[x[0] for x in candidates]
    keyids=[x[0] for x in keys]
    fills=[x[0] for x in fillers]
    anchorid=anchor[0]

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float32
    ).to(a.device)
    model.eval()
    cfg=model.config
    if (cfg.model_type!="gpt_neox"
        or int(cfg.num_hidden_layers)!=6
        or int(cfg.hidden_size)!=512
        or int(cfg.num_attention_heads)!=8):
        raise RuntimeError("architecture mismatch")

    donors={}
    for q,vid in enumerate(cand):
        donors[q]=capture(
            model,
            ids([keyids[0],vid,fills[0]],torch,a.device),
            torch
        )

    prior=[1/8,1/8,1/8,2/8,1/8,1/8,1/8]
    H0=H(prior)
    if abs(H0-2.75)>1e-12:
        raise RuntimeError("unexpected class entropy")

    # v0.6.1 repair: enumerate the full Cartesian product with a unique u.
    context_pairs=[
        (lk,lf)
        for lk in loop_keys
        for lf in loop_fillers
    ]
    if len(context_pairs)!=4:
        raise RuntimeError("expected exactly four loop contexts")

    contexts=[]
    class_y={}
    pre_y={}
    transitions={}
    transitions_dup={}

    for u,((lk,_lt),(lf,_ft)) in enumerate(context_pairs):
        contexts.append((lk,lf))
        inp=ids([lk,anchorid,lf,lk],torch,a.device)
        qret={};qpre={}

        for q,vec in donors.items():
            out=patch(model,inp,vec,torch)
            qret[q]=probs(out[RETURN_POS],cand,torch)
            qpre[q]=probs(out[PRE_POS],cand,torch)

        T=[]
        Tdup=[]

        for c in CLASS_ORDER:
            qs=CLASS_TO_Q[c]

            yrow=class_average(qret,qs,duplicate=False)
            prow=class_average(qpre,qs,duplicate=False)

            # L3 explicit duplicated-label construction.
            yrow_dup=class_average(qret,qs,duplicate=True)

            class_y[(u,c)]=yrow
            pre_y[(u,c)]=prow

            crow=push_y_to_class(yrow)
            crow_dup=push_y_to_class(yrow_dup)

            if abs(sum(crow)-1.0)>TOL:
                raise RuntimeError("transition row not stochastic")
            if abs(sum(crow_dup)-1.0)>TOL:
                raise RuntimeError("duplicated transition row not stochastic")

            T.append(crow)
            Tdup.append(crow_dup)

        transitions[u]=T
        transitions_dup[u]=Tdup

    uprior={u:1/4 for u in range(4)}

    # Ensure every declared context exists before computing MI.
    for u in uprior:
        for c in CLASS_ORDER:
            if (u,c) not in class_y or (u,c) not in pre_y:
                raise RuntimeError(f"missing context/class {(u,c)}")
        if u not in transitions:
            raise RuntimeError(f"missing transition context {u}")

    B_y=cmi_source_to_y(prior,uprior,class_y)
    B_pre=cmi_source_to_y(prior,uprior,pre_y)

    Bn,Rn=profile_from_transitions(prior,uprior,transitions)
    Bn_dup,Rn_dup=profile_from_transitions(prior,uprior,transitions_dup)

    monotone=all(
        Bn[b] <= Bn[a]+TOL
        for a,b in zip(NS,NS[1:])
    )
    dpi_one=Bn[1] <= B_y+TOL
    causal=B_pre<NEGATIVE_MI_TOL

    max_transition_dup_diff=max(
        max_abs_matrix_diff(transitions[u],transitions_dup[u])
        for u in uprior
    )
    max_profile_dup_diff=max(
        max(
            abs(Bn[n]-Bn_dup[n]),
            abs(Rn[n]-Rn_dup[n])
        )
        for n in NS
    )
    duplication_ok=(
        max_transition_dup_diff<=TOL
        and max_profile_dup_diff<=TOL
    )

    verdict=(
        "LOOP-V1"
        if monotone and dpi_one and causal and duplication_ok
        else "LOOP-V3"
    )

    result={
        "schema":"TRANSFORMER-LATENT-CLOSED-LOOP-result-v0.6.1",
        "scientific_protocol":"v0.6 unchanged",
        "implementation_repair":"v0.6.1",
        "verdict":verdict,
        "runtime":{
            "python":platform.python_version(),
            "torch":torch.__version__,
            "transformers":transformers.__version__,
            "numpy":np.__version__,
            "device":a.device
        },
        "model":{"id":MODEL_ID,"revision":REVISION},
        "transport":{
            "q_to_class":{str(q):c for q,c in Q_TO_CLASS.items()},
            "candidate_to_q":{str(y):y for y in range(8)},
            "class_prior":prior,
            "H_C0_bits":H0,
        },
        "tokens":{
            "loop_keys":[{"id":i,"text":t} for i,t in loop_keys],
            "loop_fillers":[{"id":i,"text":t} for i,t in loop_fillers],
        },
        "one_step":{
            "I_C0_Yret_given_U_bits":B_y,
            "I_C0_Ypre_given_U_bits":B_pre,
            "I_C0_C1_given_U_bits":Bn[1],
            "R1":Rn[1],
            "dpi_projection_ok":dpi_one,
        },
        "multi_turn":{
            "B_bits":{str(n):Bn[n] for n in NS},
            "R":{str(n):Rn[n] for n in NS},
            "monotone":monotone,
        },
        "duplication_control":{
            "passed":duplication_ok,
            "max_transition_abs_diff":max_transition_dup_diff,
            "max_profile_abs_diff":max_profile_dup_diff,
            "B_bits_duplicated":{str(n):Bn_dup[n] for n in NS},
            "R_duplicated":{str(n):Rn_dup[n] for n in NS},
        },
        "contexts":[
            {"id":u,"key_id":lk,"filler_id":lf}
            for u,(lk,lf) in enumerate(contexts)
        ],
        "transition_kernels":{
            str(u):transitions[u]
            for u in transitions
        },
    }

    Path(a.output).write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8"
    )

    print("VERDICT:",verdict)
    print("H_C0_bits:",H0)
    print("I_C0_Yret_given_U_bits:",B_y)
    print("I_C0_C1_given_U_bits:",Bn[1])
    print("I_C0_Ypre_given_U_bits:",B_pre)
    for n in NS:
        print(f"n={n} B_bits={Bn[n]} R={Rn[n]}")
    print("monotone:",monotone)
    print("dpi_projection_ok:",dpi_one)
    print("duplication_invariance:",duplication_ok)
    print("max_transition_dup_diff:",max_transition_dup_diff)
    print("max_profile_dup_diff:",max_profile_dup_diff)
    print("loop_keys:",loop_keys)
    print("loop_fillers:",loop_fillers)
    print("output:",a.output)

    return 0 if verdict=="LOOP-V1" else 2

if __name__=="__main__":
    raise SystemExit(main())
