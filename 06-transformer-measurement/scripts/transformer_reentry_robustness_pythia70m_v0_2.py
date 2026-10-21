#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRANSFORMER-REENTRY ROBUSTNESS v0.2.

Follow-up after v0.1. The model, revision, source hook and donor interventions
are frozen. Quotients C3/C6/C9 are built exclusively from auxiliary contexts
using key/filler indices 1..3. Return-channel holdout uses only indices 4..5.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

MODEL_ID="EleutherAI/pythia-70m-deduped"
REVISION="5ff092d907c3d8ba420d9fbef792426789eb2cd2"

SOURCE_LAYER=2
SOURCE_POS=1
RETURN_POS=3
PRE_POS=0
N_CANDIDATES=8
NEGATIVE_MI_TOL=1e-8
TOKEN_RE=re.compile(r"^ ?[A-Za-z]{2,10}$")

AUX3=((1,1),(2,2),(3,3))
AUX6=AUX3+((1,2),(2,3),(3,1))
AUX9=tuple((j,k) for j in (1,2,3) for k in (1,2,3))
HOLDOUT=tuple((j,k) for j in (4,5) for k in (4,5))

def entropy(probs):
    return -sum(p*math.log2(p) for p in probs if p>0)

def conditional_mi_from_kernels(state_probs, u_probs, kernels):
    """kernels[(u,state)] -> list p(y)."""
    joint=defaultdict(float)
    for u,pu in u_probs.items():
        for s,ps in state_probs.items():
            row=kernels[(u,s)]
            for y,p in enumerate(row):
                joint[(s,y,u)] += pu*ps*p

    p_u=defaultdict(float)
    p_su=defaultdict(float)
    p_yu=defaultdict(float)
    for (s,y,u),p in joint.items():
        p_u[u]+=p
        p_su[(s,u)]+=p
        p_yu[(y,u)]+=p

    out=0.0
    for (s,y,u),p in joint.items():
        if p<=0:
            continue
        den=p_su[(s,u)]*p_yu[(y,u)]
        if den<=0:
            continue
        out += p*math.log2((p*p_u[u])/den)
    return out

def select_tokens(tokenizer):
    special=set(tokenizer.all_special_ids)
    selected=[]
    for tid in range(len(tokenizer)):
        if tid in special:
            continue
        txt=tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not TOKEN_RE.match(txt):
            continue
        if tokenizer.encode(txt, add_special_tokens=False) != [tid]:
            continue
        selected.append((tid,txt))
        if len(selected)>=21:
            break
    if len(selected)<21:
        raise RuntimeError(f"Only {len(selected)} eligible single tokens found")
    return selected[:8], selected[8:14], selected[14:20], selected[20]

def get_layer_output_tensor(output):
    return output[0] if isinstance(output, tuple) else output

def replace_layer_output(output, patched_tensor):
    return (patched_tensor,)+output[1:] if isinstance(output, tuple) else patched_tensor

def softmax_subset(logits, token_ids, torch):
    x=logits[token_ids].detach().to(dtype=torch.float64,device="cpu")
    x=x-x.max()
    p=torch.exp(x)
    p=p/p.sum()
    return [float(v) for v in p.tolist()]

def capture_activation(model,input_ids,layer_idx,pos,torch):
    captured={}
    layer=model.gpt_neox.layers[layer_idx]
    def hook(_module,_inp,out):
        h=get_layer_output_tensor(out)
        captured["v"]=h[0,pos,:].detach().clone()
        return out
    handle=layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids,use_cache=False)
    finally:
        handle.remove()
    return captured["v"]

def run_with_patch(model,input_ids,layer_idx,pos,vector,torch):
    layer=model.gpt_neox.layers[layer_idx]
    def hook(_module,_inp,out):
        h=get_layer_output_tensor(out)
        hp=h.clone()
        hp[:,pos,:]=vector.to(device=hp.device,dtype=hp.dtype)
        return replace_layer_output(out,hp)
    handle=layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out=model(input_ids=input_ids,use_cache=False)
    finally:
        handle.remove()
    return out.logits[0]

def tensor_ids(seq,torch,device):
    return torch.tensor([seq],dtype=torch.long,device=device)

def argmax_margin(probs):
    s=sorted(probs,reverse=True)
    return s[0]-s[1]

def quotient_from_signatures(signatures):
    unique=sorted(set(signatures.values()))
    sig_to_class={s:i for i,s in enumerate(unique)}
    q_to_class={q:sig_to_class[s] for q,s in signatures.items()}
    return q_to_class

def partition_blocks(q_to_class):
    groups=defaultdict(list)
    for q,c in q_to_class.items():
        groups[c].append(q)
    return tuple(sorted(tuple(sorted(v)) for v in groups.values()))

def class_probabilities(q_to_class):
    n=len(q_to_class)
    cnt=Counter(q_to_class.values())
    return {c:k/n for c,k in cnt.items()}

def aggregate_q_rows(q_to_class,qrows):
    grouped=defaultdict(list)
    for q,row in qrows.items():
        grouped[q_to_class[q]].append(row)
    out={}
    for c,rows in grouped.items():
        out[c]=[
            sum(r[j] for r in rows)/len(rows)
            for j in range(len(rows[0]))
        ]
    return out

def duplicate_distribution(q_to_class):
    orig=class_probabilities(q_to_class)
    dup_map={}
    for q,c in q_to_class.items():
        dup_map[f"{q}a"]=c
        dup_map[f"{q}b"]=c
    dup=class_probabilities(dup_map)
    return orig,dup

def refinement_status(p3,p6,p9):
    b3=set(partition_blocks(p3))
    b6=set(partition_blocks(p6))
    b9=set(partition_blocks(p9))

    def refines(fine,coarse):
        return all(any(set(b)<=set(c) for c in coarse) for b in fine)

    if not (refines(b6,b3) and refines(b9,b6)):
        return "ROB-INCOMPATIBLE"

    if b6==b9:
        return "ROB-STABLE-PARTITION"
    return "ROB-REFINING"

def metrics_for_partition(q_to_class,raw_q_ret,u_probs):
    cprobs=class_probabilities(q_to_class)
    kernels={}
    for u,qrows in raw_q_ret.items():
        crows=aggregate_q_rows(q_to_class,qrows)
        for c,row in crows.items():
            kernels[(u,c)]=row

    K=entropy(cprobs.values())
    B=conditional_mi_from_kernels(cprobs,u_probs,kernels)
    R=None if K==0 else B/K
    if len(cprobs)>1:
        L=math.log2(len(cprobs))
        kappa=K/L
        rcap=B/L
    else:
        kappa=0.0
        rcap=0.0
    return {
        "class_count":len(cprobs),
        "class_probabilities":{str(c):p for c,p in cprobs.items()},
        "partition_blocks":[list(b) for b in partition_blocks(q_to_class)],
        "K_eff_bits":K,
        "B_reentry_bits":B,
        "R":R,
        "kappa_R":kappa,
        "R_cap":rcap,
    }

def plan():
    return {
        "model_id":MODEL_ID,
        "revision":REVISION,
        "source_layer":SOURCE_LAYER,
        "source_pos":SOURCE_POS,
        "return_pos":RETURN_POS,
        "pre_pos":PRE_POS,
        "n_candidates":N_CANDIDATES,
        "auxiliary_sets":{
            "C3":[list(x) for x in AUX3],
            "C6":[list(x) for x in AUX6],
            "C9":[list(x) for x in AUX9],
        },
        "holdout_contexts":[list(x) for x in HOLDOUT],
        "quotient_rule":"argmax-signature over auxiliary probes only",
        "raw_reference":"I(Q;Yret|U_H)",
        "negative_control":"I(Q;Ypre|U_H) < 1e-8 bit AND I(Cm;Ypre|U_H) < 1e-8 bit",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",default="TRANSFORMER_REENTRY_ROBUSTNESS_RESULT_v0.2.json")
    ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args()

    if args.plan_only:
        print(json.dumps(plan(),indent=2,sort_keys=True))
        return 0

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM

    torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()

    cfg=model.config
    if cfg.model_type!="gpt_neox":
        raise RuntimeError(f"Expected gpt_neox, got {cfg.model_type}")
    if int(cfg.num_hidden_layers)!=6 or int(cfg.hidden_size)!=512 or int(cfg.num_attention_heads)!=8:
        raise RuntimeError("Frozen architecture check failed")
    if not hasattr(model,"gpt_neox") or not hasattr(model.gpt_neox,"layers"):
        raise RuntimeError("Frozen hook path unavailable")

    candidates,keys,fillers,anchor=select_tokens(tokenizer)
    cand_ids=[x[0] for x in candidates]
    key_ids=[x[0] for x in keys]
    filler_ids=[x[0] for x in fillers]
    anchor_id=anchor[0]

    # Donors fixed exactly as v0.1.
    donors={}
    for qi,vid in enumerate(cand_ids):
        ids=tensor_ids([key_ids[0],vid,filler_ids[0]],torch,args.device)
        donors[qi]=capture_activation(model,ids,SOURCE_LAYER,SOURCE_POS,torch)

    # Compute all 9 auxiliary probe readouts once.
    aux_probs={}
    aux_argmax={}
    aux_margin={}
    for qi,vec in donors.items():
        for jk in AUX9:
            j,k=jk
            ids=tensor_ids([key_ids[j],anchor_id,filler_ids[k]],torch,args.device)
            logits=run_with_patch(model,ids,SOURCE_LAYER,SOURCE_POS,vec,torch)
            probs=softmax_subset(logits[SOURCE_POS],cand_ids,torch)
            aux_probs[(qi,j,k)]=probs
            aux_argmax[(qi,j,k)]=max(range(len(probs)),key=lambda a:probs[a])
            aux_margin[(qi,j,k)]=argmax_margin(probs)

    def make_signatures(probes):
        return {
            qi:tuple(aux_argmax[(qi,j,k)] for j,k in probes)
            for qi in donors
        }

    sig3=make_signatures(AUX3)
    sig6=make_signatures(AUX6)
    sig9=make_signatures(AUX9)
    q2c3=quotient_from_signatures(sig3)
    q2c6=quotient_from_signatures(sig6)
    q2c9=quotient_from_signatures(sig9)

    # Holdout return channel only on j,k in {4,5}.
    raw_q_ret={}
    raw_q_pre={}
    for ui,(j,k) in enumerate(HOLDOUT):
        ids=tensor_ids([key_ids[j],anchor_id,filler_ids[k],key_ids[j]],torch,args.device)
        qret={}
        qpre={}
        for qi,vec in donors.items():
            logits=run_with_patch(model,ids,SOURCE_LAYER,SOURCE_POS,vec,torch)
            qret[qi]=softmax_subset(logits[RETURN_POS],cand_ids,torch)
            qpre[qi]=softmax_subset(logits[PRE_POS],cand_ids,torch)
        raw_q_ret[ui]=qret
        raw_q_pre[ui]=qpre

    u_probs={u:1/len(HOLDOUT) for u in range(len(HOLDOUT))}
    q_probs={q:1/len(donors) for q in donors}

    kernels_q_ret={(u,q):row for u,rows in raw_q_ret.items() for q,row in rows.items()}
    kernels_q_pre={(u,q):row for u,rows in raw_q_pre.items() for q,row in rows.items()}
    BQ=conditional_mi_from_kernels(q_probs,u_probs,kernels_q_ret)
    BQpre=conditional_mi_from_kernels(q_probs,u_probs,kernels_q_pre)

    results={}
    pre_results={}
    for name,qmap in (("C3",q2c3),("C6",q2c6),("C9",q2c9)):
        m=metrics_for_partition(qmap,raw_q_ret,u_probs)
        mp=metrics_for_partition(qmap,raw_q_pre,u_probs)
        m["eta_vs_Q"]=None if BQ==0 else m["B_reentry_bits"]/BQ
        m["residual_I_Q_Y_given_C_U_bits"]=BQ-m["B_reentry_bits"]

        orig,dup=duplicate_distribution(qmap)
        m["duplication_invariance"]=all(
            abs(orig.get(c,0)-dup.get(c,0))<1e-15
            for c in set(orig)|set(dup)
        )
        results[name]=m
        pre_results[name]=mp["B_reentry_bits"]

    status=refinement_status(q2c3,q2c6,q2c9)

    causal_ok=(BQpre<NEGATIVE_MI_TOL
               and all(v<NEGATIVE_MI_TOL for v in pre_results.values())
               and all(results[n]["duplication_invariance"] for n in results))

    if not causal_ok:
        verdict="TR2-V3"
    else:
        verdict="TR2-V1"

    result={
        "schema":"TRANSFORMER-REENTRY-ROBUSTNESS-result-v0.2",
        "verdict":verdict,
        "partition_status":status,
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
        },
        "auxiliary":{
            "signatures":{
                "C3":{str(q):list(s) for q,s in sig3.items()},
                "C6":{str(q):list(s) for q,s in sig6.items()},
                "C9":{str(q):list(s) for q,s in sig9.items()},
            },
            "q_to_class":{
                "C3":{str(q):c for q,c in q2c3.items()},
                "C6":{str(q):c for q,c in q2c6.items()},
                "C9":{str(q):c for q,c in q2c9.items()},
            },
            "argmax_margins":{
                str(q):{
                    f"{j},{k}":aux_margin[(q,j,k)]
                    for j,k in AUX9
                }
                for q in donors
            },
            "min_argmax_margin":min(aux_margin.values()),
        },
        "holdout":{
            "contexts":[{"id":u,"key_index":j,"filler_index":k} for u,(j,k) in enumerate(HOLDOUT)],
            "B_Q_reentry_bits":BQ,
            "B_Q_pre_bits":BQpre,
            "quotients":results,
            "B_pre_by_quotient_bits":pre_results,
            "raw_q_return_kernels":{
                str(u):{str(q):row for q,row in rows.items()}
                for u,rows in raw_q_ret.items()
            },
            "raw_q_pre_kernels":{
                str(u):{str(q):row for q,row in rows.items()}
                for u,rows in raw_q_pre.items()
            },
        },
    }

    out=Path(args.output)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("VERDICT:",verdict)
    print("partition_status:",status)
    print("B_Q_reentry_bits:",BQ)
    print("B_Q_pre_bits:",BQpre)
    for name in ("C3","C6","C9"):
        m=results[name]
        print(name,
              "classes=",m["class_count"],
              "K_eff_bits=",m["K_eff_bits"],
              "B_reentry_bits=",m["B_reentry_bits"],
              "R=",m["R"],
              "eta_vs_Q=",m["eta_vs_Q"],
              "residual=",m["residual_I_Q_Y_given_C_U_bits"],
              "B_pre=",pre_results[name])
    print("min_argmax_margin:",min(aux_margin.values()))
    print("output:",out)

    return 0 if verdict=="TR2-V1" else 2

if __name__=="__main__":
    raise SystemExit(main())
