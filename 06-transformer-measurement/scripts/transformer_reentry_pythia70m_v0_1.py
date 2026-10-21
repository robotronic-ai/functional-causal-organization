#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRANSFORMER-REENTRY v0.1 — Pythia-70M-deduped empirical runner."""

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
N_CANDIDATES=8
NEGATIVE_MI_TOL=1e-8
TOKEN_RE=re.compile(r"^ ?[A-Za-z]{2,10}$")

def entropy(probs):
    return -sum(p*math.log2(p) for p in probs if p>0)

def conditional_mi_from_kernels(class_probs, u_probs, kernels):
    ys=len(next(iter(kernels.values())))
    joint=defaultdict(float)
    for u,pu in u_probs.items():
        for c,pc in class_probs.items():
            row=kernels[(u,c)]
            for y,p in enumerate(row):
                joint[(c,y,u)] += pu*pc*p
    p_u=defaultdict(float); p_cu=defaultdict(float); p_yu=defaultdict(float)
    for (c,y,u),p in joint.items():
        p_u[u]+=p; p_cu[(c,u)]+=p; p_yu[(y,u)]+=p
    out=0.0
    for (c,y,u),p in joint.items():
        if p<=0: continue
        den=p_cu[(c,u)]*p_yu[(y,u)]
        if den<=0: continue
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
    p=torch.exp(x); p=p/p.sum()
    return [float(v) for v in p.tolist()]

def capture_activation(model, input_ids, layer_idx, pos, torch):
    captured={}
    layer=model.gpt_neox.layers[layer_idx]
    def hook(_module,_inp,out):
        h=get_layer_output_tensor(out)
        captured["v"]=h[0,pos,:].detach().clone()
        return out
    handle=layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["v"]

def run_with_patch(model, input_ids, layer_idx, pos, vector, torch):
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

def run_plain(model,input_ids,torch):
    with torch.no_grad():
        return model(input_ids=input_ids,use_cache=False).logits[0]

def tensor_ids(seq,torch,device):
    return torch.tensor([seq],dtype=torch.long,device=device)

def aggregate_by_class(q_to_class, q_rows):
    grouped=defaultdict(list)
    for q,row in q_rows.items():
        grouped[q_to_class[q]].append(row)
    out={}
    for c,rows in grouped.items():
        out[c]=[sum(r[j] for r in rows)/len(rows) for j in range(len(rows[0]))]
    return out

def class_probabilities(q_to_class):
    n=len(q_to_class); cnt=Counter(q_to_class.values())
    return {c:k/n for c,k in cnt.items()}

def plan():
    return {
        "model_id":MODEL_ID,
        "revision":REVISION,
        "source_layer":SOURCE_LAYER,
        "source_pos":SOURCE_POS,
        "return_pos":RETURN_POS,
        "pre_pos":PRE_POS,
        "n_candidates":N_CANDIDATES,
        "aux_probe_count":3,
        "return_context_count":16,
        "quotient_rule":"tuple of auxiliary argmax candidate labels",
        "keff":"H(C|U)=H(C)",
        "b_reentry":"I(C;Yret|U)",
        "r":"B_reentry / H(C)",
        "negative_control":"I(C;Ypre|U) < 1e-8 bit",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",default="TRANSFORMER_REENTRY_RESULT_v0.1.json")
    ap.add_argument("--plan-only",action="store_true")
    args=ap.parse_args()

    if args.plan_only:
        print(json.dumps(plan(),indent=2,sort_keys=True))
        return 0

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()

    cfg=model.config
    if cfg.model_type!="gpt_neox":
        raise RuntimeError(f"Expected gpt_neox, got {cfg.model_type}")
    if int(cfg.num_hidden_layers)!=6 or int(cfg.hidden_size)!=512 or int(cfg.num_attention_heads)!=8:
        raise RuntimeError("Frozen architecture check failed")
    if not hasattr(model,"gpt_neox") or not hasattr(model.gpt_neox,"layers"):
        raise RuntimeError("Frozen hook path model.gpt_neox.layers unavailable")

    candidates,keys,fillers,anchor=select_tokens(tokenizer)
    cand_ids=[x[0] for x in candidates]
    key_ids=[x[0] for x in keys]
    filler_ids=[x[0] for x in fillers]
    anchor_id=anchor[0]

    donor_vectors={}
    for qi,vid in enumerate(cand_ids):
        ids=tensor_ids([key_ids[0],vid,filler_ids[0]],torch,args.device)
        donor_vectors[qi]=capture_activation(model,ids,SOURCE_LAYER,SOURCE_POS,torch)

    signatures={}
    auxiliary_details={}
    for qi,vec in donor_vectors.items():
        sig=[]; details=[]
        for j in (1,2,3):
            ids=tensor_ids([key_ids[j],anchor_id,filler_ids[j]],torch,args.device)
            logits=run_with_patch(model,ids,SOURCE_LAYER,SOURCE_POS,vec,torch)
            probs=softmax_subset(logits[SOURCE_POS],cand_ids,torch)
            symbol=max(range(len(probs)),key=lambda k:probs[k])
            sig.append(symbol); details.append(probs)
        signatures[qi]=tuple(sig)
        auxiliary_details[str(qi)]=details

    unique_sigs=sorted(set(signatures.values()))
    sig_to_class={s:i for i,s in enumerate(unique_sigs)}
    q_to_class={q:sig_to_class[s] for q,s in signatures.items()}
    class_probs=class_probabilities(q_to_class)

    contexts=[(kj,fk) for kj in (1,2,3,4) for fk in (1,2,3,4)]
    kernels_ret={}; kernels_pre={}
    plain_rows={}; raw_q_ret={}; raw_q_pre={}

    for ui,(kj,fk) in enumerate(contexts):
        ids=tensor_ids([key_ids[kj],anchor_id,filler_ids[fk],key_ids[kj]],torch,args.device)
        plain=run_plain(model,ids,torch)
        plain_rows[str(ui)]=softmax_subset(plain[RETURN_POS],cand_ids,torch)

        qret={}; qpre={}
        for qi,vec in donor_vectors.items():
            logits=run_with_patch(model,ids,SOURCE_LAYER,SOURCE_POS,vec,torch)
            qret[qi]=softmax_subset(logits[RETURN_POS],cand_ids,torch)
            qpre[qi]=softmax_subset(logits[PRE_POS],cand_ids,torch)

        raw_q_ret[str(ui)]={str(q):row for q,row in qret.items()}
        raw_q_pre[str(ui)]={str(q):row for q,row in qpre.items()}
        cret=aggregate_by_class(q_to_class,qret)
        cpre=aggregate_by_class(q_to_class,qpre)
        for c,row in cret.items(): kernels_ret[(ui,c)]=row
        for c,row in cpre.items(): kernels_pre[(ui,c)]=row

    u_probs={u:1/len(contexts) for u in range(len(contexts))}
    B=conditional_mi_from_kernels(class_probs,u_probs,kernels_ret)
    Bpre=conditional_mi_from_kernels(class_probs,u_probs,kernels_pre)
    K=entropy(class_probs.values())

    if len(class_probs)>1:
        nominal=math.log2(len(class_probs))
        kappa=K/nominal
        rcap=B/nominal
    else:
        kappa=0.0; rcap=0.0

    R=None if K==0 else B/K

    doubled={}
    for q,c in q_to_class.items():
        doubled[f"{q}a"]=c; doubled[f"{q}b"]=c
    dup=class_probabilities(doubled)
    duplication_ok=all(abs(class_probs.get(c,0)-dup.get(c,0))<1e-15 for c in set(class_probs)|set(dup))

    if Bpre>=NEGATIVE_MI_TOL or not duplication_ok:
        verdict="TR-V3"; publish_R=None
    elif K==0:
        verdict="TR-V2"; publish_R=None
    else:
        verdict="TR-V1"; publish_R=R

    result={
        "schema":"TRANSFORMER-REENTRY-result-v0.1",
        "verdict":verdict,
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
        "quotient":{
            "signatures":{str(q):list(s) for q,s in signatures.items()},
            "q_to_class":{str(q):c for q,c in q_to_class.items()},
            "class_probabilities":{str(c):p for c,p in class_probs.items()},
            "class_count":len(class_probs),
            "auxiliary_details":auxiliary_details,
        },
        "contexts":[{"id":i,"key_index":k,"filler_index":f} for i,(k,f) in enumerate(contexts)],
        "raw_q_return_kernels":raw_q_ret,
        "raw_q_pre_kernels":raw_q_pre,
        "plain_return_rows":plain_rows,
        "metrics":{
            "K_eff_bits":K,
            "B_reentry_bits":B,
            "R":publish_R,
            "kappa_R":kappa,
            "R_cap":rcap,
            "B_pre_bits":Bpre,
            "negative_control_threshold_bits":NEGATIVE_MI_TOL,
            "duplication_invariance":duplication_ok,
        },
    }

    out=Path(args.output)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VERDICT:",verdict)
    print("classes:",len(class_probs))
    print("K_eff_bits:",K)
    print("B_reentry_bits:",B)
    print("R:",publish_R)
    print("kappa_R:",kappa)
    print("R_cap:",rcap)
    print("B_pre_bits:",Bpre)
    print("output:",out)
    return 0 if verdict in ("TR-V1","TR-V2") else 2

if __name__=="__main__":
    raise SystemExit(main())
