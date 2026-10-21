#!/usr/bin/env python3
"""
KV-BOUND boundary theorem self-test v0.3
Pure PyTorch; no external model.

Tests the decomposition

    B_q <= H(Y_t | C_t,U_t) + C_bypass,q

with

    B_q          = I_do(S ; N_{t+1} | C_t,U_t)
    C_bypass,q   = I_do(S ; N_{t+1} | C_t,U_t,Y_t)

where S is the randomized source-intervention label and C_t is carried
persistence invariant under the intervention.

In this autonomous deterministic finite-battery test, U_t is fixed within each
history and C_t is the history itself, so all quantities are computed exactly
by empirical finite-alphabet entropies.

Cases:
  post_cycle     : only path is S -> Y -> N ; C_bypass = 0
  prewrite       : S also changes persistent K/V ; C_bypass > 0
  raw_kv         : Y fixed, direct K/V side channel ; C_bypass = B > 0
  positive_latent: Y fixed, explicit latent side channel ; C_bypass = B > 0
"""

from __future__ import annotations
import math, json
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Sequence, Tuple
import torch
import torch.nn as nn


@dataclass(frozen=True)
class Config:
    seed: int = 28017
    vocab: int = 16
    d_model: int = 8
    labels: Tuple[int, ...] = (-2,-1,1,2)
    histories: Tuple[Tuple[int,...], ...] = (
        (1,2,3,4), (4,2,7,1), (8,3,5,2), (6,9,1,3),
        (2,11,4,7), (13,3,10,1), (5,12,6,2), (14,7,2,9),
    )
    source_amp: float = 0.75
    raw_kv_amp: float = 0.75
    latent_amp: float = 1.00
    quant_step: float = 0.10
    tol: float = 1e-12


def H(vals):
    c = Counter(vals); n = len(vals)
    return -sum((m/n)*math.log2(m/n) for m in c.values())


def H_cond(xvals, yvals):
    """H(X|Y) for aligned finite samples with uniform empirical measure."""
    groups = {}
    for x,y in zip(xvals,yvals):
        groups.setdefault(y, []).append(x)
    n = len(xvals)
    return sum((len(xs)/n)*H(xs) for xs in groups.values())


def I_det_source_response(response):
    """
    With uniform source labels and deterministic response given source,
    I(S;R)=H(R).
    """
    return H(response)


def I_source_response_given_y(response, yvals):
    """
    Deterministic R,Y given S => I(S;R|Y)=H(R|Y).
    """
    return H_cond(response, yvals)


def sig(x: torch.Tensor, cfg: Config):
    return tuple(int(v) for v in torch.round(x/cfg.quant_step).to(torch.int64).tolist())


class TinyAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        d=cfg.d_model
        self.embed=nn.Embedding(cfg.vocab,d,dtype=torch.float64)
        self.Wq=nn.Linear(d,d,bias=False,dtype=torch.float64)
        self.Wk=nn.Linear(d,d,bias=False,dtype=torch.float64)
        self.Wv=nn.Linear(d,d,bias=False,dtype=torch.float64)
        self.Wo=nn.Linear(d,d,bias=False,dtype=torch.float64)
        self.lm=nn.Linear(d,cfg.vocab,bias=False,dtype=torch.float64)
        for p in self.parameters(): p.requires_grad_(False)
        g=torch.Generator(device="cpu"); g.manual_seed(cfg.seed+1)
        def u():
            x=torch.randn(d,generator=g,dtype=torch.float64)
            return x/x.norm()
        self.d_source=u(); self.d_k=u(); self.d_v=u(); self.d_latent=u()

    def full(self, ids, prewrite=None, post_cycle=None):
        ids_t=torch.tensor(ids,dtype=torch.long)
        x=self.embed(ids_t)
        if prewrite is not None:
            pos,label,amp=prewrite
            x=x.clone(); x[pos]+=label*amp*self.d_source
        q=self.Wq(x); k=self.Wk(x); v=self.Wv(x)
        scores=q@k.T/math.sqrt(self.cfg.d_model)
        T=len(ids)
        scores=scores.masked_fill(torch.triu(torch.ones(T,T,dtype=torch.bool),1),float("-inf"))
        att=torch.softmax(scores,dim=-1)
        h=x+self.Wo(att@v)
        if post_cycle is not None:
            pos,label,amp=post_cycle
            h=h.clone(); h[pos]+=label*amp*self.d_source
        logits=self.lm(h)
        return h,logits,(k.detach().clone(),v.detach().clone())

    def incremental(self, token, cache, latent=None):
        k0,v0=cache
        x=self.embed(torch.tensor([token],dtype=torch.long))[0]
        q=self.Wq(x); kn=self.Wk(x); vn=self.Wv(x)
        k=torch.cat([k0,kn[None,:]],0); v=torch.cat([v0,vn[None,:]],0)
        a=torch.softmax((q[None,:]@k.T/math.sqrt(self.cfg.d_model))[0],dim=-1)
        h=x+self.Wo(a@v)
        if latent is not None:
            label,amp=latent
            h=h+label*amp*self.d_latent
        return h,self.lm(h),(k,v)

    def raw_edit(self, cache, label, amp):
        k,v=cache; k=k.clone(); v=v.clone()
        k[-1]+=label*amp*self.d_k
        v[-1]+=label*amp*self.d_v
        return k,v


def evaluate_history(m, cfg, hist, mode):
    """
    Return aligned Y and N signatures for each randomized source label.

    Natural Y is used for post_cycle/prewrite, because the theorem decomposes
    total source->next-state information into token-mediated and bypass terms.

    raw_kv/positive_latent leave Y at its clean value by construction.
    """
    clean_h,clean_logits,clean_cache=m.full(hist)
    clean_y=int(torch.argmax(clean_logits[-1]))

    ys=[]; ns=[]

    for lab in cfg.labels:
        if mode=="post_cycle":
            # intervention after K/V writes changes current output state only
            h,logits,cache=m.full(hist,post_cycle=(len(hist)-1,lab,cfg.source_amp))
            y=int(torch.argmax(logits[-1]))
            n,_,_=m.incremental(y,cache)

        elif mode=="prewrite":
            # intervention before Q/K/V writes: changes both current output
            # and persistent representation available next cycle
            h,logits,cache=m.full(hist,prewrite=(len(hist)-1,lab,cfg.source_amp))
            y=int(torch.argmax(logits[-1]))
            n,_,_=m.incremental(y,cache)

        elif mode=="raw_kv":
            y=clean_y
            cache=m.raw_edit(clean_cache,lab,cfg.raw_kv_amp)
            n,_,_=m.incremental(y,cache)

        elif mode=="positive_latent":
            y=clean_y
            n,_,_=m.incremental(y,clean_cache,latent=(lab,cfg.latent_amp))

        else:
            raise ValueError(mode)

        ys.append(y); ns.append(sig(n,cfg))

    B = I_det_source_response(ns)
    HY = H(ys)
    Cb = I_source_response_given_y(ns,ys)
    slack = HY + Cb - B
    return {
        "Y": ys,
        "N_signature": [list(x) for x in ns],
        "B_bits": B,
        "H_Y_bits": HY,
        "C_bypass_bits": Cb,
        "rhs_bits": HY+Cb,
        "slack_bits": slack,
    }


def run(cfg=Config()):
    m=TinyAttention(cfg).eval()
    modes=("post_cycle","prewrite","raw_kv","positive_latent")
    rows=[]
    aggregate={k:[] for k in modes}

    # numerical equivalence: clean cached incremental vs full recompute
    eq=[]
    for hist in cfg.histories:
        h0,l0,c0=m.full(hist)
        y0=int(torch.argmax(l0[-1]))
        hc,lc,_=m.incremental(y0,c0)
        hf,lf,_=m.full(tuple(hist)+(y0,))
        d=max(
            float(torch.linalg.vector_norm(hc-hf[-1])),
            float(torch.linalg.vector_norm(lc-lf[-1]))
        )
        eq.append(d)

        for mode in modes:
            r=evaluate_history(m,cfg,hist,mode)
            r["history"]=list(hist); r["mode"]=mode
            rows.append(r); aggregate[mode].append(r)

    summary={
        "schema":"KV-BOUND-boundary-v0.3",
        "equivalence_max_l2":max(eq),
        "mean":{
            mode:{
                "B_bits":sum(r["B_bits"] for r in rs)/len(rs),
                "H_Y_bits":sum(r["H_Y_bits"] for r in rs)/len(rs),
                "C_bypass_bits":sum(r["C_bypass_bits"] for r in rs)/len(rs),
                "slack_bits":sum(r["slack_bits"] for r in rs)/len(rs),
            } for mode,rs in aggregate.items()
        }
    }

    # The theorem must hold cell-by-cell.
    for r in rows:
        assert r["slack_bits"] >= -cfg.tol, r

    # Clean cached/recompute equivalence.
    assert summary["equivalence_max_l2"] < 1e-10

    # Critical discriminations.
    assert all(abs(r["C_bypass_bits"]) < cfg.tol for r in aggregate["post_cycle"])
    assert any(r["C_bypass_bits"] > 0 for r in aggregate["prewrite"])
    assert all(abs(r["H_Y_bits"]) < cfg.tol for r in aggregate["raw_kv"])
    assert any(r["C_bypass_bits"] > 0 for r in aggregate["raw_kv"])
    assert all(abs(r["H_Y_bits"]) < cfg.tol for r in aggregate["positive_latent"])
    assert any(r["C_bypass_bits"] > 0 for r in aggregate["positive_latent"])

    summary["PASS"]=True
    summary["interpretation"]=(
        "The exact finite-battery inequality B <= H(Y|carry)+C_bypass holds "
        "in every cell. Post-cycle interventions have zero conditional bypass; "
        "pre-write, raw-KV and explicit latent channels produce positive bypass."
    )
    return rows,summary


if __name__=="__main__":
    rows,summary=run()
    out=Path(__file__).resolve().parent
    (out/"cells.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
