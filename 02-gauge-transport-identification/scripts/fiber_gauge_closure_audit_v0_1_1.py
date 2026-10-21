#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIBER-GAUGE-CLOSURE v0.1 exact post-P4 audit."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import importlib.util
import json
import math
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

spec=importlib.util.spec_from_file_location(
    "G",str(HERE/"opf_P4_fiber_complement_ambiguity_v0_1.py")
)
G=importlib.util.module_from_spec(spec);sys.modules["G"]=G;spec.loader.exec_module(G)

FIX=HERE/"OPF_P4_FIXTURES_v0_1.json"


def load():
    return json.loads(FIX.read_text(encoding="utf-8"))["fixtures"]


def uq_of(row):
    return {int(u):G.F(p) for u,p in row["context_q"]}


def tuple_kernel_to_dict(Kt):
    return {k:dict(v) for k,v in Kt}


def cmi(row,K):
    q=G.q_of(row);uq=uq_of(row)
    joint=defaultdict(Fraction)
    for s,ps in q.items():
        for u,pu in uq.items():
            for y,p in K[(s,u)].items():
                joint[(s,y,u)] += ps*pu*p

    pu=defaultdict(Fraction);psu=defaultdict(Fraction);pyu=defaultdict(Fraction)
    for (s,y,u),p in joint.items():
        pu[u]+=p;psu[(s,u)]+=p;pyu[(y,u)]+=p

    I=0.0
    for (s,y,u),p in joint.items():
        r=Fraction(p*pu[u],psu[(s,u)]*pyu[(y,u)])
        I += float(p)*math.log2(float(r))
    return I


def deterministic_map(row):
    K=G.K_of(row)
    out={}
    for key,d in K.items():
        pos=[(y,p) for y,p in d.items() if p]
        if len(pos)!=1 or pos[0][1]!=1:
            return None
        out[key]=pos[0][0]
    return out


def complements(S,P):
    out=[];seen=set()
    for A0 in G.set_partitions(S):
        c=G.canon(A0)
        if c in seen:continue
        seen.add(c)
        A=tuple(frozenset(b) for b in c)
        if G.orthogonal(P,A,S):
            out.append(A)
    return out


def semantic_checks(row,P,A):
    S=tuple(row["alphabet"]);U=tuple(row["contexts"])
    F=deterministic_map(row)
    hp={x:i for i,b in enumerate(P) for x in b}
    ha={x:i for i,b in enumerate(A) for x in b}
    recon={(ha[s],hp[s]):s for s in S}
    if len(recon)!=len(S):
        return False,"not_product"

    # P autonomy: for fixed (p,u), p' is independent of a.
    for p in range(len(P)):
        for u in U:
            vals=set()
            for a in range(len(A)):
                s=recon[(a,p)]
                vals.add(hp[F[(s,u)]])
            if len(vals)!=1:
                return False,"P_not_autonomous"

    # Full skew-product well-definedness is automatic from bijective coordinates,
    # but verify every (a,p,u) yields exactly one (a',p').
    for a in range(len(A)):
        for p in range(len(P)):
            for u in U:
                s=recon[(a,p)]
                y=F[(s,u)]
                _=ha[y],hp[y]

    return True,"PASS"


def main():
    rows=load()
    Pdefs={
        "P4A":(frozenset((0,2)),frozenset((1,3))),
        "P4B":tuple(frozenset((h,4+h)) for h in range(4)),
        "P4C":tuple(frozenset((h,4+h)) for h in range(4)),
    }

    print("=== FIBER-GAUGE-CLOSURE v0.1 ===")
    for fam,P in Pdefs.items():
        row=next(r for r in rows if r["fixture_id"]==fam+".01")
        S=tuple(row["alphabet"])
        gauges=complements(S,P)
        passed=[]
        for A in gauges:
            ok,why=semantic_checks(row,P,A)
            if not ok:
                print(f"{fam} rejected gauge {G.canon(A)} reason={why}")
                continue
            Kt=G.surgery(row,P,A)
            score=cmi(row,tuple_kernel_to_dict(Kt))
            passed.append((G.canon(A),score))

        vals=sorted({round(x[1],12) for x in passed})
        print(f"\n{fam}")
        print("orthogonal_gauges =",len(gauges))
        print("semantically_admissible_gauges =",len(passed))
        print("distinct_B_scores =",vals)
        for A,b in passed:
            print(" gauge=",A,"B=",f"{b:.12g}")

        if len(vals)==1:
            verdict="GC-B"
        elif len(passed)>1:
            verdict="GC-NONID"
        else:
            verdict="GC-U"
        print("family_verdict =",verdict)

    print("\nGLOBAL VERDICT: GC-NONID if any family has >1 admissible gauge with distinct B.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
