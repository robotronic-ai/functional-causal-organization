#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc diagnostic: all orthogonal fiber gauges, scalar identified set and min."""

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


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))["fixtures"]


def load_oracles(path):
    return {
        x["fixture_id"]:G.K_of({"transition":x["oracle_transition"]})
        for x in json.loads(path.read_text(encoding="utf-8"))["oracles"]
    }


def uq_of(row):
    return {int(u):G.F(p) for u,p in row["context_q"]}


def kernel_tuple_to_dict(Ktuple):
    return {k:dict(d) for k,d in Ktuple}


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


def norm_dict(K):
    return tuple((k,tuple(sorted(d.items()))) for k,d in sorted(K.items()))


def complements(S,P):
    seen=set();out=[]
    for A in G.set_partitions(S):
        c=G.canon(A)
        if c in seen:continue
        seen.add(c)
        A=tuple(frozenset(b) for b in c)
        if G.orthogonal(P,A,S):
            out.append(A)
    return out


def audit_case(name,row,P,oracle=None):
    S=tuple(row["alphabet"])
    comps=complements(S,P)
    scored=[]
    for A in comps:
        kt=G.surgery(row,P,A)
        K=kernel_tuple_to_dict(kt)
        score=cmi(row,K)
        match=(oracle is not None and norm_dict(K)==norm_dict(oracle))
        scored.append((A,score,match))

    vals=[x[1] for x in scored]
    print(f"\n{name}: gauges={len(scored)}")
    print(" distinct_scores=",sorted({round(v,12) for v in vals}))
    print(" B_min=",f"{min(vals):.12g}","B_max=",f"{max(vals):.12g}")
    if oracle is not None:
        matches=[(G.canon(A),b) for A,b,m in scored if m]
        print(" oracle_matching_gauges=",len(matches))
        print(" oracle_matches=",matches)
        if matches:
            ob=matches[0][1]
            print(" oracle_is_min=",abs(ob-min(vals))<1e-12)
            print(" oracle_is_max=",abs(ob-max(vals))<1e-12)


def main():
    hist=load(HERE/"IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json")
    p3=load(HERE/"TSIPF_P3_FIXTURES_v0_1.json")
    p4=load(HERE/"OPF_P4_FIXTURES_v0_1.json")

    histo=load_oracles(HERE/"IRP_ORACLES_IR7_IR12_v0_1_PRIVATE.json")
    p3o=load_oracles(HERE/"TSIPF_P3_ORACLES_v0_1_PRIVATE.json")
    p4o=load_oracles(HERE/"OPF_P4_ORACLES_v0_1_PRIVATE.json")

    cases=[
        ("IR10",next(r for r in hist if r["family"]=="IR10"),
         (frozenset((0,2)),frozenset((1,3))),
         histo["IR10.01"]),
        ("P3D",next(r for r in p3 if r["family"]=="P3D"),
         (frozenset((0,2,4,6)),frozenset((1,3,5,7))),
         p3o["P3D.01"]),
        ("P4A",next(r for r in p4 if r["family"]=="P4A"),
         (frozenset((0,2)),frozenset((1,3))),
         p4o["P4A.01"]),
        ("P4B",next(r for r in p4 if r["family"]=="P4B"),
         tuple(frozenset((h,4+h)) for h in range(4)),
         p4o["P4B.01"]),
        ("P4C",next(r for r in p4 if r["family"]=="P4C"),
         tuple(frozenset((h,4+h)) for h in range(4)),
         p4o["P4C.01"]),
    ]

    print("=== FULL FIBER-GAUGE MIN DIAGNOSTIC ===")
    for args in cases:
        audit_case(*args)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
