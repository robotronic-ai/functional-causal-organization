#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IPF v0.1 exact deterministic image-profile factor audit."""

from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import importlib.util
import math
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

spec=importlib.util.spec_from_file_location(
    "tsaudit", str(HERE/"irp_transformation_semigroup_audit_v0_1.py")
)
TS=importlib.util.module_from_spec(spec)
sys.modules["tsaudit"]=TS
spec.loader.exec_module(TS)


def image_of(f):
    return frozenset(f)


def ipf_partition(states,elems,eta_classes):
    J={}
    for c,members in enumerate(eta_classes):
        union=set()
        for i in members:
            union |= set(image_of(elems[i]))
        J[c]=frozenset(union)

    profiles={
        s:tuple(int(s in J[c]) for c in range(len(eta_classes)))
        for s in states
    }
    blocks=defaultdict(set)
    for s,p in profiles.items():
        blocks[p].add(s)
    part=tuple(sorted(
        (frozenset(v) for v in blocks.values()),
        key=lambda b:min(b)
    ))
    return J,profiles,part


def deterministic_transition(row):
    K={}
    for key,d in row["transition"].items():
        s,u=map(int,key.split("|"))
        pos=[(int(y),p) for y,p in d.items() if p!="0/1"]
        assert len(pos)==1 and pos[0][1]=="1/1"
        K[(s,u)]=pos[0][0]
    return K


def frac(s):
    a,b=s.split("/")
    return Fraction(int(a),int(b))


def factor_cmi(row,part):
    h={s:i for i,b in enumerate(part) for s in b}
    K=deterministic_transition(row)
    q={int(s):frac(p) for s,p in row["q"]}
    uq={int(u):frac(p) for u,p in row["context_q"]}
    joint=defaultdict(Fraction)
    for s,ps in q.items():
        for u,pu in uq.items():
            y=K[(s,u)]
            joint[(h[s],h[y],u)] += ps*pu

    pu=defaultdict(Fraction); plu=defaultdict(Fraction)
    plyu=defaultdict(Fraction); pllyu=defaultdict(Fraction)
    for (l,ly,u),p in joint.items():
        pu[u]+=p;plu[(l,u)]+=p;plyu[(ly,u)]+=p;pllyu[(l,ly,u)]+=p

    I=0.0
    for (l,ly,u),p in pllyu.items():
        r=Fraction(p*pu[u],plu[(l,u)]*plyu[(ly,u)])
        I+=float(p)*math.log2(float(r))
    return I


def fmt(part):
    return [tuple(sorted(b)) for b in part]


def main():
    rows=TS.load_rows()
    print("=== IPF v0.1 IMAGE-PROFILE FACTOR AUDIT ===")

    for fam in ("IR7","IR8","IR10","IR11","IR12"):
        row=next(r for r in rows if r["family"]==fam)
        states,gens=TS.deterministic_generators(row)
        elems=TS.closure(states,gens)
        mult=TS.multiplication_table(states,elems)
        classes,cid,qmult=TS.maximal_semilattice_congruence(elems,mult)
        J,profiles,part=ipf_partition(states,elems,classes)
        I=factor_cmi(row,part)

        print(f"\n{fam} {row['fixture_id']}")
        print("  |T_SL| =",len(classes))
        print("  class_image_unions =",{c:tuple(sorted(v)) for c,v in J.items()})
        print("  profiles =",profiles)
        print("  IPF_partition =",fmt(part))
        print(f"  I_profile = {I:.12g}")

        if fam=="IR10":
            target={frozenset((0,2)),frozenset((1,3))}
            print("  equals_historical_P =",set(part)==target)
        if fam=="IR12":
            target={frozenset((0,)),frozenset((1,2))}
            print("  equals_hidden_factor =",set(part)==target)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
