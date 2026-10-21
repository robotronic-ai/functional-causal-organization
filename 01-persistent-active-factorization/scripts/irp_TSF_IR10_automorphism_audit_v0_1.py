#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TSF-AUT v0.1 exact self-automorphism audit for canonical IR10.01."""

from itertools import permutations
from pathlib import Path
from fractions import Fraction
import json

HERE=Path(__file__).resolve().parent
PUB=HERE/"IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json"


def F(s):
    a,b=s.split("/")
    return Fraction(int(a),int(b))


def load_ir10():
    rows=json.loads(PUB.read_text(encoding="utf-8"))["fixtures"]
    return next(r for r in rows if r["fixture_id"]=="IR10.01")


def K_of(row):
    K={}
    for key,d in row["transition"].items():
        s,u=map(int,key.split("|"))
        K[(s,u)]={int(y):F(p) for y,p in d.items()}
    return K


def transported_dist(d,sp):
    return {sp[y]:p for y,p in d.items()}


def automorphisms(row):
    K=K_of(row)
    S=tuple(row["alphabet"])
    U=tuple(row["contexts"])
    hits=[]
    for ps in permutations(S):
        sp=dict(zip(S,ps))
        for pu in permutations(U):
            up=dict(zip(U,pu))
            ok=True
            for s in S:
                for u in U:
                    if transported_dist(K[(s,u)],sp) != K[(sp[s],up[u])]:
                        ok=False
                        break
                if not ok:
                    break
            if ok:
                hits.append((sp,up))
    return hits


def canon_partition(blocks):
    return tuple(sorted(
        (tuple(sorted(b)) for b in blocks),
        key=lambda x:(len(x),x)
    ))


def transport_partition(part,sp):
    return canon_partition([
        {sp[x] for x in block}
        for block in part
    ])


def orbit(part,autos):
    return {
        transport_partition(part,sp)
        for sp,_up in autos
    }


def main():
    row=load_ir10()
    autos=automorphisms(row)

    P=canon_partition([{0,2},{1,3}])
    C1=canon_partition([{0},{1,2,3}])
    C2=canon_partition([{2},{0,1,3}])
    coarse={P,C1,C2}

    AF=canon_partition([{0},{1,3},{2}])

    porbit=orbit(P,autos)
    aforbit=orbit(AF,autos)

    print("=== TSF-AUT v0.1 IR10.01 ===")
    print("O1_self_automorphism_count:",len(autos))
    print("coarse_TS_factors:",sorted(coarse))
    print("P_orbit_size:",len(porbit))
    print("P_orbit:",sorted(porbit))
    print("P_orbit_equals_all_coarse:",porbit==coarse)
    print("AF_three_class_orbit_size:",len(aforbit))
    print("AF_three_class_orbit:",sorted(aforbit))

    stabilizer=[
        (sp,up) for sp,up in autos
        if transport_partition(P,sp)==P
    ]
    print("P_stabilizer_size:",len(stabilizer))

    if porbit==coarse and len(coarse)>1:
        print("\nVERDICT: AUT-TRANSITIVE")
        print(
            "No deterministic O1-equivariant state-factor selector can choose "
            "the historical P partition uniquely from this orbit."
        )
        return 3
    if len(porbit)>1:
        print("\nVERDICT: AUT-ORBIT")
        return 2

    print("\nVERDICT: AUT-U")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
