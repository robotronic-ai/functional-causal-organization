#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS v0.1 exact transformation-semigroup audit, deterministic fixtures only."""

from pathlib import Path
from collections import defaultdict
import json

HERE=Path(__file__).resolve().parent
PUB=HERE/"IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json"


def load_rows():
    return json.loads(PUB.read_text(encoding="utf-8"))["fixtures"]


def deterministic_generators(row):
    states=tuple(row["alphabet"])
    gens={}
    for u in row["contexts"]:
        f=[]
        for s in states:
            d=row["transition"][f"{s}|{u}"]
            pos=[(int(y),p) for y,p in d.items() if p!="0/1"]
            if len(pos)!=1 or pos[0][1]!="1/1":
                return None
            f.append(pos[0][0])
        gens[int(u)]=tuple(f)
    return states,gens


def compose(f,g,states):
    # f after g
    idx={s:i for i,s in enumerate(states)}
    return tuple(f[idx[g[idx[s]]]] for s in states)


def closure(states,gens):
    elems=set(gens.values())
    changed=True
    while changed:
        changed=False
        cur=list(elems)
        for f in cur:
            for g in cur:
                h=compose(f,g,states)
                if h not in elems:
                    elems.add(h);changed=True
    return tuple(sorted(elems))


def multiplication_table(states,elems):
    index={f:i for i,f in enumerate(elems)}
    mult={}
    for i,f in enumerate(elems):
        for j,g in enumerate(elems):
            mult[(i,j)]=index[compose(f,g,states)]
    return mult


class DSU:
    def __init__(self,n):
        self.p=list(range(n))
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return False
        self.p[b]=a
        return True


def maximal_semilattice_congruence(elems,mult):
    n=len(elems)
    d=DSU(n)

    # Seed identities x ~ x^2 and xy ~ yx.
    for x in range(n):
        d.union(x,mult[(x,x)])
    for x in range(n):
        for y in range(n):
            d.union(mult[(x,y)],mult[(y,x)])

    # Congruence closure under multiplication on both sides.
    changed=True
    while changed:
        changed=False
        classes=defaultdict(list)
        for x in range(n):
            classes[d.find(x)].append(x)
        pairs=[]
        for members in classes.values():
            for a in members:
                for b in members:
                    pairs.append((a,b))
        for a,b in pairs:
            for z in range(n):
                if d.union(mult[(z,a)],mult[(z,b)]):
                    changed=True
                if d.union(mult[(a,z)],mult[(b,z)]):
                    changed=True

    classes=defaultdict(list)
    for x in range(n):
        classes[d.find(x)].append(x)
    cls=list(classes.values())
    cid={x:i for i,c in enumerate(cls) for x in c}

    qmult={}
    for i,c in enumerate(cls):
        for j,e in enumerate(cls):
            vals={cid[mult[(a,b)]] for a in c for b in e}
            assert len(vals)==1
            qmult[(i,j)]=next(iter(vals))

    # Exact semilattice check.
    for i in range(len(cls)):
        assert qmult[(i,i)]==i
        for j in range(len(cls)):
            assert qmult[(i,j)]==qmult[(j,i)]

    return cls,cid,qmult


def nontrivial_permutation_elements(states,elems):
    return [
        i for i,f in enumerate(elems)
        if len(set(f))==len(states) and any(f[k]!=states[k] for k in range(len(states)))
    ]


def main():
    rows=load_rows()
    print("=== TS v0.1 TRANSFORMATION SEMIGROUP AUDIT ===")
    for fam in ("IR7","IR8","IR10","IR11","IR12"):
        row=next(r for r in rows if r["family"]==fam)
        dg=deterministic_generators(row)
        assert dg is not None
        states,gens=dg
        elems=closure(states,gens)
        mult=multiplication_table(states,elems)
        cls,cid,qmult=maximal_semilattice_congruence(elems,mult)
        index={f:i for i,f in enumerate(elems)}
        gen_classes={u:cid[index[f]] for u,f in gens.items()}
        perms=nontrivial_permutation_elements(states,elems)

        print(f"\n{fam} {row['fixture_id']}")
        print("  |T| =",len(elems))
        print("  |T_SL| =",len(cls))
        print("  generator_classes =",gen_classes)
        print("  nontrivial_permutation_elements =",perms)
        print("  classes =",cls)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
