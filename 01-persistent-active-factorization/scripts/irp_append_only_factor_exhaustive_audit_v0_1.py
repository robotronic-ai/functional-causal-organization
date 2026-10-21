#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF v0.1 exhaustive finite audit.
Exact arithmetic, no PRNG.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import math

HERE = Path(__file__).resolve().parent
PUB = HERE / "IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json"


def frac(s):
    a,b=s.split("/")
    return Fraction(int(a),int(b))


def load_rows():
    return json.loads(PUB.read_text(encoding="utf-8"))["fixtures"]


def kernel(row):
    K={}
    for key,d in row["transition"].items():
        s,u=map(int,key.split("|"))
        K[(s,u)]={int(y):frac(p) for y,p in d.items()}
    return K


def partitions(seq):
    seq=list(seq)
    if not seq:
        yield []
        return
    first=seq[0]
    for rest in partitions(seq[1:]):
        yield [frozenset([first])] + rest
        for i in range(len(rest)):
            yield rest[:i] + [frozenset(set(rest[i])|{first})] + rest[i+1:]


def unique_partitions(seq):
    seen=set()
    out=[]
    for p in partitions(seq):
        key=tuple(sorted((tuple(sorted(b)) for b in p), key=lambda x:x[0]))
        if key not in seen:
            seen.add(key)
            out.append(tuple(frozenset(b) for b in key))
    return out


def factor_map(partition):
    return {s:i for i,b in enumerate(partition) for s in b}


def quotient_if_lumpable(row,partition):
    K=kernel(row)
    us=tuple(row["contexts"])
    h=factor_map(partition)
    n=len(partition)
    Q={}
    for u in us:
        for i,block in enumerate(partition):
            ref=None
            for s in block:
                d=defaultdict(Fraction)
                for y,p in K[(s,u)].items():
                    d[h[y]] += p
                d={j:d.get(j,Fraction(0)) for j in range(n)}
                if ref is None:
                    ref=d
                elif ref != d:
                    return None
            Q[(i,u)]=ref
    return Q


def all_partial_orders_with_bottom_and_join(n):
    elems=range(n)
    pairs=[(i,j) for i in elems for j in elems if i!=j]
    out=[]
    for mask in range(1<<len(pairs)):
        le={(i,i) for i in elems}
        for k,p in enumerate(pairs):
            if mask & (1<<k):
                le.add(p)

        if any(i!=j and (i,j) in le and (j,i) in le for i in elems for j in elems):
            continue

        bad=False
        for i in elems:
            for j in elems:
                if (i,j) not in le:
                    continue
                for k in elems:
                    if (j,k) in le and (i,k) not in le:
                        bad=True
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            continue

        bottoms=[b for b in elems if all((b,x) in le for x in elems)]
        if len(bottoms)!=1:
            continue
        bottom=bottoms[0]

        join={}
        ok=True
        for a in elems:
            for b in elems:
                upp=[x for x in elems if (a,x) in le and (b,x) in le]
                mins=[x for x in upp if not any(y!=x and (y,x) in le for y in upp)]
                if len(mins)!=1:
                    ok=False
                    break
                join[(a,b)]=mins[0]
            if not ok:
                break
        if ok:
            out.append((frozenset(le),bottom,join))
    return out


PO_CACHE={n:all_partial_orders_with_bottom_and_join(n) for n in range(1,5)}


def append_only_structures(row,partition,Q):
    n=len(partition)
    us=tuple(row["contexts"])
    hits=[]
    for le,bottom,join in PO_CACHE[n]:
        mus={}
        ok=True
        for u in us:
            mu=dict(Q[(bottom,u)])
            for l in range(n):
                pred=defaultdict(Fraction)
                for a,p in mu.items():
                    if p:
                        pred[join[(l,a)]] += p
                pred={j:pred.get(j,Fraction(0)) for j in range(n)}
                if pred != Q[(l,u)]:
                    ok=False
                    break
            if not ok:
                break
            mus[u]=mu
        if ok:
            hits.append((le,bottom,join,mus))
    return hits


def refines(p_fine,p_coarse):
    return all(any(set(bf)<=set(bc) for bc in p_coarse) for bf in p_fine)


def factor_cmi_bits(row,partition,Q):
    q={int(s):frac(p) for s,p in row["q"]}
    uq={int(u):frac(p) for u,p in row["context_q"]}
    h=factor_map(partition)
    joint=defaultdict(Fraction)
    for s,ps in q.items():
        l=h[s]
        for u,pu in uq.items():
            for lp,p in Q[(l,u)].items():
                joint[(l,lp,u)] += ps*pu*p

    pu=defaultdict(Fraction)
    plu=defaultdict(Fraction)
    plpu=defaultdict(Fraction)
    pllpu=defaultdict(Fraction)
    for (l,lp,u),p in joint.items():
        pu[u]+=p
        plu[(l,u)]+=p
        plpu[(lp,u)]+=p
        pllpu[(l,lp,u)]+=p

    I=0.0
    for (l,lp,u),p in pllpu.items():
        if not p:
            continue
        r=Fraction(p*pu[u],plu[(l,u)]*plpu[(lp,u)])
        I += float(p)*math.log2(float(r))
    return I


def main():
    rows=load_rows()
    print("=== AF v0.1 EXHAUSTIVE APPEND-ONLY FACTOR AUDIT ===")
    print("join-semilattice counts by n:", {n:len(PO_CACHE[n]) for n in PO_CACHE})

    family_summary=defaultdict(lambda:defaultdict(int))

    for row in rows:
        states=tuple(row["alphabet"])
        factors=[]
        for part in unique_partitions(states):
            if len(part)==1:
                continue
            Q=quotient_if_lumpable(row,part)
            if Q is None:
                continue
            structs=append_only_structures(row,part,Q)
            if not structs:
                continue
            I=factor_cmi_bits(row,part,Q)
            factors.append({
                "partition":part,
                "n_structures":len(structs),
                "I":I,
            })

        maximal=[]
        for f in factors:
            if not any(
                g["partition"] != f["partition"]
                and refines(g["partition"],f["partition"])
                for g in factors
            ):
                maximal.append(f)

        informative=[f for f in factors if f["I"]>1e-12]
        max_info=[f for f in maximal if f["I"]>1e-12]

        if not factors:
            verdict="AF-NONE"
        elif not informative:
            verdict="AF-SAT"
        elif len(max_info)==1:
            verdict="AF-U1"
        else:
            verdict="AF-MULTI"

        print(
            f"{row['fixture_id']:8s} {row['family']:4s} "
            f"factors={len(factors):2d} informative={len(informative):2d} "
            f"maximal={len(maximal):2d} max_info={len(max_info):2d} {verdict}"
        )
        for f in max_info[:3]:
            print("   max_info partition=",
                  [tuple(sorted(b)) for b in f["partition"]],
                  f"I={f['I']:.12g}",
                  f"structures={f['n_structures']}")

        family_summary[row["family"]][verdict]+=1

    print("\n=== FAMILY SUMMARY ===")
    for fam in ("IR7","IR8","IR9","IR10","IR11","IR12"):
        print(fam,dict(family_summary[fam]))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
