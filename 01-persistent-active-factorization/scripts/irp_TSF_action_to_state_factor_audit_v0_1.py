#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TSF v0.1 exact audit: faithful state factors of the least semilattice action quotient."""

from pathlib import Path
from collections import defaultdict
import importlib.util
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

spec=importlib.util.spec_from_file_location(
    "tsaudit", str(HERE/"irp_transformation_semigroup_audit_v0_1.py")
)
TS=importlib.util.module_from_spec(spec)
sys.modules["tsaudit"]=TS
spec.loader.exec_module(TS)


def partitions(seq):
    seq=list(seq)
    if not seq:
        yield []
        return
    first=seq[0]
    for rest in partitions(seq[1:]):
        yield [frozenset([first])] + rest
        for i in range(len(rest)):
            yield rest[:i]+[frozenset(set(rest[i])|{first})]+rest[i+1:]


def unique_partitions(seq):
    seen=set()
    out=[]
    for p in partitions(seq):
        key=tuple(sorted((tuple(sorted(b)) for b in p),key=lambda x:x[0]))
        if key not in seen:
            seen.add(key)
            out.append(tuple(frozenset(b) for b in key))
    return out


def refines(fine,coarse):
    return all(any(set(bf)<=set(bc) for bc in coarse) for bf in fine)


def state_to_block(part):
    return {s:i for i,b in enumerate(part) for s in b}


def induced_map(f,states,part):
    h=state_to_block(part)
    idx={s:i for i,s in enumerate(states)}
    out={}
    for bi,b in enumerate(part):
        vals={h[f[idx[s]]] for s in b}
        if len(vals)!=1:
            return None
        out[bi]=next(iter(vals))
    return tuple(out[i] for i in range(len(part)))


def compatible_faithful(states,elems,eta_cid,part):
    induced=[]
    for f in elems:
        m=induced_map(f,states,part)
        if m is None:
            return False,None
        induced.append(m)

    # Same eta class -> same induced state transformation.
    byclass=defaultdict(set)
    for i,m in enumerate(induced):
        byclass[eta_cid[i]].add(m)
    if any(len(ms)!=1 for ms in byclass.values()):
        return False,None

    # Faithful L action: distinct eta classes -> distinct maps.
    class_map={c:next(iter(ms)) for c,ms in byclass.items()}
    if len(set(class_map.values())) != len(class_map):
        return False,None
    return True,class_map


def extrema(parts):
    finest=[
        p for p in parts
        if not any(q!=p and refines(q,p) for q in parts)
    ]
    coarsest=[
        p for p in parts
        if not any(q!=p and refines(p,q) for q in parts)
    ]
    return finest,coarsest


def fmt(p):
    return [tuple(sorted(b)) for b in p]


def main():
    rows=TS.load_rows()
    print("=== TSF v0.1 ACTION -> STATE FACTOR AUDIT ===")

    for fam in ("IR7","IR8","IR10","IR11","IR12"):
        row=next(r for r in rows if r["family"]==fam)
        states,gens=TS.deterministic_generators(row)
        elems=TS.closure(states,gens)
        mult=TS.multiplication_table(states,elems)
        classes,cid,qmult=TS.maximal_semilattice_congruence(elems,mult)

        good=[]
        for part in unique_partitions(states):
            if len(part)==1:
                continue
            ok,cmap=compatible_faithful(states,elems,cid,part)
            if ok:
                good.append(part)

        fine,coarse=extrema(good)
        if not good:
            verdict="TSF-NONE"
        elif len(good)==1:
            verdict="TSF-U"
        elif len(fine)==1 or len(coarse)==1:
            verdict="TSF-EXTREMAL"
        else:
            verdict="TSF-MULTI"

        print(f"\n{fam} {row['fixture_id']}")
        print("  |T_SL| =",len(classes))
        print("  faithful_partitions =",len(good))
        print("  finest =",[fmt(p) for p in fine])
        print("  coarsest =",[fmt(p) for p in coarse])
        print("  verdict =",verdict)

        if fam=="IR10":
            P=(frozenset((0,2)),frozenset((1,3)))
            AF=(frozenset((0,)),frozenset((1,3)),frozenset((2,)))
            # Canonicalize order.
            P=tuple(sorted(P,key=lambda b:min(b)))
            AF=tuple(sorted(AF,key=lambda b:min(b)))
            print("  historical_P_is_faithful =",P in good)
            print("  AF_three_class_is_faithful =",AF in good)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
