#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIBER-PARTIAL-ID v0.1 post-hoc illustration."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

spec=importlib.util.spec_from_file_location(
    "D",str(HERE/"fiber_full_gauge_min_diagnostic_v0_1_1.py")
)
D=importlib.util.module_from_spec(spec);sys.modules["D"]=D;spec.loader.exec_module(D)
G=D.G


def score_set(row,P):
    S=tuple(row["alphabet"])
    comps=D.complements(S,P)
    vals=[]
    kernels=set()
    for A in comps:
        kt=G.surgery(row,P,A)
        K=D.kernel_tuple_to_dict(kt)
        vals.append(D.cmi(row,K))
        kernels.add(D.norm_dict(K))
    return comps,vals,kernels


def main():
    hist=D.load(HERE/"IRP_HARNESS_FIXTURES_IR7_IR12_v0_1.json")
    p3=D.load(HERE/"TSIPF_P3_FIXTURES_v0_1.json")
    p4=D.load(HERE/"OPF_P4_FIXTURES_v0_1.json")

    cases=[
        ("IR10",next(r for r in hist if r["family"]=="IR10"),
         (frozenset((0,2)),frozenset((1,3)))),
        ("P3D",next(r for r in p3 if r["family"]=="P3D"),
         (frozenset((0,2,4,6)),frozenset((1,3,5,7)))),
        ("P4A",next(r for r in p4 if r["family"]=="P4A"),
         (frozenset((0,2)),frozenset((1,3)))),
        ("P4B",next(r for r in p4 if r["family"]=="P4B"),
         tuple(frozenset((h,4+h)) for h in range(4))),
        ("P4C",next(r for r in p4 if r["family"]=="P4C"),
         tuple(frozenset((h,4+h)) for h in range(4))),
    ]

    print("=== FIBER-PARTIAL-ID v0.1 ===")
    for name,row,P in cases:
        comps,vals,kernels=score_set(row,P)
        exact=sorted({round(x,12) for x in vals})
        if not vals:
            status="PI-EMPTY"
        elif len(exact)==1:
            status="PI-POINT"
        else:
            status="PI-PARTIAL"
        print(f"\n{name}")
        print("admissible_J =",len(comps))
        print("distinct_kernels =",len(kernels))
        print("B_set =",exact)
        if vals:
            print("B_interval =",[round(min(vals),12),round(max(vals),12)])
        print("status =",status)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
