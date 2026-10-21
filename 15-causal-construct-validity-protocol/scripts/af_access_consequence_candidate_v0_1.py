#!/usr/bin/env python3
"""Independent AF access-consequence candidate v0.1."""
from __future__ import annotations
from fractions import Fraction
import math

def F(x):
    return x if isinstance(x,Fraction) else Fraction(str(x))

def norm(d):
    z={str(k):F(v) for k,v in d.items()}
    if sum(z.values(),Fraction(0))!=1:
        raise ValueError("distribution does not sum to one")
    return z

def loss(fam,key):
    total=0.0
    for p in fam["probes"]:
        q=float(F(p["q"]))
        truth=norm(p["truth"]); pred=norm(p[key])
        for y,py in truth.items():
            if py==0: continue
            ph=pred.get(y,Fraction(0))
            if ph<=0: return math.inf
            total += q*float(py)*(-math.log2(float(ph)))
    return total

def family_score(fam):
    if not fam.get("available",True):
        return {"status":"NA","score":None}
    L=loss(fam,"pred")
    L0=loss(fam,"null")
    # oracle uses truth distribution itself
    oracle={"probes":[]}
    for p in fam["probes"]:
        oracle["probes"].append({**p,"pred":p["truth"]})
    Lstar=loss(oracle,"pred")
    if not math.isfinite(L0) or abs(L0-Lstar)<1e-15:
        return {"status":"NA","score":None,"reason":"DEGENERATE-BASELINE"}
    return {"status":"IDENTIFIED","score":(L0-L)/(L0-Lstar),
            "L":L,"L0":L0,"Lstar":Lstar}

def profile(system):
    out={}
    for fam in system["families"]:
        out[fam["id"]]=family_score(fam)
    return {"status":"PARTIAL" if any(v["status"]=="NA" for v in out.values())
            else "IDENTIFIED","families":out}
