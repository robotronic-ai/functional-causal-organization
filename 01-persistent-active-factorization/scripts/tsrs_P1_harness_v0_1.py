#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSRS-P1 prospective exact harness.

Loads frozen P1 fixtures/oracles and evaluates a candidate exporting CANDIDATE.
Uses four deterministic anonymization seeds with independently recoded source,
target, context, variable, and role identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import importlib.util
import json
import random
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from irp_interface_v0_1_1 import WeightedIntervention, validate_emulation_kernel, validate_temporal_transport

FIX=HERE/"TSRS_P1_FIXTURES_v0_1.json"
ORA=HERE/"TSRS_P1_ORACLES_v0_1_PRIVATE.json"

SEEDS=(1729,2718,31415,65537)


def F(s):
    a,b=s.split("/")
    return Fraction(int(a),int(b))


def parse_transition(obj):
    out={}
    for key,d in obj.items():
        s,u=map(int,key.split("|"))
        out[(s,u)]={int(y):F(p) for y,p in d.items()}
    return out


def load_data():
    fixtures=json.loads(FIX.read_text(encoding="utf-8"))["fixtures"]
    oracles=json.loads(ORA.read_text(encoding="utf-8"))["oracles"]
    omap={x["fixture_id"]:parse_transition(x["oracle_transition"]) for x in oracles}
    return fixtures,omap


def permuted_codes(n,rng,base):
    vals=list(range(base,base+n))
    rng.shuffle(vals)
    return tuple(vals)


class P1Context:
    def __init__(self,row,seed):
        rng=random.Random(seed)
        self.raw_states=tuple(row["alphabet"])
        self.raw_contexts=tuple(row["contexts"])
        self.K=parse_transition(row["transition"])

        self.S=100000+rng.randrange(10000)
        self.U=200000+rng.randrange(10000)
        self.T=300000+rng.randrange(10000)

        self.sc=permuted_codes(len(self.raw_states),rng,1000)
        self.tc=permuted_codes(len(self.raw_states),rng,2000)
        self.uc=permuted_codes(len(self.raw_contexts),rng,3000)

        self.s_raw_to_code=dict(zip(self.raw_states,self.sc))
        self.s_code_to_raw={v:k for k,v in self.s_raw_to_code.items()}
        self.t_raw_to_code=dict(zip(self.raw_states,self.tc))
        self.t_code_to_raw={v:k for k,v in self.t_raw_to_code.items()}
        self.u_raw_to_code=dict(zip(self.raw_contexts,self.uc))
        self.u_code_to_raw={v:k for k,v in self.u_raw_to_code.items()}

        self._q={int(s):F(p) for s,p in row["q"]}

    def variables(self): return (self.S,self.U,self.T)
    def alphabet(self,v):
        return {self.S:self.sc,self.U:self.uc,self.T:self.tc}[v]
    def parents(self,v):
        return frozenset((self.S,self.U)) if v==self.T else frozenset()
    def roles(self): return {self.S:7001,self.T:7001,self.U:9001}
    def source_state(self): return (self.S,)
    def target_state(self): return (self.T,)
    def context_state(self): return (self.U,)
    def temporal_transport(self):
        return {
            (self.t_raw_to_code[s],):(self.s_raw_to_code[s],)
            for s in self.raw_states
        }
    def battery(self):
        return tuple(
            WeightedIntervention(((self.S,self.s_raw_to_code[s]),),p)
            for s,p in self._q.items()
        )
    def kernel(self,do,targets):
        d=dict(do)
        s=self.s_code_to_raw[d[self.S]]
        u=self.u_code_to_raw[d[self.U]]
        return {
            (self.t_raw_to_code[y],):p
            for y,p in self.K[(s,u)].items()
        }


def load_candidate(path):
    spec=importlib.util.spec_from_file_location("p1candidate",str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules["p1candidate"]=mod
    spec.loader.exec_module(mod)
    return mod.CANDIDATE


def decode_kernel(ctx,out):
    rows={}
    for r in out.rows:
        s=ctx.s_code_to_raw[r.source[0]]
        u=ctx.u_code_to_raw[r.context[0]]
        rows[(s,u)]={
            ctx.t_code_to_raw[y[0]]:Fraction(p)
            for y,p in r.target_dist.items()
        }
    return rows


def main(argv=None):
    argv=sys.argv if argv is None else argv
    candidate_path=Path(argv[1]) if len(argv)>1 else HERE/"irp_candidate_TSRS_v0_1.py"
    C=load_candidate(candidate_path)
    fixtures,oracles=load_data()

    total=passes=0
    stats={}
    first=None

    for i,row in enumerate(fixtures):
        fam=row["family"]
        stats.setdefault(fam,[0,0,[]])
        expected=oracles[row["fixture_id"]]

        for seed in SEEDS:
            ctx=P1Context(row,seed*1009+i*9176)
            validate_temporal_transport(ctx)
            out=C(ctx)
            validate_emulation_kernel(ctx,out)
            got=decode_kernel(ctx,out)
            ok=(got==expected)

            total+=1
            stats[fam][1]+=1
            if ok:
                passes+=1
                stats[fam][0]+=1
            else:
                stats[fam][2].append((row["fixture_id"],seed))
                if first is None:
                    first=(row["fixture_id"],seed)

    print(f"CANDIDATE: {C.name} {C.version}")
    print(f"EXACT_PASSES: {passes}/{total}")
    for fam in ("P1A","P1B","P1C"):
        p,t,bad=stats[fam]
        print(f"{fam}: {p}/{t} failures={bad}")
    print("FIRST_FAILURE:",first)
    print("TSRS-P1 VERDICT:", "P1-V4 FINITE NON-REFUTATION" if passes==total else "P1-V3 CANDIDATE REFUTED")
    return 0 if passes==total else 2


if __name__=="__main__":
    raise SystemExit(main())
