#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RHO-CERT exact certificate candidate v0.1.

Written after RHO_CERT_PRE_CANDIDATE_FREEZE_v0.1.zip.

Uses only:
- finite causal incoming/outgoing kernels,
- declared admissible comparison transports,
- exact alphabet bijections.

No hidden role labels or private oracles.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations


class RHOCertCandidate:
    name="RHO-EXACT-CERT"
    version="0.1"

    @staticmethod
    def F(s):
        a,b=s.split("/")
        return Fraction(int(a),int(b))

    @classmethod
    def parse_variables(cls,row):
        out={}
        for v in row["variables"]:
            incoming={
                p:{x:cls.F(prob) for x,prob in d.items()}
                for p,d in v["incoming"].items()
            }
            outgoing={
                port:{
                    x:{o:cls.F(prob) for o,prob in d.items()}
                    for x,d in table.items()
                }
                for port,table in v["outgoing"].items()
            }
            out[v["id"]]={
                "alphabet":tuple(v["alphabet"]),
                "incoming":incoming,
                "outgoing":outgoing,
            }
        return out

    @staticmethod
    def push_dist(d,g):
        out={}
        for x,p in d.items():
            y=g[x]
            out[y]=out.get(y,Fraction(0))+p
        return out

    @staticmethod
    def dist_equal(a,b):
        keys=set(a)|set(b)
        return all(a.get(k,Fraction(0))==b.get(k,Fraction(0)) for k in keys)

    @classmethod
    def incoming_ok(cls,sv,tv,t,g):
        sm=t["incoming_map"]
        if set(sm)!=set(sv["incoming"]):
            return False
        if set(sm.values())!=set(tv["incoming"]):
            return False
        for sp,tp in sm.items():
            pushed=cls.push_dist(sv["incoming"][sp],g)
            if not cls.dist_equal(pushed,tv["incoming"][tp]):
                return False
        return True

    @classmethod
    def port_ok(cls,stable,ttable,g):
        # Source intervention labels transported by g.
        if set(g[x] for x in stable)!=set(ttable):
            return False

        sout=sorted({o for d in stable.values() for o in d},key=repr)
        tout=sorted({o for d in ttable.values() for o in d},key=repr)
        if len(sout)!=len(tout):
            return False

        for perm in permutations(tout):
            h=dict(zip(sout,perm))
            good=True
            for sx,sd in stable.items():
                tx=g[sx]
                pushed=cls.push_dist(sd,h)
                if not cls.dist_equal(pushed,ttable[tx]):
                    good=False
                    break
            if good:
                return True
        return False

    @classmethod
    def outgoing_ok(cls,sv,tv,t,g):
        sm=t["outgoing_map"]
        if set(sm)!=set(sv["outgoing"]):
            return False
        if set(sm.values())!=set(tv["outgoing"]):
            return False
        for sp,tp in sm.items():
            if not cls.port_ok(sv["outgoing"][sp],tv["outgoing"][tp],g):
                return False
        return True

    @classmethod
    def transport_certifies(cls,sv,tv,t):
        sa=sv["alphabet"]; ta=tv["alphabet"]
        if len(sa)!=len(ta):
            return False
        for perm in permutations(ta):
            g=dict(zip(sa,perm))
            if cls.incoming_ok(sv,tv,t,g) and cls.outgoing_ok(sv,tv,t,g):
                return True
        return False

    @staticmethod
    def all_partitions(items):
        items=list(items)
        if not items:
            yield []
            return
        first=items[0]
        for rest in RHOCertCandidate.all_partitions(items[1:]):
            yield [{first}]+[set(b) for b in rest]
            for i in range(len(rest)):
                nr=[set(b) for b in rest]
                nr[i].add(first)
                yield nr

    @staticmethod
    def block_is_clique(block,edges):
        b=list(block)
        for i in range(len(b)):
            for j in range(i+1,len(b)):
                if frozenset((b[i],b[j])) not in edges:
                    return False
        return True

    @classmethod
    def admissible_partition(cls,part,edges):
        return all(cls.block_is_clique(b,edges) for b in part)

    @classmethod
    def maximal_partition(cls,part,edges):
        # Maximal under coarsening: no two blocks can be merged and remain a clique.
        for i in range(len(part)):
            for j in range(i+1,len(part)):
                if cls.block_is_clique(set(part[i])|set(part[j]),edges):
                    return False
        return True

    @staticmethod
    def canon_partition(part):
        return tuple(sorted(
            (tuple(sorted(b)) for b in part),
            key=lambda x:(len(x),x)
        ))

    def infer(self,row):
        if row.get("interface_insufficient",False):
            return {
                "status":"RHO-V4",
                "maximal_partition_count":0,
                "maximal_partitions":[],
            }

        vars=self.parse_variables(row)
        ids=tuple(sorted(vars))
        edges=set()

        for t in row["candidate_transports"]:
            s=t["source"]; w=t["target"]
            if s==w:
                continue
            if self.transport_certifies(vars[s],vars[w],t):
                edges.add(frozenset((s,w)))

        maximal=set()
        for part in self.all_partitions(ids):
            if not self.admissible_partition(part,edges):
                continue
            if not self.maximal_partition(part,edges):
                continue
            maximal.add(self.canon_partition(part))

        mps=sorted(maximal)
        status="RHO-V1" if len(mps)==1 else "RHO-V3"

        return {
            "status":status,
            "maximal_partition_count":len(mps),
            "maximal_partitions":[
                [list(block) for block in part]
                for part in mps
            ],
        }


CANDIDATE=RHOCertCandidate()
