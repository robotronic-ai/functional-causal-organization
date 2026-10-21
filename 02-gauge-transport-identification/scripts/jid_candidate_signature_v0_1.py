#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J-ID signature estimator v0.1.

Written after JID-P5 pre-candidate freeze.

Input:
- transition kernel
- q
- persistent partition
- residual_signature

No access to hidden A or private oracles.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import permutations, product
import ast
import math


class JIDSignatureCandidate:
    name="J-ID-SIGNATURE"
    version="0.1"

    @staticmethod
    def F(s):
        a,b=s.split("/")
        return Fraction(int(a),int(b))

    @classmethod
    def parse_K(cls,row):
        out={}
        for key,d in row["transition"].items():
            s,u=map(int,key.split("|"))
            out[(s,u)]={int(y):cls.F(p) for y,p in d.items()}
        return out

    @classmethod
    def q_of(cls,row):
        return {int(s):cls.F(p) for s,p in row["q"]}

    @classmethod
    def uq_of(cls,row):
        return {int(u):cls.F(p) for u,p in row["context_q"]}

    @staticmethod
    def encode(K):
        def fs(x): return f"{x.numerator}/{x.denominator}"
        return {
            f"{s}|{u}":{str(y):fs(p) for y,p in sorted(d.items())}
            for (s,u),d in sorted(K.items())
        }

    @staticmethod
    def signatures(row):
        return {int(s):ast.literal_eval(v) for s,v in row["residual_signature"].items()}

    @staticmethod
    def gauges_from_signature(S,P,sig):
        p0=tuple(sorted(P[0]))
        groups0=defaultdict(list)
        for s in p0:
            groups0[sig[s]].append(s)

        base_counts={repr(k):len(v) for k,v in groups0.items()}
        per_fiber=[]

        for p in range(1,len(P)):
            gp=defaultdict(list)
            for s in P[p]:
                gp[sig[s]].append(s)
            counts={repr(k):len(v) for k,v in gp.items()}
            if counts!=base_counts:
                return []

            maps=[{}]
            for label,srcs in sorted(groups0.items(),key=lambda kv:repr(kv[0])):
                new=[]
                for perm in permutations(gp[label]):
                    for m in maps:
                        mm=dict(m)
                        mm.update(dict(zip(srcs,perm)))
                        new.append(mm)
                maps=new
            per_fiber.append(maps)

        out=[]
        for choices in product(*per_fiber):
            blocks=[]
            for x in p0:
                b={x}
                for mp in choices:
                    b.add(mp[x])
                blocks.append(frozenset(b))
            out.append(tuple(blocks))
        return out

    @classmethod
    def surgery(cls,row,P,A):
        S=tuple(row["alphabet"])
        U=tuple(row["contexts"])
        K=cls.parse_K(row)
        q=cls.q_of(row)

        hp={s:i for i,b in enumerate(P) for s in b}
        ha={s:i for i,b in enumerate(A) for s in b}
        recon={(ha[s],hp[s]):s for s in S}
        if len(recon)!=len(S):
            raise ValueError("bad product")

        qA=defaultdict(Fraction)
        qAP=defaultdict(Fraction)
        for s,w in q.items():
            qA[ha[s]]+=w
            qAP[(ha[s],hp[s])]+=w

        out={}
        for s in S:
            a=ha[s]
            for u in U:
                d=defaultdict(Fraction)
                for p in range(len(P)):
                    w=qAP[(a,p)]/qA[a]
                    if not w:
                        continue
                    ss=recon[(a,p)]
                    for y,py in K[(ss,u)].items():
                        d[y]+=w*py
                out[(s,u)]=dict(d)
        return out

    @classmethod
    def cmi(cls,row,K):
        q=cls.q_of(row)
        uq=cls.uq_of(row)
        joint=defaultdict(Fraction)
        for s,ps in q.items():
            for u,pu in uq.items():
                for y,p in K[(s,u)].items():
                    joint[(s,y,u)] += ps*pu*p

        pu=defaultdict(Fraction)
        psu=defaultdict(Fraction)
        pyu=defaultdict(Fraction)
        for (s,y,u),p in joint.items():
            pu[u]+=p
            psu[(s,u)]+=p
            pyu[(y,u)]+=p

        I=0.0
        for (s,y,u),p in joint.items():
            r=Fraction(p*pu[u],psu[(s,u)]*pyu[(y,u)])
            I += float(p)*math.log2(float(r))
        return I

    @staticmethod
    def norm(K):
        return tuple((k,tuple(sorted(d.items()))) for k,d in sorted(K.items()))

    def infer(self,row):
        S=tuple(row["alphabet"])
        P=tuple(frozenset(block) for block in row["persistent_partition"])
        sig=self.signatures(row)

        gauges=self.gauges_from_signature(S,P,sig)
        if not gauges:
            return {
                "status":"PI-EMPTY",
                "compatible_J_count":0,
                "B_set_bits":[],
            }

        kernels=[]
        scores=[]
        for A in gauges:
            K=self.surgery(row,P,A)
            kernels.append(K)
            scores.append(round(self.cmi(row,K),12))

        distinct_scores=sorted(set(scores))
        unique_kernels={self.norm(K):K for K in kernels}

        if len(gauges)==1 and len(unique_kernels)==1 and len(distinct_scores)==1:
            K=next(iter(unique_kernels.values()))
            return {
                "status":"PI-POINT",
                "compatible_J_count":1,
                "B_set_bits":distinct_scores,
                "oracle_transition":self.encode(K),
            }

        return {
            "status":"PI-PARTIAL",
            "compatible_J_count":len(gauges),
            "B_set_bits":distinct_scores,
        }


CANDIDATE=JIDSignatureCandidate()
