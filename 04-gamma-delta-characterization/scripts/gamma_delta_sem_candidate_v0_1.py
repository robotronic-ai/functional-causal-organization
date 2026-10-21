#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GAMMA-DELTA-SEM exact candidate v0.1.

Written after GAMMA_DELTA_SEM_PRE_CANDIDATE_FREEZE_v0.1.zip.
"""

from __future__ import annotations

from fractions import Fraction
from collections import defaultdict
import math


class GammaDeltaSemCandidate:
    name="GAMMA-DELTA-SEM"
    version="0.1"

    @staticmethod
    def F(s):
        a,b=s.split("/")
        return Fraction(int(a),int(b))

    @classmethod
    def parse_q(cls,obj):
        return {x:cls.F(p) for x,p in obj.items()}

    @classmethod
    def parse_kernel(cls,obj):
        return {
            x:{y:cls.F(p) for y,p in row.items()}
            for x,row in obj.items()
        }

    @staticmethod
    def entropy(q):
        out=0.0
        for p in q.values():
            if p:
                out -= float(p)*math.log2(float(p))
        return out

    @classmethod
    def mi(cls,q,k):
        py=defaultdict(Fraction)
        joint={}
        for x,px in q.items():
            for y,p in k[x].items():
                joint[(x,y)] = px*p
                py[y] += px*p

        out=0.0
        for (x,y),p in joint.items():
            if not p:
                continue
            r=Fraction(p,1)/(q[x]*py[y])
            out += float(p)*math.log2(float(r))
        return out

    @classmethod
    def directional_mi(cls,d):
        q=cls.parse_q(d["input_q"])
        k=cls.parse_kernel(d["kernel"])
        return cls.mi(q,k)

    @staticmethod
    def beta(x,y):
        if x+y==0:
            return None
        return 2*min(x,y)/(x+y)

    def infer(self,row):
        cuts=[]
        for p in row["gamma_partitions"]:
            x=self.directional_mi(p["ab"])
            y=self.directional_mi(p["ba"])
            g=(x+y)/2
            b=self.beta(x,y)
            cuts.append((p["id"],g,b))

        gamma=min(g for _,g,_ in cuts)
        betas=sorted({
            round(b,12)
            for _,g,b in cuts
            if abs(g-gamma)<=1e-12 and b is not None
        })

        d=row["delta"]
        q=self.parse_q(d["q"])
        k=self.parse_kernel(d["kernel"])
        H=self.entropy(q)
        I=self.mi(q,k)
        n=sum(1 for x in d["support"] if q.get(x,Fraction(0))>0)

        if n<=1:
            kappa=0.0
            cap=0.0
        else:
            L=math.log2(n)
            kappa=H/L
            cap=I/L

        dec=None if H==0 else I/H

        return {
            "gamma":round(gamma,12),
            "beta_minimizers":betas,
            "kappa":round(kappa,12),
            "delta_dec":None if dec is None else round(dec,12),
            "delta_cap":round(cap,12),
        }


CANDIDATE=GammaDeltaSemCandidate()
