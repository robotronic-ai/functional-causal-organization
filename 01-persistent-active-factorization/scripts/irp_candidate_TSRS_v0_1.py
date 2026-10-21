#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSRS v0.1 — deterministic transformation-semilattice residual surrogate.

Exploratory post-IPF candidate. See Protocole_TSRS_v0.1_residual_surrogate.md.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product

from irp_interface_v0_1_1 import EmulationKernel, EmulationRow


def cartesian(alphabets):
    out=[()]
    for vals in alphabets:
        out=[a+(x,) for a in out for x in vals]
    return tuple(out)


def compose(f,g,states):
    idx={s:i for i,s in enumerate(states)}
    return tuple(f[idx[g[idx[s]]]] for s in states)


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


class TSRSCandidate:
    name="IRP-TSRS"
    version="0.1"

    def _orders(self,ctx):
        return tuple(ctx.source_state()),tuple(ctx.context_state()),tuple(ctx.target_state())

    def _assign(self,order,values):
        return tuple(zip(order,values))

    def _ordinary(self,ctx):
        S,U,T=self._orders(ctx)
        src=cartesian([tuple(ctx.alphabet(v)) for v in S])
        us=cartesian([tuple(ctx.alphabet(v)) for v in U]) if U else ((),)
        rows={}
        for s in src:
            for u in us:
                do=self._assign(S,s)+self._assign(U,u)
                rows[(s,u)]={tuple(y):Fraction(p) for y,p in ctx.kernel(do,T).items()}
        return S,U,T,src,us,rows

    def _det_maps(self,ctx,src,us,rows):
        tr=dict(ctx.temporal_transport())
        maps={}
        for u in us:
            m={}
            for s in src:
                pos={y:p for y,p in rows[(s,u)].items() if p}
                if len(pos)!=1:
                    return None
                y,p=next(iter(pos.items()))
                if p!=1:
                    return None
                m[s]=tuple(tr[y])
            maps[u]=tuple(m[s] for s in src)
        return maps

    def _closure(self,src,maps):
        elems=set(maps.values())
        changed=True
        while changed:
            changed=False
            cur=list(elems)
            for f in cur:
                for g in cur:
                    h=compose(f,g,src)
                    if h not in elems:
                        elems.add(h);changed=True
        return tuple(sorted(elems,key=repr))

    def _least_sl(self,src,elems):
        n=len(elems)
        index={f:i for i,f in enumerate(elems)}
        mult={(i,j):index[compose(f,g,src)]
              for i,f in enumerate(elems) for j,g in enumerate(elems)}
        d=DSU(n)
        for x in range(n):
            d.union(x,mult[(x,x)])
        for x in range(n):
            for y in range(n):
                d.union(mult[(x,y)],mult[(y,x)])

        changed=True
        while changed:
            changed=False
            classes=defaultdict(list)
            for x in range(n):
                classes[d.find(x)].append(x)
            for members in list(classes.values()):
                for a in members:
                    for b in members:
                        for z in range(n):
                            changed |= d.union(mult[(z,a)],mult[(z,b)])
                            changed |= d.union(mult[(a,z)],mult[(b,z)])

        raw=defaultdict(list)
        for x in range(n):
            raw[d.find(x)].append(x)
        classes=list(raw.values())
        cid={x:i for i,c in enumerate(classes) for x in c}

        qmult={}
        for i,c in enumerate(classes):
            for j,e in enumerate(classes):
                vals={cid[mult[(a,b)]] for a in c for b in e}
                if len(vals)!=1:
                    raise ValueError("internal congruence failure")
                qmult[(i,j)]=next(iter(vals))
        return classes,cid,qmult

    def _top(self,qmult,n):
        x=0
        for y in range(1,n):
            x=qmult[(x,y)]
        if any(qmult[(x,y)]!=x or qmult[(y,x)]!=x for y in range(n)):
            raise ValueError("finite semilattice top not recovered")
        return x

    def _residual_blocks(self,src,elems,classes,top):
        top_idem=[]
        for i in classes[top]:
            f=elems[i]
            if compose(f,f,src)==f:
                top_idem.append(f)
        if not top_idem:
            raise ValueError("top semilattice class has no idempotent representative")

        idx={s:i for i,s in enumerate(src)}
        signatures={}
        # Ordering of top_idem is globally shared; equality classes are independent
        # of which deterministic ordering is chosen.
        top_idem=tuple(sorted(top_idem,key=repr))
        for s in src:
            signatures[s]=tuple(f[idx[s]] for f in top_idem)

        blocks=defaultdict(list)
        for s,sig in signatures.items():
            blocks[sig].append(s)
        return tuple(tuple(v) for v in blocks.values())

    def _q(self,ctx):
        S=tuple(ctx.source_state())
        out=defaultdict(Fraction)
        for wi in ctx.battery():
            d=dict(wi.do)
            s=tuple(d[v] for v in S)
            out[s]+=Fraction(wi.weight)
        if sum(out.values(),Fraction(0))!=1:
            raise ValueError("q not normalized")
        return dict(out)

    def __call__(self,ctx):
        S,U,T,src,us,rows=self._ordinary(ctx)
        maps=self._det_maps(ctx,src,us,rows)

        # TSRS v0.1 is deterministic-only. Preserve unsupported stochastic cases.
        if maps is None:
            chosen=rows
        else:
            elems=self._closure(src,maps)
            classes,cid,qmult=self._least_sl(src,elems)

            if len(classes)==1:
                chosen=rows
            else:
                top=self._top(qmult,len(classes))
                blocks=self._residual_blocks(src,elems,classes,top)
                block_of={s:b for b in blocks for s in b}
                q=self._q(ctx)
                chosen={}
                for s in src:
                    b=block_of[s]
                    mass=sum(q.get(ss,Fraction(0)) for ss in b)
                    if mass<=0:
                        raise ValueError("zero q-mass residual class")
                    for u in us:
                        d=defaultdict(Fraction)
                        for ss in b:
                            ps=q.get(ss,Fraction(0))/mass
                            if not ps:
                                continue
                            for y,py in rows[(ss,u)].items():
                                d[y]+=ps*py
                        chosen[(s,u)]=dict(d)

        return EmulationKernel(
            source_order=S,context_order=U,target_order=T,
            rows=tuple(
                EmulationRow(source=s,context=u,target_dist=chosen[(s,u)])
                for s in src for u in us
            )
        )


CANDIDATE=TSRSCandidate()
