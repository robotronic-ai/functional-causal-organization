#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIBER-AUT v0.1 exhaustive 4-state audit."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import product, permutations
import math


S=(0,1,2,3)
U=(0,1)
P=(frozenset((0,2)),frozenset((1,3)))
hp={s:i for i,b in enumerate(P) for s in b}
IDENT=tuple(S)


def compose(f,g):
    return tuple(f[g[i]] for i in range(len(g)))


def conjugate(beta,f):
    inv={beta[s]:s for s in S}
    return tuple(beta[f[inv[s]]] for s in S)


def closure(gens):
    elems={IDENT,*gens.values()}
    while True:
        new={compose(f,g) for f in elems for g in elems}
        if new<=elems:
            break
        elems|=new
    return tuple(sorted(elems))


def induced(t):
    out=[]
    for block in P:
        vals={hp[t[s]] for s in block}
        if len(vals)!=1:
            return None
        out.append(next(iter(vals)))
    return tuple(out)


def mult(elems):
    idx={e:i for i,e in enumerate(elems)}
    return {(i,j):idx[compose(elems[i],elems[j])]
            for i in range(len(elems)) for j in range(len(elems))}


def sections(gens):
    T=closure(gens)
    idxT={t:i for i,t in enumerate(T)}
    mt=mult(T)
    rho={i:induced(t) for i,t in enumerate(T)}
    B=tuple(sorted(set(rho.values())))
    idxB={b:i for i,b in enumerate(B)}
    mb=mult(B)
    fibers={bi:[ti for ti in range(len(T)) if idxB[rho[ti]]==bi]
            for bi in range(len(B))}
    idB=idxB[tuple(range(len(P)))]
    idT=idxT[IDENT]
    order=[b for b in range(len(B)) if b!=idB]
    out=[]
    for vals in product(*[fibers[b] for b in order]):
        sec={idB:idT,**dict(zip(order,vals))}
        ok=True
        for i in range(len(B)):
            for j in range(len(B)):
                if mt[(sec[i],sec[j])]!=sec[mb[(i,j)]]:
                    ok=False;break
            if not ok:break
        if ok:
            out.append(sec)
    return T,B,rho,out


def trivialization(T,B,sec):
    # bottom=0 for binary {id,set1}
    p0=0
    bp={}
    for p in range(2):
        hits=[i for i,b in enumerate(B) if b[p0]==p]
        if len(hits)!=1:return None
        bp[p]=hits[0]
    bottom=tuple(sorted(P[0]))
    trans={}
    for p,bi in bp.items():
        t=T[sec[bi]]
        image=tuple(t[x] for x in bottom)
        if len(set(image))!=len(bottom) or set(image)!=set(P[p]):
            return None
        trans[p]=dict(zip(bottom,image))
    A=tuple(frozenset(trans[p][x] for p in range(2)) for x in bottom)
    return tuple(sorted(A,key=lambda b:min(b)))


def surgery(gens,A):
    ha={s:i for i,b in enumerate(A) for s in b}
    recon={(ha[s],hp[s]):s for s in S}
    if len(recon)!=4:return None
    # q uniform => q(P|A) uniform because every A block hits each P fiber once.
    out={}
    for s in S:
        a=ha[s]
        for u in U:
            d=defaultdict(Fraction)
            for p in range(2):
                ss=recon[(a,p)]
                y=gens[u][ss]
                d[y]+=Fraction(1,2)
            out[(s,u)]=dict(d)
    return out


def Bscore(K):
    # uniform qS and qU
    joint=defaultdict(Fraction)
    for s in S:
        for u in U:
            for y,p in K[(s,u)].items():
                joint[(s,y,u)] += Fraction(1,4)*Fraction(1,2)*p
    pu=defaultdict(Fraction); psu=defaultdict(Fraction); pyu=defaultdict(Fraction)
    for (s,y,u),p in joint.items():
        pu[u]+=p;psu[(s,u)]+=p;pyu[(y,u)]+=p
    val=0.0
    for (s,y,u),p in joint.items():
        r=Fraction(p*pu[u],psu[(s,u)]*pyu[(y,u)])
        val+=float(p)*math.log2(float(r))
    return val


def automorphisms(gens):
    autos=[]
    # factor-preserving state permutations
    state_perms=[]
    for p0 in permutations(P[0]):
        for p1 in permutations(P[1]):
            beta={}
            for src,dst in zip(sorted(P[0]),p0): beta[src]=dst
            for src,dst in zip(sorted(P[1]),p1): beta[src]=dst
            bt=tuple(beta[s] for s in S)
            state_perms.append(bt)

    for beta in state_perms:
        for kappa in permutations(U):
            ok=True
            for u in U:
                if conjugate(beta,gens[u]) != gens[kappa[u]]:
                    ok=False;break
            if ok:
                autos.append((beta,kappa))
    return autos


def transported_section(T,B,sec,beta):
    idxT={t:i for i,t in enumerate(T)}
    out={}
    for bi,ti in sec.items():
        ct=conjugate(beta,T[ti])
        if ct not in idxT:return None
        out[bi]=idxT[ct]
    return out


def all_lifts(base_map):
    # base_map is tuple on P, each state can target either member of required fiber
    choices=[tuple(sorted(P[base_map[hp[s]]])) for s in S]
    for ys in product(*choices):
        yield tuple(ys)


def sec_key(sec):
    return tuple(sorted(sec.items()))


def main():
    idP=(0,1)
    set1=(1,1)
    lifts0=list(all_lifts(idP))
    lifts1=list(all_lifts(set1))
    assert len(lifts0)==16 and len(lifts1)==16

    systems=0
    multi=0
    score_diff=0
    orbit_multi=0
    orbit_score_diff=0
    first_multi=None
    first_score_diff=None
    first_orbit_multi=None

    for f0 in lifts0:
        for f1 in lifts1:
            systems+=1
            gens={0:f0,1:f1}
            T,B,rho,secs=sections(gens)
            admiss=[]
            for sec in secs:
                A=trivialization(T,B,sec)
                if A is None:continue
                K=surgery(gens,A)
                admiss.append((sec,A,Bscore(K)))
            if len(admiss)<2:
                continue
            multi+=1
            if first_multi is None:
                first_multi=(f0,f1,[(a,round(b,12)) for _,a,b in admiss])

            vals={round(x[2],12) for x in admiss}
            if len(vals)>1:
                score_diff+=1
                if first_score_diff is None:
                    first_score_diff=(f0,f1,[(a,round(b,12)) for _,a,b in admiss])

            autos=automorphisms(gens)
            keys={sec_key(sec):i for i,(sec,A,b) in enumerate(admiss)}
            adj=defaultdict(set)
            for i,(sec,A,b) in enumerate(admiss):
                for beta,kappa in autos:
                    ts=transported_section(T,B,sec,beta)
                    if ts is None:continue
                    k=sec_key(ts)
                    if k in keys:
                        j=keys[k]
                        adj[i].add(j);adj[j].add(i)

            seen=set()
            has_multi_orbit=False
            bad=False
            for i in range(len(admiss)):
                if i in seen:continue
                comp=set([i]);stack=[i];seen.add(i)
                while stack:
                    x=stack.pop()
                    for y in adj[x]:
                        if y not in seen:
                            seen.add(y);comp.add(y);stack.append(y)
                if len(comp)>1:
                    has_multi_orbit=True
                    vals={round(admiss[j][2],12) for j in comp}
                    if len(vals)>1:
                        bad=True
            if has_multi_orbit:
                orbit_multi+=1
                if first_orbit_multi is None:
                    first_orbit_multi=(f0,f1,[(a,round(b,12)) for _,a,b in admiss],len(autos))
            if bad:
                orbit_score_diff+=1

    print("=== FIBER-AUT v0.1 EXHAUSTIVE AUDIT ===")
    print("systems =",systems)
    print("systems_with_2plus_admissible_sections =",multi)
    print("systems_with_section_score_differences =",score_diff)
    print("systems_with_multi_section_automorphism_orbit =",orbit_multi)
    print("systems_with_score_difference_inside_automorphism_orbit =",orbit_score_diff)
    print()
    print("first_multi_section_system =",first_multi)
    print("first_score_difference_system =",first_score_diff)
    print("first_multi_section_automorphism_orbit_system =",first_orbit_multi)
    print()
    print("VERDICT:", "AUT-B-INV" if orbit_score_diff==0 else "AUT-B-DIFF")
    return 0 if orbit_score_diff==0 else 3


if __name__=="__main__":
    raise SystemExit(main())
