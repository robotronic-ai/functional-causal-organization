#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-P4 exact audit of fiber-complement ambiguity for intended persistent P."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import json
HERE=Path(__file__).resolve().parent

def F(s):
    a,b=s.split('/');return Fraction(int(a),int(b))
def K_of(row):
    out={}
    for key,d in row['transition'].items():
        s,u=map(int,key.split('|'));out[(s,u)]={int(y):F(p) for y,p in d.items()}
    return out

def q_of(row):return {int(s):F(p) for s,p in row['q']}

def set_partitions(items):
    items=tuple(items)
    if not items:
        yield tuple();return
    x=items[0]
    for rest in set_partitions(items[1:]):
        yield (frozenset((x,)),)+rest
        for i in range(len(rest)):
            yield rest[:i]+(rest[i]|frozenset((x,)),)+rest[i+1:]

def canon(part):return tuple(sorted((tuple(sorted(b)) for b in part),key=lambda x:(len(x),x)))
def orthogonal(P,A,S):
    if len(P)*len(A)!=len(S):return False
    hp={s:i for i,b in enumerate(P) for s in b};ha={s:i for i,b in enumerate(A) for s in b}
    return len({(ha[s],hp[s]) for s in S})==len(S)

def surgery(row,P,A):
    S=tuple(row['alphabet']);U=tuple(row['contexts']);K=K_of(row);q=q_of(row)
    hp={s:i for i,b in enumerate(P) for s in b};ha={s:i for i,b in enumerate(A) for s in b}
    recon={(ha[s],hp[s]):s for s in S}
    qA=defaultdict(Fraction);qAP=defaultdict(Fraction)
    for s,w in q.items():qA[ha[s]]+=w;qAP[(ha[s],hp[s])]+=w
    out={}
    for s in S:
        a=ha[s]
        for u in U:
            d=defaultdict(Fraction)
            for p in range(len(P)):
                w=qAP[(a,p)]/qA[a]
                if not w:continue
                ss=recon[(a,p)]
                for y,py in K[(ss,u)].items():d[y]+=w*py
            out[(s,u)]=tuple(sorted(d.items()))
    return tuple((k,out[k]) for k in sorted(out))

def main():
    rows=json.loads((HERE/'OPF_P4_FIXTURES_v0_1.json').read_text())['fixtures']
    intended={
      'P4A':(frozenset((0,2)),frozenset((1,3))),
      'P4B':tuple(frozenset((h,4+h)) for h in range(4)),
      'P4C':tuple(frozenset((h,4+h)) for h in range(4)),
    }
    for fam in ('P4A','P4B','P4C'):
        row=next(r for r in rows if r['fixture_id']==fam+'.01');S=tuple(row['alphabet']);P=intended[fam]
        seen=set();comps=[]
        for A in set_partitions(S):
            c=canon(A)
            if c in seen:continue
            seen.add(c);A=tuple(frozenset(b) for b in c)
            if orthogonal(P,A,S):comps.append(A)
        kernels=defaultdict(list)
        for A in comps:kernels[surgery(row,P,A)].append(A)
        print('\n'+fam)
        print('orthogonal_set_complements=',len(comps))
        print('distinct_induced_surgeries=',len(kernels))
        for i,(ker,As) in enumerate(kernels.items(),1):
            print(' surgery',i,'num_complements=',len(As),'example_A=',canon(As[0]))
        print('unique_surgery=',len(kernels)==1)
    return 0
if __name__=='__main__':raise SystemExit(main())
