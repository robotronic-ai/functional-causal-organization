#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IR13-S1 exhaustive stochastic collision search.

Protocol: Protocole_IR13_S1_v0.1_stochastic_collision_search.md
No PRNG. Exact Fraction arithmetic.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations, product
from collections import defaultdict

STATES=(0,1,2,3)
CONTEXTS=(0,1)


def balanced_partitions():
    """
    Canonical unlabeled 2+2 partitions, returned as two frozenset blocks.
    Exactly 3 partitions of a four-element set.
    """
    out=[]
    seen=set()
    for A in combinations(STATES,2):
        A=frozenset(A)
        B=frozenset(set(STATES)-set(A))
        key=tuple(sorted((tuple(sorted(A)),tuple(sorted(B)))))
        if key in seen:
            continue
        seen.add(key)
        out.append((frozenset(key[0]),frozenset(key[1])))
    return tuple(out)


PARTS=balanced_partitions()


def transverse(p0,p1):
    sig={}
    for s in STATES:
        a=0 if s in p0[0] else 1
        b=0 if s in p1[0] else 1
        sig[s]=(a,b)
    return len(set(sig.values()))==4


def labeled_class(part, flip, s):
    raw=0 if s in part[0] else 1
    return raw ^ flip


def active_channel(p_src, p_tgt, flips_src, flips_tgt):
    K={}
    for u in CONTEXTS:
        for s in STATES:
            c=labeled_class(p_src[u],flips_src[u],s)
            # target class label includes optional flip
            wanted=c ^ flips_tgt[u]
            block=p_tgt[u][wanted]
            K[(s,u)]={y:Fraction(1,2) for y in block}
    return K


def active_family():
    out=[]
    for ps0 in PARTS:
        for ps1 in PARTS:
            if not transverse(ps0,ps1):
                continue
            for pt0 in PARTS:
                for pt1 in PARTS:
                    if not transverse(pt0,pt1):
                        continue
                    for fs in product((0,1),repeat=2):
                        for ft in product((0,1),repeat=2):
                            K=active_channel((ps0,ps1),(pt0,pt1),fs,ft)
                            out.append({
                                "kind":"active",
                                "src_parts":(ps0,ps1),
                                "tgt_parts":(pt0,pt1),
                                "flips_src":fs,
                                "flips_tgt":ft,
                                "K":K,
                                "oracle":K,
                            })
    return out


def dec(s):
    return (s//2,s%2)


def enc(a,b):
    return 2*a+b


def storage_base(refresh_by_u):
    K={}
    for s in STATES:
        a,b=dec(s)
        for u in CONTEXTS:
            refresh=refresh_by_u[u]
            d=defaultdict(Fraction)
            for r in (0,1):
                if refresh==0:
                    y=enc(r,b)
                else:
                    y=enc(a,r)
                d[y]+=Fraction(1,2)
            K[(s,u)]=dict(d)
    return K


def transport(K,sp,up):
    out={}
    invs={v:k for k,v in sp.items()}
    invu={v:k for k,v in up.items()}
    for s2 in STATES:
        s=invs[s2]
        for u2 in CONTEXTS:
            u=invu[u2]
            out[(s2,u2)]={sp[y]:p for y,p in K[(s,u)].items()}
    return out


def q_mix_oracle(K):
    O={}
    for actual in STATES:
        for u in CONTEXTS:
            d=defaultdict(Fraction)
            for ss in STATES:
                for y,p in K[(ss,u)].items():
                    d[y]+=Fraction(1,4)*p
            O[(actual,u)]=dict(d)
    return O


def storage_family():
    raw=[]
    # Contexts refresh different slots.
    for mapping in ((0,1),(1,0)):
        K=storage_base(mapping)
        raw.append((mapping,K))

    out=[]
    seen=set()
    for mapping,K in raw:
        for permS in permutations(STATES):
            sp=dict(zip(STATES,permS))
            for permU in permutations(CONTEXTS):
                up=dict(zip(CONTEXTS,permU))
                K2=transport(K,sp,up)
                key=canonical(K2)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "kind":"storage",
                    "refresh_by_u":mapping,
                    "state_perm":sp,
                    "context_perm":up,
                    "K":K2,
                    "oracle":q_mix_oracle(K2),
                })
    return out


def canonical(K):
    return tuple(
        (s,u,tuple(sorted(K[(s,u)].items())))
        for s in STATES for u in CONTEXTS
    )


def transported_dist(d,sp):
    return {sp[y]:p for y,p in d.items()}


def o1_isomorphisms(KA,KB):
    hits=[]
    for permS in permutations(STATES):
        sp=dict(zip(STATES,permS))
        for permU in permutations(CONTEXTS):
            up=dict(zip(CONTEXTS,permU))
            ok=True
            for s in STATES:
                for u in CONTEXTS:
                    if transported_dist(KA[(s,u)],sp) != KB[(sp[s],up[u])]:
                        ok=False
                        break
                if not ok:
                    break
            if ok:
                hits.append((sp,up))
    return hits


def oracle_equal_under(OA,OB,sp,up):
    for s in STATES:
        for u in CONTEXTS:
            if transported_dist(OA[(s,u)],sp) != OB[(sp[s],up[u])]:
                return False
    return True


def cmi_bits(K):
    import math
    joint=defaultdict(Fraction)
    for s in STATES:
        for u in CONTEXTS:
            for y,p in K[(s,u)].items():
                joint[(s,y,u)] += Fraction(1,8)*p
    pu=defaultdict(Fraction); psu=defaultdict(Fraction)
    pyu=defaultdict(Fraction); psyu=defaultdict(Fraction)
    for (s,y,u),p in joint.items():
        pu[u]+=p; psu[(s,u)]+=p; pyu[(y,u)]+=p; psyu[(s,y,u)]+=p
    I=0.0
    for (s,y,u),p in psyu.items():
        r=Fraction(p*pu[u],psu[(s,u)]*pyu[(y,u)])
        I+=float(p)*math.log2(float(r))
    return I


def main():
    S=storage_family()
    A=active_family()

    print("=== IR13-S1 EXHAUSTIVE SEARCH ===")
    print("storage_channels:",len(S))
    print("active_constructions:",len(A))
    print("balanced_partitions:",len(PARTS))

    witnesses=[]
    checked=0
    for i,s in enumerate(S):
        for j,a in enumerate(A):
            checked+=1
            isos=o1_isomorphisms(s["K"],a["K"])
            if not isos:
                continue
            all_conflict=all(
                not oracle_equal_under(s["oracle"],a["oracle"],sp,up)
                for sp,up in isos
            )
            if all_conflict:
                witnesses.append((i,j,s,a,isos))
                break
        if witnesses:
            break

    print("pairs_checked_until_stop:",checked)
    print("witness_count_found_before_stop:",len(witnesses))

    if not witnesses:
        print("VERDICT: S1-V0 — no collision on declared exhaustive domain")
        return 0

    i,j,s,a,isos=witnesses[0]
    print("\n=== FIRST EXACT WITNESS ===")
    print("storage_index:",i)
    print("active_index:",j)
    print("O1_isomorphism_count:",len(isos))
    sp,up=isos[0]
    print("example_state_bijection:",sp)
    print("example_context_bijection:",up)
    print("B_storage_original:",f"{cmi_bits(s['K']):.12g}")
    print("B_storage_oracle:",f"{cmi_bits(s['oracle']):.12g}")
    print("B_active_oracle:",f"{cmi_bits(a['oracle']):.12g}")
    print("all_O1_isomorphisms_oracle_conflicting:",True)

    print("\nStorage channel:")
    for k in sorted(s["K"]):
        print(k,s["K"][k])

    print("\nActive channel:")
    for k in sorted(a["K"]):
        print(k,a["K"][k])

    print("\nVERDICT:")
    print(
        "S1-V1 — exact O1 collision between preregistered storage and active "
        "construction families with incompatible frozen oracle surgeries."
    )
    print(
        "Scientific status remains conditional on accepting both construction "
        "families as legitimate functional controls."
    )
    return 3


if __name__=="__main__":
    raise SystemExit(main())
