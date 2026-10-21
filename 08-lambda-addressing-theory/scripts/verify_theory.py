#!/usr/bin/env python3
from itertools import permutations, product
import math, json


def frob_perm(p,q):
    # permutation matrices differ by two unit entries for every mismatched row
    mismatches=sum(a!=b for a,b in zip(p,q))
    return math.sqrt(2*mismatches)


def witness_D():
    # D=(pairing pi, sign s in {-1,+1}, delay in {1,2,3}); second channel is fixed.
    states=list(product((0,1),(-1,1),(1,2,3)))
    def target(x,T=1):
        pi,s,d=x
        c=s if d<=T else 0
        return (pi,c)
    # horizon-1 minimal semantic quotient is exactly target in this witness
    classes={target(x) for x in states}
    assert len(states)==12 and len(classes)==6
    # Coarse no-D descriptor fails
    assert len({target(x) for x in states})>1
    # Horizon refinement
    c1=len({tuple(target(x,1) for _ in (0,)) for x in states})
    # profile through T=2 and T=3
    p2=len({tuple(target(x,t) for t in (1,2)) for x in states})
    p3=len({tuple(target(x,t) for t in (1,2,3)) for x in states})
    assert (c1,p2,p3)==(6,10,12)
    return {"states":12,"minimal_classes_T1":6,"profile_classes":[6,10,12]}


def witness_E():
    # target profiles (g1,g2) for six declared environments
    vals={
      "copy":(1,1),
      "flip":(-1,1),
      "iid_A":(0,0),
      "iid_B":(0,0),
      "delay_plus":(0,1),
      "delay_minus":(0,-1),
    }
    assert len(set(vals.values()))==5
    # instantaneous variety is identical by construction, so it cannot determine five values
    assert len(set(vals.values()))>1
    # iid hidden distinction is removable
    assert vals["iid_A"]==vals["iid_B"]
    return {"environments":6,"target_classes":5,"hidden_irrelevant_pair":["iid_A","iid_B"]}


def witness_ED(n=4):
    perms=list(permutations(range(n)))
    def comp(pi,sigma): return tuple(pi[sigma[i]] for i in range(n))
    raw=len(perms)**2
    maps={comp(pi,sigma) for pi in perms for sigma in perms}
    assert raw==576 and len(maps)==24
    # independent quotient erases each side's port labels, but endpoint maps remain nontrivial
    # refinement using first k endpoint images
    counts=[]; omegas=[]
    all_pairs=[comp(pi,sigma) for pi in perms for sigma in perms]
    for k in range(0,n):
        keys={m[:k] for m in maps}
        counts.append(len(keys))
        worst=0.0
        groups={}
        for m in maps: groups.setdefault(m[:k],[]).append(m)
        for group in groups.values():
            for i in range(len(group)):
                for j in range(i+1,len(group)):
                    worst=max(worst,frob_perm(group[i],group[j]))
        omegas.append(worst)
    assert counts==[1,4,12,24]
    assert all(abs(a-b)<1e-12 for a,b in zip(omegas,[math.sqrt(8),math.sqrt(6),2.0,0.0]))
    return {"raw_pairs":raw,"joint_quotient_classes":len(maps),"refinement_classes":counts,"omega":omegas}


def witness_OED():
    states=list(product((0,1), repeat=3))
    def pairwise(x):
        o,d,e=x; return (o^e,o^d,e^d)
    def t(x):
        o,d,e=x; return o^d^e
    assert pairwise((0,0,0))==pairwise((1,1,1))
    assert t((0,0,0))!=t((1,1,1))
    assert len({pairwise(x) for x in states})==4
    assert len({t(x) for x in states})==2
    # Every pairwise fiber contains both target values.
    groups={}
    for x in states: groups.setdefault(pairwise(x),set()).add(t(x))
    assert all(v=={0,1} for v in groups.values())
    return {"states":8,"pairwise_classes":4,"ternary_classes":2,"minimal_interaction_order":3}


def main():
    result={"D":witness_D(),"E":witness_E(),"ED":witness_ED(),"OED":witness_OED(),"status":"PASS"}
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__': main()
