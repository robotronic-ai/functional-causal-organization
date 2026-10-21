#!/usr/bin/env python3
"""
GAMMA-CERT v0.4
Autonomous validation of an additive fractional-packing lower certificate.

No external model and no manuscript modification.

Core witness:
    d(S -> T | C) = I(X_S ; X'_T | X_C)

under a product intervention distribution q over present-state coordinates.

If S is on source side A and T∪C is on target side B, then:
    J(A->B) >= I(X_S ; X'_T | X_B) >= d(S->T|C).

A collection of active witnesses can be added fractionally as long as each
source coordinate has total packing weight <= 1. The proof follows from the
fractional Shearer/Madiman-Tetali entropy inequality plus product independence.

The directional packing LP has an exact dual. Combining that dual with binary
cut variables gives a global MILP lower certificate without enumerating cuts.
"""

from __future__ import annotations
import itertools, math, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def bits_of_int(x, n):
    return tuple((x >> i) & 1 for i in range(n))


def int_from_bits(bits):
    return sum((int(b) << i) for i,b in enumerate(bits))


def mutual_information(joint):
    joint=np.asarray(joint,dtype=float)
    joint=joint/joint.sum()
    pa=joint.sum(axis=1,keepdims=True)
    pb=joint.sum(axis=0,keepdims=True)
    out=0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            p=joint[i,j]
            if p>0:
                out += p*math.log2(p/(pa[i,0]*pb[0,j]))
    return out


class FiniteBinaryKernel:
    def __init__(self,n,transition,name):
        self.n=n
        self.transition=transition
        self.name=name

    def J(self,A,B):
        A=tuple(sorted(A)); B=tuple(sorted(B))
        valsA=list(itertools.product((0,1),repeat=len(A)))
        valsB=list(itertools.product((0,1),repeat=len(B)))
        idxA={a:i for i,a in enumerate(valsA)}
        valsOut=list(itertools.product((0,1),repeat=len(B)))
        idxOut={b:i for i,b in enumerate(valsOut)}
        total=0.0
        for b in valsB:
            joint=np.zeros((2**len(A),2**len(B)))
            for a in valsA:
                x=[0]*self.n
                for pos,val in zip(A,a): x[pos]=val
                for pos,val in zip(B,b): x[pos]=val
                for pr,y in self.transition(tuple(x)):
                    out=tuple(y[pos] for pos in B)
                    joint[idxA[a],idxOut[out]] += pr/(2**len(A))
            total += mutual_information(joint)/(2**len(B))
        return total

    def cut_capacity(self,A):
        A=set(A);B=set(range(self.n))-A
        return self.J(A,B)+self.J(B,A)

    def gamma_cut(self,A):
        A=set(A);B=set(range(self.n))-A
        return self.cut_capacity(A)/(2*min(len(A),len(B)))

    def cuts(self):
        for mask in range(1<<(self.n-1)):
            A={0}
            for k in range(1,self.n):
                if mask&(1<<(k-1)): A.add(k)
            if len(A)<self.n:
                yield A

    def gamma_exact(self):
        best=(float("inf"),None)
        for A in self.cuts():
            g=self.gamma_cut(A)
            if g<best[0]-1e-12:
                best=(g,set(A))
        return best


def deterministic_kernel(n, funcs, name):
    def tr(x):
        return [(1.0,tuple(int(f(x)) for f in funcs))]
    return FiniteBinaryKernel(n,tr,name)


def ring_copy(n=6):
    funcs=[]
    for j in range(n):
        src=(j-1)%n
        funcs.append(lambda x,src=src:x[src])
    return deterministic_kernel(n,funcs,f"ring_copy_{n}")


def xor_neighbors(n=6):
    funcs=[]
    for j in range(n):
        a=(j-1)%n;b=(j+1)%n
        funcs.append(lambda x,a=a,b=b:x[a]^x[b])
    return deterministic_kernel(n,funcs,f"xor_neighbors_{n}")


def mixed_boolean_6():
    funcs=[
        lambda x:x[1]^x[2],
        lambda x:x[0]^x[3],
        lambda x:x[1]&x[4],
        lambda x:x[2]^(x[5]&x[0]),
        lambda x:x[3]|x[1],
        lambda x:x[4]^x[0],
    ]
    return deterministic_kernel(6,funcs,"mixed_boolean_6")


def random_boolean_6(seed=0):
    rng=np.random.default_rng(seed)
    tables=[rng.integers(0,2,size=64,dtype=np.int8) for _ in range(6)]
    funcs=[]
    for table in tables:
        funcs.append(lambda x,table=table:int(table[int_from_bits(x)]))
    return deterministic_kernel(6,funcs,f"random_boolean_6_seed{seed}")


def conditional_witness_mi(K,S,T,C):
    """
    d(S->T|C)=I(X_S;X'_T|X_C) under the full product intervention q.
    """
    S=tuple(sorted(S));T=tuple(sorted(T));C=tuple(sorted(C))
    valsS=list(itertools.product((0,1),repeat=len(S)))
    valsT=list(itertools.product((0,1),repeat=len(T)))
    valsC=list(itertools.product((0,1),repeat=len(C)))
    idxS={a:i for i,a in enumerate(valsS)}
    idxT={a:i for i,a in enumerate(valsT)}
    idxC={a:i for i,a in enumerate(valsC)}
    joints=[np.zeros((2**len(S),2**len(T))) for _ in valsC]
    pc=np.zeros(len(valsC))

    for xi in range(2**K.n):
        x=bits_of_int(xi,K.n)
        px=1/(2**K.n)
        sv=tuple(x[i] for i in S)
        cv=tuple(x[i] for i in C)
        ci=idxC[cv]
        pc[ci]+=px
        for pr,y in K.transition(x):
            tv=tuple(y[j] for j in T)
            joints[ci][idxS[sv],idxT[tv]] += px*pr

    ans=0.0
    for ci,joint in enumerate(joints):
        if pc[ci]>0:
            ans += pc[ci]*mutual_information(joint)
    return ans


def build_witnesses(K,maxS,maxT,maxC,eps=1e-12):
    """
    Return witnesses (S, R, d, T, C), where R=T∪C must be on target side.
    Same (S,R) activation: retain only the strongest witness.
    """
    V=set(range(K.n))
    raw=[]
    for ss in range(1,maxS+1):
        for S in itertools.combinations(range(K.n),ss):
            rem1=V-set(S)
            for tt in range(1,maxT+1):
                for T in itertools.combinations(sorted(rem1),tt):
                    rem2=rem1-set(T)
                    for cc in range(maxC+1):
                        for C in itertools.combinations(sorted(rem2),cc):
                            d=conditional_witness_mi(K,S,T,C)
                            if d>eps:
                                R=tuple(sorted(set(T)|set(C)))
                                raw.append((tuple(S),R,float(d),tuple(T),tuple(C)))

    best={}
    for S,R,d,T,C in raw:
        key=(S,R)
        if key not in best or d>best[key][0]:
            best[key]=(d,T,C)
    return [(S,R,d,T,C) for (S,R),(d,T,C) in best.items()]


def directional_packing_value(src,tgt,witnesses):
    """
    Maximum fractional packing of active witnesses.
    Source-coordinate capacity = 1.
    """
    active=[w for w in witnesses if set(w[0])<=src and set(w[1])<=tgt]
    if not active:
        return 0.0
    m=len(active)
    c=-np.array([w[2] for w in active])
    src_list=sorted(src)
    rows=[];cols=[];data=[]
    for r,i in enumerate(src_list):
        for j,w in enumerate(active):
            if i in w[0]:
                rows.append(r);cols.append(j);data.append(1.0)
    A=coo_matrix((data,(rows,cols)),shape=(len(src_list),m)).tocsr()
    res=milp(
        c,
        integrality=np.zeros(m,dtype=int),
        bounds=Bounds(np.zeros(m),np.full(m,np.inf)),
        constraints=[LinearConstraint(A,-np.inf*np.ones(len(src_list)),np.ones(len(src_list)))],
    )
    if not res.success:
        raise RuntimeError(res.message)
    return float(-res.fun)


def packing_gamma_exhaustive(K,witnesses):
    best=(float("inf"),None)
    V=set(range(K.n))
    for A in K.cuts():
        B=V-set(A)
        num=directional_packing_value(A,B,witnesses)+directional_packing_value(B,A,witnesses)
        g=num/(2*min(len(A),len(B)))
        if g<best[0]-1e-12:
            best=(g,set(A))
    return best


def packing_gamma_dual_milp(K,witnesses):
    """
    Global min-cut of the fractional-packing lower envelope.

    For fixed cut, the packing LP dual has source prices lambda_i.
    We combine those dual variables with binary cut variables.
    """
    n=K.n
    W=[(S,R,d) for S,R,d,T,C in witnesses]
    dmax=max([d for _,_,d in W],default=1.0)

    # x_i, lambda_i(for A->B), mu_i(for B->A)
    N=3*n
    obj=np.zeros(N); obj[n:]=1.0
    integrality=np.zeros(N,dtype=int);integrality[:n]=1
    lb=np.zeros(N)
    ub=np.full(N,np.inf)
    ub[:n]=1.0
    ub[n:2*n]=dmax
    ub[2*n:]=dmax

    rows=[];cols=[];data=[];lo=[];hi=[];r=0

    for S,R,d in W:
        # Forward active iff S⊆A and R⊆B:
        # sum lambda_S >= d * [1-sum_S(1-x)-sum_R x]
        for i in S:
            rows.append(r);cols.append(n+i);data.append(1.0)
            rows.append(r);cols.append(i);data.append(-d)
        for j in R:
            rows.append(r);cols.append(j);data.append(d)
        lo.append(d*(1-len(S)));hi.append(np.inf);r+=1

        # Reverse active iff S⊆B and R⊆A.
        for i in S:
            rows.append(r);cols.append(2*n+i);data.append(1.0)
            rows.append(r);cols.append(i);data.append(d)
        for j in R:
            rows.append(r);cols.append(j);data.append(-d)
        lo.append(d*(1-len(R)));hi.append(np.inf);r+=1

    # Force dual prices to zero off their source side.
    for i in range(n):
        rows += [r,r]; cols += [n+i,i]; data += [1.0,-dmax]
        lo.append(-np.inf);hi.append(0.0);r+=1

        rows += [r,r]; cols += [2*n+i,i]; data += [1.0,dmax]
        lo.append(-np.inf);hi.append(dmax);r+=1

    Acons=coo_matrix((data,(rows,cols)),shape=(r,N)).tocsr()

    best=(float("inf"),None)
    for s in range(1,n//2+1):
        Aeq=np.zeros((1,N));Aeq[0,:n]=1.0
        res=milp(
            obj,
            integrality=integrality,
            bounds=Bounds(lb,ub),
            constraints=[
                LinearConstraint(Acons,np.array(lo),np.array(hi)),
                LinearConstraint(Aeq,[s],[s]),
            ],
            options={"time_limit":30.0,"mip_rel_gap":0.0},
        )
        if not res.success:
            raise RuntimeError(res.message)
        g=float(res.fun)/(2*s)
        if g<best[0]-1e-10:
            cut=[i for i in range(n) if res.x[i]>0.5]
            best=(g,cut)
    return best


def run():
    systems=[
        ring_copy(6),
        xor_neighbors(6),
        mixed_boolean_6(),
        random_boolean_6(0),
    ]
    configs=[
        (1,1,1),
        (2,1,1),
        (2,2,1),
        (3,2,1),
    ]

    rows=[]
    for K in systems:
        exact,exact_cut=K.gamma_exact()
        for cfg in configs:
            t0=time.time()
            W=build_witnesses(K,*cfg)
            lex,cut_ex=packing_gamma_exhaustive(K,W)
            lmilp,cut_milp=packing_gamma_dual_milp(K,W)
            assert lmilp <= exact + 1e-9
            assert abs(lex-lmilp)<1e-8
            rows.append({
                "system":K.name,
                "n":K.n,
                "max_source_block":cfg[0],
                "max_target_block":cfg[1],
                "max_context_block":cfg[2],
                "n_witnesses":len(W),
                "Gamma_exact":exact,
                "L_fractional":lex,
                "fraction_of_Gamma":(lex/exact if exact>0 else None),
                "dual_MILP_matches_exhaustive":True,
                "exact_min_cut":sorted(exact_cut),
                "certificate_min_cut":sorted(cut_ex),
                "elapsed_sec":time.time()-t0,
            })
    return rows


if __name__=="__main__":
    rows=run()
    out=Path(__file__).resolve().parent
    (out/"fractional_packing_results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
