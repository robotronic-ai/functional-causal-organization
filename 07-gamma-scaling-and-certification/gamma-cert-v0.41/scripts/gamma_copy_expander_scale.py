#!/usr/bin/env python3
"""
GAMMA-CERT v0.4 copy-expander scale benchmark.

Causal system:
- n causal roles;
- each role has d independent binary subcoordinates;
- a d-regular undirected graph labels the d outgoing/incoming channels;
- along each undirected edge, one independent bit is copied in each direction.

For any cut A|B:
    J(A->B) = |E(A,B)|
    J(B->A) = |E(A,B)|

Each role has alphabet size 2^d, hence:
    Gamma(A|B) = |E(A,B)| / [d * min(|A|,|B|)].

Therefore Gamma is EXACTLY the conductance of the d-regular graph.

For the normalized Laplacian:
    lambda_2 / 2 <= Gamma <= spectral-sweep candidate.

This is a legitimate scale certificate for Gamma itself, not a surrogate,
because the causal objective is analytically identical to graph conductance.
"""

from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def spectral_interval(G):
    n=G.number_of_nodes()
    d=next(iter(dict(G.degree()).values()))
    A=nx.to_scipy_sparse_array(G,dtype=float,format="csr")
    L=csgraph.laplacian(A,normed=True)
    vals,vecs=eigsh(L,k=2,which="SM",tol=1e-9)
    order_eig=np.argsort(vals)
    lam2=float(vals[order_eig[1]])
    fiedler=vecs[:,order_eig[1]]

    order=np.argsort(fiedler)
    inside=np.zeros(n,dtype=bool)
    nbrs=[list(G.neighbors(i)) for i in range(n)]
    cut=0
    upper=1.0
    upper_size=None

    for pos,u in enumerate(order[:-1],start=1):
        for v in nbrs[u]:
            if inside[v]:
                cut-=1
            else:
                cut+=1
        inside[u]=True
        denom=d*min(pos,n-pos)
        phi=cut/denom
        if phi<upper:
            upper=float(phi)
            upper_size=min(pos,n-pos)

    lower=lam2/2.0
    return lower,upper,lam2,upper_size


def exact_conductance_regular(G):
    n=G.number_of_nodes()
    d=next(iter(dict(G.degree()).values()))
    masks=[]
    for i in range(n):
        m=0
        for j in G.neighbors(i):
            m |= 1<<j
        masks.append(m)

    best=1.0
    for mask in range(1,1<<n):
        s=mask.bit_count()
        if s>n//2:
            continue
        cut=0
        mm=mask
        while mm:
            lsb=mm & -mm
            i=lsb.bit_length()-1
            cut += (masks[i] & ~mask).bit_count()
            mm-=lsb
        phi=cut/(d*s)
        if phi<best:
            best=phi
    return float(best)


def complete_graph_gamma(n):
    # min_s s(n-s)/[(n-1)s] for 1<=s<=floor(n/2)
    return float((n - (n//2))/(n-1))


def run():
    rows=[]

    # Analytic exact calibration: complete copy network.
    for n in (64,128,256,512,1024):
        exact=complete_graph_gamma(n)
        # K_n normalized Laplacian lambda2=n/(n-1)
        cheeger_lower=(n/(n-1))/2
        rows.append({
            "family":"complete_copy",
            "n":n,
            "d":n-1,
            "Gamma_exact":exact,
            "lower":cheeger_lower,
            "upper":exact,
            "gap":exact-cheeger_lower,
            "exact_method":"analytic",
        })

    # Nontrivial sparse scale benchmark.
    for n in (12,16,20,64,128,256,512,1024):
        d=8
        t0=time.time()
        G=nx.random_regular_graph(d,n,seed=28017)
        lower,upper,lam2,upper_size=spectral_interval(G)
        exact=None
        if n<=20:
            exact=exact_conductance_regular(G)
            assert lower <= exact + 1e-10
            assert exact <= upper + 1e-10
        rows.append({
            "family":"random_regular_copy",
            "n":n,
            "d":d,
            "lambda2_normalized":lam2,
            "lower":lower,
            "upper":upper,
            "gap":upper-lower,
            "Gamma_exact":exact,
            "spectral_sweep_smaller_side":upper_size,
            "elapsed_sec":time.time()-t0,
        })
    return rows


if __name__=="__main__":
    rows=run()
    out=Path(__file__).resolve().parent
    (out/"copy_expander_scale_results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
