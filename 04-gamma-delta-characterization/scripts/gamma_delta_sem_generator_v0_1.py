#!/usr/bin/env python3
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from collections import Counter
import json, math

HERE=Path(__file__).resolve().parent

def fs(x):
    return f"{x.numerator}/{x.denominator}"

def q_uniform(labels):
    n=len(labels)
    return {x:Fraction(1,n) for x in labels}

def perfect_channel(inputs, outputs=None):
    outputs=outputs or [f"y{i}" for i in range(len(inputs))]
    return {
        x:{y:(Fraction(1) if j==i else Fraction(0)) for j,y in enumerate(outputs)}
        for i,x in enumerate(inputs)
    }

def independent_channel(inputs, outputs=("y0","y1")):
    return {x:{outputs[0]:Fraction(1,2),outputs[1]:Fraction(1,2)} for x in inputs}

def erasure_channel(inputs, e=Fraction(1,2)):
    out={}
    for i,x in enumerate(inputs):
        d={f"y{i}":1-e,"E":e}
        out[x]={k:Fraction(v) for k,v in d.items()}
    return out

def direction(q,kernel):
    return {
        "input_q":{x:fs(p) for x,p in q.items()},
        "kernel":{
            x:{y:fs(p) for y,p in d.items()}
            for x,d in kernel.items()
        },
    }

def partition(pid,ab,ba):
    return {"id":pid,"ab":ab,"ba":ba}

def entropy(q):
    z=0.0
    for p in q.values():
        if p:
            z-=float(p)*math.log2(float(p))
    return z

def mi(q,k):
    py={}
    joint={}
    for x,px in q.items():
        for y,p in k[x].items():
            joint[(x,y)]=px*p
            py[y]=py.get(y,Fraction(0))+px*p
    z=0.0
    for (x,y),p in joint.items():
        if not p: continue
        r=float(p/(q[x]*py[y]))
        z+=float(p)*math.log2(r)
    return z

def beta(x,y):
    if x+y==0:return None
    return 2*min(x,y)/(x+y)

def delta_values(support,q,k):
    H=entropy(q)
    I=mi(q,k)
    n=sum(1 for x in support if q.get(x,Fraction(0))>0)
    L=math.log2(n) if n>1 else 0.0
    kap=(H/L) if L>0 else 0.0
    dec=(I/H) if H>0 else None
    cap=(I/L) if L>0 else 0.0
    return kap,dec,cap

def base_fixture(fam):
    b=("0","1"); qu=q_uniform(b)
    perf=perfect_channel(b,["0","1"])
    indep=independent_channel(b,("0","1"))
    erase=erasure_channel(b,Fraction(1,2))

    if fam=="G0":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,["p0","p1"])
    elif fam=="G1":
        gps=[partition("p0",direction(qu,perf),direction(qu,indep))]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,["p0","p1"])
    elif fam=="G2":
        gps=[partition("p0",direction(qu,erase),direction(qu,erase))]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,["p0","p1"])
    elif fam=="G3":
        gps=[
            partition("strong",direction(qu,perf),direction(qu,perf)),
            partition("weak",direction(qu,erase),direction(qu,indep)),
        ]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,["p0","p1"])
    elif fam=="G4":
        gps=[
            partition("unidir",direction(qu,perf),direction(qu,indep)),
            partition("balanced",direction(qu,erase),direction(qu,erase)),
        ]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,["p0","p1"])
    elif fam=="G5":
        gps=[partition("p0",direction(qu,erase),direction(qu,perf))]
        ds=("p0","p1");dq={ds[0]:Fraction(3,4),ds[1]:Fraction(1,4)};dk=erasure_channel(ds,Fraction(1,2))
    elif fam=="D0":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1","p2","p3");dq=q_uniform(ds);dk=perfect_channel(ds,list(ds))
    elif fam=="D1":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1","p2","p3")
        dq={"p0":Fraction(1,2),"p1":Fraction(1,4),"p2":Fraction(1,8),"p3":Fraction(1,8)}
        dk=perfect_channel(ds,list(ds))
    elif fam=="D2":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq=q_uniform(ds);dk=independent_channel(ds,("y0","y1"))
    elif fam=="D3":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq=q_uniform(ds);dk=erasure_channel(ds,Fraction(1,2))
    elif fam=="D4":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq={"p0":Fraction(3,4),"p1":Fraction(1,4)};dk=erasure_channel(ds,Fraction(1,2))
    elif fam=="D5":
        gps=[partition("p0",direction(qu,erase),direction(qu,perf))]
        ds=("p0","p1","p2");dq={"p0":Fraction(1,2),"p1":Fraction(1,4),"p2":Fraction(1,4)}
        dk=erasure_channel(ds,Fraction(1,2))
    elif fam=="IND1A":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,list(ds))
    elif fam=="IND1B":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq=q_uniform(ds);dk=independent_channel(ds,("y0","y1"))
    elif fam=="IND2A":
        gps=[partition("p0",direction(qu,perf),direction(qu,perf))]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,list(ds))
    elif fam=="IND2B":
        gps=[partition("p0",direction(qu,perf),direction(qu,indep))]
        ds=("p0","p1");dq=q_uniform(ds);dk=perfect_channel(ds,list(ds))
    else:
        raise ValueError(fam)

    return gps,ds,dq,dk

def recode_fixture(gps,support,q,k,seed):
    # deterministic label bijections only
    def rec_dir(d,prefix):
        ins=list(d["input_q"])
        im={x:f"{prefix}i{seed}_{j}" for j,x in enumerate(reversed(ins) if seed%2 else ins)}
        outs=sorted({y for row in d["kernel"].values() for y in row})
        om={y:f"{prefix}o{seed}_{j}" for j,y in enumerate(reversed(outs) if (seed//2)%2 else outs)}
        return {
            "input_q":{im[x]:p for x,p in d["input_q"].items()},
            "kernel":{im[x]:{om[y]:p for y,p in row.items()} for x,row in d["kernel"].items()}
        }

    rg=[]
    for i,p in enumerate(gps):
        rg.append({
            "id":f"cut{seed}_{i}",
            "ab":rec_dir(p["ab"],f"a{i}_"),
            "ba":rec_dir(p["ba"],f"b{i}_"),
        })

    sm={x:f"P{seed}_{j}" for j,x in enumerate(reversed(support) if seed%2 else support)}
    outs=sorted({y for row in k.values() for y in row})
    om={y:f"Y{seed}_{j}" for j,y in enumerate(reversed(outs) if (seed//2)%2 else outs)}
    rd={
        "support":[sm[x] for x in support],
        "q":{sm[x]:fs(q[x]) for x in support},
        "kernel":{sm[x]:{om[y]:fs(p) for y,p in row.items()} for x,row in k.items()},
    }
    return rg,rd

def oracle(gps,support,q,k):
    cut=[]
    for p in gps:
        def parse_dir(d):
            qq={x:Fraction(v) for x,v in [(a,Fraction(b.split('/')[0],b.split('/')[1])) for a,b in d["input_q"].items()]}
        # easier recompute from unreencoded source objects is done before recoding outside
    vals=[]
    for p in gps:
        def decode_dir(d):
            qq={x:Fraction(*map(int,s.split("/"))) for x,s in d["input_q"].items()}
            kk={x:{y:Fraction(*map(int,s.split("/"))) for y,s in row.items()} for x,row in d["kernel"].items()}
            return mi(qq,kk)
        x=decode_dir(p["ab"]); y=decode_dir(p["ba"])
        vals.append((p["id"],(x+y)/2,beta(x,y)))
    g=min(v[1] for v in vals)
    betas=sorted({round(v[2],12) for v in vals if abs(v[1]-g)<1e-12 and v[2] is not None})

    H=entropy(q); I=mi(q,k)
    n=sum(1 for x in support if q.get(x,Fraction(0))>0)
    L=math.log2(n) if n>1 else 0
    kap=H/L if L else 0.0
    dec=I/H if H else None
    cap=I/L if L else 0.0
    return {
        "gamma":round(g,12),
        "beta_minimizers":betas,
        "kappa":round(kap,12),
        "delta_dec":None if dec is None else round(dec,12),
        "delta_cap":round(cap,12),
    }

def main():
    fixtures=[];oracles=[]
    fams=("G0","G1","G2","G3","G4","G5","D0","D1","D2","D3","D4","D5",
          "IND1A","IND1B","IND2A","IND2B")
    for fi,fam in enumerate(fams):
        gps,supp,q,k=base_fixture(fam)
        # Build source-form encoded once for oracle helper
        source_gps=[]
        for p in gps:
            source_gps.append({
                "id":p["id"],
                "ab":p["ab"],
                "ba":p["ba"],
            })
        for r in range(2):
            seed=6000+fi*10+r
            rg,rd=recode_fixture(source_gps,supp,q,k,seed)
            fid=f"{fam}.{r+1:02d}"
            fixtures.append({"fixture_id":fid,"family":fam,"gamma_partitions":rg,"delta":rd})
            oracles.append({"fixture_id":fid,"family":fam,**oracle(source_gps,supp,q,k)})

    (HERE/"GAMMA_DELTA_SEM_FIXTURES_v0_1.json").write_text(
        json.dumps({"schema":"GD-SEM-public-v0.1","fixtures":fixtures},indent=2,sort_keys=True)+"\n")
    (HERE/"GAMMA_DELTA_SEM_ORACLES_v0_1_PRIVATE.json").write_text(
        json.dumps({"schema":"GD-SEM-oracles-v0.1","oracles":oracles},indent=2,sort_keys=True)+"\n")

    print("fixture_count:",len(fixtures))
    print("family_counts:",dict(Counter(x["family"] for x in fixtures)))
    # semantic checks
    om={o["fixture_id"]:o for o in oracles}
    print("G1 gamma/beta:",om["G1.01"]["gamma"],om["G1.01"]["beta_minimizers"])
    print("G2 gamma/beta:",om["G2.01"]["gamma"],om["G2.01"]["beta_minimizers"])
    print("D1 profile:",om["D1.01"]["kappa"],om["D1.01"]["delta_dec"],om["D1.01"]["delta_cap"])
    print("IND1 gamma equal:",om["IND1A.01"]["gamma"],om["IND1B.01"]["gamma"],
          "delta:",om["IND1A.01"]["delta_cap"],om["IND1B.01"]["delta_cap"])
    print("IND2 delta equal:",om["IND2A.01"]["delta_cap"],om["IND2B.01"]["delta_cap"],
          "gamma:",om["IND2A.01"]["gamma"],om["IND2B.01"]["gamma"])
    return 0

if __name__=="__main__":
    raise SystemExit(main())
