#!/usr/bin/env python3
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from collections import Counter
import json, math

HERE=Path(__file__).resolve().parent

def fs(x): return f"{x.numerator}/{x.denominator}"

def entropy(q):
    z=0.0
    for p in q.values():
        if p:z-=float(p)*math.log2(float(p))
    return z

def mi(q,k):
    py={}
    joint={}
    for x,px in q.items():
        for y,p in k[x].items():
            joint[(x,y)]=px*p
            py[y]=py.get(y,Fraction(0))+px*p
    out=0.0
    for (x,y),p in joint.items():
        if not p:continue
        out+=float(p)*math.log2(float(p/(q[x]*py[y])))
    return out

def beta(x,y):
    if x+y==0:return None
    return 2*min(x,y)/(x+y)

def D(q,k):
    return {"input_q":{x:fs(p) for x,p in q.items()},
            "kernel":{x:{y:fs(p) for y,p in row.items()} for x,row in k.items()}}

def P(pid,ab,ba): return {"id":pid,"ab":ab,"ba":ba}

def perf():
    q={"0":Fraction(1,2),"1":Fraction(1,2)}
    k={"0":{"0":Fraction(1),"1":Fraction(0)},
       "1":{"0":Fraction(0),"1":Fraction(1)}}
    return q,k

def indep():
    q={"0":Fraction(1,2),"1":Fraction(1,2)}
    k={"0":{"0":Fraction(1,2),"1":Fraction(1,2)},
       "1":{"0":Fraction(1,2),"1":Fraction(1,2)}}
    return q,k

def erase(e):
    q={"0":Fraction(1,2),"1":Fraction(1,2)}
    k={"0":{"0":1-e,"E":e},"1":{"1":1-e,"E":e}}
    return q,k

def asym():
    q={"0":Fraction(1,2),"1":Fraction(1,2)}
    k={"0":{"0":Fraction(7,8),"1":Fraction(1,8)},
       "1":{"0":Fraction(1,4),"1":Fraction(3,4)}}
    return q,k

def nonuniform_dir():
    q={"0":Fraction(3,4),"1":Fraction(1,4)}
    k={"0":{"0":Fraction(1),"1":Fraction(0)},
       "1":{"0":Fraction(1,4),"1":Fraction(3,4)}}
    return q,k

def delta_oracle(support,q,k):
    H=entropy(q);I=mi(q,k)
    n=sum(1 for x in support if q.get(x,Fraction(0))>0)
    L=math.log2(n) if n>1 else 0
    return (H/L if L else 0.0,
            None if H==0 else I/H,
            I/L if L else 0.0)

def gamma_oracle(parts):
    vals=[]
    for p in parts:
        def parse(d):
            q={x:Fraction(*map(int,s.split("/"))) for x,s in d["input_q"].items()}
            k={x:{y:Fraction(*map(int,s.split("/"))) for y,s in row.items()} for x,row in d["kernel"].items()}
            return mi(q,k)
        x=parse(p["ab"]);y=parse(p["ba"])
        vals.append(((x+y)/2,beta(x,y)))
    g=min(x for x,b in vals)
    bs=sorted({round(b,12) for x,b in vals if abs(x-g)<=1e-12 and b is not None})
    return g,bs

def family(fam):
    qu,kp=perf(); qi,ki=indep()
    qe12,ke12=erase(Fraction(1,2))
    qe34,ke34=erase(Fraction(3,4))
    qa,ka=asym()
    qn,kn=nonuniform_dir()

    if fam=="H0":
        # all have gamma=.5: (1,0), (.5,.5), (.25,.75)
        q25,k25=erase(Fraction(3,4))  # MI .25
        q75,k75=erase(Fraction(1,4))  # MI .75
        parts=[P("u",D(qu,kp),D(qi,ki)),
               P("b",D(qe12,ke12),D(qe12,ke12)),
               P("m",D(q25,k25),D(q75,k75))]
        supp=("a","b");dq={"a":Fraction(1,2),"b":Fraction(1,2)}
        dk={"a":{"a":Fraction(1)},"b":{"b":Fraction(1)}}
    elif fam=="H1":
        parts=[P("a",D(qa,ka),D(qn,kn))]
        supp=("a","b");dq={"a":Fraction(3,4),"b":Fraction(1,4)}
        dk={"a":{"a":Fraction(1)},"b":{"b":Fraction(1)}}
    elif fam=="H2":
        parts=[P("n",D(qn,kn),D(qn,kn)),
               P("p",D(qu,kp),D(qu,kp))]
        supp=("a","b");dq={"a":Fraction(1,2),"b":Fraction(1,2)}
        dk={"a":{"0":Fraction(3,4),"1":Fraction(1,4)},
            "b":{"0":Fraction(1,4),"1":Fraction(3,4)}}
    elif fam=="H3":
        parts=[P("z0",D(qi,ki),D(qi,ki)),P("z1",D(qi,ki),D(qi,ki))]
        supp=("a","b","c");dq={"a":Fraction(1,3),"b":Fraction(1,3),"c":Fraction(1,3)}
        dk={x:{"0":Fraction(1,2),"1":Fraction(1,2)} for x in supp}
    elif fam=="H4":
        parts=[P("p",D(qu,kp),D(qe12,ke12))]
        supp=("a","b","c")
        dq={"a":Fraction(1,2),"b":Fraction(1,3),"c":Fraction(1,6)}
        dk={x:{x:Fraction(1)} for x in supp}
    elif fam=="H5":
        parts=[P("p",D(qa,ka),D(qe34,ke34))]
        supp=tuple("abcde")
        dq={x:Fraction(1,5) for x in supp}
        e=Fraction(1,3)
        dk={x:{x:1-e,"E":e} for x in supp}
    elif fam=="H6":
        parts=[P("p",D(qu,kp),D(qn,kn))]
        supp=("a","b","c","d")
        dq={x:Fraction(1,4) for x in supp}
        dk={"a":{"0":Fraction(1)},"b":{"0":Fraction(1)},
            "c":{"1":Fraction(1)},"d":{"1":Fraction(1)}}
    elif fam=="H7":
        parts=[P("p",D(qe12,ke12),D(qe34,ke34))]
        supp=("a","b","c")
        dq={"a":Fraction(1,2),"b":Fraction(1,4),"c":Fraction(1,4)}
        dk={"a":{"0":Fraction(1)},"b":{"1":Fraction(1)},"c":{"1":Fraction(1)}}
    else: raise ValueError(fam)
    return parts,supp,dq,dk

def recode(parts,supp,q,k,seed):
    def rd(d,tag):
        ins=list(d["input_q"]);im={x:f"{tag}I{seed}_{i}" for i,x in enumerate(reversed(ins) if seed%2 else ins)}
        outs=sorted({y for row in d["kernel"].values() for y in row})
        om={y:f"{tag}O{seed}_{i}" for i,y in enumerate(reversed(outs) if (seed//2)%2 else outs)}
        return {"input_q":{im[x]:p for x,p in d["input_q"].items()},
                "kernel":{im[x]:{om[y]:p for y,p in row.items()} for x,row in d["kernel"].items()}}
    pp=[{"id":f"c{seed}_{i}","ab":rd(p["ab"],f"a{i}"),"ba":rd(p["ba"],f"b{i}")} for i,p in enumerate(parts)]
    sm={x:f"P{seed}_{i}" for i,x in enumerate(reversed(supp) if seed%2 else supp)}
    outs=sorted({y for row in k.values() for y in row})
    om={y:f"Y{seed}_{i}" for i,y in enumerate(reversed(outs) if (seed//2)%2 else outs)}
    dd={"support":[sm[x] for x in supp],
        "q":{sm[x]:fs(q[x]) for x in supp},
        "kernel":{sm[x]:{om[y]:fs(p) for y,p in row.items()} for x,row in k.items()}}
    return pp,dd

def main():
    fixtures=[];oracles=[]
    fams=tuple(f"H{i}" for i in range(8))
    for fi,fam in enumerate(fams):
        parts,supp,q,k=family(fam)
        g,bs=gamma_oracle(parts)
        kap,dec,cap=delta_oracle(supp,q,k)
        for r in range(4):
            seed=7000+fi*10+r
            pp,dd=recode(parts,supp,q,k,seed)
            fid=f"{fam}.{r+1:02d}"
            fixtures.append({"fixture_id":fid,"family":fam,"gamma_partitions":pp,"delta":dd})
            oracles.append({"fixture_id":fid,"family":fam,
                            "gamma":round(g,12),"beta_minimizers":bs,
                            "kappa":round(kap,12),
                            "delta_dec":None if dec is None else round(dec,12),
                            "delta_cap":round(cap,12)})
    (HERE/"GAMMA_DELTA_SEM_HOLDOUT_CF_FIXTURES_v0_1.json").write_text(
        json.dumps({"schema":"GD-SEM-HOLDOUT-public-v0.1","fixtures":fixtures},indent=2,sort_keys=True)+"\n")
    (HERE/"GAMMA_DELTA_SEM_HOLDOUT_CF_ORACLES_v0_1_PRIVATE.json").write_text(
        json.dumps({"schema":"GD-SEM-HOLDOUT-oracles-v0.1","oracles":oracles},indent=2,sort_keys=True)+"\n")
    print("fixture_count:",len(fixtures))
    print("family_counts:",dict(Counter(x["family"] for x in fixtures)))
    for fam in fams:
        o=next(x for x in oracles if x["family"]==fam)
        print(fam,"gamma=",o["gamma"],"betas=",o["beta_minimizers"],
              "kappa=",o["kappa"],"dec=",o["delta_dec"],"cap=",o["delta_cap"])
    return 0

if __name__=="__main__": raise SystemExit(main())
