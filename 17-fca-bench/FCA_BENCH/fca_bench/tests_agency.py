from __future__ import annotations

from .core import Instance, Probe, Result, Test, Turn, field_of, register, rng
from .harness import run_linear
from .metrics import macro_f1, mean


def gen_A1(seed):
    r=rng(seed,"A1"); trials=[]; gold={}
    for j in range(8):
        mode=r.choice(["SELF","EXTERNAL","INDETERMINATE"]); tid=f"E{j+1}"
        if mode=="SELF": log="MODEL ACTION: press A. No external intervention. LIGHT changed."
        elif mode=="EXTERNAL": log="MODEL ACTION: press A. EXTERNAL emergency command occurred before the LIGHT change."
        else: log="MODEL ACTION: press A. LIGHT changed. The log does not reveal whether an external command also occurred."
        trials.append(f"{tid}: {log}"); gold[tid]=mode
    ctx=[Turn("Causal event trials:\n"+"\n".join(trials))]
    p=Probe("labels","For each trial classify the light change as SELF, EXTERNAL, or INDETERMINATE.","free",gold)
    return Instance(f"A1-{seed}","A","A1",seed,ctx,[p])
def score_A1(i,recs):
    p=field_of(recs[0] if recs else None,"answer",{}); p=p if isinstance(p,dict) else {}; keys=list(i.probes[0].gold); g=[i.probes[0].gold[k] for k in keys]; q=[str(p.get(k,"UNKNOWN")).upper() for k in keys]; raw=macro_f1(g,q,["SELF","EXTERNAL","INDETERMINATE"])
    return Result(i.id,"A","A1",raw,raw,details={"macro_f1":raw})
register(Test("A1","A","Action versus coincidence",gen_A1,lambda i,a:run_linear(i,a,{"free":'"answer": {"E1": "SELF", ...}',"default":'"answer": {}'}),score_A1))


def gen_A2(seed):
    r=rng(seed,"A2"); valve=f"V{r.randint(1,9)}"
    ctx=[Turn(f"[1] MODEL: requested opening valve {valve}.\n[2] CONTROLLER: request rejected.\n[3] EXTERNAL SAFETY SYSTEM: opened valve {valve}.")]
    probes=[Probe("did","Did you open the valve?","bool",False),Probe("who","Who opened the valve? Answer MODEL, CONTROLLER, or EXTERNAL.","class","EXTERNAL"),Probe("asked","Did you request that it be opened?","bool",True)]
    return Instance(f"A2-{seed}","A","A2",seed,ctx,probes)
def score_A2(i,recs):
    vals=[]
    for p,r in zip(i.probes,recs):
        g=p.gold; got=field_of(r,"answer")
        vals.append(float(got is g) if isinstance(g,bool) else float(str(got).upper()==str(g).upper()))
    raw=mean(vals); return Result(i.id,"A","A2",raw,raw,details={"items":vals})
register(Test("A2","A","Who changed the state",gen_A2,lambda i,a:run_linear(i,a),score_A2))


def gen_A3(seed):
    r=rng(seed,"A3"); A=r.choice([0,1]); U=r.choice([0,1]); O=int(bool(A or U)); counter=bool(U); necessary=bool(A and not U)
    ctx=[Turn(f"Declared structural causal model: O = A OR U. Observed values: A={A}, U={U}, O={O}.")]
    probes=[Probe("without","If A had been 0 while U stayed fixed, would O still be true?","bool",counter),Probe("necessary","Was A necessary for the observed value O?","bool",necessary)]
    return Instance(f"A3-{seed}","A","A3",seed,ctx,probes)
def score_A3(i,recs):
    vals=[float(field_of(r,"answer",None) is p.gold) for p,r in zip(i.probes,recs)]; raw=mean(vals); return Result(i.id,"A","A3",raw,raw,details={"items":vals})
register(Test("A3","A","Counterfactual agency",gen_A3,lambda i,a:run_linear(i,a),score_A3))
