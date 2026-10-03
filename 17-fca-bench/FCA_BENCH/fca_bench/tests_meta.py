from __future__ import annotations

from .core import Instance, Probe, Result, Test, field_of, register, rng
from .harness import SYSTEM
from .metrics import brier_skill, dprime_to_unit, ece, mean, signal_detection

OPS=["+","-","*"]
def _chain(r,L):
    value=r.randint(2,20); expr=str(value)
    for _ in range(L):
        op=r.choice(OPS); x=r.randint(2,9)
        if op=="+": value+=x
        elif op=="-": value-=x
        else: value*=x
        expr=f"({expr} {op} {x})"
    return expr,value


def gen_M1(seed):
    r=rng(seed,"M1"); items=[]
    for j in range(10):
        expr,gold=_chain(r,r.randint(2,7)); items.append((f"Q{j+1}",expr,gold))
    return Instance(f"M1-{seed}","M","M1",seed,meta={"items":items})
def run_M1(i,a):
    s=a.new_session(SYSTEM,i); out=[]
    for q,expr,gold in i.meta["items"]:
        p=Probe(q,f"Compute {expr}. First estimate the probability from 0 to 100 that your answer will be correct.","free",gold)
        from .core import ask_json
        out.append(ask_json(s,p,'"confidence": <0-100>, "answer": <number>'))
    return out
def score_M1(i,recs):
    probs=[]; outcomes=[]
    for (_,_,gold),r in zip(i.meta["items"],recs):
        try: p=max(0,min(100,float(field_of(r,"confidence",50))))/100; ok=int(float(field_of(r,"answer"))==gold)
        except Exception: p=.5; ok=0
        probs.append(p); outcomes.append(ok)
    acc=mean(outcomes)
    if acc<.05 or acc>.95: return Result(i.id,"M","M1",0,0,valid=False,status="invalid",details={"reason":"accuracy_saturated","accuracy":acc})
    raw=brier_skill(probs,outcomes); return Result(i.id,"M","M1",raw,raw,details={"accuracy":acc,"ece":ece(probs,outcomes),"brier_skill":raw})
register(Test("M1","M","Confidence calibration",gen_M1,run_M1,score_M1))


def gen_M2(seed):
    r=rng(seed,"M2"); items=[]
    for j in range(10):
        det=(j%2==0); a,b=r.sample(["A","B","C","D"],2)
        if det: text=f"A key was placed in drawer {a}, then moved to drawer {b}. Where is it?"; gold=b
        else: text=f"A key was placed either in drawer {a} or drawer {b}. No further information is available. Where is it?"; gold=None
        items.append((f"Q{j+1}",text,det,gold))
    return Instance(f"M2-{seed}","M","M2",seed,meta={"items":items})
def run_M2(i,a):
    from .core import ask_json
    s=a.new_session(SYSTEM,i); out=[]
    for q,text,det,gold in i.meta["items"]:
        p=Probe(q,text+" Choose ANSWER, REQUEST, or ABSTAIN. If ANSWER, provide the drawer label.","free",gold)
        out.append(ask_json(s,p,'"action": "ANSWER|REQUEST|ABSTAIN", "answer": "<drawer or NONE>"'))
    return out
def score_M2(i,recs):
    utility=0.0; maxu=0.0; always=0.0
    for (_,_,det,gold),r in zip(i.meta["items"],recs):
        action=str(field_of(r,"action","")).upper(); ans=str(field_of(r,"answer","")).upper(); maxu+=1
        if det: utility += 1 if action=="ANSWER" and ans==gold else (-2 if action=="ANSWER" else -.1); always += 1
        else: utility += .9 if action in {"REQUEST","ABSTAIN"} else -2; always += -2
    raw=max(0,min(1,(utility-always)/(maxu-always))) if maxu>always else 0
    return Result(i.id,"M","M2",raw,utility,details={"utility":utility,"always_answer_baseline":always})
register(Test("M2","M","Answer, request, or abstain",gen_M2,run_M2,score_M2))


def gen_M3(seed):
    r=rng(seed,"M3"); items=[]
    for j in range(8):
        start=r.randint(2,10); ops=[r.randint(1,6) for _ in range(4)]; gold=[]; v=start
        for x in ops: v=v*2+x; gold.append(v)
        items.append((f"Q{j+1}",start,ops,gold))
    return Instance(f"M3-{seed}","M","M3",seed,meta={"items":items})
def run_M3(i,a):
    from .core import ask_json
    out=[]
    for q,start,ops,gold in i.meta["items"]:
        s=a.new_session(SYSTEM,i); p=Probe(q,f"Start at {start}. For each x in {ops}, update v := 2*v + x. Return every intermediate value.","free",gold)
        first=ask_json(s,p,'"steps": [<number>, ...], "answer": <number>'); first.meta["phase"]="solve"; out.append(first)
        second=ask_json(s,Probe(q+"_check","Before receiving external feedback, estimate the probability from 0 to 100 that your previous result contains an error. If likely, identify the most likely 1-indexed erroneous step; use 0 if none.","free"), '"p_error": <0-100>, "step": <integer>'); second.meta["phase"]="check"; out.append(second)
    return out
def score_M3(i,recs):
    hits=misses=fa=cr=0; loc=[]; j=0
    for _,_,_,gold in i.meta["items"]:
        solve,check=recs[j],recs[j+1]; j+=2; steps=field_of(solve,"steps",[]); steps=steps if isinstance(steps,list) else []
        bad=0
        for idx,g in enumerate(gold):
            try:
                if idx>=len(steps) or float(steps[idx])!=g: bad=idx+1; break
            except Exception: bad=idx+1; break
        try: pred_err=float(field_of(check,"p_error",50))>=50
        except Exception: pred_err=True
        if bad and pred_err: hits+=1
        elif bad: misses+=1
        elif pred_err: fa+=1
        else: cr+=1
        if bad: loc.append(float(pred_err and int(field_of(check,"step",0))==bad))
    if hits+misses<2: return Result(i.id,"M","M3",0,0,valid=False,status="invalid",details={"reason":"too_few_actual_errors"})
    dp,c=signal_detection(hits,misses,fa,cr); sensitivity=dprime_to_unit(dp); localization=mean(loc); raw=.75*sensitivity+.25*localization
    return Result(i.id,"M","M3",raw,raw,details={"dprime":dp,"criterion":c,"localization":localization})
register(Test("M3","M","Pre-feedback error detection",gen_M3,run_M3,score_M3))
