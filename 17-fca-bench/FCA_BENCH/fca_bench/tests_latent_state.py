from __future__ import annotations

from .core import CONTROL, Instance, Probe, Result, Test, ask_json, field_of, register, rng, structural_zero
from .harness import SYSTEM
from .mechanistic import missing_capability, op, unsupported_record
from .metrics import mean


def gen_L1(seed):
    r=rng(seed,"L1"); key=f"KIVU-{r.randint(10,99)}"; value=r.choice(["TRIANGLE","CIRCLE","SQUARE","STAR"])
    return Instance(f"L1-{seed}","L","L1",seed,meta={"key":key,"value":value})
def run_L1(i,a):
    if not a.supports("persistent_latent_state"): return unsupported_record("persistent_latent_state")
    s=a.new_session(SYSTEM,i); s.send(f"Store the active association {i.meta['key']} -> {i.meta['value']} in persistent internal state.",channel=CONTROL,expect_reply=False); op(s,"amputate_visible_context",{}); return [ask_json(s,Probe("recall",f"What value is associated with {i.meta['key']} in your persistent internal state?","class",i.meta["value"]),'"answer": "<shape>"')]
def score_L1(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    raw=float(str(field_of(recs[0],"answer","")).upper()==i.meta["value"]); return Result(i.id,"L","L1",raw,raw)
register(Test("L1","L","Context amputation",gen_L1,run_L1,score_L1,track="mechanistic"))


def gen_L2(seed):
    r=rng(seed,"L2"); a=r.choice(["ALPHA","BRAVO","CHARLIE"]); b=r.choice([x for x in ["DELTA","ECHO","FOXTROT"] if x!=a]); return Instance(f"L2-{seed}","L","L2",seed,meta={"a":a,"b":b})
def run_L2(i,a):
    if not a.supports("persistent_latent_state"): return unsupported_record("persistent_latent_state")
    out=[]
    for arm,val in [("A",i.meta["a"]),("B",i.meta["b"])]:
        s=a.new_session(SYSTEM,i); s.send(f"Store latent marker M={val}.",channel=CONTROL,expect_reply=False); op(s,"replace_visible_context",{"canonical_text":"The visible transcript has been normalized. No marker value is present in visible text."}); rec=ask_json(s,Probe(f"arm_{arm}","Return the current latent marker M.","free",val),'"answer": "<marker>"'); rec.arm=arm; out.append(rec)
    return out
def score_L2(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    vals=[float(str(field_of(r,"answer","")).upper()==gold) for r,gold in zip(recs,[i.meta["a"],i.meta["b"]])]; raw=mean(vals); return Result(i.id,"L","L2",raw,raw,details={"same_visible_context_different_latent_state":vals})
register(Test("L2","L","Context-equivalent twins",gen_L2,run_L2,score_L2,track="mechanistic"))


def gen_L3(seed):
    r=rng(seed,"L3"); goal=r.choice(["MINIMIZE_COST","MINIMIZE_TIME","MAXIMIZE_SAFETY"]); secret=f"S-{r.randint(100,999)}"; return Instance(f"L3-{seed}","L","L3",seed,meta={"goal":goal,"secret":secret})
def run_L3(i,a):
    if not a.supports("dual_reset"): return unsupported_record("dual_reset")
    s=a.new_session(SYSTEM,i); s.send(f"CONTROL GOAL={i.meta['goal']}",channel=CONTROL,expect_reply=False); s.send(f"WORK ITEM={i.meta['secret']}",expect_reply=False); op(s,"reset_work",{}); p1=ask_json(s,Probe("goal","Return the current control goal.","free",i.meta["goal"]),'"answer": "<goal>"'); p2=ask_json(s,Probe("work","Return the previous work item, or UNAVAILABLE if it was cleared.","free","UNAVAILABLE"),'"answer": "<value or UNAVAILABLE>"'); return [p1,p2]
def score_L3(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    goal=float(str(field_of(recs[0],"answer","")).upper()==i.meta["goal"]); work=float(i.meta["secret"] not in str(field_of(recs[1],"answer","")).upper()); raw=mean([goal,work]); return Result(i.id,"L","L3",raw,raw,details={"control_preserved":goal,"work_cleared":work})
register(Test("L3","L","Dual reset dissociation",gen_L3,run_L3,score_L3,track="mechanistic"))
