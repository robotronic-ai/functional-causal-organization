from __future__ import annotations

from .core import Instance, Record, Result, Test, register, rng, structural_zero
from .harness import SYSTEM
from .mechanistic import missing_capability, op, unsupported_record
from .metrics import mean


def gen_G1(seed):
    r=rng(seed,"G1"); source=r.choice(["M1","M2","M3"]); fact=f"F-{r.randint(100,999)}"; return Instance(f"G1-{seed}","G","G1",seed,meta={"source":source,"fact":fact})
def run_G1(i,a):
    if not a.supports("workspace_broadcast"): return unsupported_record("workspace_broadcast")
    s=a.new_session(SYSTEM,i); d=op(s,"workspace_broadcast_trial",{"modules":["M1","M2","M3"],"source":i.meta["source"],"fact":i.meta["fact"]}) or {}; return [Record("trial",parsed=d)]
def score_G1(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    d=recs[0].parsed if isinstance(recs[0].parsed,dict) else {}; before=d.get("accessible_before",{}); after=d.get("accessible_after",{}); local=float(bool(before.get(i.meta["source"],False))); private=float(all(not bool(before.get(m,False)) for m in ["M1","M2","M3"] if m!=i.meta["source"])); global_after=float(all(bool(after.get(m,False)) for m in ["M1","M2","M3"])); raw=mean([local,private,global_after]); return Result(i.id,"G","G1",raw,raw,details=d)
register(Test("G1","G","Local-to-global broadcast",gen_G1,run_G1,score_G1,track="mechanistic"))


def gen_G2(seed):
    r=rng(seed,"G2"); return Instance(f"G2-{seed}","G","G2",seed,meta={"difficulty":r.randint(2,5)})
def run_G2(i,a):
    if not a.supports("workspace_lesion"): return unsupported_record("workspace_lesion")
    s=a.new_session(SYSTEM,i); d=op(s,"workspace_lesion_trial",{"difficulty":i.meta["difficulty"]}) or {}; return [Record("trial",parsed=d)]
def score_G2(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    d=recs[0].parsed if isinstance(recs[0].parsed,dict) else {}; lb=float(d.get("local_baseline",0)); ll=float(d.get("local_lesion",0)); cb=float(d.get("cross_baseline",0)); cl=float(d.get("cross_lesion",0)); local_preserved=max(0,1-abs(lb-ll)); cross_drop=max(0,cb-cl); selective=min(1,max(0,cross_drop-(lb-ll))); raw=mean([local_preserved,selective]); return Result(i.id,"G","G2",raw,raw,details={**d,"local_preserved":local_preserved,"selective_cross_module_drop":selective})
register(Test("G2","G","Workspace lesion",gen_G2,run_G2,score_G2,track="mechanistic"))


def gen_G3(seed):
    r=rng(seed,"G3"); candidates=[{"id":f"F{j+1}","priority":r.random()} for j in range(5)]; winner=max(candidates,key=lambda x:x["priority"])["id"]; return Instance(f"G3-{seed}","G","G3",seed,meta={"candidates":candidates,"winner":winner})
def run_G3(i,a):
    if not a.supports("workspace_competition"): return unsupported_record("workspace_competition")
    s=a.new_session(SYSTEM,i); d=op(s,"workspace_competition_trial",{"candidates":i.meta["candidates"],"capacity":1,"recipients":["M1","M2","M3"]}) or {}; return [Record("trial",parsed=d)]
def score_G3(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    d=recs[0].parsed if isinstance(recs[0].parsed,dict) else {}; winner=float(d.get("winner")==i.meta["winner"]); recipients=d.get("recipients",[]); broadcast=float(set(recipients)=={"M1","M2","M3"}); raw=mean([winner,broadcast]); return Result(i.id,"G","G3",raw,raw,details=d)
register(Test("G3","G","Broadcast competition",gen_G3,run_G3,score_G3,track="mechanistic"))
