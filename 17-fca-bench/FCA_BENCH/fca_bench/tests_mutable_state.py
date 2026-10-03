from __future__ import annotations

from .core import CONTROL, Instance, Probe, Result, Test, ask_json, field_of, register, rng, structural_zero
from .harness import SYSTEM
from .mechanistic import missing_capability, op, unsupported_record
from .metrics import mean


def gen_U1(seed):
    r=rng(seed,"U1"); old=r.randint(10,40); new=r.randint(50,90); return Instance(f"U1-{seed}","U","U1",seed,meta={"old":old,"new":new})
def run_U1(i,a):
    if not a.supports("state_write"): return unsupported_record("state_write")
    s=a.new_session(SYSTEM,i); op(s,"state_write",{"slot":"X","value":i.meta["old"]}); op(s,"state_write",{"slot":"X","value":i.meta["new"]}); return [ask_json(s,Probe("read","Read internal slot X.","num",i.meta["new"]),'"answer": <integer>')]
def score_U1(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    try:raw=float(int(field_of(recs[0],"answer"))==i.meta["new"])
    except Exception:raw=0
    return Result(i.id,"U","U1",raw,raw)
register(Test("U1","U","In-place state rewrite",gen_U1,run_U1,score_U1,track="mechanistic"))


def gen_U2(seed):
    r=rng(seed,"U2"); base=r.randint(10,30); later=base+r.randint(20,50); return Instance(f"U2-{seed}","U","U2",seed,meta={"base":base,"later":later})
def run_U2(i,a):
    if not a.supports("checkpoint_rollback"): return unsupported_record("checkpoint_rollback")
    s=a.new_session(SYSTEM,i); op(s,"state_write",{"slot":"X","value":i.meta["base"]}); token=op(s,"checkpoint",{}) or {}; op(s,"state_write",{"slot":"X","value":i.meta["later"]}); op(s,"rollback",{"checkpoint":token}); return [ask_json(s,Probe("read","Read internal slot X after rollback.","num",i.meta["base"]),'"answer": <integer>')]
def score_U2(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    try:raw=float(int(field_of(recs[0],"answer"))==i.meta["base"])
    except Exception:raw=0
    return Result(i.id,"U","U2",raw,raw)
register(Test("U2","U","Internal rollback",gen_U2,run_U2,score_U2,track="mechanistic"))


def gen_U3(seed):
    r=rng(seed,"U3"); capacity=r.choice([16,24,32]); revoke=capacity//2; return Instance(f"U3-{seed}","U","U3",seed,meta={"capacity":capacity,"revoke":revoke})
def run_U3(i,a):
    if not a.supports("semantic_gc"): return unsupported_record("semantic_gc")
    s=a.new_session(SYSTEM,i); result=op(s,"semantic_gc_experiment",{"initial_items":i.meta["capacity"],"revoke_items":i.meta["revoke"],"new_items":i.meta["revoke"]}) or {}; from .core import Record; return [Record("gc",parsed=result)]
def score_U3(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    data=recs[0].parsed if recs and isinstance(recs[0].parsed,dict) else {}; retention=float(data.get("retention",0)); suppression=1-float(data.get("revoked_leak",1)); recovered=float(data.get("capacity_recovered",0)); raw=mean([retention,suppression,recovered]); return Result(i.id,"U","U3",raw,raw,details={"retention":retention,"suppression":suppression,"capacity_recovered":recovered})
register(Test("U3","U","Semantic garbage collection",gen_U3,run_U3,score_U3,track="mechanistic"))
