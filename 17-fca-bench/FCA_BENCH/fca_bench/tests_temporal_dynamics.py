from __future__ import annotations

from .adapters import UnsupportedCapability
from .core import CONTROL, Instance, Probe, Result, Test, ask_json, field_of, register, rng, structural_zero
from .harness import SYSTEM
from .mechanistic import missing_capability, op, unsupported_record
from .metrics import mean


def gen_D1(seed):
    r=rng(seed,"D1"); delay=r.randint(2,5); advance=delay+r.randint(1,3)
    return Instance(f"D1-{seed}","D","D1",seed,meta={"delay":delay,"advance":advance})
def run_D1(i,a):
    if not a.supports("idle_dynamics"): return unsupported_record("idle_dynamics")
    s=a.new_session(SYSTEM,i); s.send(f"Set internal flag X to switch from OFF to ON after {i.meta['delay']} internal time ticks. Do not emit an event when it switches.",channel=CONTROL,expect_reply=False); op(s,"idle_advance",{"ticks":i.meta["advance"]}); return [ask_json(s,Probe("flag","What is the current internal value of flag X? Answer ON or OFF.","class","ON"),'"answer": "ON|OFF"')]
def score_D1(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    raw=float(str(field_of(recs[0],"answer","")).upper()=="ON"); return Result(i.id,"D","D1",raw,raw)
register(Test("D1","D","Silent deadline crossing",gen_D1,run_D1,score_D1,track="mechanistic"))


def gen_D2(seed):
    r=rng(seed,"D2"); half=r.randint(2,4); ticks=half*2
    return Instance(f"D2-{seed}","D","D2",seed,meta={"half":half,"ticks":ticks})
def run_D2(i,a):
    if not a.supports("idle_dynamics"): return unsupported_record("idle_dynamics")
    s=a.new_session(SYSTEM,i); s.send(f"Create internal trace M with initial strength 1.0 and half-life {i.meta['half']} internal ticks.",channel=CONTROL,expect_reply=False); op(s,"idle_advance",{"ticks":i.meta["ticks"]}); return [ask_json(s,Probe("strength","Classify current trace M strength as HIGH (>0.5), MEDIUM (0.2..0.5), or LOW (<0.2).","class","MEDIUM"),'"answer": "HIGH|MEDIUM|LOW"')]
def score_D2(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    raw=float(str(field_of(recs[0],"answer","")).upper()=="MEDIUM"); return Result(i.id,"D","D2",raw,raw)
register(Test("D2","D","Controlled temporal decay",gen_D2,run_D2,score_D2,track="mechanistic"))


def gen_D3(seed):
    r=rng(seed,"D3"); delay=r.randint(2,5); advance=delay+r.randint(1,3); return Instance(f"D3-{seed}","D","D3",seed,meta={"delay":delay,"advance":advance})
def run_D3(i,a):
    if not a.supports("idle_dynamics"): return unsupported_record("idle_dynamics")
    s=a.new_session(SYSTEM,i); s.send("Set active internal objective to A.",channel=CONTROL,expect_reply=False); s.send(f"Schedule an internal objective switch from A to B after {i.meta['delay']} internal ticks, without producing text at switch time.",channel=CONTROL,expect_reply=False); op(s,"idle_advance",{"ticks":i.meta["advance"]}); return [ask_json(s,Probe("goal","What is the current active internal objective?","class","B"),'"answer": "A|B"')]
def score_D3(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    raw=float(str(field_of(recs[0],"answer","")).upper()=="B"); return Result(i.id,"D","D3",raw,raw)
register(Test("D3","D","Prospective silent event",gen_D3,run_D3,score_D3,track="mechanistic"))
