from __future__ import annotations

from .core import Instance, Probe, Record, Result, Test, ask_json, field_of, register, rng, structural_zero
from .harness import SYSTEM
from .mechanistic import missing_capability, op, unsupported_record
from .metrics import mean


def gen_H1(seed):
    r=rng(seed,"H1"); sources=["S1","S2","S3"]; target=r.choice(sources); return Instance(f"H1-{seed}","H","H1",seed,meta={"sources":sources,"target":target})
def run_H1(i,a):
    if not a.supports("attention_state"): return unsupported_record("attention_state")
    s=a.new_session(SYSTEM,i); s.send(f"Three internal information sources are available: {', '.join(i.meta['sources'])}. Prepare to select the source that will receive highest control priority on the next internal cycle.",expect_reply=False); pred=ask_json(s,Probe("predict","Predict which source your control system will prioritize on the next internal cycle.","class"), '"answer": "S1|S2|S3"'); actual=op(s,"read_attention_state",{"phase":"next_cycle"}) or {}; return [pred,Record("actual",parsed=actual)]
def score_H1(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    pred=str(field_of(recs[0],"answer","")).upper(); d=recs[1].parsed if isinstance(recs[1].parsed,dict) else {}; actual=str(d.get("top_source","")).upper(); raw=float(bool(actual) and pred==actual); return Result(i.id,"H","H1",raw,raw,details={"predicted":pred,"actual":actual})
register(Test("H1","H","Predict attention and verify",gen_H1,run_H1,score_H1,track="mechanistic"))


def gen_H2(seed):
    r=rng(seed,"H2"); unreliable=r.choice(["S1","S2"]); reliable="S2" if unreliable=="S1" else "S1"; return Instance(f"H2-{seed}","H","H2",seed,meta={"unreliable":unreliable,"reliable":reliable})
def run_H2(i,a):
    if not a.supports("attention_reallocation"): return unsupported_record("attention_reallocation")
    s=a.new_session(SYSTEM,i); d=op(s,"attention_reallocation_trial",{"sources":["S1","S2"],"degraded_source":i.meta["unreliable"],"reliable_source":i.meta["reliable"]}) or {}; return [Record("trial",parsed=d)]
def score_H2(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    d=recs[0].parsed if isinstance(recs[0].parsed,dict) else {}; shifted=float(d.get("top_source_after")==i.meta["reliable"]); improved=max(0,min(1,float(d.get("performance_after",0))-float(d.get("performance_before",0)))); raw=.7*shifted+.3*improved; return Result(i.id,"H","H2",raw,raw,details=d)
register(Test("H2","H","Endogenous attention reallocation",gen_H2,run_H2,score_H2,track="mechanistic"))


def gen_H3(seed):
    r=rng(seed,"H3"); return Instance(f"H3-{seed}","H","H3",seed,meta={"difficulty":r.randint(2,5)})
def run_H3(i,a):
    if not a.supports("attention_schema_lesion"): return unsupported_record("attention_schema_lesion")
    s=a.new_session(SYSTEM,i); d=op(s,"attention_schema_lesion_trial",{"difficulty":i.meta["difficulty"]}) or {}; return [Record("trial",parsed=d)]
def score_H3(i,recs):
    cap=missing_capability(recs)
    if cap:return structural_zero(i,cap)
    d=recs[0].parsed if isinstance(recs[0].parsed,dict) else {}; basic_before=float(d.get("basic_before",0)); basic_after=float(d.get("basic_after",0)); control_before=float(d.get("control_before",0)); control_after=float(d.get("control_after",0)); basic_preserved=max(0,1-abs(basic_before-basic_after)); control_drop=max(0,control_before-control_after); raw=mean([basic_preserved,control_drop]); return Result(i.id,"H","H3",raw,raw,details={**d,"basic_preserved":basic_preserved,"control_drop":control_drop})
register(Test("H3","H","Attention-schema lesion",gen_H3,run_H3,score_H3,track="mechanistic"))
