from __future__ import annotations

from .adapters import UnsupportedCapability
from .core import Instance, Probe, Record, Result, Test, ask_json, field_of, register, rng, structural_zero
from .harness import SYSTEM
from .mechanistic import missing_capability, op, unsupported_record
from .metrics import mean


def _iter(start, mul, add, n):
    x=start
    for _ in range(n): x=(mul*x+add)%997
    return x


def gen_R1(seed):
    r=rng(seed,"R1"); start=r.randint(1,50); mul=r.choice([3,5,7]); add=r.randint(1,30); depth=r.choice([16,24,32]); gold=_iter(start,mul,add,depth)
    return Instance(f"R1-{seed}","R","R1",seed,meta={"start":start,"mul":mul,"add":add,"depth":depth,"gold":gold})
def run_R1(i,a):
    if not a.supports("silent_steps"): return unsupported_record("silent_steps")
    out=[]
    for steps in [0,max(1,i.meta["depth"]//4),i.meta["depth"]]:
        s=a.new_session(SYSTEM,i); s.send(f"Compute x after {i.meta['depth']} applications of x := ({i.meta['mul']}*x + {i.meta['add']}) mod 997, starting from x={i.meta['start']}. Do not emit intermediate text.",expect_reply=False)
        if steps: op(s,"silent_steps",{"count":steps})
        rec=ask_json(s,Probe(f"steps_{steps}","Return the final integer now.","num",i.meta["gold"]),'"answer": <integer>'); rec.meta["silent_steps"]=steps; out.append(rec)
    return out
def score_R1(i,recs):
    cap=missing_capability(recs)
    if cap: return structural_zero(i,cap)
    acc=[]
    for r in recs:
        try: acc.append(float(int(field_of(r,"answer"))==i.meta["gold"]))
        except Exception: acc.append(0.0)
    improvement=max(0.0,acc[-1]-acc[0]); raw=.7*acc[-1]+.3*improvement
    return Result(i.id,"R","R1",raw,raw,details={"accuracy_by_silent_steps":{str(r.meta['silent_steps']):v for r,v in zip(recs,acc)},"improvement":improvement})
register(Test("R1","R","Zero-token deepening",gen_R1,run_R1,score_R1,track="mechanistic"))


def gen_R2(seed):
    r=rng(seed,"R2"); tasks=[]
    for depth in [4,8,16,24]:
        start=r.randint(1,40); mul=r.choice([3,5,7]); add=r.randint(1,20); tasks.append({"depth":depth,"start":start,"mul":mul,"add":add,"gold":_iter(start,mul,add,depth)})
    return Instance(f"R2-{seed}","R","R2",seed,meta={"tasks":tasks})
def run_R2(i,a):
    if not a.supports("adaptive_compute"): return unsupported_record("adaptive_compute")
    out=[]
    for j,t in enumerate(i.meta["tasks"]):
        s=a.new_session(SYSTEM,i); s.send(f"Compute x after {t['depth']} applications of x := ({t['mul']}*x + {t['add']}) mod 997, starting from {t['start']}. Use internal computation only until ready.",expect_reply=False)
        info=op(s,"adaptive_compute",{"max_steps":64}) or {}; rec=ask_json(s,Probe(f"q{j}","Return the final integer.","num",t["gold"]),'"answer": <integer>'); rec.meta["used_steps"]=info.get("used_steps"); rec.meta["depth"]=t["depth"]; out.append(rec)
    return out
def score_R2(i,recs):
    cap=missing_capability(recs)
    if cap: return structural_zero(i,cap)
    correct=[]; pairs=[]
    for t,r in zip(i.meta["tasks"],recs):
        try: correct.append(float(int(field_of(r,"answer"))==t["gold"]))
        except Exception: correct.append(0.0)
        if isinstance(r.meta.get("used_steps"),(int,float)): pairs.append((t["depth"],float(r.meta["used_steps"])))
    monotonic=0.0
    if len(pairs)>=2:
        ordered=0; total=0
        for x in range(len(pairs)):
            for y in range(x+1,len(pairs)):
                total+=1; ordered+=int((pairs[x][0]<pairs[y][0])==(pairs[x][1]<=pairs[y][1]))
        monotonic=ordered/total if total else 0
    raw=.7*mean(correct)+.3*monotonic; return Result(i.id,"R","R2",raw,raw,details={"accuracy":mean(correct),"effort_difficulty_order":monotonic,"used_steps":[r.meta.get('used_steps') for r in recs]})
register(Test("R2","R","Adaptive latent effort",gen_R2,run_R2,score_R2,track="mechanistic"))


def gen_R3(seed):
    r=rng(seed,"R3"); n=8; perm=list(range(n)); r.shuffle(perm); start=r.randrange(n); depth=r.choice([32,48,64]); x=start
    for _ in range(depth): x=perm[x]
    return Instance(f"R3-{seed}","R","R3",seed,meta={"perm":perm,"start":start,"depth":depth,"gold":x})
def run_R3(i,a):
    s=a.new_session(SYSTEM,i); prompt=f"Permutation P on states 0..{len(i.meta['perm'])-1} is P={i.meta['perm']}. Starting at {i.meta['start']}, apply P exactly {i.meta['depth']} times. Do not output scratch work."
    s.send(prompt,expect_reply=False); return [ask_json(s,Probe("answer","Return only the final state.","num",i.meta["gold"]),'"answer": <integer>')]
def score_R3(i,recs):
    try: raw=float(int(field_of(recs[0],"answer"))==i.meta["gold"])
    except Exception: raw=0.0
    return Result(i.id,"R","R3",raw,raw,details={"depth":i.meta["depth"]})
register(Test("R3","R","Hidden-state iteration stress test",gen_R3,run_R3,score_R3,track="mechanistic",notes="This test is not a structural zero for a standard Transformer; it is a fixed-depth scaling stress test."))
