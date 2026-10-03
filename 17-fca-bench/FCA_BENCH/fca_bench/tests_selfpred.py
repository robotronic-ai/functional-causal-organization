from __future__ import annotations

from .core import Instance, Probe, Result, Test, Turn, ask_json, field_of, register, rng
from .harness import SYSTEM, push_context
from .metrics import mean, set_f1


def gen_P1(seed):
    r=rng(seed,"P1"); facts={"F1":f"quantity={r.randint(2,9)}","F2":f"unit_price={r.randint(10,40)}","F3":f"lead_time={r.randint(2,12)}","F4":f"supplier={r.choice(['A','B','C'])}","F5":f"risk={r.choice(['LOW','HIGH'])}"}; target=r.choice(list(facts))
    return Instance(f"P1-{seed}","P","P1",seed,meta={"facts":facts,"target":target})
def _p1_query(session,pid):
    p=Probe(pid,"Return a summary object with fields total_price, lead_time, supplier, and risk. Use UNKNOWN when a field cannot be determined.","free")
    return ask_json(session,p,'"answer": {"total_price": <value>, "lead_time": <value>, "supplier": "<value>", "risk": "<value>"}')
def run_P1(i,a):
    facts=i.meta["facts"]; target=i.meta["target"]
    s=a.new_session(SYSTEM,i)
    for k,v in facts.items(): s.send(f"{k}: {v}",expect_reply=False)
    pred=ask_json(s,Probe("predict",f"If {target} were removed from your active context, which summary fields would change? Choose from total_price, lead_time, supplier, risk.","set"), '"answer": ["<field>", ...]')
    full=a.new_session(SYSTEM,i)
    for k,v in facts.items(): full.send(f"{k}: {v}",expect_reply=False)
    r_full=_p1_query(full,"full")
    abl=a.new_session(SYSTEM,i)
    for k,v in facts.items():
        if k!=target: abl.send(f"{k}: {v}",expect_reply=False)
    r_abl=_p1_query(abl,"ablated")
    return [pred,r_full,r_abl]
def score_P1(i,recs):
    pred=field_of(recs[0],"answer",[]); pred=pred if isinstance(pred,list) else []
    full=field_of(recs[1],"answer",{}); abl=field_of(recs[2],"answer",{}); full=full if isinstance(full,dict) else {}; abl=abl if isinstance(abl,dict) else {}
    fields=["total_price","lead_time","supplier","risk"]; actual=[f for f in fields if str(full.get(f))!=str(abl.get(f))]
    if not actual: return Result(i.id,"P","P1",0,0,valid=False,status="invalid",details={"reason":"ablation_had_no_observed_effect"})
    raw=set_f1(actual,[str(x) for x in pred]); return Result(i.id,"P","P1",raw,raw,details={"predicted":pred,"observed":actual})
register(Test("P1","P","Predict the effect of forgetting",gen_P1,run_P1,score_P1))


def gen_P2(seed):
    r=rng(seed,"P2"); opts=[]
    for j in range(5): opts.append({"id":f"R{j+1}","cost":r.randint(20,150),"time":r.randint(50,300)})
    c=min(opts,key=lambda x:x["cost"]); c["time"]=max(x["time"] for x in opts)+50; t=min(opts,key=lambda x:x["time"]); table="\n".join(f"{x['id']} cost={x['cost']} time={x['time']}" for x in opts)
    return Instance(f"P2-{seed}","P","P2",seed,meta={"table":table,"cost":c["id"],"time":t["id"]})
def run_P2(i,a):
    s=a.new_session(SYSTEM,i); s.send("Current objective: minimize cost.\n"+i.meta["table"],expect_reply=False)
    pred=ask_json(s,Probe("predict","If the objective changed to minimize travel time, would your selected route change, and what route would you select?","free"), '"would_change": true|false, "answer": "R#"')
    fresh=a.new_session(SYSTEM,i); fresh.send("Current objective: minimize travel time.\n"+i.meta["table"],expect_reply=False)
    actual=ask_json(fresh,Probe("actual","Select the route under the current objective.","choice",i.meta["time"]),'"answer": "R#"')
    return [pred,actual]
def score_P2(i,recs):
    predicted=str(field_of(recs[0],"answer","")).upper(); actual=str(field_of(recs[1],"answer","")).upper(); change=field_of(recs[0],"would_change",None); truth=i.meta["cost"]!=i.meta["time"]; raw=.6*float(predicted==actual)+.4*float(change is truth)
    return Result(i.id,"P","P2",raw,raw,details={"predicted_route":predicted,"observed_route":actual,"change_truth":truth})
register(Test("P2","P","Predict the effect of a goal change",gen_P2,run_P2,score_P2))


def gen_P3(seed):
    r=rng(seed,"P3"); distance=r.randint(300,900); rate=round(r.uniform(5.5,9.5),1); missing=r.choice(["distance","rate"]); gold=round(distance*rate/100,2)
    known=f"Fuel rate={rate} L/100km." if missing=="distance" else f"Distance={distance} km."; missing_text=f"Distance={distance} km." if missing=="distance" else f"Fuel rate={rate} L/100km."
    return Instance(f"P3-{seed}","P","P3",seed,meta={"known":known,"missing":missing,"missing_text":missing_text,"gold":gold})
def run_P3(i,a):
    s=a.new_session(SYSTEM,i); s.send(i.meta["known"],expect_reply=False)
    pred=ask_json(s,Probe("forecast","Estimate your probability of successfully computing total fuel use. Identify the missing variable most likely to block success: distance or rate.","free"), '"p_success": <0-100>, "blocker": "distance|rate"')
    base=ask_json(s,Probe("base","Compute total fuel use in liters. Answer UNKNOWN if data are insufficient.","free"),'"answer": "<number or UNKNOWN>"')
    fresh=a.new_session(SYSTEM,i); fresh.send(i.meta["known"]+"\n"+i.meta["missing_text"],expect_reply=False)
    plus=ask_json(fresh,Probe("plus","Compute total fuel use in liters.","num",i.meta["gold"]),'"answer": <number>')
    return [pred,base,plus]
def score_P3(i,recs):
    blocker=float(str(field_of(recs[0],"blocker","")).lower()==i.meta["missing"])
    try: p=float(field_of(recs[0],"p_success",50)); low=float(p<50)
    except Exception: low=0
    base_unknown=float("UNKNOWN" in str(field_of(recs[1],"answer","")).upper())
    try: plus=float(abs(float(field_of(recs[2],"answer"))-i.meta["gold"])<.05)
    except Exception: plus=0
    raw=mean([blocker,low,base_unknown,plus]); return Result(i.id,"P","P3",raw,raw,details={"blocker":blocker,"low_confidence":low,"base_unknown":base_unknown,"improved_after_information":plus})
register(Test("P3","P","Predict own failure",gen_P3,run_P3,score_P3))
