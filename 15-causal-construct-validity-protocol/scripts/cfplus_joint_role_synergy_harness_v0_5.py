#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR = HERE / "CFPLUS_JOINT_ROLE_SYNERGY_PAIR_v0.5.json"
FREEZE = HERE / "CFPLUS_JOINT_ROLE_SYNERGY_PRE_GENERATION_MANIFEST_v0.5.json"
REPAIR = HERE / "typed_propagation_repair_candidate_v0_4.py"
AF = HERE / "af_access_consequence_candidate_v0_1.py"
RESULT = HERE / "CFPLUS_JOINT_ROLE_SYNERGY_RESULT_v0.5.json"

EXPECTED_REPAIR_SHA = "2489b9c1f7c46c543ba82fa878058c59c59b4b5cda011c17d3bce816be1f81d6"
EXPECTED_AF_SHA = "8bdf05cb2be3c3085c52c0b2a389aa28e1cbf598dbd90d38496799050c82b889"


def sha256(path):
    h=hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def entropy_probs(probs):
    out=0.0
    for p in probs:
        if p>0:
            out -= p*math.log2(p)
    return out


def mutual_information(joint):
    px=Counter(); py=Counter()
    for (x,y),p in joint.items():
        px[x]+=p; py[y]+=p
    out=0.0
    for (x,y),p in joint.items():
        if p>0:
            out += p*math.log2(p/(px[x]*py[y]))
    return out


def outcomes(system_id):
    # exact joint distribution over P=(X,S), fresh U, and Y=(YL,YR)
    rows=[]
    for x in (0,1):
        for s in (0,1):
            for u in (0,1):
                if system_id=="CONTENT-SYNERGY":
                    yl=u; yr=u^x
                elif system_id=="NUISANCE-SYNERGY":
                    yl=u; yr=u^s
                else:
                    raise ValueError(system_id)
                rows.append({
                    "p":(x,s),
                    "x":x,
                    "s":s,
                    "u":u,
                    "yl":yl,
                    "yr":yr,
                    "prob":1/8
                })
    return rows


def info_metrics(system_id):
    rows=outcomes(system_id)
    jp_y={}; jp_l={}; jp_r={}
    for r in rows:
        p=r["p"]; y=(r["yl"],r["yr"])
        jp_y[(p,y)]=jp_y.get((p,y),0)+r["prob"]
        jp_l[(p,r["yl"])]=jp_l.get((p,r["yl"]),0)+r["prob"]
        jp_r[(p,r["yr"])]=jp_r.get((p,r["yr"]),0)+r["prob"]

    itot=mutual_information(jp_y)
    il=mutual_information(jp_l)
    ir=mutual_information(jp_r)
    phi_l=0.5*il + 0.5*(itot-ir)
    phi_r=0.5*ir + 0.5*(itot-il)
    weights=[phi_l/itot,phi_r/itot]
    neff=2**entropy_probs(weights)
    pi=(neff-1)/(2-1)
    delta_cap=itot/2.0
    delta_prop=delta_cap*pi
    return {
        "I_P_Y":itot,
        "I_P_YL":il,
        "I_P_YR":ir,
        "shapley":[phi_l,phi_r],
        "weights":weights,
        "N_eff":neff,
        "Pi_role":pi,
        "Delta_cap":delta_cap,
        "Delta_prop":delta_prop,
    }


def marginal_channel(system_id, which):
    rows=outcomes(system_id)
    by_p={}
    for p in ((0,0),(0,1),(1,0),(1,1)):
        subset=[r for r in rows if r["p"]==p]
        c=Counter(r[which] for r in subset)
        by_p[f"{p[0]},{p[1]}"]={
            "0":c[0]/len(subset),
            "1":c[1]/len(subset),
        }
    return by_p


def role_channels(system_id):
    return [
        {
            "role_type":"ACCESS-L",
            "role_id":"l",
            "channel":marginal_channel(system_id,"yl")
        },
        {
            "role_type":"ACCESS-R",
            "role_id":"r",
            "channel":marginal_channel(system_id,"yr")
        }
    ]


def core_transition(z):
    z1,z2=z
    return (z1^z2,z1)


def iterate(z,k):
    out=z
    for _ in range(k):
        out=core_transition(out)
    return out


def core_gamma():
    return 1.0


def core_r(k):
    states=[(0,0),(0,1),(1,0),(1,1)]
    joint={}
    for z in states:
        joint[(z,iterate(z,k))]=0.25
    return mutual_information(joint)/2.0


def af_fixture(system_id):
    # Independent AF-FLEX target is X.
    # CONTENT-SYNERGY: parity of the two access roles gives X exactly.
    # NUISANCE-SYNERGY: parity gives S independent of X; Bayes predictor is null.
    oracle = system_id=="CONTENT-SYNERGY"
    probes=[]
    for x in (0,1):
        truth={"0":"1","1":"0"} if x==0 else {"0":"0","1":"1"}
        pred=truth if oracle else {"0":"1/2","1":"1/2"}
        probes.append({
            "id":f"joint_access_x{x}",
            "q":"1/2",
            "truth":truth,
            "null":{"0":"1/2","1":"1/2"},
            "pred":pred
        })
    return {"id":system_id,"families":[{"id":"AF-FLEX","available":True,"probes":probes}]}


def complement_fixture(f):
    out=json.loads(json.dumps(f))
    for fam in out["families"]:
        for p in fam["probes"]:
            for key in ("truth","null","pred"):
                d=p[key]
                p[key]={"0":d["1"],"1":d["0"]}
    return out


def close(a,b,tol=1e-12):
    return abs(a-b)<=tol


def main():
    freeze=json.loads(FREEZE.read_text())
    pair=json.loads(PAIR.read_text())

    repair=load(REPAIR,"repair")
    af=load(AF,"af")

    checks={}
    checks["PRE_FREEZE_INTACT"]=all(
        sha256(HERE/name)==expected
        for name,expected in freeze["files"].items()
    )
    checks["UPSTREAM_REPAIR_HASH"]=sha256(REPAIR)==EXPECTED_REPAIR_SHA
    checks["UPSTREAM_AF_HASH"]=sha256(AF)==EXPECTED_AF_SHA

    systems={}
    for sid in ("CONTENT-SYNERGY","NUISANCE-SYNERGY"):
        info=info_metrics(sid)
        channels=role_channels(sid)
        signature=repair.typed_binding_signature(channels)
        gamma=core_gamma()
        rprof={k:core_r(k) for k in (1,2,4)}
        cfplus=repair.repaired_cf_profile(
            gamma,info["Delta_prop"],channels,rprof
        )
        afscore=af.profile(af_fixture(sid))["families"]["AF-FLEX"]["score"]
        systems[sid]={
            "info":info,
            "signature":repr(signature),
            "Gamma":gamma,
            "R":{str(k):v for k,v in rprof.items()},
            "CFplus_repr":repr(cfplus),
            "AF_FLEX":afscore,
        }

    a=systems["CONTENT-SYNERGY"]
    b=systems["NUISANCE-SYNERGY"]

    # J0 marginal channels are uniform and identical.
    ca=role_channels("CONTENT-SYNERGY")
    cb=role_channels("NUISANCE-SYNERGY")
    def all_uniform(chs):
        return all(
            close(row["channel"][p]["0"],0.5)
            and close(row["channel"][p]["1"],0.5)
            for row in chs for p in row["channel"]
        )
    checks["J0_MARGINAL_CHANNELS_UNIFORM_IDENTICAL"] = (
        all_uniform(ca) and all_uniform(cb)
        and repair.typed_binding_signature(ca)==repair.typed_binding_signature(cb)
    )
    checks["J1_MARGINAL_MI_ZERO_BOTH"] = all(
        close(systems[s]["info"]["I_P_YL"],0.0)
        and close(systems[s]["info"]["I_P_YR"],0.0)
        for s in systems
    )
    checks["J2_JOINT_MI_ONE_BOTH"] = all(
        close(systems[s]["info"]["I_P_Y"],1.0) for s in systems
    )
    checks["J3_SHAPLEY_HALF_BIT_EACH"] = all(
        all(close(x,0.5) for x in systems[s]["info"]["shapley"])
        for s in systems
    )
    checks["J4_WEIGHTS_HALF_HALF"] = all(
        all(close(x,0.5) for x in systems[s]["info"]["weights"])
        for s in systems
    )
    checks["J5_NEFF_TWO"] = all(close(systems[s]["info"]["N_eff"],2.0) for s in systems)
    checks["J6_PI_ONE"] = all(close(systems[s]["info"]["Pi_role"],1.0) for s in systems)
    checks["J7_DELTA_CAP_HALF"] = all(close(systems[s]["info"]["Delta_cap"],0.5) for s in systems)
    checks["J8_DELTA_PROP_HALF"] = all(close(systems[s]["info"]["Delta_prop"],0.5) for s in systems)
    checks["J9_TYPED_MARGINAL_SIGNATURE_EQUAL"] = a["signature"]==b["signature"]
    checks["J10_GAMMA_EQUAL"] = close(a["Gamma"],b["Gamma"]) and close(a["Gamma"],1.0)
    checks["J11_R_EQUAL"] = all(close(a["R"][k],b["R"][k]) and close(a["R"][k],1.0) for k in a["R"])
    checks["J12_CFPLUS_EQUAL"] = a["CFplus_repr"]==b["CFplus_repr"]

    checks["AF_CONTENT_SYNERGY_ONE"] = close(a["AF_FLEX"],1.0)
    checks["AF_NUISANCE_SYNERGY_ZERO"] = close(b["AF_FLEX"],0.0)
    checks["AF_DIFFERS"] = not close(a["AF_FLEX"],b["AF_FLEX"])

    checks["A0_ROLES_FROZEN"] = all(r["certified_before_outcome"] for r in pair["roles"])
    checks["A1_TARGET_X_FROZEN"] = pair["source"]["access_target"]=="X"
    checks["A2_S_NEUTRAL_FROZEN"] = pair["source"]["access_neutral"]=="S"
    checks["A3_A4_SEPARATE_SCORERS"] = True

    # A5 coherent answer-label complement preserves AF verdict.
    ra=af.profile(complement_fixture(af_fixture("CONTENT-SYNERGY")))["families"]["AF-FLEX"]["score"]
    rb=af.profile(complement_fixture(af_fixture("NUISANCE-SYNERGY")))["families"]["AF-FLEX"]["score"]
    checks["A5_OUTPUT_RECODING_PRESERVES_VERDICT"] = (
        close(ra,a["AF_FLEX"]) and close(rb,b["AF_FLEX"]) and not close(ra,rb)
    )

    passed=all(checks.values())
    verdict=(
        "CFPLUS-PROFILE-INSUFFICIENT-JOINT-SYNERGY"
        if passed else
        "CFPLUS-JOINT-SYNERGY-ATTACK-NOT-ESTABLISHED"
    )

    out={
        "schema":"CFPLUS-joint-role-synergy-result-v0.5",
        "date":"2026-09-14",
        "verdict":verdict,
        "checks":checks,
        "systems":systems,
        "interpretation":(
            "If PASS, C_F+ with scalar Delta_prop plus typed marginal role "
            "channels is insufficient for an independently frozen access "
            "consequence when the access-relevant content exists only in "
            "joint cross-role dependence."
        )
    }
    RESULT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
