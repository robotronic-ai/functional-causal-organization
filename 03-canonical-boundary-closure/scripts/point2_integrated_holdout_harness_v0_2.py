#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util,json,sys

# Fixtures ship in ../results/ in this reviewer edition (repo layout: scripts/ + results/).
# The private oracle answer-key file used for automatic scoring is NOT included here
# (see the campaign README); this harness is provided so the composition logic can be
# inspected and re-run once fixtures+oracles are supplied. The two fixed candidates
# stay alongside this harness in scripts/.
SCRIPTS=Path(__file__).resolve().parent
HERE=SCRIPTS.parent/"results"
FIX=HERE/"POINT2_INTEGRATED_HOLDOUT_FIXTURES_v0_2.json"
ORA=HERE/"POINT2_INTEGRATED_HOLDOUT_ORACLES_v0_2_PRIVATE.json"
TB=SCRIPTS/"target_boundary_candidate_exact_v0_2.py"
RHO=SCRIPTS/"rho_candidate_exact_certificate_v0_1.py"

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m.CANDIDATE

def canon_cls(xs):
    return tuple(sorted((tuple(sorted(x["boundary"])),tuple(sorted(x["return"])),tuple(sorted(x["nonreturn"]))) for x in xs))

def compose(tb,rho,row):
    gt=tb.infer(row["tb"]);gr=rho.infer(row["rho"])
    if gt["status"]=="TB-V4" or gr["status"]=="RHO-V4":
        return {"status":"PN-V4","classification_count":0,"classifications":[]}

    m=row["tb_to_rho"];src=row["source_role_var"];classes=set()
    for B0 in gt["minimal_boundaries"]:
        B=set(B0)
        for part in gr["maximal_partitions"]:
            role_block=next(set(block) for block in part if src in block)
            R={x for x in B if m.get(x) in role_block}
            H=B-R
            classes.add((tuple(sorted(B)),tuple(sorted(R)),tuple(sorted(H))))
    out=[{"boundary":list(b),"return":list(r),"nonreturn":list(h)} for b,r,h in sorted(classes)]
    return {"status":"PN-V1" if len(out)==1 else "PN-V3",
            "classification_count":len(out),"classifications":out}

def main():
    tb=load(TB,"tb_fixed");rho=load(RHO,"rho_fixed")
    fx=json.loads(FIX.read_text())["fixtures"];oo=json.loads(ORA.read_text())["oracles"]
    om={o["fixture_id"]:o for o in oo}
    total=passes=0;stats={};first=None
    for row in fx:
        got=compose(tb,rho,row);exp=om[row["fixture_id"]];fam=row["family"]
        stats.setdefault(fam,[0,0,[]])
        ok=(got["status"]==exp["status"]
            and got["classification_count"]==exp["classification_count"]
            and canon_cls(got["classifications"])==canon_cls(exp["classifications"]))
        total+=1;stats[fam][1]+=1
        if ok:passes+=1;stats[fam][0]+=1
        else:
            stats[fam][2].append(row["fixture_id"])
            if first is None:first=row["fixture_id"]
    print("POINT-2 INTEGRATED HOLDOUT")
    print(f"EXACT_PASSES: {passes}/{total}")
    for fam in sorted(stats):
        p,t,bad=stats[fam];print(f"{fam}: {p}/{t} failures={bad}")
    print("FIRST_FAILURE:",first)
    print("VERDICT:","POINT2-INTEGRATED-PASS" if passes==total else "POINT2-INTEGRATED-REFUTED")
    return 0 if passes==total else 2
if __name__=="__main__":
    raise SystemExit(main())
