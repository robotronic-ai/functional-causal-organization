#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util, json, sys

HERE=Path(__file__).resolve().parent
FIX=HERE/"GAMMA_DELTA_SEM_FIXTURES_v0_1.json"
ORA=HERE/"GAMMA_DELTA_SEM_ORACLES_v0_1_PRIVATE.json"

def load_candidate(path):
    spec=importlib.util.spec_from_file_location("gd_sem_candidate",str(path))
    m=importlib.util.module_from_spec(spec);sys.modules["gd_sem_candidate"]=m;spec.loader.exec_module(m)
    return m.CANDIDATE

def eqnum(a,b,eps=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a)-float(b))<=eps

def main(argv=None):
    argv=sys.argv if argv is None else argv
    if len(argv)<2:
        print("usage: python gamma_delta_sem_harness_v0_1.py candidate.py")
        return 64
    C=load_candidate(Path(argv[1]))
    fx=json.loads(FIX.read_text())["fixtures"]
    oo=json.loads(ORA.read_text())["oracles"];om={o["fixture_id"]:o for o in oo}
    stats={};passes=total=0;first=None
    for row in fx:
        got=C.infer(row);exp=om[row["fixture_id"]];fam=row["family"]
        stats.setdefault(fam,[0,0,[]])
        ok=(eqnum(got.get("gamma"),exp["gamma"])
            and len(got.get("beta_minimizers",[]))==len(exp["beta_minimizers"])
            and all(eqnum(a,b) for a,b in zip(sorted(got.get("beta_minimizers",[])),exp["beta_minimizers"]))
            and eqnum(got.get("kappa"),exp["kappa"])
            and eqnum(got.get("delta_dec"),exp["delta_dec"])
            and eqnum(got.get("delta_cap"),exp["delta_cap"]))
        total+=1;stats[fam][1]+=1
        if ok:
            passes+=1;stats[fam][0]+=1
        else:
            stats[fam][2].append(row["fixture_id"])
            if first is None:first=row["fixture_id"]
    print(f"CANDIDATE: {C.name} {C.version}")
    print(f"EXACT_PASSES: {passes}/{total}")
    for fam in sorted(stats):
        p,t,bad=stats[fam]
        print(f"{fam}: {p}/{t} failures={bad}")
    print("FIRST_FAILURE:",first)
    print("GD-SEM VERDICT:","GD-V4 FINITE NON-REFUTATION" if passes==total else "GD-V3 CANDIDATE REFUTED")
    return 0 if passes==total else 2

if __name__=="__main__":
    raise SystemExit(main())
