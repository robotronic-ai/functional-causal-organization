#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TARGET-BOUNDARY candidate-fixed holdout harness v0.1."""

from __future__ import annotations
from pathlib import Path
import importlib.util,json,sys

# Fixtures ship in ../results/ in this reviewer edition (repo layout: scripts/ + results/).
# The private oracle answer-key file used for automatic scoring is NOT included here
# (see the campaign README); this harness is provided so the candidate's exact logic
# can be inspected and re-run once fixtures+oracles are supplied.
HERE=Path(__file__).resolve().parents[1]/"results"
FIX=HERE/"TARGET_BOUNDARY_HOLDOUT_CF_FIXTURES_v0_1.json"
ORA=HERE/"TARGET_BOUNDARY_HOLDOUT_CF_ORACLES_v0_1_PRIVATE.json"

def canon(bs):
    return tuple(sorted((tuple(sorted(b)) for b in bs),key=lambda x:(len(x),x)))

def load_candidate(path):
    spec=importlib.util.spec_from_file_location("tb_candidate",str(path))
    m=importlib.util.module_from_spec(spec);sys.modules["tb_candidate"]=m;spec.loader.exec_module(m)
    return m.CANDIDATE

def main(argv=None):
    argv=sys.argv if argv is None else argv
    if len(argv)<2:
        print("usage: python target_boundary_harness_v0_1.py candidate.py")
        return 64
    C=load_candidate(Path(argv[1]))
    fx=json.loads(FIX.read_text())["fixtures"]
    oo=json.loads(ORA.read_text())["oracles"]
    om={o["fixture_id"]:o for o in oo}
    total=passes=0;stats={};first=None
    for row in fx:
        exp=om[row["fixture_id"]];got=C.infer(row);fam=row["family"]
        stats.setdefault(fam,[0,0,[]])
        ok=(got.get("status")==exp["status"]
            and got.get("minimal_boundary_count")==exp["minimal_boundary_count"]
            and canon(got.get("minimal_boundaries",[]))==canon(exp["minimal_boundaries"]))
        total+=1;stats[fam][1]+=1
        if ok:passes+=1;stats[fam][0]+=1
        else:
            stats[fam][2].append(row["fixture_id"])
            if first is None:first=row["fixture_id"]
    print(f"CANDIDATE: {C.name} {C.version}")
    print(f"EXACT_PASSES: {passes}/{total}")
    for fam in sorted(stats):
        p,t,bad=stats[fam]
        print(f"{fam}: {p}/{t} failures={bad}")
    print("FIRST_FAILURE:",first)
    print("TARGET-BOUNDARY VERDICT:",
          "TB-CF-HOLDOUT-PASS" if passes==total
          else "TB-CF-HOLDOUT-REFUTED")
    return 0 if passes==total else 2

if __name__=="__main__":
    raise SystemExit(main())
