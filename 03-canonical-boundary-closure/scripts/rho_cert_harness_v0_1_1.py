#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RHO-CERT prospective harness v0.1."""

from __future__ import annotations
from pathlib import Path
import importlib.util, json, sys

# Fixtures ship in ../results/ in this reviewer edition (repo layout: scripts/ + results/).
# The private oracle answer-key file used for automatic scoring is NOT included here
# (see the campaign README); this harness is provided so the candidate's exact logic
# can be inspected and re-run once fixtures+oracles are supplied.
HERE=Path(__file__).resolve().parents[1]/"results"
FIX=HERE/"RHO_CERT_FIXTURES_v0_1_1.json"
ORA=HERE/"RHO_CERT_ORACLES_v0_1_1_PRIVATE.json"

def canon_partition(p):
    return tuple(sorted((tuple(sorted(block)) for block in p),key=lambda x:(len(x),x)))

def canon_partitions(ps):
    return tuple(sorted(canon_partition(p) for p in ps))

def load_candidate(path):
    spec=importlib.util.spec_from_file_location("rho_candidate",str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules["rho_candidate"]=mod
    spec.loader.exec_module(mod)
    return mod.CANDIDATE

def main(argv=None):
    argv=sys.argv if argv is None else argv
    if len(argv)<2:
        print("usage: python rho_cert_harness_v0_1.py candidate.py")
        return 64

    C=load_candidate(Path(argv[1]))
    fixtures=json.loads(FIX.read_text(encoding="utf-8"))["fixtures"]
    oracles=json.loads(ORA.read_text(encoding="utf-8"))["oracles"]
    omap={o["fixture_id"]:o for o in oracles}
    stats={};total=passes=0;first=None

    for row in fixtures:
        exp=omap[row["fixture_id"]]
        got=C.infer(row)
        fam=row["family"]
        stats.setdefault(fam,[0,0,[]])

        ok=True
        if got.get("status")!=exp["status"]:
            ok=False
        if got.get("maximal_partition_count")!=exp["maximal_partition_count"]:
            ok=False
        if canon_partitions(got.get("maximal_partitions",[])) != canon_partitions(exp["maximal_partitions"]):
            ok=False

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
    print("RHO-CERT VERDICT:",
          "RHO-PROSPECTIVE FINITE NON-REFUTATION" if passes==total
          else "RHO-CANDIDATE REFUTED")
    return 0 if passes==total else 2

if __name__=="__main__":
    raise SystemExit(main())
