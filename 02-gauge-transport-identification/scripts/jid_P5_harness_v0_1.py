#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J-ID-P5 v0.1 exact prospective harness."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
FIX=HERE/"JID_P5_FIXTURES_v0_1.json"
ORA=HERE/"JID_P5_ORACLES_v0_1_PRIVATE.json"


def F(s):
    a,b=s.split("/")
    return Fraction(int(a),int(b))


def normalize_kernel(obj):
    return tuple(
        (k,tuple(sorted((str(y),F(p)) for y,p in d.items())))
        for k,d in sorted(obj.items())
    )


def load_candidate(path):
    spec=importlib.util.spec_from_file_location("jid_candidate",str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules["jid_candidate"]=mod
    spec.loader.exec_module(mod)
    return mod.CANDIDATE


def main(argv=None):
    argv=sys.argv if argv is None else argv
    if len(argv)<2:
        print("usage: python jid_P5_harness_v0_1.py candidate.py")
        return 64

    C=load_candidate(Path(argv[1]))
    fixtures=json.loads(FIX.read_text(encoding="utf-8"))["fixtures"]
    oracles=json.loads(ORA.read_text(encoding="utf-8"))["oracles"]
    omap={o["fixture_id"]:o for o in oracles}

    passes=0;total=0;stats={}
    first=None

    for row in fixtures:
        fam=row["family"]
        stats.setdefault(fam,[0,0,[]])
        exp=omap[row["fixture_id"]]
        got=C.infer(row)

        ok=True
        if got.get("status")!=exp["status"]:
            ok=False
        if got.get("compatible_J_count")!=exp["compatible_J_count"]:
            ok=False
        if tuple(got.get("B_set_bits",[]))!=tuple(exp["B_set_bits"]):
            ok=False

        if exp["status"]=="PI-POINT":
            if "oracle_transition" not in got:
                ok=False
            else:
                if normalize_kernel(got["oracle_transition"]) != normalize_kernel(exp["oracle_transition"]):
                    ok=False

        total+=1;stats[fam][1]+=1
        if ok:
            passes+=1;stats[fam][0]+=1
        else:
            stats[fam][2].append(row["fixture_id"])
            if first is None:first=row["fixture_id"]

    print(f"CANDIDATE: {C.name} {C.version}")
    print(f"EXACT_PASSES: {passes}/{total}")
    for fam in ("P5A","P5B","P5C","P5D","P5E"):
        p,t,bad=stats[fam]
        print(f"{fam}: {p}/{t} failures={bad}")
    print("FIRST_FAILURE:",first)
    print("JID-P5 VERDICT:", "P5-V4 FINITE NON-REFUTATION" if passes==total else "P5-V3 CANDIDATE REFUTED")
    return 0 if passes==total else 2


if __name__=="__main__":
    raise SystemExit(main())
