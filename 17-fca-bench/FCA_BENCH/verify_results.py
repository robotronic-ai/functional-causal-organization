"""Recompute per-test aggregates from the per-instance records of an FCA-Bench
result file and compare them with the stored values.

Usage: python verify_results.py <result.json>
"""
from __future__ import annotations

import json
import sys

from fca_bench import REGISTRY
from fca_bench.core import Result
from fca_bench.run import aggregate


def main(path: str) -> int:
    data = json.load(open(path, encoding="utf-8"))
    results = [
        Result(
            i["instance_id"], i["dim"], i["test"], i["score"], i.get("raw_score", 0.0),
            i.get("valid", True), i.get("status", "executed"), i.get("details", {}),
        )
        for i in data["instances"]
    ]
    agg = aggregate(results)
    bad = 0
    for code, stored in data["tests"].items():
        new = agg["tests"][code]
        ok = abs(new["score"] - stored["score"]) < 1e-4 and all(
            abs(a - b) < 1e-4 for a, b in zip(new["ci95"], stored["ci95"])
        )
        bad += not ok
        print(f"{code:3s} score={new['score']:.4f} ci95={new['ci95']}  {'OK' if ok else 'MISMATCH'}")
    for dim, value in data["behavioral_profile"].items():
        ok = abs(agg["behavioral_profile"][dim] - value) < 1e-4
        bad += not ok
        print(f"profile {dim} = {agg['behavioral_profile'][dim]:.4f}  {'OK' if ok else 'MISMATCH'}")
    print("verified" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
