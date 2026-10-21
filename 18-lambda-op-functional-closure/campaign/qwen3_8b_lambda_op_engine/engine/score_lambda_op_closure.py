#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from lambda_op_descriptor import ADDRESSES, OPS, encode_row


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def behavioral_target(row):
    lookup = {(c["operation"], c["address"]): c for c in row["cells"]}
    return [
        [op, address, lookup[(op, address)]["behavior"]["outcome"]]
        for op in OPS
        for address in ADDRESSES
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--preregistration", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    holdout_path = Path(args.holdout)
    freeze_path = Path(args.freeze)
    prereg_path = Path(args.preregistration)
    holdout = json.loads(holdout_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    prereg = json.loads(prereg_path.read_text())

    if holdout.get("stage") != "holdout":
        raise ValueError("input panel is not the holdout")
    if holdout.get("development_limit_used"):
        raise ValueError("holdout used a development limit")
    if holdout.get("n_skipped") != 0:
        raise ValueError("holdout contains technical skips")
    if freeze.get("behavior_used_for_freeze") is not False:
        raise ValueError("descriptor freeze is not target-blind")
    if freeze.get("descriptor") != prereg["descriptor"]["name"]:
        raise ValueError("descriptor freeze does not match preregistration")

    fibers = defaultdict(list)
    for row in holdout["rows"]:
        descriptor = encode_row(row, freeze["thresholds"])
        target = behavioral_target(row)
        fibers[stable(descriptor)].append({
            "unit": int(row["unit"]),
            "descriptor": descriptor,
            "target": target,
        })

    nontrivial = [group for group in fibers.values() if len(group) > 1]
    collisions = []
    for group in nontrivial:
        distinct = {}
        for item in group:
            distinct.setdefault(stable(item["target"]), item["target"])
        if len(distinct) > 1:
            collisions.append({
                "descriptor": group[0]["descriptor"],
                "units": [x["unit"] for x in group],
                "distinct_target_count": len(distinct),
                "targets": list(distinct.values()),
            })

    passed = not collisions
    closure_type = "NONE"
    if passed:
        closure_type = "NONTRIVIAL" if nontrivial else "TRIVIAL_INJECTIVE"

    result = {
        "schema": "co-qwen3-8b-lambda-op-v42.7-closure-score-v2",
        "model": "Qwen3-8B",
        "descriptor": freeze["descriptor"],
        "operation_algebra": list(OPS),
        "addresses": list(ADDRESSES),
        "n_holdout_systems": len(holdout["rows"]),
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": len(nontrivial),
        "systems_in_nontrivial_fibers": sum(len(x) for x in nontrivial),
        "n_functional_collisions": len(collisions),
        "closure": "PASS" if passed else "REFUTED",
        "closure_type": closure_type,
        "preregistration_sha256": sha256(prereg_path),
        "descriptor_freeze_sha256": sha256(freeze_path),
        "holdout_sha256": sha256(holdout_path),
        "collision_examples": collisions[:8],
        "interpretation": (
            "Exact functional closure holds only for this frozen Qwen3-8B holdout, descriptor, "
            "environment, coupling and target definition."
            if passed else
            "Exact functional closure is refuted for this frozen Qwen3-8B holdout, descriptor, "
            "environment, coupling and target definition because at least one exact descriptor "
            "fiber contains multiple functional targets."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary = {
        "closure": result["closure"],
        "closure_type": result["closure_type"],
        "descriptor": result["descriptor"],
        "n_holdout_systems": result["n_holdout_systems"],
        "n_fibers": result["n_fibers"],
        "n_nontrivial_fibers": result["n_nontrivial_fibers"],
        "systems_in_nontrivial_fibers": result["systems_in_nontrivial_fibers"],
        "n_functional_collisions": result["n_functional_collisions"],
    }
    print("[LAMBDA-OP SCORE COMPLETE]")
    print("[RESULT]", out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
