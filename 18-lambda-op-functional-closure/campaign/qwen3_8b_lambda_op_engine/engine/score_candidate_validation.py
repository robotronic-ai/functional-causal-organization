#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from candidate_descriptor import ADDRESSES, OPS, encode_row


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def target(row):
    lookup = {(c["operation"], c["address"]): c for c in row["cells"]}
    return tuple(
        lookup[(op, address)]["behavior"]["outcome"]
        for op in OPS
        for address in ADDRESSES
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    candidate_path = Path(args.candidate)
    panel = json.loads(panel_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if panel.get("stage") != "validation":
        raise ValueError("input is not a validation panel")
    if panel.get("n_skipped") != 0:
        raise ValueError("validation panel contains technical skips")
    specs = candidate["features"]

    fibers = defaultdict(list)
    for row in panel["rows"]:
        desc = encode_row(row, specs)
        fibers[desc].append({"unit": int(row["unit"]), "target": target(row)})

    collisions = []
    nontrivial = []
    for desc, group in fibers.items():
        if len(group) > 1:
            nontrivial.append(group)
        by_target = defaultdict(list)
        for item in group:
            by_target[stable(item["target"])].append(item["unit"])
        if len(by_target) > 1:
            collisions.append({
                "descriptor_bins": list(desc),
                "units": [x["unit"] for x in group],
                "distinct_target_count": len(by_target),
                "target_groups": [{"target": json.loads(k), "units": v} for k, v in by_target.items()],
            })

    passed = not collisions
    closure_type = "NONE"
    if passed:
        closure_type = "NONTRIVIAL" if nontrivial else "TRIVIAL_INJECTIVE"

    result = {
        "schema": "co-qwen3-8b-lambda-op-candidate-validation-score-v1",
        "model": "Qwen3-8B",
        "candidate_name": candidate["candidate_name"],
        "candidate_sha256": sha256(candidate_path),
        "candidate_origin": candidate.get("status"),
        "validation_panel_sha256": sha256(panel_path),
        "n_validation_systems": len(panel["rows"]),
        "n_selected_features": len(specs),
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": len(nontrivial),
        "systems_in_nontrivial_fibers": sum(len(x) for x in nontrivial),
        "n_functional_collisions": len(collisions),
        "closure": "PASS" if passed else "REFUTED",
        "closure_type": closure_type,
        "collision_examples": collisions[:8],
        "interpretation": (
            "This is a fresh validation of a candidate chosen during target-informed exploration. "
            "No candidate parameters were changed using this validation panel."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[LAMBDA-OP CANDIDATE VALIDATION SCORE]")
    print(json.dumps({
        "closure": result["closure"],
        "closure_type": result["closure_type"],
        "candidate_name": result["candidate_name"],
        "n_validation_systems": result["n_validation_systems"],
        "n_selected_features": result["n_selected_features"],
        "n_fibers": result["n_fibers"],
        "n_nontrivial_fibers": result["n_nontrivial_fibers"],
        "systems_in_nontrivial_fibers": result["systems_in_nontrivial_fibers"],
        "n_functional_collisions": result["n_functional_collisions"],
    }, indent=2, sort_keys=True))
    print("[RESULT]", out)


if __name__ == "__main__":
    main()
