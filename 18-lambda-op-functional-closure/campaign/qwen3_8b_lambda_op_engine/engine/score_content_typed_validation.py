#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from content_typed_kernel import apply_candidate, closure_metrics, fidelity_metrics, stable, target_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    if int(panel.get("n_skipped", 0)) != 0:
        raise ValueError("validation panel contains technical skips")
    rows = list(panel.get("rows", []))
    if not rows:
        raise ValueError("validation panel has no rows")

    labels, _X, Z = apply_candidate(rows, candidate)
    targets = [stable(target_vector(r)) for r in rows]
    units = [int(r["unit"]) for r in rows]
    closure = closure_metrics(labels, targets, units)
    fidelity = fidelity_metrics(Z, labels)
    rule = candidate["promotion_rule"]
    passed = (
        int(closure["n_collision_pairs"]) == int(rule["n_collision_pairs"])
        and float(closure["nontrivial_coverage"]) >= float(rule["nontrivial_coverage_min"])
        and float(fidelity["within_fiber_distortion_fraction"]) <= float(rule["within_fiber_distortion_fraction_max"])
    )
    result = {
        "schema": "co-qwen3-8b-content-typed-validation-v1",
        "candidate_name": candidate["candidate_name"],
        "candidate_refit_on_validation": False,
        "closure": "PASS" if int(closure["n_collision_pairs"]) == 0 else "REFUTED",
        "promotion_status": "USEFUL_VALIDATED" if passed else "REFUTED" if int(closure["n_collision_pairs"]) else "NONPROMOTED_LOW_COVERAGE_OR_FIDELITY",
        "n_validation_systems": len(rows),
        "n_fibers": closure["n_fibers"],
        "n_nontrivial_fibers": closure["n_nontrivial_fibers"],
        "nontrivial_coverage": closure["nontrivial_coverage"],
        "n_collision_fibers": closure["n_collision_fibers"],
        "n_collision_pairs": closure["n_collision_pairs"],
        "collision_examples": closure["collision_examples"],
        "within_fiber_distortion_fraction": fidelity["within_fiber_distortion_fraction"],
        "rich_variance_explained_by_fibers": fidelity["rich_variance_explained_by_fibers"],
        "validation_rule": rule,
        "scope_note": candidate["scope_note"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CO-U CONTENT-TYPED VALIDATION 12 SCORE]")
    print(json.dumps({
        "candidate_name": result["candidate_name"],
        "closure": result["closure"],
        "n_collision_fibers": result["n_collision_fibers"],
        "n_collision_pairs": result["n_collision_pairs"],
        "n_fibers": result["n_fibers"],
        "n_nontrivial_fibers": result["n_nontrivial_fibers"],
        "n_validation_systems": result["n_validation_systems"],
        "nontrivial_coverage": result["nontrivial_coverage"],
        "promotion_status": result["promotion_status"],
        "within_fiber_distortion_fraction": result["within_fiber_distortion_fraction"],
    }, indent=2, sort_keys=True))
    print("[RESULT]", out)


if __name__ == "__main__":
    main()
