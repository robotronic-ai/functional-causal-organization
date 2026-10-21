#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cross_target_channel import collision_examples, collision_metrics, descriptor, load_rows, stable, target_vector


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    candidate_path = Path(args.candidate)
    rows = load_rows([panel_path])
    candidate = json.loads(candidate_path.read_text())
    family = str(candidate["family"])
    descs = [descriptor(r, family) for r in rows]
    targets = [stable(target_vector(r)) for r in rows]
    metrics = collision_metrics(descs, targets)
    closed = int(metrics["n_collision_pairs"]) == 0
    coverage = float(metrics["nontrivial_coverage"])
    promotion = "USEFUL_VALIDATED" if closed and coverage >= 0.20 else ("NONPROMOTED_LOW_COVERAGE" if closed else "REFUTED")

    result = {
        "schema": "co-qwen3-8b-cross-target-active-channel-validation-score-v1",
        "candidate_name": candidate["candidate_name"],
        "candidate_sha256": sha256(candidate_path),
        "validation_panel_sha256": sha256(panel_path),
        "n_validation_systems": len(rows),
        **metrics,
        "closure": "PASS" if closed else "REFUTED",
        "promotion_status": promotion,
        "useful_candidate_rule": "zero collision pairs and nontrivial coverage >= 0.20",
        "collision_examples": collision_examples(rows, descs, targets),
        "interpretation": "Fresh validation of a frozen target-informed family selected on a disjoint exploratory panel. No descriptor-family tuning is performed on validation data.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CO-U CROSS-TARGET VALIDATION SCORE]")
    print(json.dumps({
        "candidate_name": result["candidate_name"],
        "closure": result["closure"],
        "promotion_status": result["promotion_status"],
        "n_validation_systems": result["n_validation_systems"],
        "n_fibers": result["n_fibers"],
        "n_nontrivial_fibers": result["n_nontrivial_fibers"],
        "nontrivial_coverage": result["nontrivial_coverage"],
        "n_collision_fibers": result["n_collision_fibers"],
        "n_collision_pairs": result["n_collision_pairs"],
    }, indent=2, sort_keys=True))
    print("[RESULT]", out)


if __name__ == "__main__":
    main()
