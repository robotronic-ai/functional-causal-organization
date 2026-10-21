#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from co_u_vq import collision_metrics, encode_rows, stable, target_vector


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    candidate_path = Path(args.candidate)
    panel = json.loads(panel_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if int(panel.get("n_skipped", 0)) != 0:
        raise ValueError("validation panel contains technical skips")
    rows = list(panel.get("rows", []))
    if not rows:
        raise ValueError("empty validation panel")

    desc = encode_rows(rows, candidate)
    targets = [stable(target_vector(row)) for row in rows]
    metrics = collision_metrics(desc, targets)

    fibers: dict[tuple[int, ...], list[int]] = defaultdict(list)
    target_map: dict[tuple[int, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row, d, t in zip(rows, desc, targets):
        unit = int(row.get("unit", -1))
        fibers[d].append(unit)
        target_map[d][t].append(unit)
    examples = []
    for d in sorted(fibers):
        groups = target_map[d]
        if len(groups) <= 1:
            continue
        examples.append({
            "descriptor_codes": list(d),
            "units": fibers[d],
            "distinct_target_count": len(groups),
            "target_groups": [
                {"target": json.loads(t), "units": units}
                for t, units in sorted(groups.items())
            ],
        })

    if int(metrics["n_collision_pairs"]) == 0:
        closure = "PASS"
        closure_type = "NONTRIVIAL" if int(metrics["n_nontrivial_fibers"]) > 0 else "TRIVIAL_INJECTIVE"
    else:
        closure = "REFUTED"
        closure_type = "NONE"

    result = {
        "schema": "co-qwen3-8b-operation-indexed-validation-score-v1",
        "model": "Qwen3-8B",
        "candidate_name": candidate["candidate_name"],
        "candidate_family": candidate["family"],
        "candidate_sha256": sha256(candidate_path),
        "validation_panel_sha256": sha256(panel_path),
        "closure": closure,
        "closure_type": closure_type,
        "n_validation_systems": len(rows),
        **metrics,
        "collision_examples": examples,
        "interpretation": "Fresh validation of a target-informed candidate whose vector-quantizer codebooks are target-blind and frozen before this panel.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CO-U VALIDATION 03 SCORE]")
    summary = {k: result[k] for k in (
        "candidate_name", "candidate_family", "closure", "closure_type",
        "n_validation_systems", "n_fibers", "n_nontrivial_fibers",
        "systems_in_nontrivial_fibers", "nontrivial_coverage",
        "n_collision_fibers", "n_collision_pairs",
    )}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[RESULT]", out)


if __name__ == "__main__":
    main()
