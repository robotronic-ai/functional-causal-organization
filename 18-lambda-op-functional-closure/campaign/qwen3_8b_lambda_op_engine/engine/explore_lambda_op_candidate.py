#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from candidate_descriptor import all_feature_names, feature_values, quantile_cutpoints

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def target(row: dict[str, Any]) -> tuple[str, ...]:
    lookup = {(c["operation"], c["address"]): c for c in row["cells"]}
    return tuple(
        str(lookup[(op, address)]["behavior"]["outcome"])
        for op in OPS
        for address in ADDRESSES
    )


def collision_metrics(encoded_rows: list[tuple[Any, str]]) -> dict[str, int]:
    fibers: dict[Any, list[str]] = defaultdict(list)
    for desc, tgt in encoded_rows:
        fibers[desc].append(tgt)
    collision_fibers = 0
    collision_pairs = 0
    nontrivial = 0
    systems_nontrivial = 0
    for tgts in fibers.values():
        if len(tgts) > 1:
            nontrivial += 1
            systems_nontrivial += len(tgts)
        counts: dict[str, int] = defaultdict(int)
        for t in tgts:
            counts[t] += 1
        if len(counts) > 1:
            collision_fibers += 1
            vals = list(counts.values())
            total_pairs = len(tgts) * (len(tgts) - 1) // 2
            same_pairs = sum(n * (n - 1) // 2 for n in vals)
            collision_pairs += total_pairs - same_pairs
    return {
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": nontrivial,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "n_collision_fibers": collision_fibers,
        "n_collision_pairs": collision_pairs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    holdout_path = Path(args.holdout)
    panel = json.loads(holdout_path.read_text())
    if panel.get("stage") != "holdout":
        raise ValueError("input is not a holdout panel")
    if panel.get("n_skipped") != 0:
        raise ValueError("holdout contains technical skips")
    rows = panel["rows"]
    if len(rows) < 2:
        raise ValueError("need at least two rows")

    names = all_feature_names()
    vectors = [feature_values(row) for row in rows]
    targets = [stable(target(row)) for row in rows]

    # Candidate discovery is deliberately target-informed. This phase is exploratory only.
    bin_grid = (2, 3, 4, 5, 6, 8, 12, 16)
    candidates = []

    for bins in bin_grid:
        cuts = {
            name: quantile_cutpoints([v[name] for v in vectors], bins)
            for name in names
        }
        encoded_by_feature = {
            name: [sum(float(v[name]) > c for c in cuts[name]) for v in vectors]
            for name in names
        }

        selected: list[str] = []
        remaining = set(names)
        current_desc = [tuple() for _ in rows]
        current_metrics = collision_metrics(list(zip(current_desc, targets)))

        steps = []
        while remaining and current_metrics["n_collision_pairs"] > 0 and len(selected) < 24:
            scored = []
            for name in sorted(remaining):
                trial = [current_desc[i] + (encoded_by_feature[name][i],) for i in range(len(rows))]
                m = collision_metrics(list(zip(trial, targets)))
                # Primary objective: remove target-discordant pairs. Secondary: retain compression.
                score = (
                    m["n_collision_pairs"],
                    m["n_collision_fibers"],
                    m["n_fibers"],
                    name,
                )
                scored.append((score, name, trial, m))
            scored.sort(key=lambda x: x[0])
            _, best_name, best_desc, best_metrics = scored[0]
            if best_metrics["n_collision_pairs"] >= current_metrics["n_collision_pairs"]:
                break
            selected.append(best_name)
            remaining.remove(best_name)
            current_desc = best_desc
            current_metrics = best_metrics
            steps.append({"feature": best_name, "metrics": best_metrics})

        specs = [{"name": name, "cutpoints": cuts[name]} for name in selected]
        record = {
            "bins_requested": bins,
            "selected_feature_count": len(selected),
            "features": specs,
            "metrics": current_metrics,
            "steps": steps,
        }
        candidates.append(record)

    def rank(c):
        m = c["metrics"]
        closed = m["n_collision_pairs"] == 0
        nontrivial = m["n_nontrivial_fibers"] > 0
        # Prefer exact closure, then nontrivial closure, then fewer features, then stronger compression.
        return (
            0 if closed else 1,
            0 if (closed and nontrivial) else 1,
            c["selected_feature_count"],
            m["n_fibers"],
            -m["systems_in_nontrivial_fibers"],
            c["bins_requested"],
        )

    candidates.sort(key=rank)
    best = candidates[0]
    m = best["metrics"]
    closed = m["n_collision_pairs"] == 0
    closure_type = "NONTRIVIAL" if closed and m["n_nontrivial_fibers"] > 0 else ("TRIVIAL_INJECTIVE" if closed else "NONE")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "EXPLORATORY_CANDIDATE_SEARCH.json"
    best_path = output_dir / "EXPLORATORY_BEST_CANDIDATE.json"

    payload = {
        "schema": "co-qwen3-8b-lambda-op-exploratory-search-v1",
        "status": "TARGET_INFORMED_EXPLORATORY",
        "source_holdout": str(holdout_path),
        "source_holdout_sha256": sha256(holdout_path),
        "n_systems": len(rows),
        "candidate_family": "greedy quantized internal-feature descriptors",
        "feature_pool_size": len(names),
        "bin_grid": list(bin_grid),
        "candidates": candidates,
    }
    all_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    best_payload = {
        "schema": "co-qwen3-8b-lambda-op-exploratory-candidate-v1",
        "status": "TARGET_INFORMED_EXPLORATORY_CANDIDATE",
        "candidate_name": "LAMBDA_OP_EXPLORE_GREEDY_QUANTIZED",
        "source_holdout_sha256": sha256(holdout_path),
        "selection_used_behavioral_target": True,
        "must_be_validated_on_fresh_data": True,
        "encoding": "Each selected scalar is assigned an integer bin using the frozen cutpoints below.",
        "features": best["features"],
        "selected_feature_count": best["selected_feature_count"],
        "exploratory_metrics": m,
        "exploratory_closure": "PASS" if closed else "NOT_FOUND",
        "exploratory_closure_type": closure_type,
        "bins_requested": best["bins_requested"],
    }
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n")

    print("[LAMBDA-OP EXPLORATION COMPLETE]")
    print("[SOURCE SYSTEMS]", len(rows))
    print("[CANDIDATE]", best_payload["candidate_name"])
    print("[FEATURES]", best_payload["selected_feature_count"])
    print("[EXPLORATORY CLOSURE]", best_payload["exploratory_closure"])
    print("[TYPE]", best_payload["exploratory_closure_type"])
    print("[FIBERS]", m["n_fibers"])
    print("[NONTRIVIAL FIBERS]", m["n_nontrivial_fibers"])
    print("[SYSTEMS IN NONTRIVIAL FIBERS]", m["systems_in_nontrivial_fibers"])
    print("[COLLISION FIBERS]", m["n_collision_fibers"])
    print("[COLLISION PAIRS]", m["n_collision_pairs"])
    print("[BEST]", best_path)


if __name__ == "__main__":
    main()
