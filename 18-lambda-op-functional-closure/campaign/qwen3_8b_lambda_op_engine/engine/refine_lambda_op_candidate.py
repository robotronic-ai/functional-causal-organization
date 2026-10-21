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
        for op in OPS for address in ADDRESSES
    )


def collision_metrics(descriptors: list[tuple[int, ...]], targets: list[str]) -> dict[str, int]:
    fibers: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for desc, tgt in zip(descriptors, targets):
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
            total_pairs = len(tgts) * (len(tgts) - 1) // 2
            same_pairs = sum(n * (n - 1) // 2 for n in counts.values())
            collision_pairs += total_pairs - same_pairs
    return {
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": nontrivial,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "n_collision_fibers": collision_fibers,
        "n_collision_pairs": collision_pairs,
    }


def load_panel(path: Path) -> tuple[list[dict[str, Any]], str]:
    panel = json.loads(path.read_text())
    if panel.get("n_skipped") != 0:
        raise ValueError(f"technical skips in {path}")
    rows = list(panel.get("rows", []))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows, str(panel.get("stage", "unknown"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    panel_paths = [Path(x) for x in args.panel]
    panel_rows: list[list[dict[str, Any]]] = []
    panel_meta = []
    rows: list[dict[str, Any]] = []
    panel_ids: list[int] = []
    for idx, path in enumerate(panel_paths):
        part, stage = load_panel(path)
        panel_rows.append(part)
        rows.extend(part)
        panel_ids.extend([idx] * len(part))
        panel_meta.append({"path": str(path), "sha256": sha256(path), "stage": stage, "n": len(part)})

    vectors = [feature_values(row) for row in rows]
    targets = [stable(target(row)) for row in rows]
    all_names = all_feature_names()
    substitute = [n for n in all_names if n.startswith("SUBSTITUTE:")]
    substitute_plus_id = substitute + [n for n in all_names if n.startswith("ID:")]
    pools = [
        ("SUBSTITUTE_ONLY", substitute),
        ("SUBSTITUTE_PLUS_ID", substitute_plus_id),
        ("ALL_INTERNAL_FEATURES", all_names),
    ]
    bin_grid = (2, 3, 4, 5, 6, 8, 12, 16)
    records = []

    for pool_name, names in pools:
        for bins in bin_grid:
            cuts = {name: quantile_cutpoints([v[name] for v in vectors], bins) for name in names}
            encoded = {
                name: [sum(float(v[name]) > c for c in cuts[name]) for v in vectors]
                for name in names
            }
            selected: list[str] = []
            remaining = set(names)
            desc = [tuple() for _ in rows]
            metrics = collision_metrics(desc, targets)
            steps = []

            while remaining and metrics["n_collision_pairs"] > 0 and len(selected) < 12:
                scored = []
                for name in sorted(remaining):
                    trial = [desc[i] + (encoded[name][i],) for i in range(len(rows))]
                    combined = collision_metrics(trial, targets)
                    per_panel = []
                    for pidx in range(len(panel_paths)):
                        ids = [i for i, x in enumerate(panel_ids) if x == pidx]
                        pm = collision_metrics([trial[i] for i in ids], [targets[i] for i in ids])
                        per_panel.append(pm)
                    score = (
                        combined["n_collision_pairs"],
                        combined["n_collision_fibers"],
                        max(x["n_collision_pairs"] for x in per_panel),
                        combined["n_fibers"],
                        name,
                    )
                    scored.append((score, name, trial, combined, per_panel))
                scored.sort(key=lambda x: x[0])
                _, best_name, best_desc, best_metrics, best_per_panel = scored[0]
                if best_metrics["n_collision_pairs"] >= metrics["n_collision_pairs"]:
                    break
                selected.append(best_name)
                remaining.remove(best_name)
                desc = best_desc
                metrics = best_metrics
                steps.append({
                    "feature": best_name,
                    "combined_metrics": best_metrics,
                    "per_panel_metrics": best_per_panel,
                })

            per_panel_final = []
            for pidx in range(len(panel_paths)):
                ids = [i for i, x in enumerate(panel_ids) if x == pidx]
                per_panel_final.append(collision_metrics([desc[i] for i in ids], [targets[i] for i in ids]))
            records.append({
                "pool": pool_name,
                "bins_requested": bins,
                "features": [{"name": name, "cutpoints": cuts[name]} for name in selected],
                "selected_feature_count": len(selected),
                "combined_metrics": metrics,
                "per_panel_metrics": per_panel_final,
                "steps": steps,
            })

    def rank(rec: dict[str, Any]):
        m = rec["combined_metrics"]
        closed = m["n_collision_pairs"] == 0
        nontrivial = m["n_nontrivial_fibers"] > 0
        all_panels_closed = all(x["n_collision_pairs"] == 0 for x in rec["per_panel_metrics"])
        pool_rank = {"SUBSTITUTE_ONLY": 0, "SUBSTITUTE_PLUS_ID": 1, "ALL_INTERNAL_FEATURES": 2}[rec["pool"]]
        return (
            0 if closed and all_panels_closed else 1,
            0 if nontrivial else 1,
            rec["selected_feature_count"],
            m["n_fibers"],
            -m["systems_in_nontrivial_fibers"],
            pool_rank,
            rec["bins_requested"],
        )

    records.sort(key=rank)
    best = records[0]
    m = best["combined_metrics"]
    closed = m["n_collision_pairs"] == 0 and all(x["n_collision_pairs"] == 0 for x in best["per_panel_metrics"])
    if closed and m["n_nontrivial_fibers"] > 0:
        closure_type = "NONTRIVIAL"
    elif closed:
        closure_type = "TRIVIAL_INJECTIVE"
    else:
        closure_type = "NONE"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_path = out_dir / "REFINEMENT_SEARCH.json"
    best_path = out_dir / "REFINED_BEST_CANDIDATE.json"
    search_payload = {
        "schema": "co-qwen3-8b-lambda-op-refinement-search-v1",
        "status": "TARGET_INFORMED_EXPLORATORY_REFINEMENT",
        "source_panels": panel_meta,
        "n_systems_combined": len(rows),
        "search_strategy": "Substitute-first greedy quantized search over pooled exploratory and validation-01 systems.",
        "feature_pools": [p[0] for p in pools],
        "bin_grid": list(bin_grid),
        "candidates": records,
    }
    search_path.write_text(json.dumps(search_payload, indent=2, sort_keys=True) + "\n")
    best_payload = {
        "schema": "co-qwen3-8b-lambda-op-refined-candidate-v1",
        "status": "TARGET_INFORMED_REFINED_CANDIDATE",
        "candidate_name": "LAMBDA_OP_REFINED_SUBSTITUTE_FIRST",
        "selection_used_behavioral_target": True,
        "source_panel_sha256": [x["sha256"] for x in panel_meta],
        "must_be_validated_on_fresh_data": True,
        "feature_pool": best["pool"],
        "bins_requested": best["bins_requested"],
        "features": best["features"],
        "selected_feature_count": best["selected_feature_count"],
        "exploratory_closure": "PASS" if closed else "NOT_FOUND",
        "exploratory_closure_type": closure_type,
        "combined_metrics": m,
        "per_panel_metrics": best["per_panel_metrics"],
    }
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n")

    print("[LAMBDA-OP REFINEMENT COMPLETE]")
    print("[SOURCE SYSTEMS]", len(rows))
    print("[POOL]", best_payload["feature_pool"])
    print("[FEATURES]", best_payload["selected_feature_count"])
    print("[BINS]", best_payload["bins_requested"])
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
