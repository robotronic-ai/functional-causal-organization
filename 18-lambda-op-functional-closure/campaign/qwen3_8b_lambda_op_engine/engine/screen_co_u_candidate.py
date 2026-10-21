#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from co_u_vq import (
    OPS,
    NON_ID_OPS,
    collision_metrics,
    deterministic_kmeans,
    encode_rows,
    fit_standardizer,
    slice_vector,
    stable,
    target_vector,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if int(data.get("n_skipped", 0)) != 0:
        raise ValueError(f"technical skips in {path}")
    rows = list(data.get("rows", []))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def fit_candidate(rows: list[dict[str, Any]], family: str, operations: list[str], k: int) -> dict[str, Any]:
    slice_models: dict[str, Any] = {}
    for op in operations:
        X = np.stack([slice_vector(row, op) for row in rows], axis=0)
        mu, sd = fit_standardizer(X)
        Xn = (X - mu) / sd
        centers, _ = deterministic_kmeans(Xn, k=k)
        slice_models[op] = {
            "k": int(centers.shape[0]),
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "centers": centers.tolist(),
        }
    return {
        "schema": "co-qwen3-8b-operation-indexed-vq-candidate-v1",
        "candidate_name": f"{family}_K{k}",
        "family": family,
        "operations": operations,
        "slice_dimension": 15,
        "slice_representation": "H1/H2/H3 ordered 5D log internal vectors; non-ID slices are log effects relative to ID at the same address.",
        "quantizer": "deterministic standardized k-means fitted without behavioral targets",
        "k_requested": int(k),
        "slice_models": slice_models,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    panel_paths = [Path(x) for x in args.panel]
    panel_rows = [load_panel(p) for p in panel_paths]
    all_rows = [row for part in panel_rows for row in part]
    all_targets = [stable(target_vector(row)) for row in all_rows]

    panel_meta = []
    offset = 0
    panel_ranges = []
    for p, rows in zip(panel_paths, panel_rows):
        panel_meta.append({"path": str(p), "sha256": sha256(p), "n": len(rows)})
        panel_ranges.append((offset, offset + len(rows)))
        offset += len(rows)

    families = [
        ("D0_ID_SLICE", ["ID"]),
        ("D1_SUBSTITUTE_SLICE", ["SUBSTITUTE"]),
        ("D2_OPERATION_INDEXED_TRANSFORMS", list(NON_ID_OPS)),
        ("D3_OPERATION_INDEXED_WITH_ID", list(OPS)),
    ]
    k_grid = (2, 3, 4, 5, 6, 8, 10, 12, 16)
    records = []

    for family, operations in families:
        for k in k_grid:
            candidate = fit_candidate(all_rows, family, operations, k)
            descriptors = encode_rows(all_rows, candidate)
            combined = collision_metrics(descriptors, all_targets)
            per_panel = []
            for (lo, hi), rows in zip(panel_ranges, panel_rows):
                per_panel.append(collision_metrics(descriptors[lo:hi], all_targets[lo:hi]))
            record = {
                "candidate": candidate,
                "combined_metrics": combined,
                "per_panel_metrics": per_panel,
                "all_panels_closed": all(int(m["n_collision_pairs"]) == 0 for m in per_panel),
            }
            records.append(record)

    def rank(rec: dict[str, Any]):
        c = rec["combined_metrics"]
        p = rec["per_panel_metrics"]
        closed = int(c["n_collision_pairs"]) == 0 and rec["all_panels_closed"]
        coverage = float(c["nontrivial_coverage"])
        useful_nontrivial = coverage >= 0.20
        return (
            0 if closed and useful_nontrivial else 1 if closed else 2,
            max(int(x["n_collision_pairs"]) for x in p),
            int(c["n_collision_pairs"]),
            0 if useful_nontrivial else 1,
            -coverage,
            len(rec["candidate"]["operations"]),
            int(c["n_fibers"]),
            int(rec["candidate"]["k_requested"]),
        )

    records.sort(key=rank)
    best = records[0]
    best_candidate = dict(best["candidate"])
    best_candidate.update({
        "status": "TARGET_INFORMED_OPERATION_INDEXED_EXPLORATORY_CANDIDATE",
        "selection_used_behavioral_target": True,
        "fit_of_codebooks_used_behavioral_target": False,
        "source_panel_sha256": [x["sha256"] for x in panel_meta],
        "source_systems": len(all_rows),
        "exploratory_metrics": best["combined_metrics"],
        "per_panel_metrics": best["per_panel_metrics"],
        "must_be_validated_on_fresh_data": True,
        "scope_note": "This screen tests an operation-indexed internal-response kernel proxy. It does not yet identify Gamma_u, Delta_u, or R_u.",
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    search_path = out / "CO_U_SCREEN_RESULT.json"
    candidate_path = out / "CO_U_BEST_CANDIDATE.json"
    search_payload = {
        "schema": "co-qwen3-8b-operation-indexed-screen-v1",
        "status": "TARGET_INFORMED_EXPLORATORY_SCREEN",
        "source_panels": panel_meta,
        "n_systems": len(all_rows),
        "families": [name for name, _ in families],
        "k_grid": list(k_grid),
        "ranking_rule": "Prefer closure on every opened panel with >=20% nontrivial coverage; otherwise minimize worst-panel and combined collision pairs while favoring compression.",
        "candidates": records,
    }
    search_path.write_text(json.dumps(search_payload, indent=2, sort_keys=True) + "\n")
    candidate_path.write_text(json.dumps(best_candidate, indent=2, sort_keys=True) + "\n")

    m = best_candidate["exploratory_metrics"]
    print("[CO-U OPERATION-INDEXED SCREEN COMPLETE]")
    print("[SOURCE SYSTEMS]", len(all_rows))
    print("[CANDIDATE]", best_candidate["candidate_name"])
    print("[FAMILY]", best_candidate["family"])
    print("[OPERATIONS]", ",".join(best_candidate["operations"]))
    print("[K]", best_candidate["k_requested"])
    print("[FIBERS]", m["n_fibers"])
    print("[NONTRIVIAL FIBERS]", m["n_nontrivial_fibers"])
    print("[NONTRIVIAL COVERAGE]", f"{float(m['nontrivial_coverage']):.4f}")
    print("[COLLISION FIBERS]", m["n_collision_fibers"])
    print("[COLLISION PAIRS]", m["n_collision_pairs"])
    print("[BEST]", candidate_path)


if __name__ == "__main__":
    main()
