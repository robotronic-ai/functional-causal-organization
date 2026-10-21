#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from co_u_causal_kernel_analysis import (
    ADDRESSES,
    OPS,
    assign_clusters,
    cell_slice_vector,
    collision_metrics,
    deterministic_kmeans,
    fit_standardizer,
    operation_kernel_vector,
    stable,
    target_vector,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_panel(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if int(data.get("n_skipped", 0)) != 0:
        raise ValueError("active kernel panel has technical skips")
    rows = list(data.get("rows", []))
    if not rows:
        raise ValueError("active kernel panel has no rows")
    return rows


def matrix_for_slice(rows: list[dict[str, Any]], spec: tuple[str, ...]) -> np.ndarray:
    if len(spec) == 1:
        op = spec[0]
        return np.stack([operation_kernel_vector(row, op) for row in rows], axis=0)
    if len(spec) == 2:
        op, address = spec
        return np.stack([cell_slice_vector(row, op, address) for row in rows], axis=0)
    raise ValueError(spec)


def fit_family(rows: list[dict[str, Any]], name: str, slices: list[tuple[str, ...]], k: int) -> dict[str, Any]:
    models = []
    for spec in slices:
        X = matrix_for_slice(rows, spec)
        mu, sd = fit_standardizer(X)
        centers, _ = deterministic_kmeans((X - mu) / sd, k=k)
        models.append({
            "slice": list(spec),
            "k": int(centers.shape[0]),
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "centers": centers.tolist(),
        })
    return {
        "schema": "co-qwen3-8b-active-causal-kernel-candidate-v1",
        "candidate_name": f"{name}_K{k}",
        "family": name,
        "k_requested": int(k),
        "slices": [list(x) for x in slices],
        "slice_models": models,
        "quantizer": "deterministic standardized k-means fitted without behavioral targets",
        "representation": "Active target-blind causal response to ZERO_SUFFIX and HEAD_ROTATE at layers 21-23 across horizons 1-2.",
    }


def encode(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    codes = []
    for model in candidate["slice_models"]:
        spec = tuple(model["slice"])
        X = matrix_for_slice(rows, spec)
        labels = assign_clusters(
            X,
            np.asarray(model["mu"], dtype=np.float64),
            np.asarray(model["sd"], dtype=np.float64),
            np.asarray(model["centers"], dtype=np.float64),
        )
        codes.append(labels)
    return [tuple(int(code[i]) for code in codes) for i in range(len(rows))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    rows = load_panel(panel)
    targets = [stable(target_vector(row)) for row in rows]

    families = [
        ("K0_SUBSTITUTE_JOINT", [("SUBSTITUTE",)]),
        ("K1_SUBSTITUTE_PER_ADDRESS", [("SUBSTITUTE", a) for a in ADDRESSES]),
        ("K2_NON_ID_PER_OPERATION", [(op,) for op in ("KEEP", "REVOKE", "SUBSTITUTE")]),
        ("K3_ALL_PER_OPERATION", [(op,) for op in OPS]),
    ]
    k_grid = (2, 3, 4, 5, 6, 8, 10, 12)
    records = []

    for name, slices in families:
        for k in k_grid:
            candidate = fit_family(rows, name, slices, k)
            descriptors = encode(rows, candidate)
            metrics = collision_metrics(descriptors, targets)
            records.append({"candidate": candidate, "metrics": metrics})

    def rank(rec: dict[str, Any]):
        m = rec["metrics"]
        closed = int(m["n_collision_pairs"]) == 0
        coverage = float(m["nontrivial_coverage"])
        useful = closed and coverage >= 0.20
        return (
            0 if useful else 1 if closed else 2,
            int(m["n_collision_pairs"]),
            0 if coverage >= 0.20 else 1,
            -coverage,
            int(m["n_fibers"]),
            len(rec["candidate"]["slices"]),
            int(rec["candidate"]["k_requested"]),
        )

    records.sort(key=rank)
    best = records[0]
    m = best["metrics"]
    useful = int(m["n_collision_pairs"]) == 0 and float(m["nontrivial_coverage"]) >= 0.20
    best_candidate = dict(best["candidate"])
    best_candidate.update({
        "status": "USEFUL_EXPLORATORY_CANDIDATE" if useful else "NO_USEFUL_CANDIDATE",
        "selection_used_behavioral_target": True,
        "fit_of_codebooks_used_behavioral_target": False,
        "source_panel_sha256": sha256(panel),
        "source_systems": len(rows),
        "exploratory_metrics": m,
        "useful_candidate_rule": "zero collision pairs and nontrivial coverage >= 0.20",
        "must_be_validated_on_fresh_data": bool(useful),
        "scope_note": "This is an active causal-kernel proxy, not yet a direct measurement of Gamma_u, Delta_u, or recurrence R_u.",
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    search_path = out / "CO_U_CAUSAL_KERNEL_SCREEN_RESULT.json"
    candidate_path = out / "CO_U_CAUSAL_KERNEL_BEST_CANDIDATE.json"
    search_path.write_text(json.dumps({
        "schema": "co-qwen3-8b-active-causal-kernel-screen-v1",
        "status": "TARGET_INFORMED_EXPLORATORY_SCREEN",
        "source_panel": str(panel),
        "source_panel_sha256": sha256(panel),
        "n_systems": len(rows),
        "families": [name for name, _ in families],
        "k_grid": list(k_grid),
        "useful_candidate_rule": "zero collision pairs and nontrivial coverage >= 0.20",
        "candidates": records,
    }, indent=2, sort_keys=True) + "\n")
    candidate_path.write_text(json.dumps(best_candidate, indent=2, sort_keys=True) + "\n")

    print("[CO-U ACTIVE CAUSAL KERNEL SCREEN COMPLETE]")
    print("[SOURCE SYSTEMS]", len(rows))
    print("[STATUS]", best_candidate["status"])
    print("[CANDIDATE]", best_candidate["candidate_name"])
    print("[FAMILY]", best_candidate["family"])
    print("[K]", best_candidate["k_requested"])
    print("[FIBERS]", m["n_fibers"])
    print("[NONTRIVIAL FIBERS]", m["n_nontrivial_fibers"])
    print("[NONTRIVIAL COVERAGE]", f"{float(m['nontrivial_coverage']):.4f}")
    print("[COLLISION FIBERS]", m["n_collision_fibers"])
    print("[COLLISION PAIRS]", m["n_collision_pairs"])
    print("[BEST]", candidate_path)


if __name__ == "__main__":
    main()
