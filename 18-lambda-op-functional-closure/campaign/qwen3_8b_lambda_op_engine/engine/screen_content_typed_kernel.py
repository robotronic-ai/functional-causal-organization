#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from content_typed_kernel import (
    closure_metrics,
    encode_matrix,
    feature_names,
    feature_vector,
    fidelity_metrics,
    quantile_edges,
    stable,
    target_vector,
)

BINS = (2, 3, 4, 6)
MAX_FEATURES = 8
COVERAGE_MIN = 0.20
DISTORTION_MAX = 0.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def family_indices(names: list[str]) -> dict[str, list[int]]:
    return {
        "SUBSTITUTE_ONLY": [i for i, n in enumerate(names) if n.startswith("SUBSTITUTE:")],
        "REVOKE_SUBSTITUTE": list(range(len(names))),
        "ZERO_SUFFIX_ONLY": [i for i, n in enumerate(names) if ":ZERO_SUFFIX:" in n],
        "SUBSTITUTE_ZERO_SUFFIX": [i for i, n in enumerate(names) if n.startswith("SUBSTITUTE:") and ":ZERO_SUFFIX:" in n],
    }


def score_specs(Z: np.ndarray, specs: list[dict[str, Any]], targets: list[str], units: list[int]) -> dict[str, Any]:
    labels = encode_matrix(Z, specs)
    return {
        "labels": labels,
        "closure": closure_metrics(labels, targets, units),
        "fidelity": fidelity_metrics(Z, labels),
    }


def search_family(Z: np.ndarray, indices: list[int], targets: list[str], units: list[int]) -> dict[str, Any]:
    specs: list[dict[str, Any]] = []
    remaining = set(indices)
    trace: list[dict[str, Any]] = []
    current = score_specs(Z, specs, targets, units)

    for step in range(1, MAX_FEATURES + 1):
        best = None
        for fi in sorted(remaining):
            for bins in BINS:
                edges = quantile_edges(Z[:, fi], bins)
                if not edges:
                    continue
                trial_specs = specs + [{"feature_index": fi, "bins_requested": bins, "edges": edges}]
                sc = score_specs(Z, trial_specs, targets, units)
                c = sc["closure"]
                f = sc["fidelity"]
                key = (
                    int(c["n_collision_pairs"]),
                    -float(c["nontrivial_coverage"]),
                    float(f["within_fiber_distortion_fraction"]),
                    int(c["n_fibers"]),
                    fi,
                    bins,
                )
                if best is None or key < best[0]:
                    best = (key, fi, bins, edges, sc)
        if best is None:
            break
        _, fi, bins, edges, current = best
        specs.append({"feature_index": fi, "bins_requested": bins, "edges": edges})
        remaining.remove(fi)
        trace.append({
            "step": step,
            "feature_index": fi,
            "bins_requested": bins,
            "edges": edges,
            "closure": current["closure"],
            "fidelity": current["fidelity"],
        })
        c = current["closure"]
        f = current["fidelity"]
        if (
            int(c["n_collision_pairs"]) == 0
            and float(c["nontrivial_coverage"]) >= COVERAGE_MIN
            and float(f["within_fiber_distortion_fraction"]) <= DISTORTION_MAX
        ):
            break

    c = current["closure"]
    f = current["fidelity"]
    useful = (
        int(c["n_collision_pairs"]) == 0
        and float(c["nontrivial_coverage"]) >= COVERAGE_MIN
        and float(f["within_fiber_distortion_fraction"]) <= DISTORTION_MAX
    )
    return {"useful": useful, "specs": specs, "trace": trace, "closure": c, "fidelity": f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    panel_path = Path(args.panel)
    panel = json.loads(panel_path.read_text())
    if int(panel.get("n_skipped", 0)) != 0:
        raise ValueError("panel contains technical skips")
    rows = list(panel.get("rows", []))
    if not rows:
        raise ValueError("panel has no rows")

    X = np.stack([feature_vector(r) for r in rows])
    names = feature_names()
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    targets = [stable(target_vector(r)) for r in rows]
    units = [int(r["unit"]) for r in rows]

    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for family, idxs in family_indices(names).items():
        res = search_family(Z, idxs, targets, units)
        rec = {
            "family": family,
            "useful": res["useful"],
            "selected_feature_count": len(res["specs"]),
            "selected_features": [
                {**s, "feature_name": names[int(s["feature_index"])]}
                for s in res["specs"]
            ],
            "closure": res["closure"],
            "fidelity": res["fidelity"],
            "trace": res["trace"],
        }
        records.append(rec)
        if res["useful"]:
            candidates.append(rec)

    if candidates:
        candidates.sort(key=lambda r: (
            -float(r["closure"]["nontrivial_coverage"]),
            float(r["fidelity"]["within_fiber_distortion_fraction"]),
            int(r["selected_feature_count"]),
            str(r["family"]),
        ))
        best = candidates[0]
        status = "USEFUL_EXPLORATORY_CANDIDATE"
        candidate = {
            "schema": "co-qwen3-8b-content-typed-candidate-v1",
            "candidate_name": f"CONTENT_TYPED_{best['family']}_{best['selected_feature_count']}F",
            "status": status,
            "family": best["family"],
            "source_panel_sha256": sha256(panel_path),
            "source_systems": len(rows),
            "feature_names": names,
            "standardizer": {"mu": mu.tolist(), "sd": sd.tolist()},
            "selected_quantized_features": [
                {k: v for k, v in s.items() if k in ("feature_index", "bins_requested", "edges")}
                for s in best["selected_features"]
            ],
            "construction_closure": best["closure"],
            "construction_fidelity": best["fidelity"],
            "descriptor_uses_baseline_outcome": False,
            "descriptor_uses_baseline_absolute_logprob": False,
            "feature_edges_fit_without_behavior": True,
            "feature_selection_used_behavioral_target": True,
            "must_be_validated_on_fresh_data": True,
            "promotion_rule": {
                "n_collision_pairs": 0,
                "nontrivial_coverage_min": COVERAGE_MIN,
                "within_fiber_distortion_fraction_max": DISTORTION_MAX,
            },
            "scope_note": "Exploratory typed content-effect refinement of Lambda within the current operation-indexed organizational proxy. It is not a scalar validation of full C_O.",
        }
    else:
        status = "NO_USEFUL_CANDIDATE"
        candidate = None

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "co-qwen3-8b-content-typed-screen-v1",
        "status": status,
        "source_systems": len(rows),
        "source_panel_sha256": sha256(panel_path),
        "feature_dimension": int(X.shape[1]),
        "records": records,
        "candidate_name": candidate["candidate_name"] if candidate else None,
        "selection_is_exploratory_and_target_informed": True,
        "interpretation": "The measured features are signed causal content effects under internal interventions. Quantile edges are target-blind; greedy feature selection is exploratory and may use the behavioral closure target.",
    }
    result_path = out / "CONTENT_TYPED_12_SCREEN_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if candidate:
        cand_path = out / "CONTENT_TYPED_12_CANDIDATE.json"
        cand_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")

    print("[CO-U CONTENT-TYPED SCREEN COMPLETE]")
    print("[SOURCE SYSTEMS]", len(rows))
    print("[FEATURE DIMENSION]", X.shape[1])
    print("[STATUS]", status)
    if candidate:
        c = candidate["construction_closure"]
        f = candidate["construction_fidelity"]
        print("[CANDIDATE]", candidate["candidate_name"])
        print("[FAMILY]", candidate["family"])
        print("[SELECTED FEATURES]", len(candidate["selected_quantized_features"]))
        print("[FIBERS]", c["n_fibers"])
        print("[NONTRIVIAL COVERAGE]", f"{float(c['nontrivial_coverage']):.4f}")
        print("[WITHIN-FIBER DISTORTION]", f"{float(f['within_fiber_distortion_fraction']):.4f}")
        print("[COLLISION PAIRS]", c["n_collision_pairs"])
        print("[CANDIDATE FILE]", out / "CONTENT_TYPED_12_CANDIDATE.json")
    print("[RESULT]", result_path)


if __name__ == "__main__":
    main()
