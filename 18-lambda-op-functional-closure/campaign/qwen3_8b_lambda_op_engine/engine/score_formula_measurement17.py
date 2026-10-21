#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from co_base_operator_profile import partition_quality
from formula_measurement17 import ADDRESSES, OPS, component_keys, family_components, family_label


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _target(row: dict[str, Any]) -> str:
    cells = {(c["operation"], c["source_address"]): c for c in row["cells"]}
    vec = [str(cells[(op, a)]["behavior"]["outcome"]) for op in OPS for a in ADDRESSES]
    return json.dumps(vec, separators=(",", ":"))


def _collision_metrics(labels: list[str], targets: list[str]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    collision_pairs = 0
    collision_fibers = 0
    n_nontrivial = 0
    systems_nontrivial = 0
    for idxs in groups.values():
        if len(idxs) > 1:
            n_nontrivial += 1
            systems_nontrivial += len(idxs)
        counts: dict[str, int] = defaultdict(int)
        for i in idxs:
            counts[targets[i]] += 1
        cp = len(idxs) * (len(idxs) - 1) // 2 - sum(v * (v - 1) // 2 for v in counts.values())
        if cp:
            collision_fibers += 1
            collision_pairs += cp
    n = len(labels)
    return {
        "n_fibers": len(groups),
        "n_collision_fibers": collision_fibers,
        "n_collision_pairs": collision_pairs,
        "n_nontrivial_fibers": n_nontrivial,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "nontrivial_coverage": systems_nontrivial / n if n else 0.0,
        "fiber_fraction": len(groups) / n if n else 0.0,
    }


def _source_rows(panel: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(r["unit"]): r for r in panel["rows"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-panel", required=True)
    ap.add_argument("--source-content-panel", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    measurement_path = Path(args.measurement_panel)
    source_path = Path(args.source_content_panel)
    panel = json.loads(measurement_path.read_text())
    source = json.loads(source_path.read_text())
    if int(panel.get("n_skipped", 0)) != 0 or not panel.get("rows"):
        raise RuntimeError("Formula Measurement 17 panel must be complete and non-empty")
    source_by_unit = _source_rows(source)

    rows = []
    targets = []
    units = []
    key_rows = []
    for row in panel["rows"]:
        su = int(row["source_unit"])
        if su not in source_by_unit:
            raise RuntimeError(f"missing source row {su}")
        target = _target(source_by_unit[su])
        keys = component_keys(row)
        rows.append(row)
        targets.append(target)
        units.append(int(row["unit"]))
        key_rows.append(keys)

    families = family_components()
    screens = {}
    for name, comps in families.items():
        labels = [family_label(keys, comps) for keys in key_rows]
        metrics = _collision_metrics(labels, targets)
        quality = partition_quality(labels, targets, permutations=1000, seed=17017)
        admissible = (
            metrics["n_collision_pairs"] == 0
            and metrics["nontrivial_coverage"] >= 0.20
            and metrics["fiber_fraction"] <= 0.80
            and quality["sufficiency_loss_H_target_given_descriptor_over_H_target"] <= 1e-12
            and quality["ami_exceeds_permutation_95"]
        )
        screens[name] = {
            "components": list(comps),
            "representation_resolution": "fixed 1/16 for normalized continuous coordinates; exact categorical signatures for Lambda and R source partitions",
            "metrics": metrics,
            "partition_quality": quality,
            "admissible_development_candidate": bool(admissible),
            "partition_labels": labels,
        }

    admissible_names = [name for name, rec in screens.items() if rec["admissible_development_candidate"]]
    selected = None
    if admissible_names:
        selected_name = min(
            admissible_names,
            key=lambda name: (
                screens[name]["metrics"]["n_fibers"],
                len(screens[name]["components"]),
                name,
            ),
        )
        selected = {
            "candidate_name": f"FORMULA17_{selected_name.replace('+', '_')}_FIXED_RHO16",
            "family": selected_name,
            "components": screens[selected_name]["components"],
            "selection_rule": "coarsest admissible preregistered family: fewer fibers, then fewer components, then lexical tie-break",
            "representation_resolution": screens[selected_name]["representation_resolution"],
            "measurement_refit_on_holdout_allowed": False,
        }

    pairwise = []
    decoded_targets = [json.loads(x) for x in targets]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if targets[i] == targets[j]:
                continue
            diff = [k for k, (a, b) in enumerate(zip(decoded_targets[i], decoded_targets[j])) if a != b]
            separating = [name for name, rec in screens.items() if rec["partition_labels"][i] != rec["partition_labels"][j]]
            component_sep = [c for c in ("GAMMA", "DELTA", "R", "LAMBDA") if screens[c]["partition_labels"][i] != screens[c]["partition_labels"][j]]
            pairwise.append({
                "unit_i": units[i],
                "unit_j": units[j],
                "differing_target_indices": diff,
                "separated_by_components_at_fixed_rho": component_sep,
                "separated_by_families": separating,
            })

    r_cells = [c for row in rows for c in row["R"]["cells"]]
    r_nontrivial = sum(int(c["n_source_classes"] > 1) for c in r_cells)
    r_control_pass = all(bool(c["negative_temporal_storage_control_pass"]) for c in r_cells)
    r_dup_pass = all(bool(c["label_duplication_invariance"]) for c in r_cells)
    r_status = "MEASURABLE" if r_nontrivial else "TRIVIAL_SOURCE_QUOTIENT_ON_DEVELOPMENT"

    status = "DEVELOPMENT_CANDIDATE_FOUND" if selected else "NO_ADMISSIBLE_FIXED_RHO_PARTITION"
    result = {
        "schema": "co-qwen3-8b-formula-measurement-17-result-v1",
        "status": status,
        "n_systems": len(rows),
        "measurement_panel_sha256": sha256(measurement_path),
        "source_content_panel_sha256": sha256(source_path),
        "generalization_level": "OP",
        "generalization_level_changed": False,
        "raw_measurement_used_behavioral_target": False,
        "target_used_only_for_declared_development_evaluation": True,
        "candidate_family_count": len(families),
        "fixed_resolution": "1/16 normalized continuous coordinates plus exact categorical signatures",
        "family_screens": screens,
        "selected_development_candidate": selected,
        "heterogeneous_pair_attribution": pairwise,
        "R_construct_check": {
            "n_cells": len(r_cells),
            "n_nontrivial_source_quotient_cells": r_nontrivial,
            "negative_temporal_storage_control_all_pass": r_control_pass,
            "label_duplication_invariance_all_pass": r_dup_pass,
            "status": r_status,
            "scope_note": "The Qwen negative control is storage-level: localized interventions must leave causally prior cached positions unchanged. It is not claimed identical to the Pythia pre-logit control.",
        },
        "selection_rule": {
            "collision_pairs": 0,
            "nontrivial_coverage_min": 0.20,
            "fiber_fraction_max": 0.80,
            "sufficiency_loss_max": 0.0,
            "ami_above_fixed_fiber_permutation_95": True,
            "coarsest_admissible_wins": True,
        },
        "promotion_status": "NOT_VALIDATED" if selected else "NO_CANDIDATE",
        "next_step": "Freeze the selected representation and run one fresh holdout only." if selected else "Do not enrich the operation algebra yet; inspect which component or quotient axis causes the fixed-rho failure.",
        "scope_note": "Finite-battery Qwen measurement at fixed OP level. Values are conditional on the declared interventions, readouts, macro roles, and resolution; they are not intrinsic universal model constants.",
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "CO_FORMULA_MEASUREMENT_17_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if selected:
        cand_path = outdir / "CO_FORMULA_MEASUREMENT_17_CANDIDATE.json"
        cand = dict(selected)
        cand.update({
            "schema": "co-qwen3-8b-formula-measurement-17-candidate-v1",
            "development_measurement_sha256": sha256(measurement_path),
            "generalization_level": "OP",
            "fixed_resolution": result["fixed_resolution"],
            "holdout_refit_allowed": False,
        })
        cand_path.write_text(json.dumps(cand, indent=2, sort_keys=True) + "\n")
    summary_path = outdir / "CO_FORMULA_MEASUREMENT_17_SUMMARY.md"
    summary_path.write_text(
        "# Qwen3-8B Formula Measurement 17\n\n"
        f"- Systems: {len(rows)}\n"
        f"- Status: {status}\n"
        f"- Candidate families evaluated: {len(families)}\n"
        f"- R non-trivial source-quotient cells: {r_nontrivial}/{len(r_cells)}\n"
        f"- R storage-level negative controls: {'PASS' if r_control_pass else 'FAIL'}\n"
        f"- R duplication invariance: {'PASS' if r_dup_pass else 'FAIL'}\n"
        + (f"- Selected development candidate: {selected['candidate_name']}\n" if selected else "- Selected development candidate: none\n")
        + "\nNo development candidate is a closure validation. One fresh holdout is required after freeze.\n"
    )
    print("[CO FORMULA MEASUREMENT 17 SCORE]")
    print("[SYSTEMS]", len(rows))
    print("[STATUS]", status)
    print("[CANDIDATE FAMILIES]", len(families))
    print("[R NONTRIVIAL CELLS]", f"{r_nontrivial}/{len(r_cells)}")
    print("[R TEMPORAL STORAGE CONTROL]", "PASS" if r_control_pass else "FAIL")
    print("[R DUPLICATION INVARIANCE]", "PASS" if r_dup_pass else "FAIL")
    if selected:
        print("[CANDIDATE]", selected["candidate_name"])
        rec = screens[selected["family"]]
        print("[FAMILY]", selected["family"])
        print("[FIBERS]", rec["metrics"]["n_fibers"])
        print("[COVERAGE]", f"{rec['metrics']['nontrivial_coverage']:.4f}")
        print("[AMI]", f"{rec['partition_quality']['adjusted_mutual_information']:.4f}")
    print("[RESULT]", result_path)
    print("[SUMMARY]", summary_path)


if __name__ == "__main__":
    main()
