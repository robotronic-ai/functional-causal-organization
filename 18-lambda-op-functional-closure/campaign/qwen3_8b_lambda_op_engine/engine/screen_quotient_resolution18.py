#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from co_base_operator_profile import adjusted_mutual_information, partition_quality
from formula_measurement17 import (
    ADDRESSES,
    OPS,
    Q_INTERVENTIONS,
    canonical_partition,
    entropy_bits,
    family_components,
    mutual_information_bits,
    stable,
)

EXPECTED_MEASUREMENT_SHA = "ea9e74daa3e7f681d8efbbd8c7fa8b3afe77cb891857383cf455db872d511561"
EXPECTED_SOURCE_SHA = "92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4"
PERMUTATIONS = 1000
PERMUTATION_SEED = 18018

# Deliberately short, declared after Phase 17 exposed quasi-injectivity at rho=1/16.
# Lambda is always reduced from ordered top-2 to ordered top-1. R is tested at
# top-1 and top-2 source quotient depth. No operation, role, or target-dependent
# symmetry is introduced.
REPRESENTATIONS = (
    {"name": "RHO8_L1_R1", "continuous_bins": 8, "lambda_rank_depth": 1, "r_rank_depth": 1},
    {"name": "RHO8_L1_R2", "continuous_bins": 8, "lambda_rank_depth": 1, "r_rank_depth": 2},
    {"name": "RHO4_L1_R1", "continuous_bins": 4, "lambda_rank_depth": 1, "r_rank_depth": 1},
    {"name": "RHO4_L1_R2", "continuous_bins": 4, "lambda_rank_depth": 1, "r_rank_depth": 2},
    {"name": "RHO2_L1_R1", "continuous_bins": 2, "lambda_rank_depth": 1, "r_rank_depth": 1},
    {"name": "RHO2_L1_R2", "continuous_bins": 2, "lambda_rank_depth": 1, "r_rank_depth": 2},
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def qbin(value: float | None, bins: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    x = min(1.0, max(0.0, float(value)))
    return str(int(round(x * int(bins))))


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


def gamma_key(row: dict[str, Any], bins: int) -> str:
    parts = []
    by = {(x["operation"], x["cut"]): x for x in row["Gamma"]["cuts"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, f"{source}|" + "+".join(a for a in ADDRESSES if a != source))]
            parts.extend([
                qbin(rec["forward_normalized"], bins),
                qbin(rec["backward_normalized"], bins),
                qbin(rec["cut_score_arithmetic_mean"], bins),
                qbin(rec["directional_balance_beta"], bins),
            ])
    return "|".join(parts)


def delta_key(row: dict[str, Any], bins: int) -> str:
    parts = []
    by = {(x["operation"], x["source_address"]): x for x in row["Delta"]["cells"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, source)]
            parts.extend([qbin(rec["kappa_q"], bins), qbin(rec["delta_q"], bins), qbin(rec["Delta_cap"], bins)])
    return "|".join(parts)


def _truncate_rank_signature(signature: Any, depth: int) -> Any:
    # R source signature: list[aux_context][ordered labels]
    return [list(x)[:depth] for x in signature]


def _reentry_at_rank(rec: dict[str, Any], depth: int, bins: int) -> tuple[str, dict[str, Any]]:
    source_signatures = [_truncate_rank_signature(sig, depth) for sig in rec["source_signatures"]]
    class_ids = canonical_partition(source_signatures)
    members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(class_ids):
        members[int(c)].append(i)
    classes = sorted(members)
    p_c = [len(members[c]) / len(class_ids) for c in classes]
    k_eff = entropy_bits(p_c)

    # Reconstruct per-intervention return distributions from the original rank-2
    # class-conditional channel. This is exact for merging old classes because the
    # stored old conditionals are class averages and class masses are known.
    old_class_ids = [int(x) for x in rec["source_class_ids"]]
    old_classes = sorted(set(old_class_ids))
    old_index = {c: j for j, c in enumerate(old_classes)}
    context_mis = []
    for ctx in rec["return_contexts"]:
        old_cond = ctx["conditionals_by_class"]
        by_q = [old_cond[old_index[c]] for c in old_class_ids]
        cond_new = []
        for c in classes:
            arr = np.asarray([by_q[i] for i in members[c]], dtype=np.float64)
            cond_new.append(np.mean(arr, axis=0).tolist())
        context_mis.append(mutual_information_bits(p_c, cond_new))
    b_reentry = float(np.mean(context_mis)) if context_mis else 0.0
    r = None if k_eff <= 1e-15 else float(b_reentry / k_eff)
    key = "|".join([
        str(len(classes)),
        qbin(None if k_eff <= 1e-15 else k_eff / math.log2(len(Q_INTERVENTIONS)), bins),
        qbin(r, bins),
        stable(class_ids),
    ])
    return key, {
        "rank_depth": depth,
        "source_class_ids": class_ids,
        "n_source_classes": len(classes),
        "K_eff_bits": k_eff,
        "B_reentry_bits": b_reentry,
        "R": r,
    }


def r_key(row: dict[str, Any], bins: int, rank_depth: int) -> str:
    parts = []
    by = {(x["operation"], x["source_address"]): x for x in row["R"]["cells"]}
    for op in OPS:
        for source in ADDRESSES:
            key, _ = _reentry_at_rank(by[(op, source)], rank_depth, bins)
            parts.append(key)
    return "|".join(parts)


def lambda_key(row: dict[str, Any], rank_depth: int) -> str:
    parts = []
    by = {(x["operation"], x["source_address"]): x for x in row["Lambda"]["cells"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, source)]
            for q_name in Q_INTERVENTIONS:
                joint = rec["joint_signature_by_intervention"][q_name]
                truncated = [list(role_rank)[:rank_depth] for role_rank in joint]
                parts.append(stable(truncated))
    return "|".join(parts)


def component_keys(row: dict[str, Any], rep: dict[str, Any]) -> dict[str, str]:
    bins = int(rep["continuous_bins"])
    return {
        "Gamma": gamma_key(row, bins),
        "Delta": delta_key(row, bins),
        "R": r_key(row, bins, int(rep["r_rank_depth"])),
        "Lambda": lambda_key(row, int(rep["lambda_rank_depth"])),
    }


def family_label(keys: dict[str, str], components: Iterable[str]) -> str:
    return stable([[name, keys[name]] for name in components])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurement-panel", required=True)
    ap.add_argument("--source-content-panel", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    measurement_path = Path(args.measurement_panel)
    source_path = Path(args.source_content_panel)
    if sha256(measurement_path) != EXPECTED_MEASUREMENT_SHA:
        raise RuntimeError("Formula Measurement 17 panel SHA mismatch")
    if sha256(source_path) != EXPECTED_SOURCE_SHA:
        raise RuntimeError("source content panel SHA mismatch")
    panel = json.loads(measurement_path.read_text())
    source = json.loads(source_path.read_text())
    source_by_unit = {int(r["unit"]): r for r in source["rows"]}
    if int(panel.get("n_skipped", 0)) != 0 or len(panel.get("rows", [])) != 12:
        raise RuntimeError("Phase 18 requires the complete 12-system Formula Measurement 17 panel")

    rows = list(panel["rows"])
    targets = []
    units = []
    for row in rows:
        su = int(row["source_unit"])
        targets.append(_target(source_by_unit[su]))
        units.append(int(row["unit"]))

    families = family_components()
    screens: list[dict[str, Any]] = []
    label_sets: list[list[str]] = []
    for rep in REPRESENTATIONS:
        key_rows = [component_keys(row, rep) for row in rows]
        for family, comps in families.items():
            labels = [family_label(keys, comps) for keys in key_rows]
            metrics = _collision_metrics(labels, targets)
            quality = partition_quality(labels, targets, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
            rec = {
                "screen_id": f"{rep['name']}__{family}",
                "representation": dict(rep),
                "family": family,
                "components": list(comps),
                "metrics": metrics,
                "partition_quality": quality,
                "partition_labels": labels,
            }
            screens.append(rec)
            label_sets.append(labels)

    # Global max-statistic permutation threshold over the entire declared Phase 18 grid.
    rng = np.random.default_rng(PERMUTATION_SEED)
    target_arr = np.asarray(targets, dtype=object)
    max_ami = []
    for _ in range(PERMUTATIONS):
        perm = target_arr[rng.permutation(len(targets))].tolist()
        max_ami.append(max(adjusted_mutual_information(labels, perm) for labels in label_sets))
    global95 = float(np.quantile(np.asarray(max_ami, dtype=np.float64), 0.95))

    admissible = []
    for rec in screens:
        m = rec["metrics"]
        q = rec["partition_quality"]
        q["global_grid_max_ami_permutation_95"] = global95
        q["ami_exceeds_global_grid_permutation_95"] = bool(q["adjusted_mutual_information"] > global95)
        ok = (
            m["n_collision_pairs"] == 0
            and m["nontrivial_coverage"] >= 0.20
            and m["fiber_fraction"] <= 0.80
            and q["sufficiency_loss_H_target_given_descriptor_over_H_target"] <= 1e-12
            and q["ami_exceeds_permutation_95"]
            and q["ami_exceeds_global_grid_permutation_95"]
        )
        rec["admissible_exploratory_candidate"] = bool(ok)
        if ok:
            admissible.append(rec)

    selected = None
    if admissible:
        best = min(
            admissible,
            key=lambda rec: (
                rec["metrics"]["n_fibers"],
                len(rec["components"]),
                -float(1.0 / int(rec["representation"]["continuous_bins"])),
                int(rec["representation"]["r_rank_depth"]),
                rec["screen_id"],
            ),
        )
        selected = {
            "schema": "co-qwen3-8b-quotient-resolution-18-candidate-v1",
            "status": "USEFUL_EXPLORATORY_CANDIDATE",
            "candidate_name": f"Q18_{best['screen_id']}",
            "family": best["family"],
            "components": best["components"],
            "representation": best["representation"],
            "metrics": best["metrics"],
            "partition_quality": best["partition_quality"],
            "measurement_panel_sha256": EXPECTED_MEASUREMENT_SHA,
            "source_content_panel_sha256": EXPECTED_SOURCE_SHA,
            "generalization_level": "OP",
            "operation_algebra_changed": False,
            "candidate_grid_was_fixed_before_phase18_target_evaluation": True,
            "phase18_is_adaptive_after_phase17": True,
            "holdout_refit_allowed": False,
        }

    status = "USEFUL_EXPLORATORY_CANDIDATE" if selected else "NO_ADMISSIBLE_QUOTIENT_RESOLUTION_CANDIDATE"
    result = {
        "schema": "co-qwen3-8b-quotient-resolution-18-result-v1",
        "status": status,
        "n_systems": len(rows),
        "measurement_panel_sha256": EXPECTED_MEASUREMENT_SHA,
        "source_content_panel_sha256": EXPECTED_SOURCE_SHA,
        "generalization_level": "OP",
        "operation_algebra_changed": False,
        "search_axis": ["Q", "rho"],
        "phase17_failure_mode": "QUASI_INJECTIVE_AT_FIXED_RHO16",
        "adaptive_after_phase17": True,
        "raw_measurement_refit": False,
        "behavior_used_to_construct_candidate_grid": False,
        "behavior_used_only_to_evaluate_declared_grid": True,
        "representation_grid": [dict(x) for x in REPRESENTATIONS],
        "representation_count": len(REPRESENTATIONS),
        "family_count": len(families),
        "total_declared_screens": len(screens),
        "permutations": PERMUTATIONS,
        "global_max_ami_permutation_95": global95,
        "selection_rule": {
            "collision_pairs": 0,
            "nontrivial_coverage_min": 0.20,
            "fiber_fraction_max": 0.80,
            "sufficiency_loss_max": 0.0,
            "ami_above_partition_permutation_95": True,
            "ami_above_global_grid_max_permutation_95": True,
            "coarsest_admissible_wins": True,
        },
        "selected_candidate": selected,
        "screens": screens,
        "next_step": (
            "Freeze this exact quotient-resolution rule and open one fresh holdout; no representation refit."
            if selected else
            "Do not enrich the operation algebra. The declared rho/rank quotient grid was insufficient; inspect a target-independent semantic quotient of Lambda or macro-grain rho next."
        ),
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "CO_QUOTIENT_RESOLUTION_18_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if selected:
        cand_path = outdir / "CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json"
        cand_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")

    # Compact console shortlist: coarsest zero-collision screens even if they miss AMI/coverage.
    shortlist = sorted(
        screens,
        key=lambda rec: (
            rec["metrics"]["n_collision_pairs"],
            rec["metrics"]["n_fibers"],
            -rec["partition_quality"]["adjusted_mutual_information"],
            rec["screen_id"],
        ),
    )[:5]
    print("[CO QUOTIENT-RESOLUTION 18 COMPLETE]")
    print("[SOURCE SYSTEMS]", len(rows))
    print("[STATUS]", status)
    print("[REPRESENTATIONS]", len(REPRESENTATIONS))
    print("[FAMILIES]", len(families))
    print("[SCREENS]", len(screens))
    print("[GLOBAL AMI PERM95]", f"{global95:.4f}")
    if selected:
        print("[CANDIDATE]", selected["candidate_name"])
        print("[FAMILY]", selected["family"])
        print("[REPRESENTATION]", json.dumps(selected["representation"], sort_keys=True))
        print("[FIBERS]", selected["metrics"]["n_fibers"])
        print("[NONTRIVIAL COVERAGE]", f"{selected['metrics']['nontrivial_coverage']:.4f}")
        print("[COLLISION PAIRS]", selected["metrics"]["n_collision_pairs"])
        print("[AMI]", f"{selected['partition_quality']['adjusted_mutual_information']:.4f}")
    else:
        for i, rec in enumerate(shortlist, 1):
            print(
                f"[NEAR {i}] {rec['screen_id']} fibers={rec['metrics']['n_fibers']} "
                f"coverage={rec['metrics']['nontrivial_coverage']:.4f} collisions={rec['metrics']['n_collision_pairs']} "
                f"AMI={rec['partition_quality']['adjusted_mutual_information']:.4f}"
            )
    print("[RESULT]", result_path)
    if selected:
        print("[CANDIDATE FILE]", outdir / "CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json")


if __name__ == "__main__":
    main()
