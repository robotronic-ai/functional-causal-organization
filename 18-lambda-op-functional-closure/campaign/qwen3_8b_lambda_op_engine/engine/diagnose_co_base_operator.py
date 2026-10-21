#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from co_base_operator_profile import (
    ADDRESSES,
    OPS,
    compact_search,
    delta_profile,
    delta_vector,
    gamma_profile,
    gamma_vector,
    r_profile,
    r_vector,
    routing_tensor,
    source_content_feature_vector,
    target_vector,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source_rows_by_unit(panel: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(r["unit"]): r for r in panel["rows"]}


def _gamma_scalar(g: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(x["gamma_min_bidirectional"]) for x in g["per_operation"]], dtype=np.float64)


def _delta_only(d: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([float(x[key]) for x in d["per_operation"]], dtype=np.float64)


def _r_cycle_only(r: dict[str, Any]) -> np.ndarray:
    vals = []
    for op in r["per_operation"]:
        vals.extend(float(x) for x in op["r_cycle_candidate"])
    return np.asarray(vals, dtype=np.float64)


def _r_k1_only(r: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(op["r_cycle_candidate"][0]) for op in r["per_operation"]], dtype=np.float64)


def _lambda_scalar_norm(v: np.ndarray) -> np.ndarray:
    # Existing typed Lambda proxy has REVOKE 12D then SUBSTITUTE 12D.
    return np.asarray([float(np.linalg.norm(v[:12])), float(np.linalg.norm(v[12:]))], dtype=np.float64)


def _names_gamma_profile() -> list[str]:
    names = []
    for op in OPS:
        for cut in ("H1|H2+H3", "H2|H1+H3", "H3|H1+H2"):
            names.extend([f"Gamma:{op}:{cut}:forward", f"Gamma:{op}:{cut}:backward"])
    return names


def _names_delta_factors() -> list[str]:
    names = []
    for op in OPS:
        names.extend([
            f"Delta:{op}:cap_effective_rank",
            f"Delta:{op}:pi_role",
            f"Delta:{op}:role_coverage",
            f"Delta:{op}:product",
        ])
    return names


def _names_r_cycle() -> list[str]:
    return [f"R:{op}:cycle:k{k}" for op in OPS for k in (1, 2, 3, 4)]


def _names_r_full() -> list[str]:
    out = []
    for op in OPS:
        out.extend([f"R:{op}:cycle:k{k}" for k in (1, 2, 3, 4)])
        out.extend([f"R:{op}:persistence:k{k}" for k in (1, 2, 3, 4)])
    return out


def _names_lambda() -> list[str]:
    names = []
    for op in ("REVOKE", "SUBSTITUTE"):
        for address in ADDRESSES:
            for intervention in ("ZERO_SUFFIX", "HEAD_ROTATE"):
                names.extend([
                    f"Lambda:{op}:{address}:{intervention}:DONOR_MINUS_ORIGINAL",
                    f"Lambda:{op}:{address}:{intervention}:NONE_MINUS_ORIGINAL",
                ])
    return names


def _decorate(search: dict[str, Any], names: list[str]) -> dict[str, Any]:
    out = dict(search)
    if out.get("status") == "COMPACT_CLOSED_QUOTIENT_FOUND":
        for spec in out["selected_features"]:
            j = int(spec["feature_index"])
            spec["feature_name"] = names[j]
    return out


def _family_search(Xs: list[np.ndarray], targets: list[str], names: list[str]) -> dict[str, Any]:
    if not Xs:
        return {
            "status": "NOT_MEASURABLE",
            "reason": "no measured systems are available for this family",
            "metrics": {
                "n_fibers": 0,
                "n_collision_fibers": 0,
                "n_collision_pairs": 0,
                "n_nontrivial_fibers": 0,
                "systems_in_nontrivial_fibers": 0,
                "nontrivial_coverage": 0.0,
            },
        }
    X = np.stack(Xs)
    return _decorate(compact_search(X, targets, max_features=5), names)


def _closed_useful(rec: dict[str, Any]) -> bool:
    q = rec.get("partition_quality", {})
    return (
        rec.get("status") == "COMPACT_CLOSED_QUOTIENT_FOUND"
        and rec["metrics"]["n_collision_pairs"] == 0
        and rec["metrics"]["nontrivial_coverage"] >= 0.20
        and bool(q.get("ami_exceeds_permutation_95", False))
        and float(q.get("sufficiency_loss_H_target_given_descriptor_over_H_target", 1.0)) <= 1e-12
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-panel", required=True)
    ap.add_argument("--source-content-panel", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    routing_path = Path(args.routing_panel)
    source_path = Path(args.source_content_panel)
    routing = json.loads(routing_path.read_text())
    source = json.loads(source_path.read_text())
    if not routing.get("rows"):
        skips = routing.get("skips", [])
        details = "; ".join(str(x.get("error", "unknown")) for x in skips[:3])
        raise RuntimeError(
            "Base-Operator routing panel contains zero measured rows. "
            f"n_skipped={routing.get('n_skipped', len(skips))}. "
            f"First skip causes: {details or 'not recorded'}"
        )
    if int(routing.get("n_skipped", 0)) != 0:
        raise RuntimeError(
            "Base-Operator routing panel contains technical skips; "
            "component attribution is not scored on a partial panel."
        )
    source_by_unit = _source_rows_by_unit(source)

    profiles = []
    gamma_rows = []
    gamma_scalar_rows = []
    delta_rows = []
    delta_product_rows = []
    delta_min_rows = []
    delta_mean_rows = []
    r_cycle_rows = []
    r_k1_rows = []
    r_full_rows = []
    lambda_rows = []
    lambda_scalar_rows = []
    targets = []
    units = []

    for row in routing["rows"]:
        source_unit = int(row["source_unit"])
        if source_unit not in source_by_unit:
            raise RuntimeError(f"missing source unit {source_unit}")
        src = source_by_unit[source_unit]
        rt = routing_tensor(row)
        g = gamma_profile(rt)
        d = delta_profile(rt)
        r = r_profile(rt)
        l = source_content_feature_vector(src)
        target = target_vector(src)

        gv = gamma_vector(g)
        dv = delta_vector(d, "product")
        rv_cycle = _r_cycle_only(r)
        rv_full = r_vector(r, include_persistence=True)

        profiles.append({
            "unit": int(row["unit"]),
            "source_unit": source_unit,
            "bindings": row["bindings"],
            "donor_values": row["donor_values"],
            "Gamma": g,
            "Delta": d,
            "R": r,
            "Lambda": {
                "estimator": "LAMBDA_CONTENT_TYPED_12_EXISTING_PROXY",
                "status": "EXISTING_OPERATION_INDEXED_TYPED_PROXY",
                "feature_vector": l.tolist(),
                "note": "Reused from the opened Content-Typed Validation 13 panel; no new Lambda fit is performed here.",
            },
            "behavior_target": list(target),
        })
        gamma_rows.append(gv)
        gamma_scalar_rows.append(_gamma_scalar(g))
        delta_rows.append(dv)
        delta_product_rows.append(_delta_only(d, "delta_product"))
        delta_min_rows.append(_delta_only(d, "delta_min"))
        delta_mean_rows.append(_delta_only(d, "delta_normalized_mean"))
        r_cycle_rows.append(rv_cycle)
        r_k1_rows.append(_r_k1_only(r))
        r_full_rows.append(rv_full)
        lambda_rows.append(l)
        lambda_scalar_rows.append(_lambda_scalar_norm(l))
        targets.append(json.dumps(list(target), separators=(",", ":")))
        units.append(int(row["unit"]))

    families: dict[str, tuple[list[np.ndarray], list[str]]] = {
        "GAMMA_PROFILE": (gamma_rows, _names_gamma_profile()),
        "GAMMA_SCALAR_MIN": (gamma_scalar_rows, [f"Gamma:{op}:min" for op in OPS]),
        "DELTA_FACTORS_PRODUCT": (delta_rows, _names_delta_factors()),
        "DELTA_PRODUCT_ONLY": (delta_product_rows, [f"Delta:{op}:product" for op in OPS]),
        "DELTA_MIN_ONLY": (delta_min_rows, [f"Delta:{op}:min" for op in OPS]),
        "DELTA_MEAN_ONLY": (delta_mean_rows, [f"Delta:{op}:mean" for op in OPS]),
        "R_CYCLE_K1_ONLY": (r_k1_rows, [f"R:{op}:cycle:k1" for op in OPS]),
        "R_CYCLE_PROFILE": (r_cycle_rows, _names_r_cycle()),
        "R_CYCLE_PLUS_PERSISTENCE": (r_full_rows, _names_r_full()),
        "LAMBDA_TYPED": (lambda_rows, _names_lambda()),
        "LAMBDA_SCALAR_NORM": (lambda_scalar_rows, ["Lambda:REVOKE:norm", "Lambda:SUBSTITUTE:norm"]),
    }

    # Primary component definitions used for attribution.
    primary = {
        "Gamma": (gamma_rows, _names_gamma_profile()),
        "Delta": (delta_rows, _names_delta_factors()),
        "R": (r_cycle_rows, _names_r_cycle()),
        "Lambda": (lambda_rows, _names_lambda()),
    }
    for a, b in (("Gamma", "Delta"), ("Gamma", "R"), ("Gamma", "Lambda"), ("Delta", "R"), ("Delta", "Lambda"), ("R", "Lambda")):
        Xa, na = primary[a]
        Xb, nb = primary[b]
        families[f"{a.upper()}+{b.upper()}"] = ([np.concatenate([x, y]) for x, y in zip(Xa, Xb)], na + nb)
    full_rows = [np.concatenate([g, d, r, l]) for g, d, r, l in zip(gamma_rows, delta_rows, r_cycle_rows, lambda_rows)]
    full_names = _names_gamma_profile() + _names_delta_factors() + _names_r_cycle() + _names_lambda()
    families["FULL_CO_PROFILE"] = (full_rows, full_names)

    screens = {name: _family_search(Xs, targets, names) for name, (Xs, names) in families.items()}

    singles = [k for k in ("GAMMA_PROFILE", "DELTA_FACTORS_PRODUCT", "R_CYCLE_PROFILE", "LAMBDA_TYPED") if _closed_useful(screens[k])]
    pairs = [k for k in ("GAMMA+DELTA", "GAMMA+R", "GAMMA+LAMBDA", "DELTA+R", "DELTA+LAMBDA", "R+LAMBDA") if _closed_useful(screens[k])]
    full_ok = _closed_useful(screens["FULL_CO_PROFILE"])
    if len(singles) == 1:
        diagnosis = {"status": "LOCALIZED", "candidate_component": singles[0], "basis": "one primary component alone admits a compact closed quotient on the pilot"}
    elif len(singles) > 1:
        diagnosis = {"status": "DIFFUSE", "candidate_components": singles, "basis": "multiple primary components independently admit compact closed quotients on the pilot"}
    elif pairs:
        diagnosis = {"status": "INTERACTIVE", "candidate_pairs": pairs, "basis": "no primary component alone closes, but at least one two-component combination does"}
    elif full_ok:
        diagnosis = {"status": "DIFFUSE", "basis": "closure appears only in the full four-component profile on the pilot"}
    else:
        diagnosis = {"status": "UNRESOLVED", "basis": "no tested compact quotient closes the pilot; do not force a component attribution"}

    target_lists = [json.loads(t) for t in targets]
    pairwise = []
    primary_screen_names = {
        "Gamma": "GAMMA_PROFILE",
        "Delta": "DELTA_FACTORS_PRODUCT",
        "R": "R_CYCLE_PROFILE",
        "Lambda": "LAMBDA_TYPED",
    }
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            if targets[i] == targets[j]:
                continue
            differing = [k for k, (a, b) in enumerate(zip(target_lists[i], target_lists[j])) if a != b]
            separated = []
            primary_sep = []
            for family_name, rec in screens.items():
                labels = rec.get("partition_labels")
                if rec.get("status") == "COMPACT_CLOSED_QUOTIENT_FOUND" and labels and labels[i] != labels[j]:
                    separated.append(family_name)
            for comp, family_name in primary_screen_names.items():
                rec = screens[family_name]
                labels = rec.get("partition_labels")
                if rec.get("status") == "COMPACT_CLOSED_QUOTIENT_FOUND" and labels and labels[i] != labels[j]:
                    primary_sep.append(comp)
            pairwise.append({
                "unit_i": units[i],
                "unit_j": units[j],
                "differing_target_indices": differing,
                "separated_by_primary_components": primary_sep,
                "separated_by_screen_families": separated,
            })

    formula_checks = {
        "Gamma_profile_vs_scalar": {
            "profile": screens["GAMMA_PROFILE"],
            "scalar_min": screens["GAMMA_SCALAR_MIN"],
            "interpretation": "If scalar_min loses closure or coverage relative to the cut profile, scalarization is implicated rather than Gamma as a family.",
        },
        "Delta_composition": {
            "product": screens["DELTA_PRODUCT_ONLY"],
            "minimum": screens["DELTA_MIN_ONLY"],
            "normalized_mean": screens["DELTA_MEAN_ONLY"],
            "factors_plus_product": screens["DELTA_FACTORS_PRODUCT"],
            "interpretation": "Alternative compositions are screened without changing operation algebra; no variant is promoted from this pilot alone.",
        },
        "R_horizon": {
            "k1_only": screens["R_CYCLE_K1_ONLY"],
            "k1_to_k4": screens["R_CYCLE_PROFILE"],
            "cycle_plus_persistence_control": screens["R_CYCLE_PLUS_PERSISTENCE"],
            "interpretation": "This tests whether a single horizon is insufficient and keeps persistence separate from algebraic cycle return.",
        },
        "Lambda_typed_vs_scalar": {
            "typed": screens["LAMBDA_TYPED"],
            "scalar_norm": screens["LAMBDA_SCALAR_NORM"],
            "interpretation": "This directly tests excessive scalarization of the existing typed content-effect Lambda proxy.",
        },
    }

    result = {
        "schema": "co-qwen3-8b-base-operator-component-audit-v1",
        "status": "EXPLORATORY_COMPONENT_AUDIT_COMPLETE",
        "n_systems": len(profiles),
        "operation_algebra": list(OPS),
        "generalization_level": "OP",
        "generalization_level_changed_in_this_phase": False,
        "global_scalar_co_constructed": False,
        "routing_panel_sha256": sha256(routing_path),
        "source_content_panel_sha256": sha256(source_path),
        "component_profiles": profiles,
        "family_screens": screens,
        "heterogeneous_pair_attribution": pairwise,
        "pairwise_attribution_note": "Pairwise separation is reported only for compact screen partitions, never for the nearly-injective continuous fallback.",
        "formula_checks": formula_checks,
        "diagnostic_attribution": diagnosis,
        "diagnostic_status_vocabulary": ["LOCALIZED", "INTERACTIVE", "DIFFUSE", "UNRESOLVED"],
        "selection_contract": {
            "target_facing_metrics": [
                "sufficiency_loss_H_target_given_descriptor_over_H_target",
                "adjusted_mutual_information",
                "AMI permutation control at fixed descriptor fiber sizes",
                "descriptor fiber count / partition entropy"
            ],
            "screen_admissible_requires": {
                "n_collision_pairs": 0,
                "nontrivial_coverage_min": 0.20,
                "sufficiency_loss_max": 0.0,
                "ami_above_permutation_95": True
            },
            "coarseness_rule": "Among admissible exploratory partitions, prefer fewer fibers before extra descriptor detail.",
            "warning": "The pilot search is target-informed and is not a holdout promotion. AMI is evaluation-only and never defines roles, symmetries, raw component coordinates, or measurement grain."
        },
        "design_axes": {
            "intervention_algebra_A": "OP fixed in Phase 15",
            "quotient_symmetry_Q": "not varied in Phase 15",
            "resolution_rho": "declared by macro roles, localized token spans, horizons, and quantization contract; not selected from A-dagger",
            "principle": "algebra richness and quotient strength are distinct axes"
        },
        "scope_note": "This is an exploratory Qwen estimator audit of Gamma, Delta, R, and Lambda at fixed OP level. Candidate estimators are not asserted to be the manuscript quantities until separately construct-validated.",
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "CO_BASE_OPERATOR_15_RESULT.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = outdir / "CO_BASE_OPERATOR_15_SUMMARY.md"
    summary.write_text(
        "# Qwen3-8B C_O Base-Operator Component Audit 15\n\n"
        f"- Systems: {len(profiles)}\n"
        f"- Generalization level: OP (fixed)\n"
        f"- Diagnostic status: {diagnosis['status']}\n"
        f"- Global scalar C_O constructed: no\n"
        "- Gamma: role-cut bidirectional response estimator candidate\n"
        "- Delta: intervention-richness x role-propagation estimator candidate, with minimum and mean alternatives reported\n"
        "- R: algebraic role-cycle profile k=1..4; persistence reported separately\n"
        "- Lambda: existing typed content-effect proxy reused from the opened Validation 13 panel\n"
        "\nThis pilot diagnoses estimator/component sufficiency at fixed OP level; it is not a validation of full manuscript C_O.\n"
    )
    print("[CO BASE-OP COMPONENT AUDIT COMPLETE]")
    print("[SYSTEMS]", len(profiles))
    print("[DIAGNOSTIC STATUS]", diagnosis["status"])
    if diagnosis.get("candidate_component"):
        print("[LOCALIZED COMPONENT]", diagnosis["candidate_component"])
    if diagnosis.get("candidate_pairs"):
        print("[INTERACTIVE PAIRS]", ",".join(diagnosis["candidate_pairs"]))
    print("[RESULT]", path)
    print("[SUMMARY]", summary)


if __name__ == "__main__":
    main()
