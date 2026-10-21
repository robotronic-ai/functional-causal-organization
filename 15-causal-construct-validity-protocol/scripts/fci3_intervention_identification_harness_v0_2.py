#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "FCI3_INTERVENTION_IDENTIFICATION_FIXTURES_v0.2.json"
MANIFEST = HERE / "FCI3_INTERVENTION_IDENTIFICATION_FIXTURE_MANIFEST_v0.2.json"
RESULT = HERE / "FCI3_INTERVENTION_IDENTIFICATION_RESULT_v0.2.json"


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def interval_inside(ci, eps):
    lo, hi = ci
    return -eps <= lo and hi <= eps


def interval_outside(ci, eps):
    lo, hi = ci
    return lo > eps or hi < -eps


def equivalence_status(intervals, epsilons):
    all_inside = True
    any_outside = False
    for key, ci in intervals.items():
        eps = epsilons[key]
        if not interval_inside(ci, eps):
            all_inside = False
        if interval_outside(ci, eps):
            any_outside = True

    if all_inside:
        return "EQUIVALENT"
    if any_outside:
        return "NOT-EQUIVALENT"
    return "UNRESOLVED"


def pair_status(case, profile_eps, af_eps):
    if not case["rivals_equivalent"]:
        return "RIVAL-MISMATCH"

    p = equivalence_status(case["profile_diffs_ci"], profile_eps)

    if p == "NOT-EQUIVALENT":
        return "PROFILE-MISMATCH"
    if p == "UNRESOLVED":
        return "PROFILE-UNRESOLVED"

    if case["sentinel_changed"]:
        return "EXCLUSION-UNSUPPORTED"

    a = equivalence_status(case["af_diffs_ci"], af_eps)
    if a == "EQUIVALENT":
        return "FIBER-CONSISTENT"
    if a == "NOT-EQUIVALENT":
        return "FIBER-VIOLATION"
    return "AF-UNRESOLVED"


def instrument_status(case, beta_eps):
    ci = case["beta_diff_ci"]
    if interval_inside(ci, beta_eps):
        return "INSTRUMENT-CONSISTENT"
    if interval_outside(ci, beta_eps):
        return "INSTRUMENT-DISAGREEMENT"
    return "INSTRUMENT-UNRESOLVED"


def main():
    data = json.loads(FIXTURES.read_text())
    manifest = json.loads(MANIFEST.read_text())

    checks = {
        "FIXTURE_HASH_INTACT": sha256(FIXTURES) == manifest["fixtures_sha256"]
    }

    rows = []
    all_match = True
    for case in data["cases"]:
        p_status = pair_status(
            case,
            data["profile_epsilon"],
            data["af_epsilon"],
        )
        i_status = instrument_status(
            case,
            data["beta_epsilon"],
        )
        p_match = p_status == case["expected_pair_status"]
        i_match = i_status == case["expected_instrument_status"]
        all_match = all_match and p_match and i_match
        rows.append({
            "id": case["id"],
            "observed_pair_status": p_status,
            "expected_pair_status": case["expected_pair_status"],
            "pair_match": p_match,
            "observed_instrument_status": i_status,
            "expected_instrument_status": case["expected_instrument_status"],
            "instrument_match": i_match,
        })

    checks["ALL_FROZEN_CASES_CLASSIFIED_AS_EXPECTED"] = all_match
    passed = all(checks.values())

    verdict = (
        "FCI3-IDENTIFICATION-ADJUDICATOR-PASS"
        if passed
        else "FCI3-IDENTIFICATION-ADJUDICATOR-REVIEW"
    )

    out = {
        "schema": "FCI3-intervention-identification-result-v0.2",
        "date": "2026-09-14",
        "verdict": verdict,
        "checks": checks,
        "cases": rows,
        "interpretation": (
            "The adjudicator distinguishes same-profile fiber consistency, "
            "fiber violation, exclusion failure, rival mismatch, profile "
            "uncertainty, and convergent-instrument disagreement on frozen "
            "synthetic fixtures. This validates protocol logic only, not CF "
            "construct validity."
        )
    }
    RESULT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
