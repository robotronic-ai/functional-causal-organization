#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AF = HERE / "af_access_consequence_candidate_v0_1.py"
WITNESS = HERE / "CF_AF_IDENTIFICATION_WITNESSES_v0.1.json"
WM = HERE / "CF_AF_IDENTIFICATION_WITNESS_MANIFEST_v0.1.json"
RESULT = HERE / "CF_AF_IDENTIFICATION_LIMITS_RESULT_v0.1.json"


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_af():
    spec = importlib.util.spec_from_file_location("af", AF)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def complement_system(system):
    out = copy.deepcopy(system)
    for fam in out["families"]:
        if not fam.get("available", True):
            continue
        for p in fam["probes"]:
            keys = list(p["truth"].keys())
            if len(keys) != 2:
                raise ValueError("binary recoding fixture expected")
            a, b = keys
            for field in ("truth", "null", "pred"):
                d = p[field]
                p[field] = {a: d[b], b: d[a]}
    return out


def all_oracle(profile):
    vals = []
    for fam_id in ("AF-FLEX", "AF-DELAY", "AF-UPDATE", "AF-STATUS"):
        row = profile["families"][fam_id]
        vals.append(row["status"] == "IDENTIFIED" and math.isclose(row["score"], 1.0, abs_tol=1e-12))
    return all(vals)


def main():
    af = load_af()
    data = json.loads(WITNESS.read_text(encoding="utf-8"))
    wm = json.loads(WM.read_text(encoding="utf-8"))

    checks = {}

    checks["N0_AF_SCORER_HASH"] = sha256(AF) == wm["af_candidate_sha256"]
    checks["WITNESS_HASH_INTACT"] = sha256(WITNESS) == wm["witness_sha256"]

    oracle_system = data["af_behavior"]
    oracle_profile = af.profile(oracle_system)
    checks["N1_ORACLE_AF_PROFILE"] = all_oracle(oracle_profile)

    # N2: acyclic witness has no same-role return path.
    r_w = data["witnesses"]["R_ACYCLIC_COMPILED"]
    r_zero = all(float(v) == 0.0 for v in r_w["expected_R"].values())
    no_paths = len(r_w["same_role_return_paths"]) == 0
    checks["N2_ZERO_RECURRENCE_WITH_ORACLE_AF"] = r_zero and no_paths and all_oracle(oracle_profile)

    # N3: Gamma exact on zero cross-cut channels.
    g_w = data["witnesses"]["GAMMA_PARALLEL"]
    j_ab = float(g_w["cross_cut_channels"]["A_to_B_bits"])
    j_ba = float(g_w["cross_cut_channels"]["B_to_A_bits"])
    # Any positive structural denominator leaves zero numerator -> Gamma=0.
    gamma = 0.0 if j_ab == 0.0 and j_ba == 0.0 else None
    checks["N3_ZERO_GAMMA_WITH_ORACLE_AF"] = gamma == 0.0 and all_oracle(oracle_profile)

    # N4: one certified role with positive information gives Pi_role := 0.
    d_w = data["witnesses"]["DELTA_SINGLE_ROLE"]
    m = len(d_w["certified_roles"])
    i_py = float(d_w["I_P_Y_bits"])
    delta_cap = float(d_w["Delta_cap"])
    pi_role = 0.0 if m == 1 and i_py > 0 else None
    delta_prop = delta_cap * pi_role if pi_role is not None else None
    flex = oracle_profile["families"]["AF-FLEX"]["score"]
    checks["N4_ZERO_DELTA_PROP_WITH_AF_FLEX_ORACLE"] = (
        math.isclose(flex, 1.0, abs_tol=1e-12)
        and pi_role == 0.0
        and delta_prop == 0.0
    )

    # N5: coherent answer-label bijection.
    recoded = complement_system(oracle_system)
    recoded_profile = af.profile(recoded)
    inv = True
    for fam_id in ("AF-FLEX", "AF-DELAY", "AF-UPDATE", "AF-STATUS"):
        s0 = oracle_profile["families"][fam_id]["score"]
        s1 = recoded_profile["families"][fam_id]["score"]
        inv = inv and math.isclose(s0, s1, abs_tol=1e-12)
    checks["N5_ANSWER_LABEL_RECODING"] = inv

    passed = all(checks.values())
    verdict = "AF-NOT-STRUCTURALLY-IDENTIFYING" if passed else "AF-IDENTIFICATION-ATTACK-REVIEW"

    out = {
        "schema": "CF-AF-identification-limits-result-v0.1",
        "date": "2026-09-14",
        "verdict": verdict,
        "checks": checks,
        "af_profile": oracle_profile,
        "structural_witnesses": {
            "R_ACYCLIC_COMPILED": {
                "R": r_w["expected_R"],
                "same_role_return_paths": r_w["same_role_return_paths"],
            },
            "GAMMA_PARALLEL": {
                "Gamma": gamma,
                "cross_cut_channels": g_w["cross_cut_channels"],
            },
            "DELTA_SINGLE_ROLE": {
                "m": m,
                "I_P_Y_bits": i_py,
                "Pi_role": pi_role,
                "Delta_cap": delta_cap,
                "Delta_prop": delta_prop,
            },
        },
        "interpretation": (
            "The frozen AF consequence battery can be maximized by fixtures "
            "with zero recurrence, zero integration, or zero propagated-role "
            "coverage respectively. Therefore AF behavior does not identify "
            "the individual CF coordinates. This is a statement about the AF "
            "indicator profile, not about the latent construct A."
        )
    }
    RESULT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
