#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR = HERE / "FCI2_ACCESS_ROLE_BINDING_PAIR_v0.3.json"
FREEZE = HERE / "FCI2_ACCESS_ROLE_BINDING_PRE_GENERATION_MANIFEST_v0.3.json"
AF_PATH = HERE / "af_access_consequence_candidate_v0_1.py"
RESULT = HERE / "FCI2_ACCESS_ROLE_BINDING_RESULT_v0.3.json"

EXPECTED_AF_SHA256 = "8bdf05cb2be3c3085c52c0b2a389aa28e1cbf598dbd90d38496799050c82b889"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def entropy(probs):
    out = 0.0
    for p in probs:
        if p > 0:
            out -= p * math.log2(p)
    return out


def mutual_information(joint):
    # joint maps (x,y) -> p
    px = Counter()
    py = Counter()
    for (x, y), p in joint.items():
        px[x] += p
        py[y] += p
    out = 0.0
    for (x, y), p in joint.items():
        if p > 0:
            out += p * math.log2(p / (px[x] * py[y]))
    return out


def source_states():
    return [(0,0),(0,1),(1,0),(1,1)]


def readout(binding, p):
    xa, xb = p
    values = {"X_A": xa, "X_B": xb}
    return (values[binding["TASK-A"]], values[binding["TASK-B"]])


def delta_metrics(binding):
    states = source_states()
    q = 1.0 / len(states)

    joint_py = {}
    joint_pya = {}
    joint_pyb = {}
    for p in states:
        y = readout(binding, p)
        ya, yb = y
        joint_py[(p, y)] = joint_py.get((p, y), 0.0) + q
        joint_pya[(p, ya)] = joint_pya.get((p, ya), 0.0) + q
        joint_pyb[(p, yb)] = joint_pyb.get((p, yb), 0.0) + q

    i_total = mutual_information(joint_py)
    i_a = mutual_information(joint_pya)
    i_b = mutual_information(joint_pyb)

    # For two players, Shapley:
    # phi_A = 1/2 I(P;YA) + 1/2 [I(P;YA,YB)-I(P;YB)]
    # phi_B analogously.
    phi_a = 0.5 * i_a + 0.5 * (i_total - i_b)
    phi_b = 0.5 * i_b + 0.5 * (i_total - i_a)

    weights = [phi_a / i_total, phi_b / i_total]
    h2 = entropy(weights)
    n_eff = 2 ** h2
    pi = n_eff - 1.0  # denominator m-1 = 1 for m=2
    delta_cap = i_total / 2.0
    delta_prop = delta_cap * pi

    return {
        "I_P_Y": i_total,
        "I_P_TASK_A": i_a,
        "I_P_TASK_B": i_b,
        "shapley": [phi_a, phi_b],
        "weights": weights,
        "N_eff": n_eff,
        "Pi_role": pi,
        "Delta_cap": delta_cap,
        "Delta_prop": delta_prop
    }


def core_transition(z):
    z1, z2 = z
    return (z1 ^ z2, z1)


def iterate(z, k):
    out = z
    for _ in range(k):
        out = core_transition(out)
    return out


def gamma_core():
    # J(Z1 -> Z2) under do(Z2=b):
    # Z2' = Z1 exactly, so one bit.
    j_12 = 1.0

    # J(Z2 -> Z1) under do(Z1=a):
    # Z1' = a XOR Z2, a bijection of Z2, so one bit.
    j_21 = 1.0

    gamma = (j_12 + j_21) / (2.0 * 1.0)
    return {"J_Z1_to_Z2": j_12, "J_Z2_to_Z1": j_21, "Gamma": gamma}


def recurrence_core(k):
    states = [(0,0),(0,1),(1,0),(1,1)]
    q = 0.25
    joint = {}
    for z in states:
        zk = iterate(z, k)
        joint[(z, zk)] = joint.get((z, zk), 0.0) + q
    mi = mutual_information(joint)
    k_eff = 2.0
    return {"k": k, "I_return": mi, "K_eff": k_eff, "R": mi / k_eff}


def load_af():
    if sha256(AF_PATH) != EXPECTED_AF_SHA256:
        raise RuntimeError("AF scorer hash mismatch")
    spec = importlib.util.spec_from_file_location("af", AF_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def af_flex_fixture(system_id, binding):
    # Four equally weighted typed probes: each task role is queried for both
    # values of its required component.  The aligned system has the required
    # component and gives the oracle distribution.  The cross-bound system has
    # only the independent other bit; its Bayes-optimal predictor of the
    # required bit is therefore uniform, equal to the content-blind null.
    probes = []
    aligned = (
        binding["TASK-A"] == "X_A"
        and binding["TASK-B"] == "X_B"
    )
    for role in ("TASK-A", "TASK-B"):
        for x in (0, 1):
            truth = {"0": "1", "1": "0"} if x == 0 else {"0": "0", "1": "1"}
            pred = truth if aligned else {"0": "1/2", "1": "1/2"}
            probes.append({
                "id": f"{role}_x{x}",
                "q": "1/4",
                "truth": truth,
                "null": {"0": "1/2", "1": "1/2"},
                "pred": pred
            })
    return {
        "id": system_id,
        "families": [{
            "id": "AF-FLEX",
            "available": True,
            "probes": probes
        }]
    }


def close(a, b, tol=1e-12):
    return abs(a-b) <= tol


def main():
    pair = json.loads(PAIR.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    af = load_af()

    # Integrity of frozen pre-generation material.
    freeze_ok = True
    for name, expected in freeze["files"].items():
        freeze_ok &= sha256(HERE / name) == expected

    gamma = gamma_core()
    r = {k: recurrence_core(k) for k in (1,2,4)}

    systems = {}
    for s in pair["systems"]:
        d = delta_metrics(s["binding"])
        af_profile = af.profile(af_flex_fixture(s["id"], s["binding"]))
        af_score = af_profile["families"]["AF-FLEX"]["score"]
        systems[s["id"]] = {
            "delta": d,
            "Gamma": gamma["Gamma"],
            "R": {str(k): r[k]["R"] for k in r},
            "AF_FLEX": af_score
        }

    A = systems["ALIGNED"]
    B = systems["CROSSBOUND"]

    checks = {
        "PRE_FREEZE_INTACT": freeze_ok,
        "B0_I_P_Y_2_ALIGNED": close(A["delta"]["I_P_Y"], 2.0),
        "B0_I_P_Y_2_CROSSBOUND": close(B["delta"]["I_P_Y"], 2.0),
        "B1_DELTA_CAP_1_BOTH": close(A["delta"]["Delta_cap"], 1.0) and close(B["delta"]["Delta_cap"], 1.0),
        "B2_SHAPLEY_1BIT_EACH_BOTH": all(close(x,1.0) for x in A["delta"]["shapley"] + B["delta"]["shapley"]),
        "B3_WEIGHTS_HALF_HALF_BOTH": all(close(x,0.5) for x in A["delta"]["weights"] + B["delta"]["weights"]),
        "B4_NEFF_2_BOTH": close(A["delta"]["N_eff"],2.0) and close(B["delta"]["N_eff"],2.0),
        "B5_PI_1_BOTH": close(A["delta"]["Pi_role"],1.0) and close(B["delta"]["Pi_role"],1.0),
        "B6_DELTA_PROP_1_BOTH": close(A["delta"]["Delta_prop"],1.0) and close(B["delta"]["Delta_prop"],1.0),
        "B7_GAMMA_1_BOTH": close(A["Gamma"],1.0) and close(B["Gamma"],1.0),
        "B8_R_1_2_4_1_BOTH": all(close(v,1.0) for v in list(A["R"].values()) + list(B["R"].values())),
        "B9_COMPRESSED_CF_EQUAL": (
            close(A["Gamma"],B["Gamma"])
            and close(A["delta"]["Delta_prop"],B["delta"]["Delta_prop"])
            and all(close(A["R"][k],B["R"][k]) for k in A["R"])
        ),
        "AF_ALIGNED_1": close(A["AF_FLEX"],1.0),
        "AF_CROSSBOUND_0": close(B["AF_FLEX"],0.0),
        "AF_DIFFERS": not close(A["AF_FLEX"],B["AF_FLEX"]),
        "C0_C2_ALL_ROLES_CERTIFIED_EX_ANTE": all(r["certified_before_outcome"] for r in pair["access_roles"]),
        "C3_CROSS_TYPE_SWAP_FORBIDDEN": pair["cross_type_permutation_admissible"] is False,
    }

    # C4: coherent within-role label recoding. Binary complement applied to both
    # truth and prediction leaves the AF score unchanged.
    def complement_dist(d):
        return {"0": d["1"], "1": d["0"]}

    def recoded_af(system_id, binding):
        fixture = af_flex_fixture(system_id, binding)
        for fam in fixture["families"]:
            for p in fam["probes"]:
                p["truth"] = complement_dist(p["truth"])
                p["null"] = complement_dist(p["null"])
                p["pred"] = complement_dist(p["pred"])
        return af.profile(fixture)["families"]["AF-FLEX"]["score"]

    a_rec = recoded_af("ALIGNED-R", pair["systems"][0]["binding"])
    b_rec = recoded_af("CROSSBOUND-R", pair["systems"][1]["binding"])
    checks["C4_WITHIN_ROLE_RECODING_PRESERVES_VERDICT"] = (
        close(a_rec, A["AF_FLEX"])
        and close(b_rec, B["AF_FLEX"])
        and not close(a_rec,b_rec)
    )

    passed = all(checks.values())
    verdict = (
        "FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING"
        if passed
        else "FCI2-ATTACK-NOT-ESTABLISHED"
    )

    result = {
        "schema": "FCI2-access-role-binding-result-v0.3",
        "date": "2026-09-14",
        "verdict": verdict,
        "checks": checks,
        "core": {
            "Gamma": gamma,
            "R": {str(k): r[k] for k in r}
        },
        "systems": systems,
        "interpretation": (
            "If PASS, the compressed numerical profile "
            "(Gamma, Delta_prop, R(k)) is not sufficient for the independently "
            "defined typed AF-FLEX consequence. The broader causal family C is "
            "not refuted. No consciousness-phenomenology claim is made."
        )
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
