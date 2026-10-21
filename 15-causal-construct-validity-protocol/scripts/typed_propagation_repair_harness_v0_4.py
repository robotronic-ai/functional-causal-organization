#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "FCI2_TYPED_PROPAGATION_REPAIR_PRE_CANDIDATE_MANIFEST_v0.4.json"
WITNESS = HERE / "FCI2_ACCESS_ROLE_BINDING_PAIR_v0.3.json"
WITNESS_RESULT = HERE / "FCI2_ACCESS_ROLE_BINDING_RESULT_v0.3.json"
CANDIDATE = HERE / "typed_propagation_repair_candidate_v0_4.py"
RESULT = HERE / "FCI2_TYPED_PROPAGATION_REPAIR_RESULT_v0.4.json"


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_candidate():
    spec = importlib.util.spec_from_file_location("repair", CANDIDATE)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def deterministic_channel(binding_component):
    """
    Source keys are typed (X_A,X_B) bit pairs encoded as 'a,b'.
    Channel outputs the selected component deterministically.
    """
    out = {}
    for xa in (0,1):
        for xb in (0,1):
            y = xa if binding_component == "X_A" else xb
            out[f"{xa},{xb}"] = (
                {"0":1.0,"1":0.0} if y == 0
                else {"0":0.0,"1":1.0}
            )
    return out


def role_channels_for(binding, renamed=False, duplicate_task_a=False):
    rows = [
        {
            "role_type":"TASK-A",
            "role_id":"alpha" if renamed else "TASK-A-instance",
            "channel":deterministic_channel(binding["TASK-A"])
        },
        {
            "role_type":"TASK-B",
            "role_id":"beta" if renamed else "TASK-B-instance",
            "channel":deterministic_channel(binding["TASK-B"])
        }
    ]
    if duplicate_task_a:
        rows.append({
            "role_type":"TASK-A",
            "role_id":"TASK-A-duplicate",
            "channel":deterministic_channel(binding["TASK-A"])
        })
    return rows


def complement_outputs(role_channels):
    out=[]
    for item in role_channels:
        channel={}
        for key,d in item["channel"].items():
            channel[key]={"0":d["1"],"1":d["0"]}
        out.append({**item,"channel":channel})
    return out


def main():
    freeze=json.loads(FREEZE.read_text())
    witness=json.loads(WITNESS.read_text())
    wr=json.loads(WITNESS_RESULT.read_text())
    c=load_candidate()

    checks={}
    checks["PRE_REPAIR_FREEZE_INTACT"] = (
        sha256(HERE/"FCI2_ACCESS_ROLE_BINDING_CLOSURE_v0.3.md")
        == freeze["files"]["FCI2_ACCESS_ROLE_BINDING_CLOSURE_v0.3.md"]
        and sha256(HERE/"Protocol_FCI2_TYPED_PROPAGATION_REPAIR_v0.4_FROZEN.md")
        == freeze["files"]["Protocol_FCI2_TYPED_PROPAGATION_REPAIR_v0.4_FROZEN.md"]
        and sha256(WITNESS_RESULT) == freeze["upstream_witness_result_sha256"]
    )

    aligned_binding=witness["systems"][0]["binding"]
    cross_binding=witness["systems"][1]["binding"]

    aligned=role_channels_for(aligned_binding)
    cross=role_channels_for(cross_binding)

    sig_a=c.typed_binding_signature(aligned)
    sig_b=c.typed_binding_signature(cross)

    checks["R0_BINDING_SIGNATURE_DIFFERS"] = sig_a != sig_b

    r_profile={1:1.0,2:1.0,4:1.0}
    cf_a=c.repaired_cf_profile(1.0,1.0,aligned,r_profile)
    cf_b=c.repaired_cf_profile(1.0,1.0,cross,r_profile)
    checks["R1_REPAIRED_CF_DIFFERS"] = cf_a != cf_b
    checks["R2_SCALAR_DELTA_UNCHANGED_EQUAL"] = (
        cf_a["D_q"]["Delta_prop"] == 1.0
        and cf_b["D_q"]["Delta_prop"] == 1.0
    )

    # R3 coherent within-role answer-label complement.
    checks["R3_WITHIN_ROLE_RECODING_INVARIANT"] = (
        c.typed_binding_signature(complement_outputs(aligned)) == sig_a
        and c.typed_binding_signature(complement_outputs(cross)) == sig_b
    )

    # R4 role identifier renaming.
    checks["R4_ROLE_ID_RENAMING_INVARIANT"] = (
        c.typed_binding_signature(role_channels_for(aligned_binding,renamed=True)) == sig_a
        and c.typed_binding_signature(role_channels_for(cross_binding,renamed=True)) == sig_b
    )

    # R5 cross-type swap must remain distinct: swap channels while keeping role types.
    swapped_types = [
        {"role_type":"TASK-A","role_id":"x","channel":deterministic_channel("X_B")},
        {"role_type":"TASK-B","role_id":"y","channel":deterministic_channel("X_A")},
    ]
    checks["R5_CROSS_TYPE_SWAP_NOT_EQUIVALENT"] = (
        c.typed_binding_signature(swapped_types) != sig_a
    )

    # R6 representation-level duplicate inside TASK-A collapses.
    checks["R6_DUPLICATION_WITHIN_ROLE_INVARIANT"] = (
        c.typed_binding_signature(
            role_channels_for(aligned_binding,duplicate_task_a=True)
        ) == sig_a
    )

    # R7 one-role homogeneous scalar gate remains zero.
    one_role=[
        {"role_type":"ROLE","role_id":"r0","channel":deterministic_channel("X_A")}
    ]
    one_d=c.repaired_delta_object(0.0,one_role)
    checks["R7_ONE_ROLE_SCALAR_GATE_ZERO"] = one_d["Delta_prop"] == 0.0

    passed=all(checks.values())
    verdict=(
        "FCI2-TYPED-PROPAGATION-REPAIR-PASS"
        if passed else
        "FCI2-TYPED-PROPAGATION-REPAIR-REVIEW"
    )

    out={
        "schema":"FCI2-typed-propagation-repair-result-v0.4",
        "date":"2026-09-14",
        "verdict":verdict,
        "checks":checks,
        "aligned_signature":repr(sig_a),
        "crossbound_signature":repr(sig_b),
        "candidate_sha256":sha256(CANDIDATE),
        "interpretation":(
            "The typed intervention-response signature repairs the frozen "
            "v0.3 content-to-role binding witness without changing the scalar "
            "Delta_prop gate. This is a witness-specific repair, not universal "
            "profile sufficiency."
        )
    }
    RESULT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
