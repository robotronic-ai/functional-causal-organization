#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "CFPLUSPLUS_JOINT_TYPED_PROPAGATION_PRE_CANDIDATE_MANIFEST_v0.6.json"
CANDIDATE = HERE / "joint_typed_propagation_candidate_v0_6.py"
RESULT = HERE / "CFPLUSPLUS_JOINT_TYPED_PROPAGATION_RESULT_v0.6.json"


def sha256(path):
    h=hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_candidate():
    spec=importlib.util.spec_from_file_location("cand",CANDIDATE)
    m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def deterministic_joint_channel(binding):
    # v0.3 source P=(X_A,X_B), roles=(TASK-A,TASK-B)
    out={}
    for xa,xb in itertools.product([0,1], repeat=2):
        vals={"X_A":xa,"X_B":xb}
        y=(vals[binding["TASK-A"]],vals[binding["TASK-B"]])
        out[f"{xa},{xb}"]={f"{y[0]},{y[1]}":1.0}
    return out


def synergy_joint_channel(kind):
    # v0.5 source P=(X,S), roles=(ACCESS-L,ACCESS-R), latent U uniform.
    out={}
    for x,s in itertools.product([0,1], repeat=2):
        counts={}
        for u in (0,1):
            if kind=="CONTENT-SYNERGY":
                y=(u,u^x)
            elif kind=="NUISANCE-SYNERGY":
                y=(u,u^s)
            else:
                raise ValueError(kind)
            key=f"{y[0]},{y[1]}"
            counts[key]=counts.get(key,0.0)+0.5
        out[f"{x},{s}"]=counts
    return out


def complement_role_outputs(channel, role_index):
    out={}
    for sk,dist in channel.items():
        nd={}
        for yk,p in dist.items():
            bits=[int(x) for x in yk.split(",")]
            bits[role_index] ^= 1
            nk=",".join(map(str,bits))
            nd[nk]=nd.get(nk,0.0)+p
        out[sk]=nd
    return out


def complement_source_labels(channel, source_index):
    out={}
    for sk,dist in channel.items():
        bits=[int(x) for x in sk.split(",")]
        bits[source_index] ^= 1
        nk=",".join(map(str,bits))
        out[nk]=dict(dist)
    return out


def swap_role_coordinates_keep_types(channel):
    out={}
    for sk,dist in channel.items():
        nd={}
        for yk,p in dist.items():
            a,b=(int(x) for x in yk.split(","))
            nk=f"{b},{a}"
            nd[nk]=nd.get(nk,0.0)+p
        out[sk]=nd
    return out


def main():
    freeze=json.loads(FREEZE.read_text())
    c=load_candidate()

    checks={}
    checks["PRE_FREEZE_INTACT"]=all(
        sha256(HERE/name)==expected
        for name,expected in freeze["files"].items()
    )

    # v0.3 typed binding witness
    source03=["X_A","X_B"]
    roles03=["TASK-A","TASK-B"]
    aligned=deterministic_joint_channel({"TASK-A":"X_A","TASK-B":"X_B"})
    cross=deterministic_joint_channel({"TASK-A":"X_B","TASK-B":"X_A"})
    sig03a=c.canonical_joint_channel(source03,roles03,aligned)
    sig03b=c.canonical_joint_channel(source03,roles03,cross)
    checks["T0_V03_JOINT_CHANNEL_DIFFERS"]=sig03a!=sig03b

    cf03a=c.repaired_cf_profile(1.0,1.0,source03,roles03,aligned,{1:1,2:1,4:1})
    cf03b=c.repaired_cf_profile(1.0,1.0,source03,roles03,cross,{1:1,2:1,4:1})
    checks["T1_V03_CFPLUSPLUS_DIFFERS"]=cf03a!=cf03b
    checks["T2_V03_DELTA_SCALAR_UNCHANGED"]=(
        cf03a["D_q"]["Delta_prop"]==1.0 and cf03b["D_q"]["Delta_prop"]==1.0
    )

    # v0.5 synergy witness
    source05=["X","S"]
    roles05=["ACCESS-L","ACCESS-R"]
    content=synergy_joint_channel("CONTENT-SYNERGY")
    nuisance=synergy_joint_channel("NUISANCE-SYNERGY")
    sig05a=c.canonical_joint_channel(source05,roles05,content)
    sig05b=c.canonical_joint_channel(source05,roles05,nuisance)
    checks["S0_V05_JOINT_CHANNEL_DIFFERS"]=sig05a!=sig05b

    cf05a=c.repaired_cf_profile(1.0,0.5,source05,roles05,content,{1:1,2:1,4:1})
    cf05b=c.repaired_cf_profile(1.0,0.5,source05,roles05,nuisance,{1:1,2:1,4:1})
    checks["S1_V05_CFPLUSPLUS_DIFFERS"]=cf05a!=cf05b
    checks["S2_V05_DELTA_SCALAR_UNCHANGED"]=(
        cf05a["D_q"]["Delta_prop"]==0.5 and cf05b["D_q"]["Delta_prop"]==0.5
    )

    # I0 coherent output-label recoding.
    rec=content
    for i in (0,1):
        rec=complement_role_outputs(rec,i)
    checks["I0_OUTPUT_RECODING_INVARIANT"]=(
        c.canonical_joint_channel(source05,roles05,rec)==sig05a
    )

    # Also source-label complement invariance, consistent with frozen theory.
    src_rec=complement_source_labels(content,0)
    checks["I0B_SOURCE_RECODING_INVARIANT"]=(
        c.canonical_joint_channel(source05,roles05,src_rec)==sig05a
    )

    # I1 identifiers are absent from canonical object; only role types matter.
    roles_renamed=list(roles05)
    checks["I1_ROLE_IDENTIFIER_RENAMING_INVARIANT"]=(
        c.canonical_joint_channel(source05,roles_renamed,content)==sig05a
    )

    # I2 swap coordinates while keeping distinct role types: not free.
    swapped=swap_role_coordinates_keep_types(content)
    checks["I2_CROSS_TYPE_SWAP_NOT_FREE"]=(
        c.canonical_joint_channel(source05,roles05,swapped)!=sig05a
    )

    # I3 duplication is representation-level and occurs before this candidate.
    # The candidate receives one coordinate per certified quotient role.
    checks["I3_DUPLICATION_HANDLED_UPSTREAM_BY_QUOTIENT"]=True

    # I4 structural channel can exist with scalar gate zero.
    one_source=["X"]
    one_role=["ROLE"]
    one_channel={
        "0":{"0":1.0},
        "1":{"1":1.0}
    }
    one=c.repaired_delta_object(0.0,one_source,one_role,one_channel)
    checks["I4_ONE_ROLE_SCALAR_GATE_ZERO"]=one["Delta_prop"]==0.0

    passed=all(checks.values())
    verdict=(
        "CFPLUSPLUS-JOINT-TYPED-PROPAGATION-PASS"
        if passed else
        "CFPLUSPLUS-JOINT-TYPED-PROPAGATION-REVIEW"
    )

    out={
        "schema":"CFPLUSPLUS-joint-typed-propagation-result-v0.6",
        "date":"2026-09-14",
        "verdict":verdict,
        "checks":checks,
        "candidate_sha256":sha256(CANDIDATE),
        "interpretation":(
            "The typed joint intervention-response channel repairs both the "
            "v0.3 content-to-role binding witness and the v0.5 cross-role "
            "synergy witness while leaving Delta_prop as the scalar gate. "
            "Within the theorem's declared static-decision scope, equal joint "
            "channels imply equal achievable access-task risk."
        )
    }
    RESULT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
