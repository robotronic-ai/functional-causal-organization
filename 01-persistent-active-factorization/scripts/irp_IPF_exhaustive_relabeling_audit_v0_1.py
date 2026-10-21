#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IPF v0.1 exhaustive relabeling audit.

For each canonical deterministic family IR7, IR8, IR10, IR11, IR12:
- enumerate every permutation of state labels;
- enumerate every permutation of context labels;
- transport the transition system exactly;
- recompute TS -> least semilattice congruence -> IPF;
- verify that the recovered IPF partition is exactly the transported original
  partition.

No PRNG.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location(
    "tsaudit", str(HERE / "irp_transformation_semigroup_audit_v0_1.py")
)
TS = importlib.util.module_from_spec(spec)
sys.modules["tsaudit"] = TS
spec.loader.exec_module(TS)

spec2 = importlib.util.spec_from_file_location(
    "ipfaudit", str(HERE / "irp_image_profile_factor_audit_v0_1.py")
)
IPF = importlib.util.module_from_spec(spec2)
sys.modules["ipfaudit"] = IPF
spec2.loader.exec_module(IPF)


def canon_partition(part):
    return tuple(sorted(
        (tuple(sorted(b)) for b in part),
        key=lambda x: (len(x), x)
    ))


def compute_ipf_partition(row):
    states, gens = TS.deterministic_generators(row)
    elems = TS.closure(states, gens)
    mult = TS.multiplication_table(states, elems)
    classes, cid, qmult = TS.maximal_semilattice_congruence(elems, mult)
    _J, _profiles, part = IPF.ipf_partition(states, elems, classes)
    return canon_partition(part)


def relabel_row(row, state_perm, context_perm):
    S = tuple(row["alphabet"])
    U = tuple(row["contexts"])
    sp = dict(zip(S, state_perm))
    up = dict(zip(U, context_perm))

    transition = {}
    for key, dist in row["transition"].items():
        s, u = map(int, key.split("|"))
        nd = {str(sp[int(y)]): p for y, p in dist.items()}
        transition[f"{sp[s]}|{up[u]}"] = nd

    # Keep public schema fields needed by TS/IPF.
    out = dict(row)
    out["alphabet"] = list(state_perm)
    out["contexts"] = list(context_perm)
    out["transition"] = transition
    return out, sp, up


def transport_partition(part, sp):
    return canon_partition(
        [{sp[x] for x in block} for block in part]
    )


def main():
    rows = TS.load_rows()
    print("=== IPF v0.1 EXHAUSTIVE RELABELING AUDIT ===")

    grand = 0
    for fam in ("IR7", "IR8", "IR10", "IR11", "IR12"):
        row = next(r for r in rows if r["family"] == fam)
        base = compute_ipf_partition(row)

        S = tuple(row["alphabet"])
        U = tuple(row["contexts"])

        checked = 0
        for ps in permutations(S):
            for pu in permutations(U):
                rr, sp, up = relabel_row(row, ps, pu)
                got = compute_ipf_partition(rr)
                expected = transport_partition(base, sp)
                if got != expected:
                    print("[FAIL]", fam)
                    print(" state_perm=", sp)
                    print(" context_perm=", up)
                    print(" expected=", expected)
                    print(" got=", got)
                    return 2
                checked += 1

        grand += checked
        print(f"[PASS] {fam}: {checked} exhaustive state/context relabelings")

    print("TOTAL_RELABELLINGS_CHECKED:", grand)
    print("VERDICT: exact finite equivariance of IPF on all canonical deterministic controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
