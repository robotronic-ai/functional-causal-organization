#!/usr/bin/env python3
"""Exhaustive U0-style linear corollary check over all 3x3 GF(2) dynamics.

Fixed source interface E injects two content basis vectors into coordinates 1 and 2.
Declared readouts are the first two coordinate functionals.

Checks the classic reachable/observable identity:
minimal predictive dimension = reachable dimension - unobservable dimension
                            = generalized Hankel rank.
"""

import json
from pathlib import Path

from common_gf2 import all_binary_matrices, gf2_rank_rows, mat_mul_rows, mat_pow_rows, mat_vec_mul_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "u0_linear_corollary.json"
N = 3
E_COLS = [0b001, 0b010]
D_ROWS = [0b001, 0b010]


def span_vectors(vectors, width=N):
    basis = []
    for v in vectors:
        if gf2_rank_rows(basis + [v], width) > gf2_rank_rows(basis, width):
            basis.append(v)
    return basis


def all_span_elements(basis):
    out = []
    for mask in range(1 << len(basis)):
        v = 0
        for i, b in enumerate(basis):
            if (mask >> i) & 1:
                v ^= b
        out.append(v)
    return sorted(set(out))


def main():
    total = 0
    failures = 0
    dim_hist = {}

    for A in all_binary_matrices(N):
        total += 1
        # Reachable subspace from E using k=0..2 (enough in dimension 3).
        reach_generators = []
        for k in range(N):
            Ak = mat_pow_rows(A, k, N)
            for e in E_COLS:
                reach_generators.append(mat_vec_mul_rows(Ak, e))
        R_basis = span_vectors(reach_generators)
        R_elements = all_span_elements(R_basis)
        r_reach = len(R_basis)

        # Unobservable reachable vectors: D A^n x = 0 for n=0..2.
        unobs = []
        for x in R_elements:
            ok = True
            for n in range(N):
                An = mat_pow_rows(A, n, N)
                y = mat_vec_mul_rows(An, x)
                if any(((d & y).bit_count() & 1) for d in D_ROWS):
                    ok = False
                    break
            if ok:
                unobs.append(x)
        N_basis = span_vectors(unobs)
        predicted_dim = r_reach - len(N_basis)

        # Block Hankel rows (d,n), columns (e,k), n,k=0..2.
        rows = []
        for d in D_ROWS:
            for n in range(N):
                row_bits = 0
                col_idx = 0
                for e in E_COLS:
                    for k in range(N):
                        Ank = mat_pow_rows(A, n + k, N)
                        y = mat_vec_mul_rows(Ank, e)
                        bit = (d & y).bit_count() & 1
                        row_bits |= bit << col_idx
                        col_idx += 1
                rows.append(row_bits)
        hankel_rank = gf2_rank_rows(rows, len(E_COLS) * N)
        dim_hist[predicted_dim] = dim_hist.get(predicted_dim, 0) + 1
        if hankel_rank != predicted_dim:
            failures += 1

    result = {
        "status": "PASS" if failures == 0 else "FAIL",
        "universe": {
            "field": "F2",
            "state_dimension": 3,
            "dynamics_count": total,
            "source_interface": ["e1", "e2"],
            "declared_readouts": ["x1", "x2"],
        },
        "checks": {
            "hankel_rank_equals_reachable_minus_unobservable_failures": failures
        },
        "minimal_predictive_dimension_histogram": {str(k): v for k, v in sorted(dim_hist.items())},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
