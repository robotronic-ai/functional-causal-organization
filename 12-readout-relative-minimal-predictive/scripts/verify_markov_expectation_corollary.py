#!/usr/bin/env python3
"""Exact rational check of the finite-state Markov expectation corollary.

Universe:
- two states,
- two control kernels,
- each row has P(next=1) in {0, 1/2, 1},
- 9 kernels and 81 ordered control-kernel pairs,
- source sets {0}, {1}, {0,1},
- declared readout d(x)=1[x=1].

All arithmetic is exact using fractions.Fraction.
"""

import json
from fractions import Fraction
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "markov_expectation_corollary.json"
VALS = [Fraction(0), Fraction(1, 2), Fraction(1)]
READOUT = (Fraction(0), Fraction(1))


def rank(rows):
    rows = [list(r) for r in rows]
    if not rows:
        return 0
    m = len(rows)
    n = len(rows[0])
    r = 0
    for c in range(n):
        pivot = next((i for i in range(r, m) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(m):
            if i != r and rows[i][c] != 0:
                a = rows[i][c]
                rows[i] = [x - a * y for x, y in zip(rows[i], rows[r])]
        r += 1
        if r == m:
            break
    return r


def independent_add(vec, basis):
    if rank(basis + [vec]) > rank(basis):
        basis.append(tuple(vec))
        return True
    return False


def kernels():
    # Kernel represented by p_x = P(next=1 | current=x).
    return [(p0, p1) for p0 in VALS for p1 in VALS]


def U_apply(f, K):
    f0, f1 = f
    p0, p1 = K
    return (
        (1 - p0) * f0 + p0 * f1,
        (1 - p1) * f0 + p1 * f1,
    )


def koopman_closure(Ks):
    basis = []
    independent_add(READOUT, basis)
    changed = True
    while changed:
        changed = False
        for f in list(basis):
            for K in Ks:
                if independent_add(U_apply(f, K), basis):
                    changed = True
    return basis


def coordinates(vec, basis):
    r = len(basis)
    if r == 0:
        if any(vec):
            raise ValueError("nonzero vector in zero basis")
        return tuple()
    # Solve B^T c = vec by brute Gaussian elimination on the augmented system.
    # Equations are state coordinates (2 equations, r unknowns).
    rows = []
    for state in range(2):
        rows.append([basis[j][state] for j in range(r)] + [vec[state]])
    row = 0
    pivots = []
    for col in range(r):
        pivot = next((i for i in range(row, len(rows)) if rows[i][col] != 0), None)
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        pv = rows[row][col]
        rows[row] = [x / pv for x in rows[row]]
        for i in range(len(rows)):
            if i != row and rows[i][col] != 0:
                a = rows[i][col]
                rows[i] = [x - a * y for x, y in zip(rows[i], rows[row])]
        pivots.append((row, col))
        row += 1
    coeff = [Fraction(0)] * r
    for rr, cc in pivots:
        coeff[cc] = rows[rr][-1]
    # Verify.
    recon = [sum(coeff[j] * basis[j][s] for j in range(r)) for s in range(2)]
    if tuple(recon) != tuple(vec):
        raise ValueError("vector not in basis span")
    return tuple(coeff)


def dual_matrix(K, W):
    # M[j][m] are coordinates of U(w_j) in W basis.
    coeff_rows = [coordinates(U_apply(w, K), W) for w in W]
    # Predictive state v stores values v(w_j). After transition:
    # v'_j = v(U w_j) = sum_m coeff_rows[j][m] v_m.
    return coeff_rows


def apply_dual(M, v):
    return tuple(sum(M[j][m] * v[m] for m in range(len(v))) for j in range(len(M)))


def source_vector(state, W):
    return tuple(w[state] for w in W)


def predictive_closure(source_states, Ms):
    states = []
    queue = []
    for s in source_states:
        v = s
        if independent_add(v, states):
            queue.append(v)
    while queue:
        v = queue.pop(0)
        for M in Ms:
            nv = apply_dual(M, v)
            if independent_add(nv, states):
                queue.append(nv)
    return states


def source_sets():
    return [(0,), (1,), (0, 1)]


def main():
    Ks = kernels()
    total = 0
    closure_failures = 0
    dual_invariance_failures = 0
    hankel_failures = 0
    dim_hist = {}

    for Ka in Ks:
        for Kb in Ks:
            controls = (Ka, Kb)
            W = koopman_closure(controls)
            for f in W:
                for K in controls:
                    if rank(W + [U_apply(f, K)]) != rank(W):
                        closure_failures += 1
            Ms = [dual_matrix(K, W) for K in controls]

            for ss in source_sets():
                total += 1
                initials = [source_vector(s, W) for s in ss]
                Vbasis = predictive_closure(initials, Ms)
                pred_dim = rank(Vbasis)
                dim_hist[pred_dim] = dim_hist.get(pred_dim, 0) + 1

                # Every dual transition must preserve the predictive span.
                for v in Vbasis:
                    for M in Ms:
                        nv = apply_dual(M, v)
                        if rank(Vbasis + [nv]) != rank(Vbasis):
                            dual_invariance_failures += 1

                # Build a finite Hankel pairing from a spanning row basis W and spanning column basis V.
                H = []
                for f in W:
                    c = coordinates(f, W)
                    H.append([sum(c[j] * v[j] for j in range(len(W))) for v in Vbasis])
                if rank(H) != pred_dim:
                    hankel_failures += 1

    failures = closure_failures + dual_invariance_failures + hankel_failures
    result = {
        "status": "PASS" if failures == 0 else "FAIL",
        "universe": {
            "states": 2,
            "controls": 2,
            "row_probability_grid": ["0", "1/2", "1"],
            "kernels": len(Ks),
            "ordered_kernel_pairs": len(Ks) ** 2,
            "source_sets_per_pair": len(source_sets()),
            "total_pair_source_cases": total,
            "declared_readout": "indicator(state == 1)",
            "arithmetic": "exact rational",
        },
        "checks": {
            "markov_koopman_closure_failures": closure_failures,
            "predictive_dual_invariance_failures": dual_invariance_failures,
            "hankel_rank_equals_predictive_dimension_failures": hankel_failures,
        },
        "minimal_predictive_dimension_histogram": {str(k): v for k, v in sorted(dim_hist.items())},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
