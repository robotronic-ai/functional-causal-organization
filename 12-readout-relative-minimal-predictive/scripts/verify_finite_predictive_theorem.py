#!/usr/bin/env python3
"""Exhaustive finite-state check of the readout-relative predictive construction.

Universe:
- X = F2^2, represented by four states 0..3.
- All 4^4 = 256 deterministic maps F:X->X.
- All 15 non-empty source subsets E(Z) of X.
- Declared readout space D = span{x1, x2}.

Checks:
1. Koopman closure W is invariant.
2. Predictive behavioral equivalence is F-invariant.
3. Distinct behavioral classes are separated by some original readout at a future delay.
4. The evaluation-state linear dynamics is well defined.
5. Minimal predictive dimension equals a generalized finite Hankel rank.
"""

import json
from itertools import product
from pathlib import Path

from common_gf2 import (
    closure_under_koopman,
    compose_function_bits,
    eval_signature,
    function_to_bits,
    gf2_rank_rows,
    rank_binary_matrix,
    reachable_states,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "finite_predictive_theorem.json"
NSTATES = 4

# State bits: state = x1 + 2*x2.
X1 = function_to_bits([(x >> 0) & 1 for x in range(NSTATES)])
X2 = function_to_bits([(x >> 1) & 1 for x in range(NSTATES)])
D_BASIS = [X1, X2]


def all_maps():
    yield from product(range(NSTATES), repeat=NSTATES)


def source_subsets():
    for mask in range(1, 1 << NSTATES):
        yield tuple(x for x in range(NSTATES) if (mask >> x) & 1)


def future_readout_witness(F, x, y):
    """Return a (readout_index, delay) separating x,y, or None."""
    for delay in range(NSTATES):
        for ridx, d in enumerate(D_BASIS):
            f = d
            for _ in range(delay):
                f = compose_function_bits(f, F, NSTATES)
            if ((f >> x) & 1) != ((f >> y) & 1):
                return ridx, delay
    return None


def enumerate_relation_vectors(columns):
    """Yield coefficient masks a with sum a_i*columns[i] = 0 over F2."""
    m = len(columns)
    width = len(columns[0]) if columns else 0
    for a in range(1 << m):
        accum = [0] * width
        for i in range(m):
            if (a >> i) & 1:
                accum = [u ^ v for u, v in zip(accum, columns[i])]
        if not any(accum):
            yield a


def main():
    total_cases = 0
    closure_failures = 0
    quotient_invariance_failures = 0
    separation_failures = 0
    dynamics_well_defined_failures = 0
    hankel_rank_failures = 0
    dim_hist = {}
    quotient_class_hist = {}

    for F in all_maps():
        W = closure_under_koopman(D_BASIS, F, NSTATES)
        # Closure check.
        for f in W:
            uf = compose_function_bits(f, F, NSTATES)
            if gf2_rank_rows(W + [uf], NSTATES) != gf2_rank_rows(W, NSTATES):
                closure_failures += 1

        for sources in source_subsets():
            total_cases += 1
            R = reachable_states(F, sources)
            sig = {x: eval_signature(x, W) for x in R}

            # Behavioral equivalence classes.
            classes = {}
            for x in R:
                classes.setdefault(sig[x], []).append(x)
            quotient_class_hist[len(classes)] = quotient_class_hist.get(len(classes), 0) + 1

            # F-invariance and separation by original readout at some future delay.
            for xs in classes.values():
                for x in xs:
                    for y in xs:
                        if sig[F[x]] != sig[F[y]]:
                            quotient_invariance_failures += 1
            reps = [xs[0] for xs in classes.values()]
            for i, x in enumerate(reps):
                for y in reps[i + 1:]:
                    if future_readout_witness(F, x, y) is None:
                        separation_failures += 1

            # Predictive linear state vectors are evaluation signatures.
            Vcols = [list(sig[x]) for x in R]
            Vnext = [list(sig[F[x]]) for x in R]
            pred_dim = rank_binary_matrix(Vcols)
            dim_hist[pred_dim] = dim_hist.get(pred_dim, 0) + 1

            # Any linear relation among current predictive states must hold after one step.
            for a in enumerate_relation_vectors(Vcols):
                accum = [0] * len(W)
                for i in range(len(R)):
                    if (a >> i) & 1:
                        accum = [u ^ v for u, v in zip(accum, Vnext[i])]
                if any(accum):
                    dynamics_well_defined_failures += 1
                    break

            # Finite Hankel: rows are d o F^n, n=0..3; columns are reached states.
            hankel_rows = []
            for d in D_BASIS:
                f = d
                for _n in range(NSTATES):
                    hankel_rows.append([((f >> x) & 1) for x in R])
                    f = compose_function_bits(f, F, NSTATES)
            hankel_rank = rank_binary_matrix(hankel_rows)
            if hankel_rank != pred_dim:
                hankel_rank_failures += 1

    result = {
        "status": "PASS" if not any([
            closure_failures,
            quotient_invariance_failures,
            separation_failures,
            dynamics_well_defined_failures,
            hankel_rank_failures,
        ]) else "FAIL",
        "universe": {
            "state_space": "F2^2",
            "deterministic_maps": 256,
            "nonempty_source_subsets_per_map": 15,
            "total_map_source_cases": total_cases,
            "readout_basis": ["x1", "x2"],
        },
        "checks": {
            "koopman_closure_failures": closure_failures,
            "behavioral_quotient_F_invariance_failures": quotient_invariance_failures,
            "behavioral_class_separation_failures": separation_failures,
            "predictive_linear_dynamics_well_defined_failures": dynamics_well_defined_failures,
            "hankel_rank_equals_predictive_dimension_failures": hankel_rank_failures,
        },
        "predictive_dimension_histogram": {str(k): v for k, v in sorted(dim_hist.items())},
        "behavioral_quotient_class_count_histogram": {str(k): v for k, v in sorted(quotient_class_hist.items())},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
