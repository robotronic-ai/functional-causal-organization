#!/usr/bin/env python3
"""Exhaustive controlled finite-state verification of the predictive theorem.

Universe:
- three abstract states,
- two deterministic controls a,b,
- all 27^2 = 729 ordered pairs of transition maps,
- all 7 non-empty source subsets,
- one declared binary readout d = 1[state == 1].

The Koopman closure is taken under both controls and therefore under every
finite control word.
"""

import json
from itertools import product
from pathlib import Path

from common_gf2 import function_to_bits, gf2_rank_rows, rank_binary_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controlled_finite_theorem.json"
NSTATES = 3
READOUT = function_to_bits([0, 1, 0])


def all_maps():
    return list(product(range(NSTATES), repeat=NSTATES))


def source_subsets():
    for mask in range(1, 1 << NSTATES):
        yield tuple(x for x in range(NSTATES) if (mask >> x) & 1)


def compose_bits(f_bits, F):
    return function_to_bits([((f_bits >> F[x]) & 1) for x in range(NSTATES)])


def closure_functions(Fs):
    """Return linear basis and discovered word-generated functions."""
    basis = []
    generators = {READOUT: ""}
    queue = [READOUT]

    def add_basis(v):
        before = gf2_rank_rows(basis, NSTATES)
        after = gf2_rank_rows(basis + [v], NSTATES)
        if after > before:
            basis.append(v)

    add_basis(READOUT)
    while queue:
        f = queue.pop(0)
        word = generators[f]
        for label, F in enumerate(Fs):
            uf = compose_bits(f, F)
            if uf not in generators:
                generators[uf] = word + str(label)
                queue.append(uf)
            add_basis(uf)
    return basis, generators


def reachable(Fs, sources):
    seen = set(sources)
    queue = list(sources)
    while queue:
        x = queue.pop(0)
        for F in Fs:
            y = F[x]
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return sorted(seen)


def sig(x, basis):
    return tuple((f >> x) & 1 for f in basis)


def relation_masks(columns):
    m = len(columns)
    width = len(columns[0]) if columns else 0
    for mask in range(1 << m):
        acc = [0] * width
        for i in range(m):
            if (mask >> i) & 1:
                acc = [a ^ b for a, b in zip(acc, columns[i])]
        if not any(acc):
            yield mask


def main():
    maps = all_maps()
    total = 0
    closure_failures = 0
    congruence_failures = 0
    separation_failures = 0
    dynamics_failures = 0
    hankel_failures = 0
    dim_hist = {}

    for Fa in maps:
        for Fb in maps:
            Fs = (Fa, Fb)
            W, word_generators = closure_functions(Fs)

            for f in W:
                for F in Fs:
                    uf = compose_bits(f, F)
                    if gf2_rank_rows(W + [uf], NSTATES) != gf2_rank_rows(W, NSTATES):
                        closure_failures += 1

            for sources in source_subsets():
                total += 1
                R = reachable(Fs, sources)
                signatures = {x: sig(x, W) for x in R}
                classes = {}
                for x in R:
                    classes.setdefault(signatures[x], []).append(x)

                # Congruence under each control.
                for xs in classes.values():
                    for x in xs:
                        for y in xs:
                            for F in Fs:
                                if signatures[F[x]] != signatures[F[y]]:
                                    congruence_failures += 1

                # Distinct classes must differ on at least one word-generated original readout.
                reps = [xs[0] for xs in classes.values()]
                for i, x in enumerate(reps):
                    for y in reps[i + 1:]:
                        if not any(((f >> x) & 1) != ((f >> y) & 1) for f in word_generators):
                            separation_failures += 1

                Vcols = [list(signatures[x]) for x in R]
                pred_dim = rank_binary_matrix(Vcols)
                dim_hist[pred_dim] = dim_hist.get(pred_dim, 0) + 1

                # Each control must induce a well-defined linear map on predictive states.
                for F in Fs:
                    Vnext = [list(signatures[F[x]]) for x in R]
                    for a in relation_masks(Vcols):
                        acc = [0] * len(W)
                        for i in range(len(R)):
                            if (a >> i) & 1:
                                acc = [u ^ v for u, v in zip(acc, Vnext[i])]
                        if any(acc):
                            dynamics_failures += 1
                            break

                # Controlled Hankel rows: all discovered d o F_word functions.
                hrows = [[((f >> x) & 1) for x in R] for f in word_generators]
                if rank_binary_matrix(hrows) != pred_dim:
                    hankel_failures += 1

    failures = sum([
        closure_failures,
        congruence_failures,
        separation_failures,
        dynamics_failures,
        hankel_failures,
    ])
    result = {
        "status": "PASS" if failures == 0 else "FAIL",
        "universe": {
            "states": 3,
            "controls": 2,
            "transition_map_pairs": len(maps) ** 2,
            "nonempty_source_subsets_per_pair": (1 << NSTATES) - 1,
            "total_pair_source_cases": total,
            "declared_readout": "indicator(state == 1)",
            "field": "F2",
        },
        "checks": {
            "multi_koopman_closure_failures": closure_failures,
            "controlled_behavioral_congruence_failures": congruence_failures,
            "behavioral_separation_failures": separation_failures,
            "controlled_predictive_dynamics_well_defined_failures": dynamics_failures,
            "controlled_hankel_rank_failures": hankel_failures,
        },
        "minimal_predictive_dimension_histogram": {str(k): v for k, v in sorted(dim_hist.items())},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
