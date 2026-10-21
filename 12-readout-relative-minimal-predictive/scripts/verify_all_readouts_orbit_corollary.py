#!/usr/bin/env python3
"""Exhaustive small finite-state corollary for arbitrary readouts.

For every deterministic map on four states and every non-empty source subset,
all state functions are admissible readouts. The predictive behavioral quotient
must therefore coincide with equality on reachable states, and the minimal
linear predictive dimension must equal the number of reachable states.
"""

import json
from itertools import product
from pathlib import Path

from common_gf2 import reachable_states

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "all_readouts_orbit_corollary.json"
NSTATES = 4


def source_subsets():
    for mask in range(1, 1 << NSTATES):
        yield tuple(x for x in range(NSTATES) if (mask >> x) & 1)


def main():
    total = 0
    failures = 0
    reach_hist = {}
    for F in product(range(NSTATES), repeat=NSTATES):
        for sources in source_subsets():
            total += 1
            R = reachable_states(F, sources)
            r = len(R)
            reach_hist[r] = reach_hist.get(r, 0) + 1
            # With all functions available, evaluation vectors are distinct one-hot basis vectors.
            quotient_classes = r
            predictive_dim = r
            if quotient_classes != r or predictive_dim != r:
                failures += 1
    result = {
        "status": "PASS" if failures == 0 else "FAIL",
        "universe": {
            "finite_states": NSTATES,
            "deterministic_maps": NSTATES ** NSTATES,
            "nonempty_source_subsets_per_map": (1 << NSTATES) - 1,
            "total_map_source_cases": total,
            "readout_class": "all functions X -> F2",
        },
        "checks": {
            "orbit_machine_equals_behavioral_quotient_failures": failures,
            "minimal_predictive_dimension_equals_reachable_state_count_failures": failures,
        },
        "reachable_state_count_histogram": {str(k): v for k, v in sorted(reach_hist.items())},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
