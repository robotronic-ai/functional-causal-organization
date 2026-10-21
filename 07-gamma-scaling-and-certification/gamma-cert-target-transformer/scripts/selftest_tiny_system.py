#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gamma_certificate_from_trials import certify


def ring_transition(x: np.ndarray) -> np.ndarray:
    return np.roll(x, shift=1, axis=1)


def make_split(
    n_roles: int,
    n_contexts: int,
    repeats_per_context: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    context = []

    for context_id in range(n_contexts):
        for _ in range(repeats_per_context):
            rows.append(
                rng.integers(
                    0,
                    2,
                    size=n_roles,
                    dtype=np.int8,
                )
            )
            context.append(context_id)

    x = np.asarray(rows, dtype=np.int8)
    y = ring_transition(x)
    return x, y, np.asarray(context, dtype=np.int64)


def main() -> None:
    config = {
        "delta_lower": 0.025,
        "delta_upper": 0.025,
        "top_k_per_source": 2,
        "clip_probability": 0.001,
        "max_candidate_cuts": 24,
        "v1_max_absolute_width": 0.10,
    }

    split_specs = [
        (0, 11001),
        (1, 12001),
        (2, 13001),
        (3, 14001),
    ]

    all_x = []
    all_y = []
    all_context = []
    all_split = []

    for split_id, seed in split_specs:
        x, y, context = make_split(
            n_roles=10,
            n_contexts=8,
            repeats_per_context=300,
            seed=seed,
        )
        all_x.append(x)
        all_y.append(y)
        all_context.append(context)
        all_split.append(
            np.full(len(x), split_id, dtype=np.int8)
        )

    trials = {
        "x": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "context": np.concatenate(all_context),
        "split": np.concatenate(all_split),
    }

    quotient_gate = {
        "passed": True,
        "method": "exact_identity_macrostate",
        "holdout_factorization_violations": 0,
    }

    result = certify(trials, config, quotient_gate)
    exact_gamma = 0.2
    result["selftest_exact_gamma"] = exact_gamma
    result["selftest_interval_covers_exact"] = bool(
        result.get("gamma_lower", 0.0) <= exact_gamma <= result.get("gamma_upper", 1.0)
    )
    result["selftest_pass"] = result["selftest_interval_covers_exact"]

    output = Path(__file__).resolve().parent / "selftest_results.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
