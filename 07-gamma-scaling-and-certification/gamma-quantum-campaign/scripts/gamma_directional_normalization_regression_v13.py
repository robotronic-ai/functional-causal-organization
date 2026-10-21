#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(
        -p * math.log2(p)
        - (1.0 - p) * math.log2(1.0 - p)
    )


def mutual_information(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    total = float(joint.sum())
    if total <= 0.0:
        return 0.0

    joint = joint / total
    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)
    product = p_a @ p_b

    mask = joint > 0.0
    ratio = np.divide(
        joint,
        product,
        out=np.ones_like(joint),
        where=mask,
    )

    return float(
        np.sum(
            joint[mask]
            * np.log2(ratio[mask])
        )
    )


@dataclass
class Kernel:
    n: int
    transition: np.ndarray
    name: str

    def __post_init__(self) -> None:
        self.transition = np.asarray(
            self.transition,
            dtype=np.float64,
        )

        expected = (
            2**self.n,
            2**self.n,
        )
        if self.transition.shape != expected:
            raise ValueError(
                f"{self.name}: expected transition shape "
                f"{expected}, got {self.transition.shape}."
            )

        row_sums = self.transition.sum(axis=1)
        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"{self.name}: transition rows do not sum to one."
            )

    def cuts(self):
        full = set(range(self.n))

        for mask in range(
            1 << (self.n - 1)
        ):
            side_a = {0}

            for index in range(
                1,
                self.n,
            ):
                if mask & (
                    1 << (index - 1)
                ):
                    side_a.add(index)

            if len(side_a) < self.n:
                yield (
                    side_a,
                    full - side_a,
                )

    def directional_information(
        self,
        source: set[int],
        target: set[int],
    ) -> float:
        source = tuple(
            sorted(source)
        )
        target = tuple(
            sorted(target)
        )

        total = 0.0

        for target_fixed in range(
            2 ** len(target)
        ):
            joint = np.zeros(
                (
                    2 ** len(source),
                    2 ** len(target),
                ),
                dtype=np.float64,
            )

            for source_value in range(
                2 ** len(source)
            ):
                x_bits = np.zeros(
                    self.n,
                    dtype=np.int8,
                )

                for position, index in enumerate(
                    source
                ):
                    x_bits[index] = (
                        source_value
                        >> position
                    ) & 1

                for position, index in enumerate(
                    target
                ):
                    x_bits[index] = (
                        target_fixed
                        >> position
                    ) & 1

                x_code = int(
                    sum(
                        int(value) << index
                        for index, value in enumerate(
                            x_bits
                        )
                    )
                )

                for y_code, probability in enumerate(
                    self.transition[
                        x_code
                    ]
                ):
                    if probability <= 0.0:
                        continue

                    target_future = 0

                    for position, index in enumerate(
                        target
                    ):
                        bit = (
                            y_code
                            >> index
                        ) & 1
                        target_future |= (
                            int(bit)
                            << position
                        )

                    joint[
                        source_value,
                        target_future,
                    ] += (
                        probability
                        / (
                            2 ** len(source)
                        )
                    )

            total += (
                mutual_information(
                    joint
                )
                / (
                    2 ** len(target)
                )
            )

        return float(total)

    def old_cut_gamma(
        self,
        side_a: set[int],
        side_b: set[int],
    ) -> dict:
        j_ab = self.directional_information(
            side_a,
            side_b,
        )
        j_ba = self.directional_information(
            side_b,
            side_a,
        )

        denominator = 2.0 * min(
            len(side_a),
            len(side_b),
        )

        value = (
            j_ab + j_ba
        ) / denominator

        return {
            "j_a_to_b_bits": float(j_ab),
            "j_b_to_a_bits": float(j_ba),
            "denominator_bits": float(
                denominator
            ),
            "gamma_cut": float(value),
        }

    def directional_cut_gamma(
        self,
        side_a: set[int],
        side_b: set[int],
    ) -> dict:
        j_ab = self.directional_information(
            side_a,
            side_b,
        )
        j_ba = self.directional_information(
            side_b,
            side_a,
        )

        h_a = float(len(side_a))
        h_b = float(len(side_b))

        eta_ab = (
            j_ab / h_a
            if h_a > 0.0
            else 0.0
        )
        eta_ba = (
            j_ba / h_b
            if h_b > 0.0
            else 0.0
        )

        value = 0.5 * (
            eta_ab + eta_ba
        )

        return {
            "j_a_to_b_bits": float(j_ab),
            "j_b_to_a_bits": float(j_ba),
            "source_entropy_a_bits": h_a,
            "source_entropy_b_bits": h_b,
            "efficiency_a_to_b": float(
                eta_ab
            ),
            "efficiency_b_to_a": float(
                eta_ba
            ),
            "gamma_cut": float(value),
        }

    def exact_metric(
        self,
        kind: str,
    ) -> dict:
        rows = []

        for side_a, side_b in self.cuts():
            if kind == "old":
                result = self.old_cut_gamma(
                    side_a,
                    side_b,
                )
            elif kind == "directional":
                result = (
                    self.directional_cut_gamma(
                        side_a,
                        side_b,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown metric kind: {kind}"
                )

            rows.append(
                {
                    "side_a": sorted(
                        side_a
                    ),
                    "side_b": sorted(
                        side_b
                    ),
                    **result,
                }
            )

        minimum = min(
            row["gamma_cut"]
            for row in rows
        )

        minimizing = [
            row
            for row in rows
            if abs(
                row["gamma_cut"]
                - minimum
            ) <= 1e-12
        ]

        return {
            "gamma": float(minimum),
            "minimizing_cuts": minimizing,
            "all_cuts_bounded": bool(
                all(
                    -1e-12
                    <= row["gamma_cut"]
                    <= 1.0 + 1e-12
                    for row in rows
                )
            ),
            "cuts": rows,
        }


def deterministic_kernel(
    n: int,
    functions,
    name: str,
) -> Kernel:
    transition = np.zeros(
        (2**n, 2**n),
        dtype=np.float64,
    )

    for x_code in range(
        2**n
    ):
        x_bits = np.asarray(
            [
                (
                    x_code
                    >> index
                ) & 1
                for index in range(n)
            ],
            dtype=np.int8,
        )

        y_bits = np.asarray(
            [
                int(function(x_bits))
                for function in functions
            ],
            dtype=np.int8,
        )

        y_code = int(
            sum(
                int(value) << index
                for index, value in enumerate(
                    y_bits
                )
            )
        )

        transition[
            x_code,
            y_code,
        ] = 1.0

    return Kernel(
        n=n,
        transition=transition,
        name=name,
    )


def ring_copy(
    n: int = 5,
) -> Kernel:
    functions = []

    for target in range(n):
        source = (
            target - 1
        ) % n

        functions.append(
            lambda x, source=source: int(
                x[source]
            )
        )

    return deterministic_kernel(
        n,
        functions,
        f"ring_copy_{n}",
    )


def xor_neighbors(
    n: int = 5,
) -> Kernel:
    functions = []

    for target in range(n):
        left = (
            target - 1
        ) % n
        right = (
            target + 1
        ) % n

        functions.append(
            lambda x, left=left, right=right: int(
                x[left]
                ^ x[right]
            )
        )

    return deterministic_kernel(
        n,
        functions,
        f"xor_neighbors_{n}",
    )


def mixed_boolean_5() -> Kernel:
    functions = [
        lambda x: int(
            x[1] ^ x[2]
        ),
        lambda x: int(
            x[0] ^ x[3]
        ),
        lambda x: int(
            x[1] & x[4]
        ),
        lambda x: int(
            x[2]
            ^ (
                x[4]
                & x[0]
            )
        ),
        lambda x: int(
            x[3] | x[1]
        ),
    ]

    return deterministic_kernel(
        5,
        functions,
        "mixed_boolean_5",
    )


def shared_noise_synergy_3() -> Kernel:
    transition = np.zeros(
        (8, 8),
        dtype=np.float64,
    )

    for x_code in range(8):
        x = np.asarray(
            [
                (
                    x_code
                    >> index
                ) & 1
                for index in range(3)
            ],
            dtype=np.int8,
        )

        for random_bit in (
            0,
            1,
        ):
            y = np.asarray(
                [
                    random_bit ^ x[1],
                    random_bit ^ x[2],
                    random_bit ^ x[0],
                ],
                dtype=np.int8,
            )

            y_code = int(
                sum(
                    int(value) << index
                    for index, value in enumerate(
                        y
                    )
                )
            )

            transition[
                x_code,
                y_code,
            ] += 0.5

    return Kernel(
        3,
        transition,
        "shared_noise_synergy_3",
    )


def identity_kernel(
    n: int,
) -> Kernel:
    return deterministic_kernel(
        n,
        [
            (
                lambda x, index=index: int(
                    x[index]
                )
            )
            for index in range(n)
        ],
        f"identity_{n}",
    )


def feedforward_two_role() -> Kernel:
    return deterministic_kernel(
        2,
        [
            lambda x: 0,
            lambda x: int(
                x[0]
            ),
        ],
        "feedforward_two_role",
    )


def bidirectional_swap_two_role() -> Kernel:
    return deterministic_kernel(
        2,
        [
            lambda x: int(
                x[1]
            ),
            lambda x: int(
                x[0]
            ),
        ],
        "bidirectional_swap_two_role",
    )


def partial_reverse_two_role(
    reverse_strength: float,
) -> Kernel:
    p = float(
        reverse_strength
    )

    if not (
        0.0
        <= p
        <= 1.0
    ):
        raise ValueError(
            "reverse_strength must lie in [0,1]."
        )

    transition = np.zeros(
        (4, 4),
        dtype=np.float64,
    )

    for x_code in range(4):
        a = (
            x_code
            >> 0
        ) & 1
        b = (
            x_code
            >> 1
        ) & 1

        # B' always receives A.
        future_b = a

        # A' is B with probability p, and an independent fair bit
        # with probability 1-p.
        for future_a in (
            0,
            1,
        ):
            probability = (
                p
                * float(
                    future_a == b
                )
                + (
                    1.0 - p
                )
                * 0.5
            )

            y_code = (
                int(future_a)
                | (
                    int(future_b)
                    << 1
                )
            )

            transition[
                x_code,
                y_code,
            ] += probability

    return Kernel(
        2,
        transition,
        (
            "partial_reverse_two_role_"
            f"{p:.3f}"
        ),
    )


def random_stochastic_kernel(
    n: int,
    seed: int,
) -> Kernel:
    rng = np.random.default_rng(
        seed
    )

    transition = rng.dirichlet(
        np.ones(
            2**n,
            dtype=np.float64,
        ),
        size=2**n,
    )

    return Kernel(
        n,
        transition,
        f"random_stochastic_n{n}_seed{seed}",
    )


def permute_kernel(
    kernel: Kernel,
    permutation: tuple[int, ...],
) -> Kernel:
    if sorted(
        permutation
    ) != list(
        range(
            kernel.n
        )
    ):
        raise ValueError(
            "Permutation is invalid."
        )

    transition = np.zeros_like(
        kernel.transition
    )

    def permute_code(
        code: int,
    ) -> int:
        bits = [
            (
                code
                >> index
            ) & 1
            for index in range(
                kernel.n
            )
        ]

        permuted = [
            bits[
                permutation[
                    index
                ]
            ]
            for index in range(
                kernel.n
            )
        ]

        return int(
            sum(
                bit << index
                for index, bit in enumerate(
                    permuted
                )
            )
        )

    for x_code in range(
        2**kernel.n
    ):
        x_new = permute_code(
            x_code
        )

        for y_code, probability in enumerate(
            kernel.transition[
                x_code
            ]
        ):
            if probability <= 0.0:
                continue

            y_new = permute_code(
                y_code
            )

            transition[
                x_new,
                y_new,
            ] += probability

    return Kernel(
        kernel.n,
        transition,
        (
            kernel.name
            + "_permuted"
        ),
    )


def evaluate_reference_suite() -> dict:
    systems = [
        ring_copy(5),
        xor_neighbors(5),
        mixed_boolean_5(),
        shared_noise_synergy_3(),
    ]

    expected_old = {
        "ring_copy_5": 0.5,
        "xor_neighbors_5": 1.0,
        "mixed_boolean_5": (
            0.6639097655573916
        ),
        "shared_noise_synergy_3": 0.5,
    }

    rows = []

    for kernel in systems:
        old = kernel.exact_metric(
            "old"
        )
        directional = kernel.exact_metric(
            "directional"
        )

        rows.append(
            {
                "system": kernel.name,
                "old_gamma": old[
                    "gamma"
                ],
                "expected_old_gamma": (
                    expected_old[
                        kernel.name
                    ]
                ),
                "old_reference_match": bool(
                    abs(
                        old["gamma"]
                        - expected_old[
                            kernel.name
                        ]
                    )
                    <= 1e-12
                ),
                "directional_gamma": (
                    directional[
                        "gamma"
                    ]
                ),
                "directional_minimizing_cuts": [
                    {
                        "side_a": row[
                            "side_a"
                        ],
                        "side_b": row[
                            "side_b"
                        ],
                        "j_a_to_b_bits": row[
                            "j_a_to_b_bits"
                        ],
                        "j_b_to_a_bits": row[
                            "j_b_to_a_bits"
                        ],
                        "gamma_cut": row[
                            "gamma_cut"
                        ],
                    }
                    for row in (
                        directional[
                            "minimizing_cuts"
                        ]
                    )
                ],
                "directional_bounded": (
                    directional[
                        "all_cuts_bounded"
                    ]
                ),
            }
        )

    return {
        "rows": rows,
        "all_old_reference_values_reproduced": bool(
            all(
                row[
                    "old_reference_match"
                ]
                for row in rows
            )
        ),
        "all_directional_values_bounded": bool(
            all(
                row[
                    "directional_bounded"
                ]
                for row in rows
            )
        ),
    }


def evaluate_anchor_properties() -> dict:
    anchors = {
        "disconnected_identity_3": (
            identity_kernel(3),
            0.0,
        ),
        "feedforward_two_role": (
            feedforward_two_role(),
            0.5,
        ),
        "bidirectional_swap_two_role": (
            bidirectional_swap_two_role(),
            1.0,
        ),
    }

    rows = {}

    for name, (
        kernel,
        expected,
    ) in anchors.items():
        result = kernel.exact_metric(
            "directional"
        )

        rows[name] = {
            "gamma": result[
                "gamma"
            ],
            "expected": expected,
            "passed": bool(
                abs(
                    result["gamma"]
                    - expected
                )
                <= 1e-12
            ),
        }

    return {
        "rows": rows,
        "passed": bool(
            all(
                row["passed"]
                for row in (
                    rows.values()
                )
            )
        ),
    }


def evaluate_monotonicity() -> dict:
    strengths = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    rows = []

    for strength in strengths:
        kernel = (
            partial_reverse_two_role(
                strength
            )
        )
        result = kernel.exact_metric(
            "directional"
        )

        expected_reverse_information = (
            1.0
            - h2(
                (
                    1.0
                    - strength
                )
                / 2.0
            )
        )
        expected_gamma = 0.5 * (
            1.0
            + expected_reverse_information
        )

        rows.append(
            {
                "reverse_strength": strength,
                "gamma": result[
                    "gamma"
                ],
                "expected_gamma": float(
                    expected_gamma
                ),
                "expected_match": bool(
                    abs(
                        result[
                            "gamma"
                        ]
                        - expected_gamma
                    )
                    <= 1e-12
                ),
            }
        )

    values = [
        row["gamma"]
        for row in rows
    ]

    monotonic = all(
        values[index + 1]
        >= values[index]
        - 1e-12
        for index in range(
            len(values) - 1
        )
    )

    return {
        "rows": rows,
        "monotonic_non_decreasing": bool(
            monotonic
        ),
        "analytic_values_match": bool(
            all(
                row[
                    "expected_match"
                ]
                for row in rows
            )
        ),
        "passed": bool(
            monotonic
            and all(
                row[
                    "expected_match"
                ]
                for row in rows
            )
        ),
    }


def evaluate_permutation_invariance() -> dict:
    systems = [
        ring_copy(5),
        xor_neighbors(5),
        mixed_boolean_5(),
        shared_noise_synergy_3(),
    ]

    permutations = {
        5: (
            2,
            4,
            1,
            0,
            3,
        ),
        3: (
            2,
            0,
            1,
        ),
    }

    rows = []

    for kernel in systems:
        original = kernel.exact_metric(
            "directional"
        )["gamma"]

        permuted = permute_kernel(
            kernel,
            permutations[
                kernel.n
            ],
        ).exact_metric(
            "directional"
        )["gamma"]

        rows.append(
            {
                "system": kernel.name,
                "original_gamma": float(
                    original
                ),
                "permuted_gamma": float(
                    permuted
                ),
                "absolute_difference": float(
                    abs(
                        original
                        - permuted
                    )
                ),
                "passed": bool(
                    abs(
                        original
                        - permuted
                    )
                    <= 1e-12
                ),
            }
        )

    return {
        "rows": rows,
        "passed": bool(
            all(
                row["passed"]
                for row in rows
            )
        ),
    }


def evaluate_random_boundedness(
    seeds: list[int],
) -> dict:
    rows = []

    for n in (
        3,
        4,
    ):
        for seed in seeds:
            kernel = random_stochastic_kernel(
                n,
                seed,
            )
            result = kernel.exact_metric(
                "directional"
            )

            rows.append(
                {
                    "system": kernel.name,
                    "gamma": result[
                        "gamma"
                    ],
                    "all_cuts_bounded": result[
                        "all_cuts_bounded"
                    ],
                }
            )

    return {
        "rows": rows,
        "passed": bool(
            all(
                row[
                    "all_cuts_bounded"
                ]
                and 0.0
                <= row["gamma"]
                <= 1.0
                for row in rows
            )
        ),
    }


def evaluate_ring_size_sweep() -> dict:
    rows = []

    for n in range(
        3,
        9,
    ):
        kernel = ring_copy(n)

        old = kernel.exact_metric(
            "old"
        )
        directional = kernel.exact_metric(
            "directional"
        )

        rows.append(
            {
                "role_count": n,
                "old_gamma": old[
                    "gamma"
                ],
                "directional_gamma": (
                    directional[
                        "gamma"
                    ]
                ),
                "directional_minimizing_cut_sizes": [
                    [
                        len(row["side_a"]),
                        len(row["side_b"]),
                    ]
                    for row in (
                        directional[
                            "minimizing_cuts"
                        ]
                    )
                ],
            }
        )

    return {
        "rows": rows,
        "interpretation": (
            "This sweep is diagnostic, not a pass/fail criterion. "
            "It shows how the two normalizations respond to cut "
            "imbalance as system size changes."
        ),
    }


def evaluate_equal_capacity_agreement() -> dict:
    systems = [
        ring_copy(4),
        xor_neighbors(4),
        random_stochastic_kernel(
            4,
            301,
        ),
        random_stochastic_kernel(
            4,
            302,
        ),
    ]

    rows = []

    for kernel in systems:
        for side_a, side_b in kernel.cuts():
            if len(side_a) != len(side_b):
                continue

            old = kernel.old_cut_gamma(
                side_a,
                side_b,
            )["gamma_cut"]

            directional = (
                kernel.directional_cut_gamma(
                    side_a,
                    side_b,
                )["gamma_cut"]
            )

            rows.append(
                {
                    "system": kernel.name,
                    "side_a": sorted(
                        side_a
                    ),
                    "side_b": sorted(
                        side_b
                    ),
                    "old_gamma_cut": float(
                        old
                    ),
                    "directional_gamma_cut": float(
                        directional
                    ),
                    "absolute_difference": float(
                        abs(
                            old
                            - directional
                        )
                    ),
                    "passed": bool(
                        abs(
                            old
                            - directional
                        )
                        <= 1e-12
                    ),
                }
            )

    return {
        "rows": rows,
        "passed": bool(
            all(
                row["passed"]
                for row in rows
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact regression audit for directional-entropy "
            "normalization of Gamma."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "gamma_directional_normalization_regression_v13.json"
        ),
    )
    args = parser.parse_args()

    reference = (
        evaluate_reference_suite()
    )
    anchors = (
        evaluate_anchor_properties()
    )
    monotonicity = (
        evaluate_monotonicity()
    )
    permutation = (
        evaluate_permutation_invariance()
    )
    boundedness = (
        evaluate_random_boundedness(
            [
                401,
                402,
                403,
                404,
                405,
            ]
        )
    )
    ring_sweep = (
        evaluate_ring_size_sweep()
    )
    equal_capacity = (
        evaluate_equal_capacity_agreement()
    )

    hard_checks = {
        "old_reference_reproduction": (
            reference[
                "all_old_reference_values_reproduced"
            ]
        ),
        "directional_reference_boundedness": (
            reference[
                "all_directional_values_bounded"
            ]
        ),
        "anchor_systems": anchors[
            "passed"
        ],
        "monotonic_added_reverse_coupling": (
            monotonicity[
                "passed"
            ]
        ),
        "permutation_invariance": (
            permutation[
                "passed"
            ]
        ),
        "random_kernel_boundedness": (
            boundedness[
                "passed"
            ]
        ),
        "equal_capacity_agreement_with_old_metric": (
            equal_capacity[
                "passed"
            ]
        ),
    }

    result = {
        "experiment": (
            "gamma_directional_normalization_regression_v13"
        ),
        "definition_under_test": (
            "Gamma_dir(pi) = 0.5 * "
            "[J(A->B')/H_q(A) + J(B->A')/H_q(B)]."
        ),
        "uniform_binary_role_battery": (
            "For these exact tests, H_q(side) equals the number "
            "of independent binary roles on that side."
        ),
        "reference_suite": reference,
        "anchor_properties": anchors,
        "monotonicity": monotonicity,
        "permutation_invariance": (
            permutation
        ),
        "random_boundedness": (
            boundedness
        ),
        "equal_capacity_agreement": (
            equal_capacity
        ),
        "ring_size_sweep": (
            ring_sweep
        ),
        "hard_checks": hard_checks,
        "all_hard_checks_passed": bool(
            all(
                hard_checks.values()
            )
        ),
        "semantic_change_flag": True,
        "semantic_change_note": (
            "The directional normalization intentionally differs from "
            "the previous minimum-side-capacity normalization on "
            "unequal-capacity cuts. This is not treated as a software "
            "failure. It requires a scientific decision about whether "
            "Gamma should measure smaller-side saturation or the mean "
            "efficiency of both causal directions."
        ),
    }

    Path(
        args.output
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    compact = {
        "output": args.output,
        "all_hard_checks_passed": (
            result[
                "all_hard_checks_passed"
            ]
        ),
        "hard_checks": (
            hard_checks
        ),
        "reference_directional_gamma": {
            row["system"]: row[
                "directional_gamma"
            ]
            for row in (
                reference["rows"]
            )
        },
        "ring_size_sweep": (
            ring_sweep["rows"]
        ),
        "semantic_change_flag": True,
    }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
