#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import gamma_directional_normalization_regression_v13 as v13


def replicate_bits(
    macro_code: int,
    macro_roles: int,
    replicas: int,
) -> int:
    physical_code = 0

    for role in range(macro_roles):
        bit = (
            macro_code
            >> role
        ) & 1

        for replica in range(replicas):
            physical_index = (
                role * replicas
                + replica
            )
            physical_code |= (
                int(bit)
                << physical_index
            )

    return int(physical_code)


def extract_physical_target_code(
    macro_future_code: int,
    target_roles: tuple[int, ...],
    replicas: int,
) -> int:
    output_code = 0
    position = 0

    for role in target_roles:
        bit = (
            macro_future_code
            >> role
        ) & 1

        for _ in range(replicas):
            output_code |= (
                int(bit)
                << position
            )
            position += 1

    return int(output_code)


def coherent_directional_information(
    kernel: v13.Kernel,
    source_roles: tuple[int, ...],
    target_roles: tuple[int, ...],
    replicas: int,
) -> float:
    total = 0.0

    for target_fixed in range(
        2 ** len(target_roles)
    ):
        joint = np.zeros(
            (
                2 ** len(source_roles),
                2 ** (
                    len(target_roles)
                    * replicas
                ),
            ),
            dtype=np.float64,
        )

        for source_value in range(
            2 ** len(source_roles)
        ):
            macro_bits = np.zeros(
                kernel.n,
                dtype=np.int8,
            )

            for position, role in enumerate(
                source_roles
            ):
                macro_bits[role] = (
                    source_value
                    >> position
                ) & 1

            for position, role in enumerate(
                target_roles
            ):
                macro_bits[role] = (
                    target_fixed
                    >> position
                ) & 1

            macro_code = int(
                sum(
                    int(bit) << index
                    for index, bit in enumerate(
                        macro_bits
                    )
                )
            )

            for future_code, probability in enumerate(
                kernel.transition[
                    macro_code
                ]
            ):
                if probability <= 0.0:
                    continue

                physical_target = (
                    extract_physical_target_code(
                        future_code,
                        target_roles,
                        replicas,
                    )
                )

                joint[
                    source_value,
                    physical_target,
                ] += (
                    probability
                    / (
                        2 ** len(source_roles)
                    )
                )

        total += (
            v13.mutual_information(
                joint
            )
            / (
                2 ** len(target_roles)
            )
        )

    return float(total)


def coherent_cut_gamma(
    kernel: v13.Kernel,
    side_a: tuple[int, ...],
    side_b: tuple[int, ...],
    replicas: int,
) -> dict:
    j_ab = coherent_directional_information(
        kernel,
        side_a,
        side_b,
        replicas,
    )
    j_ba = coherent_directional_information(
        kernel,
        side_b,
        side_a,
        replicas,
    )

    # Under the coherent intervention battery, each replicated role
    # still carries exactly one source bit. The source entropy is
    # therefore the number of macro roles, not the number of physical
    # replica coordinates.
    h_a = float(len(side_a))
    h_b = float(len(side_b))

    gamma = 0.5 * (
        j_ab / h_a
        + j_ba / h_b
    )

    return {
        "j_a_to_b_bits": float(j_ab),
        "j_b_to_a_bits": float(j_ba),
        "source_entropy_a_bits": h_a,
        "source_entropy_b_bits": h_b,
        "physical_source_coordinates_a": int(
            len(side_a) * replicas
        ),
        "physical_source_coordinates_b": int(
            len(side_b) * replicas
        ),
        "gamma_cut": float(gamma),
    }


def coherent_exact_gamma(
    kernel: v13.Kernel,
    replicas: int,
) -> dict:
    rows = []

    for side_a, side_b in kernel.cuts():
        side_a_tuple = tuple(
            sorted(side_a)
        )
        side_b_tuple = tuple(
            sorted(side_b)
        )

        row = coherent_cut_gamma(
            kernel,
            side_a_tuple,
            side_b_tuple,
            replicas,
        )

        rows.append(
            {
                "side_a_macro_roles": list(
                    side_a_tuple
                ),
                "side_b_macro_roles": list(
                    side_b_tuple
                ),
                **row,
            }
        )

    gamma = min(
        row["gamma_cut"]
        for row in rows
    )

    minimizing = [
        row
        for row in rows
        if abs(
            row["gamma_cut"]
            - gamma
        ) <= 1e-12
    ]

    return {
        "gamma": float(gamma),
        "minimizing_cuts": minimizing,
        "all_cuts_bounded": bool(
            all(
                -1e-12
                <= row["gamma_cut"]
                <= 1.0 + 1e-12
                for row in rows
            )
        ),
    }


def evaluate_system(
    kernel: v13.Kernel,
    replicas_list: list[int],
) -> dict:
    latent = kernel.exact_metric(
        "directional"
    )

    rows = []

    for replicas in replicas_list:
        replicated = coherent_exact_gamma(
            kernel,
            replicas,
        )

        rows.append(
            {
                "replicas_per_macro_role": replicas,
                "physical_role_count": int(
                    kernel.n * replicas
                ),
                "gamma": replicated[
                    "gamma"
                ],
                "absolute_difference_from_latent": float(
                    abs(
                        replicated["gamma"]
                        - latent["gamma"]
                    )
                ),
                "all_cuts_bounded": (
                    replicated[
                        "all_cuts_bounded"
                    ]
                ),
                "passed": bool(
                    abs(
                        replicated["gamma"]
                        - latent["gamma"]
                    )
                    <= 1e-12
                ),
            }
        )

    return {
        "system": kernel.name,
        "latent_macro_role_count": (
            kernel.n
        ),
        "latent_directional_gamma": (
            latent["gamma"]
        ),
        "replication_rows": rows,
        "passed": bool(
            all(
                row["passed"]
                and row[
                    "all_cuts_bounded"
                ]
                for row in rows
            )
        ),
    }


def entropy_miscount_diagnostic(
    kernel: v13.Kernel,
    replicas: int,
) -> dict:
    latent = kernel.exact_metric(
        "directional"
    )

    rows = []

    for cut in latent["cuts"]:
        side_a = tuple(
            cut["side_a"]
        )
        side_b = tuple(
            cut["side_b"]
        )

        j_ab = cut[
            "j_a_to_b_bits"
        ]
        j_ba = cut[
            "j_b_to_a_bits"
        ]

        coherent_gamma = 0.5 * (
            j_ab / len(side_a)
            + j_ba / len(side_b)
        )

        incorrect_physical_count_gamma = (
            0.5
            * (
                j_ab
                / (
                    len(side_a)
                    * replicas
                )
                + j_ba
                / (
                    len(side_b)
                    * replicas
                )
            )
        )

        rows.append(
            {
                "side_a": list(side_a),
                "side_b": list(side_b),
                "coherent_source_entropy_gamma": float(
                    coherent_gamma
                ),
                "incorrect_coordinate_count_gamma": float(
                    incorrect_physical_count_gamma
                ),
                "shrink_factor": float(
                    incorrect_physical_count_gamma
                    / coherent_gamma
                ) if coherent_gamma > 0.0 else None,
            }
        )

    correct = min(
        row[
            "coherent_source_entropy_gamma"
        ]
        for row in rows
    )
    incorrect = min(
        row[
            "incorrect_coordinate_count_gamma"
        ]
        for row in rows
    )

    return {
        "replicas_per_macro_role": (
            replicas
        ),
        "correct_gamma": float(
            correct
        ),
        "incorrect_gamma_if_entropy_is_replaced_by_physical_coordinate_count": float(
            incorrect
        ),
        "ratio_incorrect_to_correct": float(
            incorrect / correct
        ) if correct > 0.0 else None,
        "interpretation": (
            "Replication invariance requires H_q(source) from the "
            "actual coherent intervention distribution. Replacing "
            "source entropy with the number of physical coordinates "
            "would spuriously divide Gamma by the replica count."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact quotient/refactorization audit for directional "
            "Gamma under coherent replicated intervention roles."
        )
    )
    parser.add_argument(
        "--output",
        default=(
            "gamma_directional_quotient_invariance_v14.json"
        ),
    )
    args = parser.parse_args()

    systems = [
        v13.ring_copy(5),
        v13.xor_neighbors(5),
        v13.mixed_boolean_5(),
        v13.shared_noise_synergy_3(),
        v13.feedforward_two_role(),
        v13.bidirectional_swap_two_role(),
    ]

    replicas_list = [
        1,
        2,
        3,
        5,
    ]

    rows = [
        evaluate_system(
            kernel,
            replicas_list,
        )
        for kernel in systems
    ]

    diagnostics = {
        kernel.name: (
            entropy_miscount_diagnostic(
                kernel,
                replicas=3,
            )
        )
        for kernel in (
            v13.ring_copy(5),
            v13.xor_neighbors(5),
            v13.mixed_boolean_5(),
        )
    }

    result = {
        "experiment": (
            "gamma_directional_quotient_invariance_v14"
        ),
        "definition_under_test": (
            "Gamma_dir(pi) = 0.5 * "
            "[J(A->B')/H_q(A) + J(B->A')/H_q(B)]."
        ),
        "replication_semantics": (
            "Each macro role is replaced by multiple perfectly coherent "
            "physical copies. The intervention algebra still contains "
            "one independent bit per macro role, so H_q is unchanged."
        ),
        "systems": rows,
        "entropy_miscount_diagnostic": (
            diagnostics
        ),
        "all_replication_invariance_checks_passed": bool(
            all(
                row["passed"]
                for row in rows
            )
        ),
        "scientific_implication": (
            "Directional normalization is representation-invariant "
            "under coherent role replication when source entropy is "
            "computed from the declared intervention distribution q. "
            "Physical coordinate count is not a valid substitute for "
            "H_q."
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

    print(
        json.dumps(
            {
                "output": args.output,
                "all_replication_invariance_checks_passed": (
                    result[
                        "all_replication_invariance_checks_passed"
                    ]
                ),
                "systems": {
                    row["system"]: {
                        "latent_directional_gamma": (
                            row[
                                "latent_directional_gamma"
                            ]
                        ),
                        "replication_gammas": [
                            {
                                "replicas": item[
                                    "replicas_per_macro_role"
                                ],
                                "gamma": item[
                                    "gamma"
                                ],
                                "difference": item[
                                    "absolute_difference_from_latent"
                                ],
                            }
                            for item in row[
                                "replication_rows"
                            ]
                        ],
                    }
                    for row in rows
                },
                "entropy_miscount_diagnostic": (
                    diagnostics
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
