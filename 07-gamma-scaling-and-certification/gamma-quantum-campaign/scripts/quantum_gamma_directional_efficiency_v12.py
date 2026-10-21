#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import quantum_gamma_kv_experiment_v1 as v1
import quantum_gamma_block_multihorizon_v10 as v10


def parse_int_list(value: str) -> list[int]:
    result = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )
    if not result or min(result) < 1:
        raise ValueError(
            "All list values must be positive integers."
        )
    return result


def cut_name(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> str:
    return (
        "+".join(
            v10.BLOCK_NAMES[index]
            for index in left
        )
        + " | "
        + "+".join(
            v10.BLOCK_NAMES[index]
            for index in right
        )
    )


def directional_efficiency_gamma(
    classes: dict,
    horizon: int,
) -> dict:
    rows = []

    for left, right in v10.CUTS:
        j_left_to_right = v10.directional_information(
            classes=classes,
            source_indices=left,
            target_indices=right,
            horizon=horizon,
        )
        j_right_to_left = v10.directional_information(
            classes=classes,
            source_indices=right,
            target_indices=left,
            horizon=horizon,
        )

        # The intervention battery is uniform over independent binary
        # block assignments. Therefore each source block contributes
        # exactly one bit of intervention entropy.
        h_left = float(len(left))
        h_right = float(len(right))

        efficiency_left_to_right = (
            j_left_to_right / h_left
            if h_left > 0.0
            else 0.0
        )
        efficiency_right_to_left = (
            j_right_to_left / h_right
            if h_right > 0.0
            else 0.0
        )

        gamma_cut = 0.5 * (
            efficiency_left_to_right
            + efficiency_right_to_left
        )

        rows.append(
            {
                "cut": cut_name(left, right),
                "left_blocks": [
                    v10.BLOCK_NAMES[index]
                    for index in left
                ],
                "right_blocks": [
                    v10.BLOCK_NAMES[index]
                    for index in right
                ],
                "source_entropy_left_bits": h_left,
                "source_entropy_right_bits": h_right,
                "j_left_to_right_bits": float(
                    j_left_to_right
                ),
                "j_right_to_left_bits": float(
                    j_right_to_left
                ),
                "efficiency_left_to_right": float(
                    efficiency_left_to_right
                ),
                "efficiency_right_to_left": float(
                    efficiency_right_to_left
                ),
                "gamma_cut": float(
                    gamma_cut
                ),
                "bounded": bool(
                    -1e-12
                    <= gamma_cut
                    <= 1.0 + 1e-12
                ),
            }
        )

    gamma = min(
        row["gamma_cut"]
        for row in rows
    )

    minimizing = [
        row["cut"]
        for row in rows
        if abs(
            row["gamma_cut"] - gamma
        ) <= 1e-12
    ]

    return {
        "gamma": float(gamma),
        "minimizing_cuts": minimizing,
        "all_cuts_bounded": bool(
            all(row["bounded"] for row in rows)
        ),
        "cuts": rows,
    }


def summarize_contexts(
    rows: list[dict],
) -> dict:
    gamma_values = np.asarray(
        [
            row["gamma"]
            for row in rows
        ],
        dtype=np.float64,
    )

    minimizing = Counter()
    for row in rows:
        for cut in row["minimizing_cuts"]:
            minimizing[cut] += 1

    cut_summary = []

    for cut_index in range(
        len(v10.CUTS)
    ):
        cut_rows = [
            row["cuts"][cut_index]
            for row in rows
        ]

        cut_summary.append(
            {
                "cut": cut_rows[0]["cut"],
                "mean_j_left_to_right_bits": float(
                    np.mean(
                        [
                            row[
                                "j_left_to_right_bits"
                            ]
                            for row in cut_rows
                        ]
                    )
                ),
                "mean_j_right_to_left_bits": float(
                    np.mean(
                        [
                            row[
                                "j_right_to_left_bits"
                            ]
                            for row in cut_rows
                        ]
                    )
                ),
                "mean_efficiency_left_to_right": float(
                    np.mean(
                        [
                            row[
                                "efficiency_left_to_right"
                            ]
                            for row in cut_rows
                        ]
                    )
                ),
                "mean_efficiency_right_to_left": float(
                    np.mean(
                        [
                            row[
                                "efficiency_right_to_left"
                            ]
                            for row in cut_rows
                        ]
                    )
                ),
                "mean_gamma_cut": float(
                    np.mean(
                        [
                            row["gamma_cut"]
                            for row in cut_rows
                        ]
                    )
                ),
                "minimum_gamma_cut": float(
                    np.min(
                        [
                            row["gamma_cut"]
                            for row in cut_rows
                        ]
                    )
                ),
                "maximum_gamma_cut": float(
                    np.max(
                        [
                            row["gamma_cut"]
                            for row in cut_rows
                        ]
                    )
                ),
            }
        )

    return {
        "mean_context_gamma": float(
            gamma_values.mean()
        ),
        "median_context_gamma": float(
            np.median(gamma_values)
        ),
        "minimum_context_gamma": float(
            gamma_values.min()
        ),
        "maximum_context_gamma": float(
            gamma_values.max()
        ),
        "context_gamma_values": [
            float(value)
            for value in gamma_values.tolist()
        ],
        "minimizing_cut_frequency": dict(
            minimizing
        ),
        "all_contexts_bounded": bool(
            all(
                row["all_cuts_bounded"]
                for row in rows
            )
        ),
        "cuts": cut_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a closure-invariant bidirectional causal "
            "efficiency normalization for the three-block Quantum "
            "intervention battery."
        )
    )
    parser.add_argument(
        "--root",
        default="/mnt/c/Users/Merien/pycharm/llama/quantum",
    )
    parser.add_argument(
        "--model-dir",
        default=(
            "/mnt/c/Users/Merien/pycharm/llama/quantum/"
            "quantum_vibe_thinker"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12122034,
    )
    parser.add_argument(
        "--flow",
        choices=("s_attn",),
        default="s_attn",
    )
    parser.add_argument(
        "--contexts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--closure-depths",
        default="1,2,4,8",
    )
    parser.add_argument(
        "--evaluation-horizons",
        default="1,2,4,8",
    )
    parser.add_argument(
        "--base-tolerance",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--output",
        default=(
            "quantum_gamma_directional_efficiency_v12.json"
        ),
    )
    args = parser.parse_args()

    closure_depths = parse_int_list(
        args.closure_depths
    )
    evaluation_horizons = parse_int_list(
        args.evaluation_horizons
    )

    maximum_depth = max(
        max(closure_depths),
        max(evaluation_horizons),
    )

    v1.set_determinism(
        args.seed
    )

    root = Path(
        args.root
    ).resolve()
    model_dir = Path(
        args.model_dir
    ).resolve()

    model, tokenizer, config = v1.load_model(
        model_dir,
        root,
        args.device,
    )
    n_layers = len(
        model.layers
    )

    contexts = v10.prepare_contexts(
        model=model,
        tokenizer=tokenizer,
        flow=args.flow,
        n_layers=n_layers,
        device=args.device,
        context_count=args.contexts,
        max_prompt_tokens=(
            args.max_prompt_tokens
        ),
    )

    context_rollouts = []
    maximum_noise = 0.0

    for context in contexts:
        noise = v10.maximum_repeat_noise(
            model=model,
            context=context,
            flow=args.flow,
            n_layers=n_layers,
            device=args.device,
            max_horizon=maximum_depth,
        )
        tolerance = max(
            float(args.base_tolerance),
            10.0 * float(noise),
        )
        maximum_noise = max(
            maximum_noise,
            float(noise),
        )

        rollouts = {
            pattern: v10.rollout_pattern(
                model=model,
                context=context,
                bits=pattern,
                flow=args.flow,
                n_layers=n_layers,
                device=args.device,
                max_horizon=maximum_depth,
            )
            for pattern in v10.PATTERNS
        }

        context_rollouts.append(
            {
                "context_id": int(
                    context["context_id"]
                ),
                "word_a": context["word_a"],
                "word_b": context["word_b"],
                "tolerance": float(
                    tolerance
                ),
                "rollouts": rollouts,
            }
        )

    by_closure = {}

    for closure_depth in closure_depths:
        closure_horizons = list(
            range(1, closure_depth + 1)
        )

        evaluation = {}

        for horizon in evaluation_horizons:
            if horizon > closure_depth:
                continue

            context_rows = []

            for item in context_rollouts:
                classes = (
                    v10.build_global_horizon_classes(
                        rollouts=item[
                            "rollouts"
                        ],
                        horizons=closure_horizons,
                        tolerance=item[
                            "tolerance"
                        ],
                    )
                )

                gamma = (
                    directional_efficiency_gamma(
                        classes=classes,
                        horizon=horizon,
                    )
                )

                context_rows.append(
                    {
                        "context_id": item[
                            "context_id"
                        ],
                        "word_a": item[
                            "word_a"
                        ],
                        "word_b": item[
                            "word_b"
                        ],
                        **gamma,
                    }
                )

            evaluation[
                str(horizon)
            ] = {
                "contexts": context_rows,
                "summary": summarize_contexts(
                    context_rows
                ),
            }

        by_closure[
            str(closure_depth)
        ] = {
            "closure_depth": int(
                closure_depth
            ),
            "evaluation": evaluation,
        }

    closure_invariance = {}

    for horizon in evaluation_horizons:
        eligible = [
            depth
            for depth in closure_depths
            if depth >= horizon
        ]

        if not eligible:
            continue

        gamma_means = [
            by_closure[str(depth)][
                "evaluation"
            ][str(horizon)][
                "summary"
            ][
                "mean_context_gamma"
            ]
            for depth in eligible
        ]

        cut_records = []

        for cut_index in range(
            len(v10.CUTS)
        ):
            rows = [
                by_closure[str(depth)][
                    "evaluation"
                ][str(horizon)][
                    "summary"
                ][
                    "cuts"
                ][cut_index]
                for depth in eligible
            ]

            cut_records.append(
                {
                    "cut": rows[0]["cut"],
                    "j_left_to_right_range": float(
                        max(
                            row[
                                "mean_j_left_to_right_bits"
                            ]
                            for row in rows
                        )
                        - min(
                            row[
                                "mean_j_left_to_right_bits"
                            ]
                            for row in rows
                        )
                    ),
                    "j_right_to_left_range": float(
                        max(
                            row[
                                "mean_j_right_to_left_bits"
                            ]
                            for row in rows
                        )
                        - min(
                            row[
                                "mean_j_right_to_left_bits"
                            ]
                            for row in rows
                        )
                    ),
                    "gamma_cut_range": float(
                        max(
                            row[
                                "mean_gamma_cut"
                            ]
                            for row in rows
                        )
                        - min(
                            row[
                                "mean_gamma_cut"
                            ]
                            for row in rows
                        )
                    ),
                    "gamma_cut_values": [
                        float(
                            row[
                                "mean_gamma_cut"
                            ]
                        )
                        for row in rows
                    ],
                }
            )

        closure_invariance[
            str(horizon)
        ] = {
            "closure_depths": eligible,
            "mean_context_gamma_values": [
                float(value)
                for value in gamma_means
            ],
            "mean_context_gamma_range": float(
                max(gamma_means)
                - min(gamma_means)
            ),
            "cuts": cut_records,
        }

    result = {
        "experiment": (
            "quantum_gamma_directional_efficiency_v12"
        ),
        "flow": args.flow,
        "blocks": v10.BLOCKS,
        "context_count": len(
            contexts
        ),
        "maximum_repeat_noise": float(
            maximum_noise
        ),
        "definition": (
            "For each cut, Gamma_bidirectional is one half of the "
            "sum of the two directional causal-information efficiencies: "
            "J(A->B')/H_q(A) and J(B->A')/H_q(B)."
        ),
        "boundedness": (
            "Each directional mutual information is bounded by the "
            "entropy of its intervened source, so every directional "
            "efficiency and every Gamma cut lie in [0,1]."
        ),
        "relation_to_previous_normalization": (
            "When the two sides have equal source entropy, this "
            "normalization equals (J_forward + J_reverse)/(2H), "
            "which is the equal-capacity form of the previous metric."
        ),
        "source_entropy": (
            "The present q is uniform over independent binary block "
            "assignments, so H_q(side) equals the number of source "
            "blocks on that side in bits."
        ),
        "closures": by_closure,
        "closure_invariance": (
            closure_invariance
        ),
    }

    Path(args.output).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    compact = {
        "output": args.output,
        "maximum_repeat_noise": float(
            maximum_noise
        ),
        "closure_invariance": (
            closure_invariance
        ),
        "deepest_closure_summary": (
            by_closure[
                str(max(closure_depths))
            ][
                "evaluation"
            ]
        ),
    }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
