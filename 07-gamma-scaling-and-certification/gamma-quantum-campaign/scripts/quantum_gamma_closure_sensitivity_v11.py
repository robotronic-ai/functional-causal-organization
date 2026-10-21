#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
            "All integer list values must be positive."
        )
    return result


def cut_label(row: dict) -> str:
    return (
        "+".join(row["left_blocks"])
        + " | "
        + "+".join(row["right_blocks"])
    )


def mean_cut_table(
    context_rows: list[dict],
) -> list[dict]:
    output = []

    for cut_index in range(len(v10.CUTS)):
        rows = [
            context["cuts"][cut_index]
            for context in context_rows
        ]

        output.append(
            {
                "cut": cut_label(rows[0]),
                "mean_j_left_to_right_bits": float(
                    np.mean(
                        [
                            row["j_left_to_right_bits"]
                            for row in rows
                        ]
                    )
                ),
                "mean_j_right_to_left_bits": float(
                    np.mean(
                        [
                            row["j_right_to_left_bits"]
                            for row in rows
                        ]
                    )
                ),
                "mean_denominator_bits": float(
                    np.mean(
                        [
                            row["denominator_bits"]
                            for row in rows
                        ]
                    )
                ),
                "mean_gamma_cut": float(
                    np.mean(
                        [
                            row["gamma_cut"]
                            for row in rows
                        ]
                    )
                ),
                "minimum_gamma_cut": float(
                    np.min(
                        [
                            row["gamma_cut"]
                            for row in rows
                        ]
                    )
                ),
                "maximum_gamma_cut": float(
                    np.max(
                        [
                            row["gamma_cut"]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    return output


def block_alphabet_summary(
    context_classes: list[dict],
) -> dict:
    summary = {}

    for block_name in v10.BLOCK_NAMES:
        sizes = [
            classes[block_name][
                "alphabet_size"
            ]
            for classes in context_classes
        ]

        summary[block_name] = {
            "minimum": int(np.min(sizes)),
            "median": float(np.median(sizes)),
            "maximum": int(np.max(sizes)),
            "values": [
                int(value)
                for value in sizes
            ],
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how finite-battery Gamma changes when the same raw "
            "K/V macro alphabet is closed to progressively deeper "
            "autoregressive horizons."
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
        default=11112033,
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
        default="1,2,4",
    )
    parser.add_argument(
        "--base-tolerance",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--output",
        default="quantum_gamma_closure_sensitivity_v11.json",
    )
    args = parser.parse_args()

    closure_depths = parse_int_list(
        args.closure_depths
    )
    evaluation_horizons = parse_int_list(
        args.evaluation_horizons
    )

    maximum_depth = max(
        closure_depths
    )

    if max(evaluation_horizons) > maximum_depth:
        raise ValueError(
            "Every evaluation horizon must be at or below the "
            "maximum closure depth."
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
        maximum_noise = max(
            maximum_noise,
            noise,
        )

        tolerance = max(
            float(args.base_tolerance),
            10.0 * float(noise),
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
                "context": context,
                "noise": float(noise),
                "tolerance": float(tolerance),
                "rollouts": rollouts,
            }
        )

    closure_reports = {}

    for closure_depth in closure_depths:
        closure_horizons = list(
            range(1, closure_depth + 1)
        )

        context_classes = []
        per_horizon_rows = {
            horizon: []
            for horizon in evaluation_horizons
            if horizon <= closure_depth
        }
        class_audits = []

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

            context_classes.append(
                classes
            )

            audit_passed = all(
                classes[block][
                    "class_audit"
                ]["passed"]
                for block in v10.BLOCK_NAMES
            )
            class_audits.append(
                bool(audit_passed)
            )

            for horizon in (
                per_horizon_rows
            ):
                gamma = (
                    v10.exact_gamma_at_horizon(
                        classes=classes,
                        horizons=closure_horizons,
                        horizon=horizon,
                    )
                )
                per_horizon_rows[
                    horizon
                ].append(
                    gamma
                )

        evaluation_summary = {}

        for horizon, rows in (
            per_horizon_rows.items()
        ):
            gamma_values = [
                row["gamma"]
                for row in rows
            ]

            minimizing_counts = {
                index: 0
                for index in range(
                    len(v10.CUTS)
                )
            }

            for row in rows:
                for index in row[
                    "minimizing_cut_indices"
                ]:
                    minimizing_counts[
                        int(index)
                    ] += 1

            evaluation_summary[
                str(horizon)
            ] = {
                "mean_context_gamma": float(
                    np.mean(
                        gamma_values
                    )
                ),
                "minimum_context_gamma": float(
                    np.min(
                        gamma_values
                    )
                ),
                "maximum_context_gamma": float(
                    np.max(
                        gamma_values
                    )
                ),
                "context_gamma_values": [
                    float(value)
                    for value in (
                        gamma_values
                    )
                ],
                "mean_cuts": mean_cut_table(
                    rows
                ),
                "minimizing_cut_counts": {
                    cut_label(
                        rows[0][
                            "cuts"
                        ][index]
                    ): int(count)
                    for index, count in (
                        minimizing_counts.items()
                    )
                    if count > 0
                },
                "all_contexts_bounded": bool(
                    all(
                        row[
                            "all_cuts_bounded"
                        ]
                        for row in rows
                    )
                ),
            }

        closure_reports[
            str(closure_depth)
        ] = {
            "closure_depth": int(
                closure_depth
            ),
            "closure_horizons": (
                closure_horizons
            ),
            "all_class_audits_passed": bool(
                all(class_audits)
            ),
            "block_alphabet_sizes": (
                block_alphabet_summary(
                    context_classes
                )
            ),
            "evaluation": (
                evaluation_summary
            ),
        }

    invariance = {}

    for horizon in evaluation_horizons:
        eligible_depths = [
            depth
            for depth in closure_depths
            if depth >= horizon
        ]

        if len(eligible_depths) < 2:
            continue

        cut_records = []

        for cut_index in range(
            len(v10.CUTS)
        ):
            j_lr = []
            j_rl = []
            denominators = []
            gammas = []

            for depth in eligible_depths:
                row = closure_reports[
                    str(depth)
                ][
                    "evaluation"
                ][str(horizon)][
                    "mean_cuts"
                ][cut_index]

                j_lr.append(
                    row[
                        "mean_j_left_to_right_bits"
                    ]
                )
                j_rl.append(
                    row[
                        "mean_j_right_to_left_bits"
                    ]
                )
                denominators.append(
                    row[
                        "mean_denominator_bits"
                    ]
                )
                gammas.append(
                    row[
                        "mean_gamma_cut"
                    ]
                )

            cut_records.append(
                {
                    "cut": closure_reports[
                        str(
                            eligible_depths[0]
                        )
                    ][
                        "evaluation"
                    ][str(horizon)][
                        "mean_cuts"
                    ][cut_index][
                        "cut"
                    ],
                    "closure_depths": (
                        eligible_depths
                    ),
                    "j_left_to_right_range": float(
                        max(j_lr)
                        - min(j_lr)
                    ),
                    "j_right_to_left_range": float(
                        max(j_rl)
                        - min(j_rl)
                    ),
                    "denominator_range": float(
                        max(denominators)
                        - min(denominators)
                    ),
                    "gamma_range": float(
                        max(gammas)
                        - min(gammas)
                    ),
                    "j_left_to_right_values": [
                        float(value)
                        for value in j_lr
                    ],
                    "j_right_to_left_values": [
                        float(value)
                        for value in j_rl
                    ],
                    "denominator_values": [
                        float(value)
                        for value in (
                            denominators
                        )
                    ],
                    "gamma_values": [
                        float(value)
                        for value in gammas
                    ],
                }
            )

        invariance[
            str(horizon)
        ] = cut_records

    result = {
        "experiment": (
            "quantum_gamma_closure_sensitivity_v11"
        ),
        "flow": args.flow,
        "blocks": v10.BLOCKS,
        "context_count": len(
            contexts
        ),
        "closure_depths": (
            closure_depths
        ),
        "evaluation_horizons": (
            evaluation_horizons
        ),
        "maximum_repeat_noise": float(
            maximum_noise
        ),
        "purpose": (
            "Test whether directional causal information is stable "
            "while Gamma changes only because raw K/V support "
            "cardinality grows with closure depth."
        ),
        "interpretation": (
            "If J is closure-invariant but denominators grow and Gamma "
            "shrinks, numerical K/V identity is not a stable causal "
            "macro-grain for cardinality-normalized Gamma."
        ),
        "closures": (
            closure_reports
        ),
        "fixed_horizon_invariance": (
            invariance
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
        "maximum_repeat_noise": float(
            maximum_noise
        ),
        "closure_summary": {
            str(depth): {
                "block_alphabet_sizes": (
                    closure_reports[
                        str(depth)
                    ][
                        "block_alphabet_sizes"
                    ]
                ),
                "evaluation": {
                    horizon: {
                        "mean_context_gamma": (
                            values[
                                "mean_context_gamma"
                            ]
                        ),
                        "minimizing_cut_counts": (
                            values[
                                "minimizing_cut_counts"
                            ]
                        ),
                    }
                    for horizon, values in (
                        closure_reports[
                            str(depth)
                        ][
                            "evaluation"
                        ].items()
                    )
                },
            }
            for depth in closure_depths
        },
        "fixed_horizon_invariance": (
            invariance
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
