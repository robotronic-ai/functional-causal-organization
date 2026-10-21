#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import quantum_gamma_kv_experiment_v1 as v1


BLOCKS = {
    "entry_0_4": list(range(0, 5)),
    "core_5_30": list(range(5, 31)),
    "terminal_31_35": list(range(31, 36)),
}
BLOCK_NAMES = list(BLOCKS.keys())
PATTERNS = list(itertools.product((0, 1), repeat=3))
CUTS = [
    ((0,), (1, 2)),
    ((1,), (0, 2)),
    ((2,), (0, 1)),
]


def entropy_uniform(signatures: list[tuple]) -> float:
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    total = float(len(signatures))
    return float(
        -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )
    )


def block_distance(
    left: list[torch.Tensor],
    right: list[torch.Tensor],
    layers: list[int],
) -> float:
    return float(
        max(
            v1.normalized_distance(left[layer], right[layer])
            for layer in layers
        )
    )


def pairwise_matrix(
    states: list[list[torch.Tensor]],
    layers: list[int],
) -> np.ndarray:
    n = len(states)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            value = block_distance(
                states[i],
                states[j],
                layers,
            )
            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


def connected_components(
    matrix: np.ndarray,
    tolerance: float,
) -> list[int]:
    n = int(matrix.shape[0])
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] <= tolerance:
                union(i, j)

    mapping = {}
    labels = []

    for i in range(n):
        root = find(i)
        if root not in mapping:
            mapping[root] = len(mapping)
        labels.append(mapping[root])

    return labels


def audit_class_diameters(
    matrix: np.ndarray,
    labels: list[int],
    tolerance: float,
) -> dict:
    by_class = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[int(label)].append(index)

    diameters = {}
    passed = True

    for label, indices in by_class.items():
        if len(indices) <= 1:
            diameter = 0.0
        else:
            diameter = float(
                max(
                    matrix[i, j]
                    for i in indices
                    for j in indices
                )
            )
        diameters[str(label)] = diameter
        if diameter > tolerance:
            passed = False

    return {
        "passed": bool(passed),
        "maximum_class_diameter": float(
            max(diameters.values(), default=0.0)
        ),
        "class_diameters": diameters,
    }


def current_state(
    model,
    n_layers: int,
) -> list[torch.Tensor]:
    lengths = [
        int(model.kv_cache.get_seq_length(layer))
        for layer in range(n_layers)
    ]
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"Cache lengths differ: {lengths}"
        )
    return v1.response_vectors(
        model.kv_cache,
        lengths[0],
        n_layers,
    )


def initialize_pattern(
    model,
    context: dict,
    bits: tuple[int, int, int],
) -> list[torch.Tensor]:
    v1.restore_cache(
        model.kv_cache,
        context["snapshot_a"],
    )
    v1.reset_transient_state(model)

    for block_index, block_name in enumerate(BLOCK_NAMES):
        if int(bits[block_index]) == 1:
            v1.patch_last_kv(
                model.kv_cache,
                context["snapshot_b"],
                BLOCKS[block_name],
            )

    return current_state(
        model,
        len(model.layers),
    )


def forward_token(
    model,
    token_id: int,
    flow: str,
    device: str,
):
    token = torch.tensor(
        [[int(token_id)]],
        dtype=torch.long,
        device=device,
    )
    v1.reset_transient_state(model)

    with torch.inference_mode():
        return model(
            input_ids=token,
            past_key_values=model.kv_cache,
            use_cache=True,
            current_flow=flow,
        )


def rollout_pattern(
    model,
    context: dict,
    bits: tuple[int, int, int],
    flow: str,
    n_layers: int,
    device: str,
    max_horizon: int,
) -> dict:
    current = initialize_pattern(
        model,
        context,
        bits,
    )

    token_id = int(context["probe_token_id"])
    future_states = []
    emitted_tokens = []

    for _ in range(max_horizon):
        output = forward_token(
            model,
            token_id,
            flow,
            device,
        )

        future_states.append(
            current_state(
                model,
                n_layers,
            )
        )

        logits = v1.extract_logits(output)
        token_id = int(
            torch.argmax(
                logits[:, -1, :],
                dim=-1,
            ).item()
        )
        emitted_tokens.append(token_id)

    return {
        "current_state": current,
        "future_states": future_states,
        "emitted_tokens": emitted_tokens,
    }


def prepare_contexts(
    model,
    tokenizer,
    flow: str,
    n_layers: int,
    device: str,
    context_count: int,
    max_prompt_tokens: int,
) -> list[dict]:
    raw = v1.candidate_contexts(
        tokenizer=tokenizer,
        device=device,
        max_contexts=context_count,
        max_tokens=max_prompt_tokens,
    )

    contexts = []

    for context_id, (
        word_a,
        word_b,
        ids_a,
        ids_b,
    ) in enumerate(raw):
        contexts.append(
            v1.prepare_context(
                model=model,
                tokenizer=tokenizer,
                word_a=word_a,
                word_b=word_b,
                ids_a=ids_a,
                ids_b=ids_b,
                context_id=context_id,
                flow=flow,
                n_layers=n_layers,
                device=device,
            )
        )

    return contexts


def maximum_repeat_noise(
    model,
    context: dict,
    flow: str,
    n_layers: int,
    device: str,
    max_horizon: int,
) -> float:
    first = rollout_pattern(
        model,
        context,
        (0, 0, 0),
        flow,
        n_layers,
        device,
        max_horizon,
    )
    second = rollout_pattern(
        model,
        context,
        (0, 0, 0),
        flow,
        n_layers,
        device,
        max_horizon,
    )

    values = []

    for block_name in BLOCK_NAMES:
        layers = BLOCKS[block_name]
        values.append(
            block_distance(
                first["current_state"],
                second["current_state"],
                layers,
            )
        )

        for horizon in range(max_horizon):
            values.append(
                block_distance(
                    first["future_states"][horizon],
                    second["future_states"][horizon],
                    layers,
                )
            )

    return float(max(values, default=0.0))


def build_global_horizon_classes(
    rollouts: dict[tuple[int, int, int], dict],
    horizons: list[int],
    tolerance: float,
) -> dict:
    result = {}

    for block_name in BLOCK_NAMES:
        layers = BLOCKS[block_name]

        labels_metadata = []
        states = []

        for pattern in PATTERNS:
            states.append(
                rollouts[pattern]["current_state"]
            )
            labels_metadata.append(
                ("current", 0, pattern)
            )

        for horizon in horizons:
            for pattern in PATTERNS:
                states.append(
                    rollouts[pattern]["future_states"][horizon - 1]
                )
                labels_metadata.append(
                    ("future", horizon, pattern)
                )

        matrix = pairwise_matrix(
            states,
            layers,
        )
        labels = connected_components(
            matrix,
            tolerance,
        )
        audit = audit_class_diameters(
            matrix,
            labels,
            tolerance,
        )

        current_map = {}
        future_maps = {
            str(horizon): {}
            for horizon in horizons
        }

        for metadata, label in zip(
            labels_metadata,
            labels,
        ):
            time_kind, horizon, pattern = metadata
            key = "".join(map(str, pattern))

            if time_kind == "current":
                current_map[key] = int(label)
            else:
                future_maps[str(horizon)][key] = int(label)

        result[block_name] = {
            "alphabet_size": int(len(set(labels))),
            "current_class_by_pattern": current_map,
            "future_class_by_horizon_pattern": future_maps,
            "class_audit": audit,
            "maximum_pairwise_distance": float(
                matrix.max()
            ),
        }

    return result


def pattern_key(
    pattern: tuple[int, int, int],
) -> str:
    return "".join(map(str, pattern))


def side_signature(
    classes: dict,
    block_indices: tuple[int, ...],
    pattern: tuple[int, int, int],
    time_kind: str,
    horizon: int | None = None,
) -> tuple:
    key = pattern_key(pattern)
    values = []

    for index in block_indices:
        block_name = BLOCK_NAMES[index]
        block = classes[block_name]

        if time_kind == "current":
            label = block[
                "current_class_by_pattern"
            ][key]
        else:
            label = block[
                "future_class_by_horizon_pattern"
            ][str(horizon)][key]

        values.append(int(label))

    return tuple(values)


def side_global_alphabet_size(
    classes: dict,
    block_indices: tuple[int, ...],
    horizons: list[int],
) -> int:
    signatures = set()

    for pattern in PATTERNS:
        signatures.add(
            side_signature(
                classes,
                block_indices,
                pattern,
                "current",
            )
        )

        for horizon in horizons:
            signatures.add(
                side_signature(
                    classes,
                    block_indices,
                    pattern,
                    "future",
                    horizon,
                )
            )

    return int(len(signatures))


def directional_information(
    classes: dict,
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    horizon: int,
) -> float:
    source_assignments = list(
        itertools.product(
            (0, 1),
            repeat=len(source_indices),
        )
    )
    target_assignments = list(
        itertools.product(
            (0, 1),
            repeat=len(target_indices),
        )
    )

    conditional_entropies = []

    for target_bits in target_assignments:
        signatures = []

        for source_bits in source_assignments:
            full = [0, 0, 0]

            for position, index in enumerate(source_indices):
                full[index] = int(source_bits[position])

            for position, index in enumerate(target_indices):
                full[index] = int(target_bits[position])

            pattern = tuple(full)

            signatures.append(
                side_signature(
                    classes,
                    target_indices,
                    pattern,
                    "future",
                    horizon,
                )
            )

        conditional_entropies.append(
            entropy_uniform(signatures)
        )

    return float(
        np.mean(conditional_entropies)
    )


def exact_gamma_at_horizon(
    classes: dict,
    horizons: list[int],
    horizon: int,
) -> dict:
    rows = []

    for left, right in CUTS:
        j_lr = directional_information(
            classes,
            left,
            right,
            horizon,
        )
        j_rl = directional_information(
            classes,
            right,
            left,
            horizon,
        )

        k_left = side_global_alphabet_size(
            classes,
            left,
            horizons,
        )
        k_right = side_global_alphabet_size(
            classes,
            right,
            horizons,
        )

        if k_left <= 1 or k_right <= 1:
            denominator = 0.0
            gamma_cut = 0.0
        else:
            denominator = 2.0 * min(
                math.log2(k_left),
                math.log2(k_right),
            )
            gamma_cut = (
                j_lr + j_rl
            ) / denominator

        rows.append(
            {
                "left_blocks": [
                    BLOCK_NAMES[index]
                    for index in left
                ],
                "right_blocks": [
                    BLOCK_NAMES[index]
                    for index in right
                ],
                "left_global_alphabet_size": k_left,
                "right_global_alphabet_size": k_right,
                "j_left_to_right_bits": float(j_lr),
                "j_right_to_left_bits": float(j_rl),
                "denominator_bits": float(denominator),
                "gamma_cut": float(gamma_cut),
                "bounded": bool(
                    gamma_cut <= 1.0 + 1e-12
                ),
            }
        )

    gamma = min(
        row["gamma_cut"]
        for row in rows
    )

    minimizing_indices = [
        index
        for index, row in enumerate(rows)
        if abs(
            row["gamma_cut"] - gamma
        ) <= 1e-12
    ]

    return {
        "gamma": float(gamma),
        "all_cuts_bounded": bool(
            all(row["bounded"] for row in rows)
        ),
        "minimizing_cut_indices": minimizing_indices,
        "cuts": rows,
    }


def summarize_contexts(
    reports: list[dict],
    horizons: list[int],
) -> dict:
    summary = {}

    for horizon in horizons:
        key = str(horizon)
        gamma_values = [
            report["horizons"][key]["gamma"]
            for report in reports
        ]

        cut_frequency = Counter()
        for report in reports:
            indices = report["horizons"][key][
                "minimizing_cut_indices"
            ]
            for index in indices:
                cut = CUTS[index]
                label = (
                    f"{BLOCK_NAMES[cut[0][0]]}"
                    if len(cut[0]) == 1
                    else "+".join(
                        BLOCK_NAMES[i]
                        for i in cut[0]
                    )
                )
                cut_frequency[label] += 1

        cut_rows = []
        for cut_index in range(len(CUTS)):
            rows = [
                report["horizons"][key]["cuts"][cut_index]
                for report in reports
            ]
            cut_rows.append(
                {
                    "left_blocks": rows[0]["left_blocks"],
                    "right_blocks": rows[0]["right_blocks"],
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

        summary[key] = {
            "mean_context_gamma": float(
                np.mean(gamma_values)
            ),
            "median_context_gamma": float(
                np.median(gamma_values)
            ),
            "minimum_context_gamma": float(
                np.min(gamma_values)
            ),
            "maximum_context_gamma": float(
                np.max(gamma_values)
            ),
            "context_gamma_values": [
                float(value)
                for value in gamma_values
            ],
            "all_contexts_bounded": bool(
                all(
                    report["horizons"][key][
                        "all_cuts_bounded"
                    ]
                    for report in reports
                )
            ),
            "minimizing_cut_frequency": dict(
                cut_frequency
            ),
            "cuts": cut_rows,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact per-context Gamma with one causal macro alphabet "
            "closed jointly across all requested horizons."
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10102032)
    parser.add_argument(
        "--flow",
        choices=("s_attn",),
        default="s_attn",
    )
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--horizons",
        default="1,2,4",
    )
    parser.add_argument(
        "--base-tolerance",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--output",
        default="quantum_gamma_block_multihorizon_v10.json",
    )
    args = parser.parse_args()

    v1.set_determinism(args.seed)

    horizons = sorted(
        {
            int(value.strip())
            for value in args.horizons.split(",")
            if value.strip()
        }
    )

    if not horizons or min(horizons) < 1:
        raise ValueError(
            "Horizons must be positive integers."
        )

    root = Path(args.root).resolve()
    model_dir = Path(args.model_dir).resolve()

    model, tokenizer, config = v1.load_model(
        model_dir,
        root,
        args.device,
    )
    n_layers = len(model.layers)

    contexts = prepare_contexts(
        model=model,
        tokenizer=tokenizer,
        flow=args.flow,
        n_layers=n_layers,
        device=args.device,
        context_count=args.contexts,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    max_horizon = max(horizons)
    context_reports = []

    for context in contexts:
        noise = maximum_repeat_noise(
            model=model,
            context=context,
            flow=args.flow,
            n_layers=n_layers,
            device=args.device,
            max_horizon=max_horizon,
        )
        tolerance = max(
            float(args.base_tolerance),
            10.0 * noise,
        )

        rollouts = {
            pattern: rollout_pattern(
                model=model,
                context=context,
                bits=pattern,
                flow=args.flow,
                n_layers=n_layers,
                device=args.device,
                max_horizon=max_horizon,
            )
            for pattern in PATTERNS
        }

        classes = build_global_horizon_classes(
            rollouts=rollouts,
            horizons=horizons,
            tolerance=tolerance,
        )

        class_audit_passed = bool(
            all(
                classes[block]["class_audit"]["passed"]
                for block in BLOCK_NAMES
            )
        )

        horizon_reports = {
            str(horizon): exact_gamma_at_horizon(
                classes=classes,
                horizons=horizons,
                horizon=horizon,
            )
            for horizon in horizons
        }

        context_reports.append(
            {
                "context_id": int(context["context_id"]),
                "word_a": context["word_a"],
                "word_b": context["word_b"],
                "token_length": int(context["token_length"]),
                "probe_token_id": int(context["probe_token_id"]),
                "maximum_repeat_noise": float(noise),
                "equivalence_tolerance": float(tolerance),
                "class_audit_passed": class_audit_passed,
                "block_classes": classes,
                "horizons": horizon_reports,
            }
        )

    summary = summarize_contexts(
        context_reports,
        horizons,
    )

    result = {
        "experiment": "quantum_gamma_block_multihorizon_v10",
        "estimand": (
            "Per-context exact finite-battery Gamma with a single "
            "architecture-block macro alphabet closed jointly across "
            "all requested horizons."
        ),
        "flow": args.flow,
        "blocks": BLOCKS,
        "horizons": horizons,
        "context_semantics": (
            "Each prompt context is treated as an exogenous operating "
            "condition U. Gamma is computed exactly conditional on U=u. "
            "Cross-context summaries describe the distribution of those "
            "conditional Gamma values and are not relabeled as one "
            "unconditional Gamma."
        ),
        "source_battery": (
            "Uniform over all 2^3 coherent donor-A/donor-B block "
            "assignments."
        ),
        "macro_alphabet": (
            "Within each fixed context, equivalence classes are built "
            "jointly over the current states and every requested future "
            "horizon. The same alphabet and denominator are therefore "
            "used when comparing horizons."
        ),
        "equivalence": (
            "Deterministic K/V states are grouped only within the "
            "repeat-noise tolerance. Connected-component classes are "
            "audited so every within-class diameter must remain within "
            "that tolerance."
        ),
        "scope": (
            "Exact for the finite three-block intervention battery "
            "conditional on each tested context. It is not a universal "
            "full-state Gamma claim."
        ),
        "context_count": len(contexts),
        "contexts": context_reports,
        "summary": summary,
        "all_class_audits_passed": bool(
            all(
                context["class_audit_passed"]
                for context in context_reports
            )
        ),
        "maximum_repeat_noise": float(
            max(
                context["maximum_repeat_noise"]
                for context in context_reports
            )
        ),
    }

    Path(args.output).write_text(
        json.dumps(result, indent=2)
    )

    print(
        json.dumps(
            {
                "output": args.output,
                "context_count": len(contexts),
                "horizons": horizons,
                "all_class_audits_passed": result[
                    "all_class_audits_passed"
                ],
                "maximum_repeat_noise": result[
                    "maximum_repeat_noise"
                ],
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
