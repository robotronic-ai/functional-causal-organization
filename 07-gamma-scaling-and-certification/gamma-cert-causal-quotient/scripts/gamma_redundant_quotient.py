#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def entropy_codes(codes: np.ndarray) -> float:
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_mask(
    x_states: np.ndarray,
    y_states: np.ndarray,
    mask: int,
    dimension: int,
) -> float:
    x_masked = x_states & mask
    y_masked = y_states & mask
    joint = x_masked | (y_masked.astype(np.uint64) << np.uint64(dimension))
    return entropy_codes(joint) - entropy_codes(x_masked)


def exact_gamma_deterministic(
    y_states: np.ndarray,
    dimension: int,
) -> tuple[float, int]:
    x_states = np.arange(2**dimension, dtype=np.uint64)
    full_mask = (1 << dimension) - 1
    best = (float("inf"), None)

    for rest in range(1 << (dimension - 1)):
        side_a = 1 | (rest << 1)
        if side_a == full_mask:
            continue
        side_b = full_mask ^ side_a

        size_a = side_a.bit_count()
        size_b = dimension - size_a

        j_ab = conditional_entropy_mask(
            x_states,
            y_states,
            side_b,
            dimension,
        )
        j_ba = conditional_entropy_mask(
            x_states,
            y_states,
            side_a,
            dimension,
        )

        gamma = (j_ab + j_ba) / (2 * min(size_a, size_b))
        if gamma < best[0] - 1e-12:
            best = (gamma, side_a)

    return float(best[0]), int(best[1])


def latent_ring_next(z: np.ndarray) -> np.ndarray:
    return np.roll(z, shift=1, axis=1)


def encode_replicas(z: np.ndarray, replicas: int) -> np.ndarray:
    return np.repeat(z, replicas, axis=1)


class RedundantTransition(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_redundant_nullspace(
    latent_roles: int,
    replicas: int,
    seed: int,
    null_scale: float = 0.0,
    steps: int = 2200,
) -> tuple[RedundantTransition, float]:
    latent = np.array(
        [
            [(state >> i) & 1 for i in range(latent_roles)]
            for state in range(2**latent_roles)
        ],
        dtype=np.float32,
    )
    x = encode_replicas(latent, replicas)
    y = encode_replicas(latent_ring_next(latent), replicas)

    torch.manual_seed(seed)
    model = RedundantTransition(latent_roles * replicas)

    # Add a component orthogonal to the coherent replica subspace.
    # Inside each replica group, the added coefficients sum to zero.
    if null_scale > 0 and replicas > 1:
        rng = np.random.default_rng(seed + 777)
        with torch.no_grad():
            weight = model.linear.weight.detach().numpy()
            for out_index in range(latent_roles * replicas):
                for role in range(latent_roles):
                    vector = rng.normal(size=replicas)
                    vector = vector - vector.mean()
                    start = role * replicas
                    stop = start + replicas
                    weight[out_index, start:stop] += null_scale * vector
            model.linear.weight.copy_(
                torch.tensor(weight, dtype=model.linear.weight.dtype)
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.BCEWithLogitsLoss()
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        prediction = (model(x_tensor).numpy() > 0).astype(np.int8)

    accuracy = float((prediction == y.astype(np.int8)).mean())
    return model, accuracy


def eval_all_micro(
    model: RedundantTransition,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_states = np.arange(2**dimension, dtype=np.uint64)
    x = (
        (
            x_states[:, None]
            >> np.arange(dimension, dtype=np.uint64)
        )
        & 1
    ).astype(np.float32)

    with torch.inference_mode():
        y = (model(torch.tensor(x)).numpy() > 0).astype(np.uint64)

    y_states = (
        y * (1 << np.arange(dimension, dtype=np.uint64))
    ).sum(axis=1).astype(np.uint64)

    return x_states, y_states, y.astype(np.int8)


def eval_coherent_macro(
    model: RedundantTransition,
    latent_roles: int,
    replicas: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    latent = np.array(
        [
            [(state >> i) & 1 for i in range(latent_roles)]
            for state in range(2**latent_roles)
        ],
        dtype=np.float32,
    )
    x = encode_replicas(latent, replicas)

    with torch.inference_mode():
        y = (model(torch.tensor(x)).numpy() > 0).astype(np.int8)

    grouped = y.reshape(len(latent), latent_roles, replicas)
    unanimity = np.all(grouped == grouped[:, :, 0:1], axis=2)
    macro = (grouped.mean(axis=2) >= 0.5).astype(np.uint64)

    y_states = (
        macro * (1 << np.arange(latent_roles, dtype=np.uint64))
    ).sum(axis=1).astype(np.uint64)

    return (
        latent.astype(np.int8),
        y,
        macro,
        y_states,
        float(unanimity.mean()),
    )


def exact_gamma_macro(
    y_states: np.ndarray,
    latent_roles: int,
) -> tuple[float, int]:
    return exact_gamma_deterministic(y_states, latent_roles)


def decode_majority_bits(
    x: np.ndarray,
    latent_roles: int,
    replicas: int,
) -> np.ndarray:
    threshold = math.ceil(replicas / 2)
    return (
        x.reshape(len(x), latent_roles, replicas).sum(axis=2)
        >= threshold
    ).astype(np.int8)


def raw_quotient_sufficiency(
    model: RedundantTransition,
    latent_roles: int,
    replicas: int,
) -> dict:
    dimension = latent_roles * replicas
    x_states = np.arange(2**dimension, dtype=np.uint64)

    x = (
        (
            x_states[:, None]
            >> np.arange(dimension, dtype=np.uint64)
        )
        & 1
    ).astype(np.int8)

    quotient_state = decode_majority_bits(
        x,
        latent_roles,
        replicas,
    )
    quotient_code = (
        quotient_state * (1 << np.arange(latent_roles))
    ).sum(axis=1).astype(int)

    with torch.inference_mode():
        y = (
            model(torch.tensor(x.astype(np.float32))).numpy() > 0
        ).astype(np.int8)

    future_macro = decode_majority_bits(
        y,
        latent_roles,
        replicas,
    )
    future_code = (
        future_macro * (1 << np.arange(latent_roles))
    ).sum(axis=1).astype(int)

    ambiguous_classes = 0
    max_future_count = 1
    examples = []

    for q in range(2**latent_roles):
        values = np.unique(future_code[quotient_code == q])
        if len(values) > 1:
            ambiguous_classes += 1
            max_future_count = max(max_future_count, len(values))

            if len(examples) < 5:
                indices = np.where(quotient_code == q)[0]
                first = int(indices[0])
                second = next(
                    int(index)
                    for index in indices
                    if future_code[index] != future_code[first]
                )
                examples.append(
                    {
                        "macro_class": q,
                        "micro_state_1": int(x_states[first]),
                        "future_macro_1": int(future_code[first]),
                        "micro_state_2": int(x_states[second]),
                        "future_macro_2": int(future_code[second]),
                    }
                )

    return {
        "raw_state_count": int(len(x_states)),
        "ambiguous_macro_classes": int(ambiguous_classes),
        "fraction_macro_classes_ambiguous": float(
            ambiguous_classes / (2**latent_roles)
        ),
        "max_distinct_future_macrostates_within_one_class": int(
            max_future_count
        ),
        "quotient_sufficient_under_raw_algebra": bool(
            ambiguous_classes == 0
        ),
        "examples": examples,
    }


def coherent_minimality(
    model: RedundantTransition,
    latent_roles: int,
    replicas: int,
) -> dict:
    latent = np.array(
        [
            [(state >> i) & 1 for i in range(latent_roles)]
            for state in range(2**latent_roles)
        ],
        dtype=np.int8,
    )
    x = encode_replicas(latent, replicas).astype(np.float32)

    with torch.inference_mode():
        y = (model(torch.tensor(x)).numpy() > 0).astype(np.int8)

    future_macro = decode_majority_bits(
        y,
        latent_roles,
        replicas,
    )

    essential = []
    for role in range(latent_roles):
        found = False
        for state in range(2**latent_roles):
            paired = state ^ (1 << role)
            if paired < state:
                continue
            if np.any(future_macro[state] != future_macro[paired]):
                found = True
                break
        essential.append(found)

    return {
        "coherent_macrostate_combinations_tested": int(2**latent_roles),
        "causally_sufficient_on_coherent_algebra": True,
        "each_macro_role_essential": essential,
        "minimal_on_coherent_algebra": bool(all(essential)),
    }


def run_experiment() -> dict:
    latent_roles = 4
    replicas = 3
    scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    rows = []

    for index, scale in enumerate(scales):
        start = time.time()

        model, accuracy = train_redundant_nullspace(
            latent_roles,
            replicas,
            seed=35000 + index,
            null_scale=scale,
        )

        _, micro_next, _ = eval_all_micro(
            model,
            latent_roles * replicas,
        )
        gamma_raw, raw_cut = exact_gamma_deterministic(
            micro_next,
            latent_roles * replicas,
        )

        _, _, _, macro_next, unanimity = eval_coherent_macro(
            model,
            latent_roles,
            replicas,
        )
        gamma_macro, macro_cut = exact_gamma_macro(
            macro_next,
            latent_roles,
        )

        rows.append(
            {
                "nullspace_scale": scale,
                "coherent_training_accuracy": accuracy,
                "coherent_output_unanimity": unanimity,
                "gamma_raw_micro": gamma_raw,
                "gamma_macro_quotient": gamma_macro,
                "raw_min_cut_size": min(
                    raw_cut.bit_count(),
                    latent_roles * replicas - raw_cut.bit_count(),
                ),
                "macro_min_cut_size": min(
                    macro_cut.bit_count(),
                    latent_roles - macro_cut.bit_count(),
                ),
                "coherent_minimality": coherent_minimality(
                    model,
                    latent_roles,
                    replicas,
                ),
                "raw_algebra_quotient_test": raw_quotient_sufficiency(
                    model,
                    latent_roles,
                    replicas,
                ),
                "elapsed_seconds": time.time() - start,
            }
        )

    return {
        "experiment": "redundant_causal_quotient_nullspace_stress",
        "latent_roles": latent_roles,
        "replicas_per_role": replicas,
        "rows": rows,
    }


if __name__ == "__main__":
    result = run_experiment()
    target = (
        Path(__file__).resolve().parent
        / "nullspace_stress_results.json"
    )
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
