#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression


def binary_entropy(p: float) -> float:
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def clopper_pearson_upper(errors: int, total: int, alpha: float) -> float:
    if total <= 0 or errors >= total:
        return 1.0
    if errors == 0:
        return float(1.0 - alpha ** (1.0 / total))
    return float(beta.ppf(1.0 - alpha, errors + 1, total - errors))


def empirical_bernstein_upper(
    losses: np.ndarray,
    alpha: float,
    upper_bound: float,
) -> float:
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if n < 2:
        return upper_bound
    mean = float(losses.mean())
    variance = float(losses.var(ddof=1))
    log_term = math.log(2.0 / alpha)
    radius = (
        math.sqrt(2.0 * variance * log_term / n)
        + 7.0 * upper_bound * log_term / (3.0 * (n - 1))
    )
    return min(upper_bound, mean + radius)


def one_hot_context(context_ids: np.ndarray, levels: np.ndarray) -> np.ndarray:
    index = {int(value): i for i, value in enumerate(levels.tolist())}
    output = np.zeros((len(context_ids), len(levels)), dtype=float)
    for row, value in enumerate(context_ids):
        output[row, index[int(value)]] = 1.0
    return output


def fit_binary_decoder(
    source_bit: np.ndarray,
    response_bit: np.ndarray,
    context_ids: np.ndarray,
    context_levels: np.ndarray,
) -> LogisticRegression:
    features = np.column_stack(
        [
            response_bit.astype(float),
            one_hot_context(context_ids, context_levels),
        ]
    )
    model = LogisticRegression(
        C=10.0,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(features, source_bit.astype(int))
    return model


def decoder_error(
    model: LogisticRegression,
    source_bit: np.ndarray,
    response_bit: np.ndarray,
    context_ids: np.ndarray,
    context_levels: np.ndarray,
) -> int:
    features = np.column_stack(
        [
            response_bit.astype(float),
            one_hot_context(context_ids, context_levels),
        ]
    )
    prediction = model.predict(features)
    return int(np.sum(prediction != source_bit.astype(int)))


def build_lower_graph(
    discovery_x: np.ndarray,
    discovery_y: np.ndarray,
    discovery_context: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    holdout_context: np.ndarray,
    delta_lower: float,
    top_k_per_source: int,
) -> tuple[np.ndarray, dict]:
    n_roles = discovery_x.shape[1]
    context_levels = np.unique(
        np.concatenate([discovery_context, holdout_context])
    )

    candidates = []
    for source in range(n_roles):
        for target in range(n_roles):
            if source == target:
                continue
            model = fit_binary_decoder(
                discovery_x[:, source],
                discovery_y[:, target],
                discovery_context,
                context_levels,
            )
            discovery_errors = decoder_error(
                model,
                discovery_x[:, source],
                discovery_y[:, target],
                discovery_context,
                context_levels,
            )
            discovery_error_rate = discovery_errors / len(discovery_x)
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "model": model,
                    "discovery_error_rate": discovery_error_rate,
                }
            )

    selected = []
    for source in range(n_roles):
        local = [item for item in candidates if item["source"] == source]
        local.sort(key=lambda item: item["discovery_error_rate"])
        selected.extend(local[:top_k_per_source])

    alpha_edge = delta_lower / max(1, len(selected))
    directed_lcb = np.zeros((n_roles, n_roles), dtype=float)
    edge_records = []

    for item in selected:
        source = item["source"]
        target = item["target"]
        errors = decoder_error(
            item["model"],
            holdout_x[:, source],
            holdout_y[:, target],
            holdout_context,
            context_levels,
        )
        p_upper = clopper_pearson_upper(
            errors,
            len(holdout_x),
            alpha_edge,
        )
        information_lcb = max(
            0.0,
            1.0 - binary_entropy(min(p_upper, 0.5)),
        )
        directed_lcb[source, target] = information_lcb
        edge_records.append(
            {
                "source": source,
                "target": target,
                "holdout_errors": errors,
                "holdout_total": int(len(holdout_x)),
                "error_rate_upper": p_upper,
                "information_lcb_bits": information_lcb,
            }
        )

    out_degree = np.sum(directed_lcb > 0.0, axis=1)
    safe_directed = np.zeros_like(directed_lcb)
    for source in range(n_roles):
        if out_degree[source] > 0:
            safe_directed[source] = (
                directed_lcb[source] / out_degree[source]
            )

    symmetric = safe_directed + safe_directed.T

    metadata = {
        "selected_edge_count": len(selected),
        "positive_directed_edge_count": int(
            np.sum(directed_lcb > 0.0)
        ),
        "edge_records": edge_records,
    }
    return symmetric, metadata


def spectral_gamma_lower(weight_matrix: np.ndarray) -> dict:
    degrees = weight_matrix.sum(axis=1)
    if np.any(degrees <= 1e-15):
        return {
            "gamma_lower": 0.0,
            "lambda2_normalized": 0.0,
            "minimum_weighted_degree": float(degrees.min()),
            "fiedler_vector": None,
        }

    laplacian = csgraph.laplacian(weight_matrix, normed=True)
    eigenvalues, eigenvectors = eigsh(
        laplacian,
        k=2,
        which="SM",
        tol=1e-9,
    )
    order = np.argsort(eigenvalues)
    lambda2 = float(eigenvalues[order[1]])
    fiedler = eigenvectors[:, order[1]]
    minimum_degree = float(degrees.min())

    gamma_lower = minimum_degree * lambda2 / 4.0

    return {
        "gamma_lower": float(gamma_lower),
        "lambda2_normalized": lambda2,
        "minimum_weighted_degree": minimum_degree,
        "fiedler_vector": fiedler.tolist(),
    }


def candidate_cuts_from_fiedler(
    fiedler_vector: np.ndarray,
) -> list[tuple[int, ...]]:
    order = np.argsort(fiedler_vector)
    return [
        tuple(sorted(order[:size].tolist()))
        for size in range(1, len(order))
    ]


def fit_autoregressive_side(
    x_side: np.ndarray,
    y_side: np.ndarray,
    context_ids: np.ndarray,
    context_levels: np.ndarray,
    clip_probability: float,
) -> dict:
    n_outputs = y_side.shape[1]
    models = []

    context_features = one_hot_context(context_ids, context_levels)

    for output_index in range(n_outputs):
        previous = y_side[:, :output_index].astype(float)
        features = np.column_stack(
            [
                x_side.astype(float),
                context_features,
                previous,
            ]
        )
        target = y_side[:, output_index].astype(int)

        if np.all(target == target[0]):
            models.append(
                {
                    "type": "constant",
                    "value": int(target[0]),
                }
            )
        else:
            model = LogisticRegression(
                C=10.0,
                max_iter=1000,
                solver="lbfgs",
            )
            model.fit(features, target)
            models.append(
                {
                    "type": "logistic",
                    "model": model,
                }
            )

    return {
        "models": models,
        "clip_probability": float(clip_probability),
    }


def autoregressive_log_losses(
    trained: dict,
    x_side: np.ndarray,
    y_side: np.ndarray,
    context_ids: np.ndarray,
    context_levels: np.ndarray,
) -> np.ndarray:
    context_features = one_hot_context(context_ids, context_levels)
    clip_probability = trained["clip_probability"]
    losses = np.zeros(len(x_side), dtype=float)

    for output_index, item in enumerate(trained["models"]):
        previous = y_side[:, :output_index].astype(float)
        features = np.column_stack(
            [
                x_side.astype(float),
                context_features,
                previous,
            ]
        )
        target = y_side[:, output_index].astype(int)

        if item["type"] == "constant":
            probability_one = np.full(
                len(x_side),
                float(item["value"]),
            )
        else:
            probability_one = item["model"].predict_proba(features)[:, 1]

        probability_one = np.clip(
            probability_one,
            clip_probability,
            1.0 - clip_probability,
        )
        probability_target = np.where(
            target == 1,
            probability_one,
            1.0 - probability_one,
        )
        losses += -np.log2(probability_target)

    return losses


def discovery_cut_score(
    cut_a: tuple[int, ...],
    discovery_x: np.ndarray,
    discovery_y: np.ndarray,
    discovery_context: np.ndarray,
    context_levels: np.ndarray,
    clip_probability: float,
) -> float:
    n_roles = discovery_x.shape[1]
    set_a = set(cut_a)
    cut_b = tuple(i for i in range(n_roles) if i not in set_a)

    trained_b = fit_autoregressive_side(
        discovery_x[:, cut_b],
        discovery_y[:, cut_b],
        discovery_context,
        context_levels,
        clip_probability,
    )
    trained_a = fit_autoregressive_side(
        discovery_x[:, cut_a],
        discovery_y[:, cut_a],
        discovery_context,
        context_levels,
        clip_probability,
    )

    loss_b = autoregressive_log_losses(
        trained_b,
        discovery_x[:, cut_b],
        discovery_y[:, cut_b],
        discovery_context,
        context_levels,
    ).mean()
    loss_a = autoregressive_log_losses(
        trained_a,
        discovery_x[:, cut_a],
        discovery_y[:, cut_a],
        discovery_context,
        context_levels,
    ).mean()

    denominator = 2.0 * min(len(cut_a), len(cut_b))
    return float((loss_a + loss_b) / denominator)


def select_candidate_cut(
    fiedler_vector: np.ndarray,
    discovery_x: np.ndarray,
    discovery_y: np.ndarray,
    discovery_context: np.ndarray,
    clip_probability: float,
    max_candidate_cuts: int,
) -> tuple[int, ...]:
    context_levels = np.unique(discovery_context)
    cuts = candidate_cuts_from_fiedler(fiedler_vector)

    if len(cuts) > max_candidate_cuts:
        indices = np.linspace(
            0,
            len(cuts) - 1,
            max_candidate_cuts,
            dtype=int,
        )
        cuts = [cuts[index] for index in indices]

    scored = [
        (
            discovery_cut_score(
                cut,
                discovery_x,
                discovery_y,
                discovery_context,
                context_levels,
                clip_probability,
            ),
            cut,
        )
        for cut in cuts
    ]
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def gamma_upper_for_cut(
    cut_a: tuple[int, ...],
    discovery_x: np.ndarray,
    discovery_y: np.ndarray,
    discovery_context: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    holdout_context: np.ndarray,
    delta_upper: float,
    clip_probability: float,
) -> dict:
    n_roles = discovery_x.shape[1]
    set_a = set(cut_a)
    cut_b = tuple(i for i in range(n_roles) if i not in set_a)

    context_levels = np.unique(
        np.concatenate([discovery_context, holdout_context])
    )

    trained_a = fit_autoregressive_side(
        discovery_x[:, cut_a],
        discovery_y[:, cut_a],
        discovery_context,
        context_levels,
        clip_probability,
    )
    trained_b = fit_autoregressive_side(
        discovery_x[:, cut_b],
        discovery_y[:, cut_b],
        discovery_context,
        context_levels,
        clip_probability,
    )

    loss_a = autoregressive_log_losses(
        trained_a,
        holdout_x[:, cut_a],
        holdout_y[:, cut_a],
        holdout_context,
        context_levels,
    )
    loss_b = autoregressive_log_losses(
        trained_b,
        holdout_x[:, cut_b],
        holdout_y[:, cut_b],
        holdout_context,
        context_levels,
    )

    upper_bound_a = len(cut_a) * (
        -math.log2(clip_probability)
    )
    upper_bound_b = len(cut_b) * (
        -math.log2(clip_probability)
    )

    ce_upper_a = empirical_bernstein_upper(
        loss_a,
        delta_upper / 2.0,
        upper_bound_a,
    )
    ce_upper_b = empirical_bernstein_upper(
        loss_b,
        delta_upper / 2.0,
        upper_bound_b,
    )

    denominator = 2.0 * min(len(cut_a), len(cut_b))
    gamma_upper = min(
        1.0,
        (ce_upper_a + ce_upper_b) / denominator,
    )

    return {
        "gamma_upper": float(gamma_upper),
        "cut_a": list(cut_a),
        "cut_b": list(cut_b),
        "cut_smaller_side_size": int(
            min(len(cut_a), len(cut_b))
        ),
        "cross_entropy_upper_a_bits": float(ce_upper_a),
        "cross_entropy_upper_b_bits": float(ce_upper_b),
    }


def load_trials(path: Path) -> dict:
    data = np.load(path)
    required = [
        "x",
        "y",
        "context",
        "split",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing array: {key}")

    x = data["x"].astype(np.int8)
    y = data["y"].astype(np.int8)
    context = data["context"].astype(np.int64)
    split = data["split"].astype(np.int8)

    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if len(context) != len(x) or len(split) != len(x):
        raise ValueError("Trial arrays must have matching row counts.")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must contain binary macro-role states.")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain binary macro-role responses.")

    return {
        "x": x,
        "y": y,
        "context": context,
        "split": split,
    }


def certify(
    trials: dict,
    config: dict,
    quotient_gate: dict,
) -> dict:
    if not quotient_gate.get("passed", False):
        return {
            "status": "NA",
            "reason": "The causal minimal-state quotient gate did not pass.",
        }

    x = trials["x"]
    y = trials["y"]
    context = trials["context"]
    split = trials["split"]

    lower_discovery_mask = split == 0
    lower_holdout_mask = split == 1
    upper_discovery_mask = split == 2
    upper_holdout_mask = split == 3

    for mask, label in [
        (lower_discovery_mask, "lower_discovery"),
        (lower_holdout_mask, "lower_holdout"),
        (upper_discovery_mask, "upper_discovery"),
        (upper_holdout_mask, "upper_holdout"),
    ]:
        if int(mask.sum()) == 0:
            raise ValueError(f"Split {label} is empty.")

    graph, lower_metadata = build_lower_graph(
        x[lower_discovery_mask],
        y[lower_discovery_mask],
        context[lower_discovery_mask],
        x[lower_holdout_mask],
        y[lower_holdout_mask],
        context[lower_holdout_mask],
        config["delta_lower"],
        config["top_k_per_source"],
    )

    lower = spectral_gamma_lower(graph)
    if lower["fiedler_vector"] is None:
        return {
            "status": "V3",
            "gamma_lower": 0.0,
            "gamma_upper": 1.0,
            "interval_width": 1.0,
            "reason": "The certified lower graph is disconnected.",
            "lower_metadata": lower_metadata,
        }

    fiedler = np.asarray(lower["fiedler_vector"], dtype=float)

    candidate_cut = select_candidate_cut(
        fiedler,
        x[upper_discovery_mask],
        y[upper_discovery_mask],
        context[upper_discovery_mask],
        config["clip_probability"],
        config["max_candidate_cuts"],
    )

    upper = gamma_upper_for_cut(
        candidate_cut,
        x[upper_discovery_mask],
        y[upper_discovery_mask],
        context[upper_discovery_mask],
        x[upper_holdout_mask],
        y[upper_holdout_mask],
        context[upper_holdout_mask],
        config["delta_upper"],
        config["clip_probability"],
    )

    gamma_lower = float(lower["gamma_lower"])
    gamma_upper = float(upper["gamma_upper"])
    width = gamma_upper - gamma_lower

    precision_pass = (
        width <= config["v1_max_absolute_width"]
    )

    return {
        "status": "V1" if precision_pass else "V3",
        "confidence_level": 1.0
        - config["delta_lower"]
        - config["delta_upper"],
        "gamma_lower": gamma_lower,
        "gamma_upper": gamma_upper,
        "interval_width": float(width),
        "v1_max_absolute_width": config[
            "v1_max_absolute_width"
        ],
        "candidate_cut": upper,
        "spectral_lower": {
            "lambda2_normalized": lower["lambda2_normalized"],
            "minimum_weighted_degree": lower[
                "minimum_weighted_degree"
            ],
        },
        "lower_metadata": lower_metadata,
        "quotient_gate": quotient_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--quotient-gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trials = load_trials(Path(args.trials))
    config = json.loads(Path(args.config).read_text())
    quotient_gate = json.loads(Path(args.quotient_gate).read_text())

    result = certify(trials, config, quotient_gate)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
