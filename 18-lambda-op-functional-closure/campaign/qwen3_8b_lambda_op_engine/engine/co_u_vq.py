#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
NON_ID_OPS = ("KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
BASE_FEATURES = (
    "primary_k_rms",
    "primary_v_rms",
    "reference_k_rms",
    "reference_v_rms",
    "corridor_ratio",
)
EPS = 1e-12


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cell_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in row["cells"]:
        key = (str(cell["operation"]), str(cell["address"]))
        if key in out:
            raise ValueError(f"duplicate cell {key} in unit={row.get('unit')}")
        out[key] = cell
    expected = {(op, address) for op in OPS for address in ADDRESSES}
    if set(out) != expected:
        raise ValueError(f"invalid operation-address grid in unit={row.get('unit')}")
    return out


def target_vector(row: dict[str, Any]) -> tuple[str, ...]:
    cells = cell_map(row)
    return tuple(
        str(cells[(op, address)]["behavior"]["outcome"])
        for op in OPS for address in ADDRESSES
    )


def _log(x: float) -> float:
    return math.log(max(float(x), EPS))


def slice_vector(row: dict[str, Any], operation: str) -> np.ndarray:
    """Return one target-blind 15D operation slice.

    ID uses absolute log internal measurements. Non-ID operations use log effects
    relative to ID at the same address. Address identity is preserved by the fixed
    H1,H2,H3 ordering.
    """
    cells = cell_map(row)
    values: list[float] = []
    for address in ADDRESSES:
        cur = cells[(operation, address)]["internal"]
        base = cells[("ID", address)]["internal"]
        for feat in BASE_FEATURES:
            if operation == "ID":
                values.append(_log(cur[feat]))
            else:
                values.append(_log(cur[feat]) - _log(base[feat]))
    return np.asarray(values, dtype=np.float64)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def deterministic_kmeans(X: np.ndarray, k: int, seed: int = 202609271011, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic NumPy k-means with farthest-point initialization."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        raise ValueError("empty matrix")
    k = max(1, min(int(k), n))

    norms = np.sum(X * X, axis=1)
    first = int(np.argmin(np.abs(norms - np.median(norms))))
    chosen = [first]
    min_d2 = np.sum((X - X[first]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_d2))
        if idx in chosen:
            for candidate in np.argsort(-min_d2):
                if int(candidate) not in chosen:
                    idx = int(candidate)
                    break
        chosen.append(idx)
        d2 = np.sum((X - X[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)

    centers = X[np.asarray(chosen, dtype=int)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        if shift < 1e-10:
            break
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    return centers, labels


def assign_clusters(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, centers: np.ndarray) -> np.ndarray:
    Xn = (np.asarray(X, dtype=np.float64) - mu) / sd
    d2 = ((Xn[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def collision_metrics(descriptors: list[tuple[int, ...]], targets: list[str]) -> dict[str, float | int]:
    fibers: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for desc, tgt in zip(descriptors, targets):
        fibers[desc].append(tgt)
    collision_fibers = 0
    collision_pairs = 0
    nontrivial = 0
    systems_nontrivial = 0
    for tgts in fibers.values():
        if len(tgts) > 1:
            nontrivial += 1
            systems_nontrivial += len(tgts)
        counts: dict[str, int] = defaultdict(int)
        for t in tgts:
            counts[t] += 1
        if len(counts) > 1:
            collision_fibers += 1
            total_pairs = len(tgts) * (len(tgts) - 1) // 2
            same_pairs = sum(n * (n - 1) // 2 for n in counts.values())
            collision_pairs += total_pairs - same_pairs
    n = len(descriptors)
    return {
        "n_systems": n,
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": nontrivial,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "nontrivial_coverage": (systems_nontrivial / n) if n else 0.0,
        "n_collision_fibers": collision_fibers,
        "n_collision_pairs": collision_pairs,
    }


def encode_rows(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    family = str(candidate["family"])
    ops = list(candidate["operations"])
    models = candidate["slice_models"]
    codes: dict[str, np.ndarray] = {}
    for op in ops:
        X = np.stack([slice_vector(row, op) for row in rows], axis=0)
        spec = models[op]
        mu = np.asarray(spec["mu"], dtype=np.float64)
        sd = np.asarray(spec["sd"], dtype=np.float64)
        centers = np.asarray(spec["centers"], dtype=np.float64)
        codes[op] = assign_clusters(X, mu, sd, centers)
    out = []
    for i in range(len(rows)):
        out.append(tuple(int(codes[op][i]) for op in ops))
    return out
