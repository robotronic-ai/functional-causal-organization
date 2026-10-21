#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
INTERVENTIONS = ("ZERO_SUFFIX", "HEAD_ROTATE")
HORIZONS = (1, 2)
EPS = 1e-12


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def target_vector(row: dict[str, Any]) -> tuple[str, ...]:
    cells = {(str(c["operation"]), str(c["address"])): c for c in row["cells"]}
    expected = {(op, address) for op in OPS for address in ADDRESSES}
    if set(cells) != expected:
        raise ValueError(f"invalid behavior grid in unit={row.get('unit')}")
    return tuple(str(cells[(op, address)]["behavior"]["outcome"]) for op in OPS for address in ADDRESSES)


def kernel_cell_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cells = {(str(c["operation"]), str(c["address"])): c for c in row["cells"]}
    expected = {(op, address) for op in OPS for address in ADDRESSES}
    if set(cells) != expected:
        raise ValueError(f"invalid kernel grid in unit={row.get('unit')}")
    return cells


def cell_kernel_vector(cell: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    kernel = cell["causal_kernel"]
    for kind in INTERVENTIONS:
        recs = sorted(kernel[kind], key=lambda x: int(x["horizon"]))
        if [int(x["horizon"]) for x in recs] != list(HORIZONS):
            raise ValueError("invalid horizon records")
        for rec in recs:
            values.extend([
                math.log(max(float(rec["final_hidden_rel_rms"]), EPS)),
                math.log(max(float(rec["primary_block_rel_rms"]), EPS)),
                math.log(max(float(rec["post_primary_block_rel_rms"]), EPS)),
                math.log(max(float(rec["reference_block_rel_rms"]), EPS)),
            ])
    return np.asarray(values, dtype=np.float64)


def operation_kernel_vector(row: dict[str, Any], operation: str) -> np.ndarray:
    cells = kernel_cell_map(row)
    return np.concatenate([cell_kernel_vector(cells[(operation, address)]) for address in ADDRESSES], axis=0)


def cell_slice_vector(row: dict[str, Any], operation: str, address: str) -> np.ndarray:
    return cell_kernel_vector(kernel_cell_map(row)[(operation, address)])


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def deterministic_kmeans(X: np.ndarray, k: int, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
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
    for iteration in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels) and iteration > 0:
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
    for desc, target in zip(descriptors, targets):
        fibers[desc].append(target)
    collision_fibers = 0
    collision_pairs = 0
    nontrivial = 0
    systems_nontrivial = 0
    for tgts in fibers.values():
        if len(tgts) > 1:
            nontrivial += 1
            systems_nontrivial += len(tgts)
        counts: dict[str, int] = defaultdict(int)
        for tgt in tgts:
            counts[tgt] += 1
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
