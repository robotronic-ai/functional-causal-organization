#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
CONTENT_OPS = ("REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
INTERVENTIONS = ("ZERO_SUFFIX", "HEAD_ROTATE")
MARGINS = ("DONOR_MINUS_ORIGINAL", "NONE_MINUS_ORIGINAL")


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cell_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cells = {(str(c["operation"]), str(c["source_address"])): c for c in row["cells"]}
    expected = {(op, address) for op in OPS for address in ADDRESSES}
    if set(cells) != expected:
        raise ValueError(f"invalid operation-address grid in unit={row.get('unit')}")
    return cells


def target_vector(row: dict[str, Any]) -> tuple[str, ...]:
    cells = cell_map(row)
    return tuple(str(cells[(op, address)]["behavior"]["outcome"]) for op in OPS for address in ADDRESSES)


def feature_names() -> list[str]:
    names: list[str] = []
    for op in CONTENT_OPS:
        for address in ADDRESSES:
            for intervention in INTERVENTIONS:
                for margin in MARGINS:
                    names.append(f"{op}:{address}:{intervention}:{margin}")
    return names


def feature_vector(row: dict[str, Any]) -> np.ndarray:
    cells = cell_map(row)
    vals: list[float] = []
    for op in CONTENT_OPS:
        for address in ADDRESSES:
            effects = cells[(op, address)]["content_effect_kernel"]
            for intervention in INTERVENTIONS:
                rec = effects[intervention]
                vals.append(float(rec["delta_margin_donor_vs_original"]))
                vals.append(float(rec["delta_margin_none_vs_original"]))
    return np.asarray(vals, dtype=np.float64)


def _pair_collisions(targets: list[str]) -> int:
    counts: dict[str, int] = defaultdict(int)
    for t in targets:
        counts[t] += 1
    n = len(targets)
    return n * (n - 1) // 2 - sum(v * (v - 1) // 2 for v in counts.values())


def closure_metrics(labels: list[str], targets: list[str], units: list[int]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    collision_pairs = 0
    collision_fibers = 0
    nontrivial_fibers = 0
    systems_nontrivial = 0
    examples: list[dict[str, Any]] = []
    for label, idxs in groups.items():
        ts = [targets[i] for i in idxs]
        if len(idxs) > 1:
            nontrivial_fibers += 1
            systems_nontrivial += len(idxs)
        cp = _pair_collisions(ts)
        if cp:
            collision_pairs += cp
            collision_fibers += 1
            if len(examples) < 12:
                by_target: dict[str, list[int]] = defaultdict(list)
                for i in idxs:
                    by_target[targets[i]].append(units[i])
                examples.append({
                    "fiber": label,
                    "members": [
                        {"target": json.loads(t), "units": us}
                        for t, us in sorted(by_target.items())
                    ],
                })
    n = len(labels)
    return {
        "n_systems": n,
        "n_fibers": len(groups),
        "n_nontrivial_fibers": nontrivial_fibers,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "nontrivial_coverage": systems_nontrivial / n if n else 0.0,
        "n_collision_fibers": collision_fibers,
        "n_collision_pairs": collision_pairs,
        "collision_examples": examples,
    }


def fidelity_metrics(Z: np.ndarray, labels: list[str]) -> dict[str, Any]:
    overall = Z.mean(axis=0)
    tss = float(np.sum((Z - overall) ** 2))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    wss = 0.0
    sizes: list[int] = []
    for idxs in groups.values():
        block = Z[idxs]
        center = block.mean(axis=0)
        wss += float(np.sum((block - center) ** 2))
        sizes.append(len(idxs))
    frac = wss / tss if tss > 1e-12 else 0.0
    return {
        "within_fiber_distortion_fraction": frac,
        "rich_variance_explained_by_fibers": 1.0 - frac,
        "fiber_sizes": sorted(sizes, reverse=True),
    }


def quantile_edges(values: np.ndarray, bins: int) -> list[float]:
    if bins < 2:
        raise ValueError("bins must be >=2")
    qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    raw = [float(np.quantile(values, q)) for q in qs]
    out: list[float] = []
    for x in raw:
        if not out or x > out[-1] + 1e-12:
            out.append(x)
    return out


def bin_value(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="right"))


def encode_matrix(X: np.ndarray, specs: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for row in X:
        parts = [str(bin_value(float(row[int(s["feature_index"])]), list(s["edges"]))) for s in specs]
        labels.append("|".join(parts) if parts else "ALL")
    return labels


def apply_candidate(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    X = np.stack([feature_vector(r) for r in rows])
    mu = np.asarray(candidate["standardizer"]["mu"], dtype=np.float64)
    sd = np.asarray(candidate["standardizer"]["sd"], dtype=np.float64)
    Z = (X - mu) / sd
    labels = encode_matrix(Z, list(candidate["selected_quantized_features"]))
    return labels, X, Z
