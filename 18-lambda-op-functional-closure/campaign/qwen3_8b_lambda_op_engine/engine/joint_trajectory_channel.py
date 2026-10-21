#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

from cross_target_channel import ADDRESSES, HORIZONS, METRICS, OPS, cell_map, stable, target_vector
from co_u_vq import deterministic_kmeans, fit_standardizer, assign_clusters

EPS = 1e-12


def _log(x: float) -> float:
    return math.log(max(float(x), EPS))


def _records(row: dict[str, Any], op: str, source: str, target: str) -> dict[int, dict[str, Any]]:
    cell = cell_map(row)[(op, source)]
    recs = cell['cross_target_kernel'][target]['ZERO_SUFFIX']
    return {int(r['horizon']): r for r in recs}


def source_trajectory_vector(row: dict[str, Any], op: str, source: str) -> np.ndarray:
    """24D target-blind typed trajectory for one operation/source slice.

    For each target and internal metric, retain a log-level and an ordered
    horizon-2 minus horizon-1 log change. This preserves quantitative routing
    strength and temporal direction while keeping H1/H2/H3 identity explicit.
    """
    vals: list[float] = []
    for target in ADDRESSES:
        rec = _records(row, op, source, target)
        for metric in METRICS:
            l1 = _log(rec[1][metric])
            l2 = _log(rec[2][metric])
            vals.append(0.5 * (l1 + l2))
            vals.append(l2 - l1)
    return np.asarray(vals, dtype=np.float64)


def operation_trajectory_vector(row: dict[str, Any], op: str) -> np.ndarray:
    return np.concatenate([source_trajectory_vector(row, op, s) for s in ADDRESSES])


def full_trajectory_vector(row: dict[str, Any]) -> np.ndarray:
    return np.concatenate([operation_trajectory_vector(row, op) for op in OPS])


def _fit_matrix(X: np.ndarray, k: int) -> tuple[dict[str, Any], np.ndarray]:
    mu, sd = fit_standardizer(X)
    Xn = (X - mu) / sd
    centers, labels = deterministic_kmeans(Xn, k)
    spec = {
        'mu': mu.tolist(),
        'sd': sd.tolist(),
        'centers': centers.tolist(),
        'k': int(centers.shape[0]),
        'dimension': int(X.shape[1]),
    }
    return spec, labels


def fit_candidate(rows: list[dict[str, Any]], family: str, k: int) -> tuple[dict[str, Any], list[tuple[int, ...]]]:
    models: dict[str, Any] = {}
    codes: dict[str, np.ndarray] = {}

    if family == 'FULL':
        X = np.stack([full_trajectory_vector(r) for r in rows])
        spec, labels = _fit_matrix(X, k)
        models['FULL'] = spec
        descs = [(int(labels[i]),) for i in range(len(rows))]
        components = ['FULL']
    elif family == 'BY_OPERATION':
        components = list(OPS)
        for op in OPS:
            X = np.stack([operation_trajectory_vector(r, op) for r in rows])
            spec, labels = _fit_matrix(X, k)
            models[op] = spec
            codes[op] = labels
        descs = [tuple(int(codes[op][i]) for op in OPS) for i in range(len(rows))]
    elif family == 'SUBSTITUTE_BY_SOURCE':
        components = [f'SUBSTITUTE:{s}' for s in ADDRESSES]
        for source in ADDRESSES:
            key = f'SUBSTITUTE:{source}'
            X = np.stack([source_trajectory_vector(r, 'SUBSTITUTE', source) for r in rows])
            spec, labels = _fit_matrix(X, k)
            models[key] = spec
            codes[key] = labels
        descs = [tuple(int(codes[key][i]) for key in components) for i in range(len(rows))]
    elif family == 'ALL_BY_OP_SOURCE':
        components = [f'{op}:{s}' for op in OPS for s in ADDRESSES]
        for op in OPS:
            for source in ADDRESSES:
                key = f'{op}:{source}'
                X = np.stack([source_trajectory_vector(r, op, source) for r in rows])
                spec, labels = _fit_matrix(X, k)
                models[key] = spec
                codes[key] = labels
        descs = [tuple(int(codes[key][i]) for key in components) for i in range(len(rows))]
    else:
        raise ValueError(family)

    candidate = {
        'schema': 'co-qwen3-8b-joint-typed-trajectory-candidate-v1',
        'candidate_name': f'JT_{family}_K{k}',
        'family': family,
        'k_requested': int(k),
        'components': components,
        'models': models,
        'feature_transform': 'LOG_LEVEL_DELTA',
        'experimental_intervention': 'ZERO_SUFFIX',
        'horizons': [1, 2],
        'metrics': list(METRICS),
        'addresses': list(ADDRESSES),
        'operations': list(OPS),
        'fit_of_codebooks_used_behavioral_target': False,
    }
    return candidate, descs


def _vector_for_component(row: dict[str, Any], family: str, component: str) -> np.ndarray:
    if family == 'FULL':
        return full_trajectory_vector(row)
    if family == 'BY_OPERATION':
        return operation_trajectory_vector(row, component)
    if family in {'SUBSTITUTE_BY_SOURCE', 'ALL_BY_OP_SOURCE'}:
        op, source = component.split(':', 1)
        return source_trajectory_vector(row, op, source)
    raise ValueError(family)


def encode_rows(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    family = str(candidate['family'])
    components = list(candidate['components'])
    models = candidate['models']
    codes: dict[str, np.ndarray] = {}
    for component in components:
        X = np.stack([_vector_for_component(r, family, component) for r in rows])
        spec = models[component]
        mu = np.asarray(spec['mu'], dtype=np.float64)
        sd = np.asarray(spec['sd'], dtype=np.float64)
        centers = np.asarray(spec['centers'], dtype=np.float64)
        codes[component] = assign_clusters(X, mu, sd, centers)
    return [tuple(int(codes[c][i]) for c in components) for i in range(len(rows))]


def collision_metrics(descs: list[tuple[int, ...]], targets: list[str]) -> dict[str, int | float]:
    fibers: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for d, t in zip(descs, targets):
        fibers[d].append(t)
    collision_fibers = 0
    collision_pairs = 0
    nontrivial_fibers = 0
    systems_nontrivial = 0
    for tgts in fibers.values():
        if len(tgts) > 1:
            nontrivial_fibers += 1
            systems_nontrivial += len(tgts)
        counts: dict[str, int] = defaultdict(int)
        for t in tgts:
            counts[t] += 1
        if len(counts) > 1:
            collision_fibers += 1
            total = len(tgts) * (len(tgts) - 1) // 2
            same = sum(n * (n - 1) // 2 for n in counts.values())
            collision_pairs += total - same
    n = len(descs)
    return {
        'n_systems': n,
        'n_fibers': len(fibers),
        'n_nontrivial_fibers': nontrivial_fibers,
        'systems_in_nontrivial_fibers': systems_nontrivial,
        'nontrivial_coverage': systems_nontrivial / n if n else 0.0,
        'n_collision_fibers': collision_fibers,
        'n_collision_pairs': collision_pairs,
    }


def collision_examples(rows: list[dict[str, Any]], descs: list[tuple[int, ...]], targets: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, d in enumerate(descs):
        groups[d].append(i)
    out = []
    for d, idxs in groups.items():
        tg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i in idxs:
            tg[targets[i]].append({'panel': rows[i].get('_panel_tag'), 'unit': int(rows[i]['unit'])})
        if len(tg) <= 1:
            continue
        out.append({
            'descriptor': list(d),
            'members': [
                {'target': json.loads(t), 'systems': systems}
                for t, systems in sorted(tg.items())
            ],
        })
    return out
