#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from cross_target_channel import ADDRESSES, HORIZONS, METRICS, OPS, descriptor, load_rows

EPS = 1e-12


def _log(x: float) -> float:
    return math.log(max(float(x), EPS))


def rich_vector(row: dict) -> np.ndarray:
    cells = {(str(c['operation']), str(c['source_address'])): c for c in row['cells']}
    vals: list[float] = []
    for op in OPS:
        for source in ADDRESSES:
            cell = cells[(op, source)]
            for target in ADDRESSES:
                records = cell['cross_target_kernel'][target]['ZERO_SUFFIX']
                by_h = {int(r['horizon']): r for r in records}
                for horizon in HORIZONS:
                    rec = by_h[horizon]
                    for metric in METRICS:
                        vals.append(_log(rec[metric]))
    return np.asarray(vals, dtype=np.float64)


def _slices() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    op_indices: dict[str, list[int]] = {op: [] for op in OPS}
    op_source_indices: dict[str, list[int]] = {f'{op}:{s}': [] for op in OPS for s in ADDRESSES}
    idx = 0
    for op in OPS:
        for source in ADDRESSES:
            for _target in ADDRESSES:
                for _horizon in HORIZONS:
                    for _metric in METRICS:
                        op_indices[op].append(idx)
                        op_source_indices[f'{op}:{source}'].append(idx)
                        idx += 1
    return (
        {k: np.asarray(v, dtype=int) for k, v in op_indices.items()},
        {k: np.asarray(v, dtype=int) for k, v in op_source_indices.items()},
    )


OP_IDX, OP_SOURCE_IDX = _slices()


def summary_global_mean(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=1, keepdims=True)


def summary_global_mean_std(X: np.ndarray) -> np.ndarray:
    return np.stack([X.mean(axis=1), X.std(axis=1)], axis=1)


def summary_operation_moments(X: np.ndarray) -> np.ndarray:
    cols = []
    for op in OPS:
        block = X[:, OP_IDX[op]]
        cols.extend([block.mean(axis=1), block.std(axis=1)])
    return np.stack(cols, axis=1)


def summary_op_source_moments(X: np.ndarray) -> np.ndarray:
    cols = []
    for op in OPS:
        for source in ADDRESSES:
            block = X[:, OP_SOURCE_IDX[f'{op}:{source}']]
            cols.extend([block.mean(axis=1), block.std(axis=1)])
    return np.stack(cols, axis=1)


def summary_diag_offdiag_temporal(rows: list[dict]) -> np.ndarray:
    out = []
    for row in rows:
        cells = {(str(c['operation']), str(c['source_address'])): c for c in row['cells']}
        vals: list[float] = []
        for op in OPS:
            for source in ADDRESSES:
                cell = cells[(op, source)]
                diag_h = []
                off_h = []
                for horizon in HORIZONS:
                    diag_rec = {int(r['horizon']): r for r in cell['cross_target_kernel'][source]['ZERO_SUFFIX']}[horizon]
                    diag = float(np.mean([_log(diag_rec[m]) for m in METRICS]))
                    off_vals = []
                    for target in ADDRESSES:
                        if target == source:
                            continue
                        rec = {int(r['horizon']): r for r in cell['cross_target_kernel'][target]['ZERO_SUFFIX']}[horizon]
                        off_vals.extend([_log(rec[m]) for m in METRICS])
                    off = float(np.mean(off_vals))
                    diag_h.append(diag)
                    off_h.append(off)
                vals.extend([
                    0.5 * (diag_h[0] + diag_h[1]),
                    0.5 * (off_h[0] + off_h[1]),
                    diag_h[1] - diag_h[0],
                    off_h[1] - off_h[0],
                ])
        out.append(vals)
    return np.asarray(out, dtype=np.float64)


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    sa = ra.std()
    sb = rb.std()
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            vals.append(float(np.linalg.norm(X[i] - X[j])))
    return np.asarray(vals, dtype=np.float64)


def _pairwise_hamming(descs: list[tuple[int, ...]]) -> np.ndarray:
    vals = []
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            a = descs[i]
            b = descs[j]
            vals.append(sum(x != y for x, y in zip(a, b)) / max(1, len(a)))
    return np.asarray(vals, dtype=np.float64)


def _knn_overlap(X: np.ndarray, S: np.ndarray, k: int = 3) -> float:
    n = len(X)
    if n <= 1:
        return 1.0
    k = min(k, n - 1)
    scores = []
    for i in range(n):
        dx = np.linalg.norm(X - X[i], axis=1)
        ds = np.linalg.norm(S - S[i], axis=1)
        dx[i] = np.inf
        ds[i] = np.inf
        nx = set(np.argsort(dx)[:k].tolist())
        ns = set(np.argsort(ds)[:k].tolist())
        scores.append(len(nx & ns) / k)
    return float(np.mean(scores))


def _loocv_r2(X: np.ndarray, S: np.ndarray, alpha: float = 1.0) -> float:
    n = len(X)
    sse = 0.0
    base = 0.0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr = X[mask]
        Str = S[mask]
        x_mu = Xtr.mean(axis=0)
        x_sd = Xtr.std(axis=0)
        x_sd = np.where(x_sd > 1e-12, x_sd, 1.0)
        s_mu = Str.mean(axis=0)
        s_sd = Str.std(axis=0)
        s_sd = np.where(s_sd > 1e-12, s_sd, 1.0)
        Xn = (Xtr - x_mu) / x_sd
        Sn = (Str - s_mu) / s_sd
        si = (S[i] - s_mu) / s_sd
        yi = (X[i] - x_mu) / x_sd
        gram = Sn.T @ Sn + alpha * np.eye(Sn.shape[1])
        coef = np.linalg.solve(gram, Sn.T @ Xn)
        pred = si @ coef
        sse += float(np.sum((yi - pred) ** 2))
        base += float(np.sum(yi ** 2))
    return float(1.0 - sse / base) if base > 0 else 0.0


def _continuous_metrics(X: np.ndarray, S: np.ndarray) -> dict:
    Xz = _standardize(X)
    Sz = _standardize(S)
    dx = _pairwise_euclidean(Xz)
    ds = _pairwise_euclidean(Sz)
    return {
        'dimension': int(S.shape[1]),
        'pairwise_distance_spearman': _spearman(dx, ds),
        'knn3_overlap': _knn_overlap(Xz, Sz, 3),
        'loocv_r2_reconstruct_rich': _loocv_r2(X, S, alpha=1.0),
    }


def _categorical_metrics(X: np.ndarray, descs: list[tuple[int, ...]]) -> dict:
    Xz = _standardize(X)
    dx = _pairwise_euclidean(Xz)
    dh = _pairwise_hamming(descs)
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, d in enumerate(descs):
        groups.setdefault(d, []).append(i)
    center = Xz.mean(axis=0)
    tss = float(np.sum((Xz - center) ** 2))
    wss = 0.0
    for idxs in groups.values():
        block = Xz[idxs]
        c = block.mean(axis=0)
        wss += float(np.sum((block - c) ** 2))
    return {
        'dimension': len(descs[0]) if descs else 0,
        'n_fibers': len(groups),
        'nontrivial_coverage': sum(len(v) for v in groups.values() if len(v) > 1) / len(descs),
        'pairwise_distance_spearman': _spearman(dx, dh),
        'within_fiber_distortion_fraction': wss / tss if tss > 0 else 0.0,
        'rich_variance_explained_by_fibers': 1.0 - (wss / tss if tss > 0 else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    rows = load_rows([panel])
    X = np.stack([rich_vector(r) for r in rows])

    summaries: dict[str, Callable[[], np.ndarray]] = {
        'GLOBAL_LOG_MEAN_1D': lambda: summary_global_mean(X),
        'GLOBAL_LOG_MEAN_STD_2D': lambda: summary_global_mean_std(X),
        'OPERATION_MOMENTS_8D': lambda: summary_operation_moments(X),
        'OP_SOURCE_MOMENTS_24D': lambda: summary_op_source_moments(X),
        'DIAG_OFFDIAG_TEMPORAL_48D': lambda: summary_diag_offdiag_temporal(rows),
    }
    metrics = {name: _continuous_metrics(X, fn()) for name, fn in summaries.items()}

    topology_descs = [descriptor(r, 'ALL_FLOW_H2_FORWARD') for r in rows]
    metrics['ALL_FLOW_H2_FORWARD_CATEGORICAL'] = _categorical_metrics(X, topology_descs)

    result = {
        'schema': 'co-qwen3-8b-metrology-compression-fidelity-v1',
        'status': 'EXPLORATORY_METROLOGY_AUDIT',
        'source_panel': str(panel),
        'n_systems': len(rows),
        'rich_proxy': {
            'name': 'CROSS_TARGET_ZERO_SUFFIX_LOG_TRAJECTORY',
            'dimension': int(X.shape[1]),
            'axes': ['operation', 'source_address', 'target_address', 'horizon', 'internal_metric'],
            'target_blind': True,
        },
        'full_co_scalarization_auditable': False,
        'reason_full_co_not_auditable': 'This panel does not contain certified Gamma, Delta, or R measurements. The audit covers the current Lambda/active-kernel proxy only.',
        'scalarization_contract': 'C_O is a four-component profile, not a default scalar. One-dimensional summaries below are diagnostics only.',
        'metrics': metrics,
        'interpretation_rule': {
            'distance_spearman': 'Higher is better; 1 means pairwise geometry is rank-preserved.',
            'knn3_overlap': 'Higher is better; 1 means local neighborhoods are preserved.',
            'loocv_r2_reconstruct_rich': 'Higher is better; <=0 means no out-of-sample reconstruction gain over the train-mean baseline.',
            'within_fiber_distortion_fraction': 'Lower is better; 0 means no rich variation remains inside categorical fibers.',
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[COMPRESSION FIDELITY AUDIT COMPLETE]')
    print('[SYSTEMS]', len(rows))
    print('[RICH DIMENSION]', X.shape[1])
    for name, rec in metrics.items():
        print('[SUMMARY]', name, json.dumps(rec, sort_keys=True))
    print('[RESULT]', out)


if __name__ == '__main__':
    main()
