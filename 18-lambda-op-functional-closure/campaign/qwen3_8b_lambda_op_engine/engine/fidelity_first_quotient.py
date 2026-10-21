#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_compression_fidelity import rich_vector
from cross_target_channel import load_rows, stable, target_vector
from co_u_vq import deterministic_kmeans

PCA_DIMS = (1, 2, 4, 6, 8, 12, 16, 20, 22)
K_VALUES = (2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20)
PCA_GATE = {
    'pairwise_distance_spearman_min': 0.95,
    'knn3_overlap_min': 0.75,
    'loocv_r2_reconstruct_rich_min': 0.60,
}
QUOTIENT_GATE = {
    'within_fiber_distortion_fraction_max': 0.20,
    'nontrivial_coverage_min': 0.20,
}
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def _fit_pca(Z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    d_eff = max(1, min(int(d), Z.shape[0] - 1, Z.shape[1]))
    _u, _s, vt = np.linalg.svd(Z, full_matrices=False)
    components = vt[:d_eff]
    scores = Z @ components.T
    return components, scores


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
    if ra.std() <= EPS or rb.std() <= EPS:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    vals: list[float] = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            vals.append(float(np.linalg.norm(X[i] - X[j])))
    return np.asarray(vals, dtype=np.float64)


def _knn_overlap(X: np.ndarray, S: np.ndarray, k: int = 3) -> float:
    n = len(X)
    if n <= 1:
        return 1.0
    k = min(k, n - 1)
    vals = []
    for i in range(n):
        dx = np.linalg.norm(X - X[i], axis=1)
        ds = np.linalg.norm(S - S[i], axis=1)
        dx[i] = np.inf
        ds[i] = np.inf
        nx = set(np.argsort(dx)[:k].tolist())
        ns = set(np.argsort(ds)[:k].tolist())
        vals.append(len(nx & ns) / k)
    return float(np.mean(vals))


def _loocv_pca_reconstruction_r2(X: np.ndarray, d: int) -> float:
    n = len(X)
    sse = 0.0
    base = 0.0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr = X[mask]
        mu, sd = _fit_standardizer(Xtr)
        Ztr = (Xtr - mu) / sd
        zi = (X[i] - mu) / sd
        components, _ = _fit_pca(Ztr, d)
        pred = (zi @ components.T) @ components
        sse += float(np.sum((zi - pred) ** 2))
        base += float(np.sum(zi ** 2))
    return float(1.0 - sse / base) if base > EPS else 0.0


def pca_metrics(X: np.ndarray, d: int) -> tuple[dict[str, Any], dict[str, Any]]:
    mu, sd = _fit_standardizer(X)
    Z = (X - mu) / sd
    components, scores = _fit_pca(Z, d)
    dx = _pairwise_distances(Z)
    ds = _pairwise_distances(scores)
    total_var = float(np.sum(Z ** 2))
    recon = scores @ components
    resid = float(np.sum((Z - recon) ** 2))
    metrics = {
        'dimension': int(components.shape[0]),
        'pairwise_distance_spearman': _spearman(dx, ds),
        'knn3_overlap': _knn_overlap(Z, scores, 3),
        'loocv_r2_reconstruct_rich': _loocv_pca_reconstruction_r2(X, int(components.shape[0])),
        'construction_variance_retained': 1.0 - resid / total_var if total_var > EPS else 1.0,
    }
    spec = {
        'mu': mu.tolist(),
        'sd': sd.tolist(),
        'components': components.tolist(),
    }
    return metrics, spec


def pca_qualified(m: dict[str, Any]) -> bool:
    return (
        float(m['pairwise_distance_spearman']) >= PCA_GATE['pairwise_distance_spearman_min']
        and float(m['knn3_overlap']) >= PCA_GATE['knn3_overlap_min']
        and float(m['loocv_r2_reconstruct_rich']) >= PCA_GATE['loocv_r2_reconstruct_rich_min']
    )


def _cluster_fidelity(Z: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    center = Z.mean(axis=0)
    tss = float(np.sum((Z - center) ** 2))
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        groups[int(lab)].append(i)
    wss = 0.0
    nontrivial = 0
    systems_nontrivial = 0
    sizes = []
    for idxs in groups.values():
        block = Z[idxs]
        c = block.mean(axis=0)
        wss += float(np.sum((block - c) ** 2))
        sizes.append(len(idxs))
        if len(idxs) > 1:
            nontrivial += 1
            systems_nontrivial += len(idxs)
    frac = wss / tss if tss > EPS else 0.0
    return {
        'n_fibers': len(groups),
        'n_nontrivial_fibers': nontrivial,
        'systems_in_nontrivial_fibers': systems_nontrivial,
        'nontrivial_coverage': systems_nontrivial / len(Z) if len(Z) else 0.0,
        'within_fiber_distortion_fraction': frac,
        'rich_variance_explained_by_fibers': 1.0 - frac,
        'fiber_sizes': sorted(sizes, reverse=True),
    }


def quotient_qualified(m: dict[str, Any]) -> bool:
    return (
        float(m['within_fiber_distortion_fraction']) <= QUOTIENT_GATE['within_fiber_distortion_fraction_max']
        and float(m['nontrivial_coverage']) >= QUOTIENT_GATE['nontrivial_coverage_min']
    )


def _closure_metrics(labels: np.ndarray, targets: list[str]) -> dict[str, Any]:
    groups: dict[int, list[str]] = defaultdict(list)
    for lab, tgt in zip(labels.tolist(), targets):
        groups[int(lab)].append(tgt)
    collision_fibers = 0
    collision_pairs = 0
    examples = []
    for lab, tgts in groups.items():
        counts: dict[str, int] = defaultdict(int)
        for tgt in tgts:
            counts[tgt] += 1
        if len(counts) > 1:
            collision_fibers += 1
            total = len(tgts) * (len(tgts) - 1) // 2
            same = sum(v * (v - 1) // 2 for v in counts.values())
            collision_pairs += total - same
            if len(examples) < 8:
                examples.append({
                    'fiber': int(lab),
                    'target_counts': dict(sorted(counts.items())),
                })
    return {
        'n_collision_fibers': collision_fibers,
        'n_collision_pairs': collision_pairs,
        'collision_examples': examples,
    }


def encode_rich_matrix(X: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    mu = np.asarray(candidate['rich_standardizer']['mu'], dtype=np.float64)
    sd = np.asarray(candidate['rich_standardizer']['sd'], dtype=np.float64)
    components = np.asarray(candidate['pca']['components'], dtype=np.float64)
    centers = np.asarray(candidate['quotient']['centers'], dtype=np.float64)
    Z = (X - mu) / sd
    scores = Z @ components.T
    d2 = ((scores[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def encode_rows(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> np.ndarray:
    X = np.stack([rich_vector(r) for r in rows])
    return encode_rich_matrix(X, candidate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    rows = load_rows([panel])
    X = np.stack([rich_vector(r) for r in rows])
    targets = [stable(target_vector(r)) for r in rows]

    pca_records = []
    specs: dict[int, dict[str, Any]] = {}
    for d in PCA_DIMS:
        if d >= len(rows):
            continue
        m, spec = pca_metrics(X, d)
        rec = {'dimension_requested': d, 'metrics': m, 'qualified': pca_qualified(m)}
        pca_records.append(rec)
        specs[int(m['dimension'])] = spec

    qualified_pca = [r for r in pca_records if r['qualified']]
    selected_pca = None
    quotient_records = []
    candidate = None
    status = 'NO_FIDELITY_QUALIFIED_REPRESENTATION'

    if qualified_pca:
        qualified_pca.sort(key=lambda r: (int(r['metrics']['dimension']), -float(r['metrics']['loocv_r2_reconstruct_rich'])))
        selected_pca = qualified_pca[0]
        d = int(selected_pca['metrics']['dimension'])
        spec = specs[d]
        mu = np.asarray(spec['mu'], dtype=np.float64)
        sd = np.asarray(spec['sd'], dtype=np.float64)
        components = np.asarray(spec['components'], dtype=np.float64)
        Z = (X - mu) / sd
        scores = Z @ components.T

        fitted: dict[int, np.ndarray] = {}
        for k in K_VALUES:
            if k > len(rows):
                continue
            centers, labels = deterministic_kmeans(scores, k)
            m = _cluster_fidelity(Z, labels)
            rec = {
                'k_requested': int(k),
                'k_realized': int(centers.shape[0]),
                'metrics': m,
                'qualified': quotient_qualified(m),
            }
            quotient_records.append(rec)
            fitted[int(k)] = centers

        qualified_q = [r for r in quotient_records if r['qualified']]
        if qualified_q:
            qualified_q.sort(key=lambda r: (int(r['k_realized']), float(r['metrics']['within_fiber_distortion_fraction'])))
            best_q = qualified_q[0]
            k = int(best_q['k_requested'])
            centers = fitted[k]
            labels = np.argmin(((scores[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
            closure = _closure_metrics(labels, targets)
            if int(closure['n_collision_pairs']) == 0:
                status = 'FIDELITY_QUALIFIED_EXPLORATORY_CANDIDATE'
            else:
                status = 'FIDELITY_QUALIFIED_BUT_CLOSURE_REFUTED'
            candidate = {
                'schema': 'co-qwen3-8b-fidelity-first-candidate-v1',
                'candidate_name': f'FIDELITY_FIRST_PCA{d}_K{int(centers.shape[0])}',
                'status': status,
                'source_panel_sha256': sha256(panel),
                'source_systems': len(rows),
                'rich_proxy': 'CROSS_TARGET_ZERO_SUFFIX_LOG_TRAJECTORY_288D',
                'rich_standardizer': {'mu': mu.tolist(), 'sd': sd.tolist()},
                'pca': {
                    'dimension': d,
                    'components': components.tolist(),
                    'construction_metrics': selected_pca['metrics'],
                    'gate': PCA_GATE,
                },
                'quotient': {
                    'kind': 'KMEANS_ON_PCA_SCORES',
                    'k': int(centers.shape[0]),
                    'centers': centers.tolist(),
                    'construction_fidelity': best_q['metrics'],
                    'gate': QUOTIENT_GATE,
                },
                'behavioral_target_used_to_define_features': False,
                'behavioral_target_used_to_fit_representation': False,
                'behavioral_target_used_to_select_pca_dimension': False,
                'behavioral_target_used_to_select_quotient_k': False,
                'behavioral_target_inspected_only_after_freeze': True,
                'construction_closure': closure,
                'must_be_validated_on_fresh_data': status == 'FIDELITY_QUALIFIED_EXPLORATORY_CANDIDATE',
                'scope_note': 'This candidate represents the current cross-target Lambda/active-kernel proxy only. It is not a scalar C_O and does not include certified Gamma, Delta, or R.',
                'precision_note': 'Repeatability was clean in the current bfloat16 CUDA stack. Systematic bias relative to float32 remains unmeasured.',
            }
        else:
            status = 'NO_FIDELITY_QUALIFIED_QUOTIENT'

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'co-qwen3-8b-fidelity-first-screen-v1',
        'status': status,
        'source_panel': str(panel),
        'source_panel_sha256': sha256(panel),
        'n_systems': len(rows),
        'rich_dimension': int(X.shape[1]),
        'selection_is_target_blind': True,
        'pca_gate': PCA_GATE,
        'quotient_gate': QUOTIENT_GATE,
        'pca_records': pca_records,
        'selected_pca': selected_pca,
        'quotient_records': quotient_records,
        'candidate_name': candidate['candidate_name'] if candidate else None,
        'construction_closure': candidate['construction_closure'] if candidate else None,
        'interpretation': 'Representation and quotient are selected from rich-proxy fidelity only. Behavioral closure is inspected after the quotient is frozen.',
    }
    (out / 'FIDELITY_FIRST_10_SCREEN_RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if candidate is not None:
        (out / 'FIDELITY_FIRST_10_CANDIDATE.json').write_text(json.dumps(candidate, indent=2, sort_keys=True) + '\n')

    print('[CO-U FIDELITY-FIRST SCREEN COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[RICH DIMENSION]', X.shape[1])
    print('[STATUS]', status)
    if selected_pca is not None:
        m = selected_pca['metrics']
        print('[PCA DIM]', m['dimension'])
        print('[PCA SPEARMAN]', f"{float(m['pairwise_distance_spearman']):.4f}")
        print('[PCA KNN3]', f"{float(m['knn3_overlap']):.4f}")
        print('[PCA LOOCV R2]', f"{float(m['loocv_r2_reconstruct_rich']):.4f}")
    if candidate is not None:
        q = candidate['quotient']['construction_fidelity']
        c = candidate['construction_closure']
        print('[CANDIDATE]', candidate['candidate_name'])
        print('[K]', candidate['quotient']['k'])
        print('[NONTRIVIAL COVERAGE]', f"{float(q['nontrivial_coverage']):.4f}")
        print('[WITHIN-FIBER DISTORTION]', f"{float(q['within_fiber_distortion_fraction']):.4f}")
        print('[COLLISION PAIRS]', c['n_collision_pairs'])
    print('[RESULT]', out / 'FIDELITY_FIRST_10_SCREEN_RESULT.json')
    if candidate is not None:
        print('[CANDIDATE FILE]', out / 'FIDELITY_FIRST_10_CANDIDATE.json')


if __name__ == '__main__':
    main()
