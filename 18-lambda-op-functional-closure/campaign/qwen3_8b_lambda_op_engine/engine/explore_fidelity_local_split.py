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
from co_u_vq import deterministic_kmeans
from cross_target_channel import load_rows, stable, target_vector

EPS = 1e-12
K_VALUES = (16, 18, 20)
COVERAGE_MIN = 0.20
DISTORTION_MAX = 0.20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _pair_collisions(targets: list[str]) -> int:
    counts: dict[str, int] = defaultdict(int)
    for t in targets:
        counts[t] += 1
    n = len(targets)
    return n * (n - 1) // 2 - sum(v * (v - 1) // 2 for v in counts.values())


def _closure(labels: list[str], targets: list[str], units: list[int]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    collision_pairs = 0
    collision_fibers = 0
    examples = []
    for lab, idxs in groups.items():
        ts = [targets[i] for i in idxs]
        pairs = _pair_collisions(ts)
        if pairs:
            collision_pairs += pairs
            collision_fibers += 1
            if len(examples) < 12:
                by_target: dict[str, list[int]] = defaultdict(list)
                for i in idxs:
                    by_target[targets[i]].append(units[i])
                examples.append({
                    'fiber': lab,
                    'members': [
                        {'target': json.loads(t), 'units': us}
                        for t, us in sorted(by_target.items())
                    ],
                })
    return {
        'n_collision_fibers': collision_fibers,
        'n_collision_pairs': collision_pairs,
        'collision_examples': examples,
    }


def _fidelity(Z: np.ndarray, labels: list[str]) -> dict[str, Any]:
    overall = Z.mean(axis=0)
    tss = float(np.sum((Z - overall) ** 2))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
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


def _leaf(path: str) -> dict[str, Any]:
    return {'kind': 'LEAF', 'leaf': path}


def _build_tree(scores: np.ndarray, targets: list[str], idxs: list[int], path: str = '') -> dict[str, Any]:
    ts = [targets[i] for i in idxs]
    if len(set(ts)) <= 1 or len(idxs) <= 1:
        return _leaf(path or 'R')

    best = None
    d = scores.shape[1]
    for dim in range(d):
        vals = sorted(set(float(scores[i, dim]) for i in idxs))
        if len(vals) <= 1:
            continue
        thresholds = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
        for thr in thresholds:
            left = [i for i in idxs if float(scores[i, dim]) <= thr]
            right = [i for i in idxs if float(scores[i, dim]) > thr]
            if not left or not right:
                continue
            cp = _pair_collisions([targets[i] for i in left]) + _pair_collisions([targets[i] for i in right])
            retained = (len(left) if len(left) > 1 else 0) + (len(right) if len(right) > 1 else 0)
            balance = min(len(left), len(right))
            key = (cp, -retained, -balance, dim, thr)
            if best is None or key < best[0]:
                best = (key, dim, thr, left, right)
    if best is None:
        # Continuous PCA scores should make this unreachable for distinct rows.
        return _leaf(path or 'R')
    _, dim, thr, left, right = best
    return {
        'kind': 'SPLIT',
        'dimension': int(dim),
        'threshold': float(thr),
        'left': _build_tree(scores, targets, left, path + 'L'),
        'right': _build_tree(scores, targets, right, path + 'R'),
    }


def _tree_leaf(tree: dict[str, Any], score: np.ndarray) -> str:
    node = tree
    while node['kind'] == 'SPLIT':
        if float(score[int(node['dimension'])]) <= float(node['threshold']):
            node = node['left']
        else:
            node = node['right']
    return str(node['leaf'])


def encode_scores(scores: np.ndarray, candidate: dict[str, Any]) -> list[str]:
    centers = np.asarray(candidate['base_quotient']['centers'], dtype=np.float64)
    base = np.argmin(((scores[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    splits = candidate.get('local_splits', {})
    labels = []
    for i, b in enumerate(base.tolist()):
        key = str(int(b))
        if key in splits:
            labels.append(f'b{b}|{_tree_leaf(splits[key], scores[i])}')
        else:
            labels.append(f'b{b}')
    return labels


def encode_rows(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    X = np.stack([rich_vector(r) for r in rows])
    mu = np.asarray(candidate['rich_standardizer']['mu'], dtype=np.float64)
    sd = np.asarray(candidate['rich_standardizer']['sd'], dtype=np.float64)
    components = np.asarray(candidate['pca']['components'], dtype=np.float64)
    Z = (X - mu) / sd
    scores = Z @ components.T
    return encode_scores(scores, candidate), X, Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--fidelity-candidate', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    frozen = json.loads(Path(args.fidelity_candidate).read_text())
    rows = load_rows([panel])
    X = np.stack([rich_vector(r) for r in rows])
    targets = [stable(target_vector(r)) for r in rows]
    units = [int(r['unit']) for r in rows]

    mu = np.asarray(frozen['rich_standardizer']['mu'], dtype=np.float64)
    sd = np.asarray(frozen['rich_standardizer']['sd'], dtype=np.float64)
    components = np.asarray(frozen['pca']['components'], dtype=np.float64)
    Z = (X - mu) / sd
    scores = Z @ components.T

    records = []
    candidates = []
    for k in K_VALUES:
        centers, base = deterministic_kmeans(scores, k)
        base_labels = [f'b{int(x)}' for x in base.tolist()]
        base_fid = _fidelity(Z, base_labels)
        base_cl = _closure(base_labels, targets, units)
        qualified = (
            base_fid['nontrivial_coverage'] >= COVERAGE_MIN
            and base_fid['within_fiber_distortion_fraction'] <= DISTORTION_MAX
        )
        records.append({
            'kind': 'PURE_KMEANS', 'k': k, 'qualified': qualified,
            'fidelity': base_fid, 'closure': base_cl,
        })
        if qualified and base_cl['n_collision_pairs'] == 0:
            candidates.append({
                'mode': 'PURE_KMEANS', 'k': k, 'centers': centers,
                'splits': {}, 'fidelity': base_fid, 'closure': base_cl,
            })

        if not qualified or base_cl['n_collision_pairs'] == 0:
            continue

        groups: dict[int, list[int]] = defaultdict(list)
        for i, b in enumerate(base.tolist()):
            groups[int(b)].append(i)
        splits: dict[str, Any] = {}
        for b, idxs in groups.items():
            if _pair_collisions([targets[i] for i in idxs]) > 0:
                splits[str(b)] = _build_tree(scores, targets, idxs)

        tmp_candidate = {
            'base_quotient': {'centers': centers.tolist()},
            'local_splits': splits,
        }
        split_labels = encode_scores(scores, tmp_candidate)
        split_fid = _fidelity(Z, split_labels)
        split_cl = _closure(split_labels, targets, units)
        split_ok = (
            split_cl['n_collision_pairs'] == 0
            and split_fid['nontrivial_coverage'] >= COVERAGE_MIN
            and split_fid['within_fiber_distortion_fraction'] <= DISTORTION_MAX
        )
        records.append({
            'kind': 'COLLISION_GUIDED_LOCAL_SPLIT', 'k': k, 'qualified': split_ok,
            'n_split_base_fibers': len(splits), 'local_splits': splits,
            'fidelity': split_fid, 'closure': split_cl,
        })
        if split_ok:
            candidates.append({
                'mode': 'COLLISION_GUIDED_LOCAL_SPLIT', 'k': k,
                'centers': centers, 'splits': splits,
                'fidelity': split_fid, 'closure': split_cl,
            })

    if candidates:
        candidates.sort(key=lambda c: (
            -float(c['fidelity']['nontrivial_coverage']),
            float(c['fidelity']['within_fiber_distortion_fraction']),
            len(c['splits']), int(c['k']),
        ))
        best = candidates[0]
        status = 'USEFUL_EXPLORATORY_CANDIDATE'
        name = f"CO_EXPLORE_PCA20_K{best['k']}"
        if best['mode'] == 'COLLISION_GUIDED_LOCAL_SPLIT':
            name += '_LOCAL_SPLIT'
        candidate = {
            'schema': 'co-qwen3-8b-exploratory-co-candidate-v1',
            'candidate_name': name,
            'status': status,
            'selection_mode': best['mode'],
            'source_panel_sha256': sha256(panel),
            'source_systems': len(rows),
            'rich_proxy': frozen['rich_proxy'],
            'rich_standardizer': frozen['rich_standardizer'],
            'pca': frozen['pca'],
            'base_quotient': {
                'kind': 'KMEANS_ON_FROZEN_PCA20',
                'k': int(best['k']),
                'centers': best['centers'].tolist(),
            },
            'local_splits': best['splits'],
            'construction_fidelity': best['fidelity'],
            'construction_closure': best['closure'],
            'behavioral_target_used_to_select_candidate': True,
            'behavioral_target_used_to_select_local_splits': best['mode'] == 'COLLISION_GUIDED_LOCAL_SPLIT',
            'representation_before_local_split_was_target_blind': True,
            'must_be_validated_on_fresh_data': True,
            'promotion_rule': {
                'n_collision_pairs': 0,
                'nontrivial_coverage_min': COVERAGE_MIN,
                'within_fiber_distortion_fraction_max': DISTORTION_MAX,
            },
            'scope_note': 'Exploratory candidate for the current operation-indexed cross-target organizational proxy. It is not a certified scalar of the full four-component C_O profile.',
        }
    else:
        status = 'NO_USEFUL_EXPLORATORY_CANDIDATE'
        candidate = None

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'co-qwen3-8b-exploratory-co-search-v1',
        'status': status,
        'source_systems': len(rows),
        'source_panel_sha256': sha256(panel),
        'frozen_pca_dimension': int(components.shape[0]),
        'search_k_values': list(K_VALUES),
        'records': records,
        'candidate_name': candidate['candidate_name'] if candidate else None,
        'search_is_target_informed': True,
        'interpretation': 'The search is exploratory. It first scores finer fidelity-qualified K values, then permits local PCA splits only inside construction collision fibers. Any selected candidate requires a new independent validation panel.',
    }
    (out / 'CO_EXPLORATORY_SEARCH_11_RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if candidate:
        (out / 'CO_EXPLORATORY_CANDIDATE_11.json').write_text(json.dumps(candidate, indent=2, sort_keys=True) + '\n')

    print('[CO EXPLORATORY SEARCH 11 COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[STATUS]', status)
    if candidate:
        f = candidate['construction_fidelity']
        c = candidate['construction_closure']
        print('[CANDIDATE]', candidate['candidate_name'])
        print('[MODE]', candidate['selection_mode'])
        print('[BASE K]', candidate['base_quotient']['k'])
        print('[LOCAL SPLIT FIBERS]', len(candidate['local_splits']))
        print('[FIBERS]', f['n_fibers'])
        print('[NONTRIVIAL COVERAGE]', f"{float(f['nontrivial_coverage']):.4f}")
        print('[WITHIN-FIBER DISTORTION]', f"{float(f['within_fiber_distortion_fraction']):.4f}")
        print('[COLLISION PAIRS]', c['n_collision_pairs'])
        print('[CANDIDATE FILE]', out / 'CO_EXPLORATORY_CANDIDATE_11.json')
    print('[RESULT]', out / 'CO_EXPLORATORY_SEARCH_11_RESULT.json')


if __name__ == '__main__':
    main()
