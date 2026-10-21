#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from content_typed_kernel import (
    closure_metrics,
    encode_matrix,
    feature_names,
    feature_vector,
    fidelity_metrics,
    quantile_edges,
    stable,
    target_vector,
)

BASE_BINS = (2, 3, 4, 5, 6, 8)
SEED_BINS = (2, 3, 4, 5, 6, 8, 10, 12)
MAX_FEATURES = 8
BEAM_WIDTH = 72
COVERAGE_MIN = 0.20
MIN_PANEL_COVERAGE = 0.20
DISTORTION_MAX = 0.25
EXPECTED_HASHES = {
    'construction_12': '9b2b9ec96836d39383852ac666ddc2c4aa362473ebed7e9b52d24270e6a8d90a',
    'opened_validation_12': '09b7c65d30754f999fe1d93ed53ca2df4eba982ae1ac4c4ba3f740a86c8f4d40',
    'opened_validation_13': '92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4',
}
SEED13_SPECS = ((6, 8), (22, 3), (23, 8))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def score_specs(
    Z: np.ndarray,
    specs: tuple[tuple[int, int], ...],
    edge_map: dict[tuple[int, int], list[float]],
    targets: list[str],
    units: list[int],
) -> dict[str, Any]:
    expanded = [
        {'feature_index': fi, 'bins_requested': bins, 'edges': edge_map[(fi, bins)]}
        for fi, bins in specs
    ]
    labels = encode_matrix(Z, expanded)
    return {
        'labels': labels,
        'closure': closure_metrics(labels, targets, units),
        'fidelity': fidelity_metrics(Z, labels),
        'expanded_specs': expanded,
    }


def panel_metrics(
    Z: np.ndarray,
    labels: list[str],
    targets: list[str],
    units: list[int],
    idxs: list[int],
) -> dict[str, Any]:
    return {
        'closure': closure_metrics([labels[i] for i in idxs], [targets[i] for i in idxs], [units[i] for i in idxs]),
        'fidelity': fidelity_metrics(Z[idxs], [labels[i] for i in idxs]),
    }


def augmented_score(
    Z: np.ndarray,
    specs: tuple[tuple[int, int], ...],
    edge_map: dict[tuple[int, int], list[float]],
    targets: list[str],
    units: list[int],
    panels: dict[str, list[int]],
) -> dict[str, Any]:
    sc = score_specs(Z, specs, edge_map, targets, units)
    per_panel = {name: panel_metrics(Z, sc['labels'], targets, units, idxs) for name, idxs in panels.items()}
    sc['per_panel'] = per_panel
    sc['minimum_open_panel_coverage'] = min(float(v['closure']['nontrivial_coverage']) for v in per_panel.values())
    sc['maximum_open_panel_distortion'] = max(float(v['fidelity']['within_fiber_distortion_fraction']) for v in per_panel.values())
    return sc


def useful(sc: dict[str, Any]) -> bool:
    c = sc['closure']
    f = sc['fidelity']
    return (
        int(c['n_collision_pairs']) == 0
        and float(c['nontrivial_coverage']) >= COVERAGE_MIN
        and float(f['within_fiber_distortion_fraction']) <= DISTORTION_MAX
        and float(sc['minimum_open_panel_coverage']) >= MIN_PANEL_COVERAGE
    )


def rank_key(sc: dict[str, Any], n_features: int, specs: tuple[tuple[int, int], ...]) -> tuple[Any, ...]:
    c = sc['closure']
    f = sc['fidelity']
    return (
        0 if useful(sc) else 1,
        int(c['n_collision_pairs']),
        int(c['n_collision_fibers']),
        0 if float(sc['minimum_open_panel_coverage']) >= MIN_PANEL_COVERAGE else 1,
        0 if float(f['within_fiber_distortion_fraction']) <= DISTORTION_MAX else 1,
        -float(sc['minimum_open_panel_coverage']),
        -float(c['nontrivial_coverage']),
        float(f['within_fiber_distortion_fraction']),
        n_features,
        specs,
    )


def beam_search(
    Z: np.ndarray,
    allowed_indices: list[int],
    bins_values: tuple[int, ...],
    edge_map: dict[tuple[int, int], list[float]],
    targets: list[str],
    units: list[int],
    panels: dict[str, list[int]],
    initial_specs: tuple[tuple[int, int], ...] = tuple(),
    max_features: int = MAX_FEATURES,
) -> list[dict[str, Any]]:
    initial = tuple(sorted(initial_specs))
    beam = [(initial, augmented_score(Z, initial, edge_map, targets, units, panels))]
    seen = {initial}
    found: list[dict[str, Any]] = []
    if useful(beam[0][1]):
        found.append({'specs': initial, 'score': beam[0][1]})
        return found

    depth = len(initial)
    while depth < max_features:
        depth += 1
        expanded: list[tuple[tuple[tuple[int, int], ...], dict[str, Any]]] = []
        for specs, _old in beam:
            used = {fi for fi, _bins in specs}
            for fi in allowed_indices:
                if fi in used:
                    continue
                for bins in bins_values:
                    if not edge_map.get((fi, bins)):
                        continue
                    nxt = tuple(sorted(specs + ((fi, bins),)))
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    sc = augmented_score(Z, nxt, edge_map, targets, units, panels)
                    expanded.append((nxt, sc))
                    if useful(sc):
                        found.append({'specs': nxt, 'score': sc})
        if not expanded:
            break
        expanded.sort(key=lambda item: rank_key(item[1], len(item[0]), item[0]))
        beam = expanded[:BEAM_WIDTH]
        if found:
            break
    return found


def load_panel(path: Path, label: str, expected_n: int) -> dict[str, Any]:
    actual = sha256(path)
    if actual != EXPECTED_HASHES[label]:
        raise ValueError(f'unexpected {label} panel hash: {actual}')
    data = json.loads(path.read_text())
    if int(data.get('n_skipped', 0)) != 0:
        raise ValueError(f'{label} panel contains technical skips')
    rows = list(data.get('rows', []))
    if len(rows) != expected_n:
        raise ValueError(f'unexpected {label} system count: {len(rows)}')
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--construction-panel', required=True)
    ap.add_argument('--opened-validation12-panel', required=True)
    ap.add_argument('--opened-validation13-panel', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    paths = {
        'construction_12': Path(args.construction_panel),
        'opened_validation_12': Path(args.opened_validation12_panel),
        'opened_validation_13': Path(args.opened_validation13_panel),
    }
    datasets = {
        'construction_12': load_panel(paths['construction_12'], 'construction_12', 32),
        'opened_validation_12': load_panel(paths['opened_validation_12'], 'opened_validation_12', 64),
        'opened_validation_13': load_panel(paths['opened_validation_13'], 'opened_validation_13', 64),
    }

    rows: list[dict[str, Any]] = []
    panels: dict[str, list[int]] = {}
    start = 0
    for name in ('construction_12', 'opened_validation_12', 'opened_validation_13'):
        block = list(datasets[name]['rows'])
        rows.extend(block)
        panels[name] = list(range(start, start + len(block)))
        start += len(block)

    X = np.stack([feature_vector(r) for r in rows])
    names = feature_names()
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    targets = [stable(target_vector(r)) for r in rows]
    units = list(range(1, len(rows) + 1))

    edge_map: dict[tuple[int, int], list[float]] = {}
    for fi in range(Z.shape[1]):
        for bins in sorted(set(BASE_BINS + SEED_BINS)):
            edge_map[(fi, bins)] = quantile_edges(Z[:, fi], bins)

    searches: list[tuple[str, list[dict[str, Any]]]] = []
    all_indices = list(range(len(names)))
    substitute_indices = [i for i, n in enumerate(names) if n.startswith('SUBSTITUTE:')]

    seed_found = beam_search(
        Z, all_indices, SEED_BINS, edge_map, targets, units, panels,
        initial_specs=SEED13_SPECS, max_features=MAX_FEATURES,
    )
    searches.append(('SEED13_EXPANSION', seed_found))

    substitute_found: list[dict[str, Any]] = []
    if not seed_found:
        substitute_found = beam_search(
            Z, substitute_indices, BASE_BINS, edge_map, targets, units, panels,
            initial_specs=tuple(), max_features=MAX_FEATURES,
        )
    searches.append(('SUBSTITUTE_ONLY_GLOBAL', substitute_found))

    global_found: list[dict[str, Any]] = []
    if not seed_found and not substitute_found:
        global_found = beam_search(
            Z, all_indices, BASE_BINS, edge_map, targets, units, panels,
            initial_specs=tuple(), max_features=MAX_FEATURES,
        )
    searches.append(('REVOKE_SUBSTITUTE_GLOBAL', global_found))

    candidates: list[dict[str, Any]] = []
    for mode, items in searches:
        for item in items:
            candidates.append({'mode': mode, **item})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidate = None
    if candidates:
        candidates.sort(key=lambda r: (
            len(r['specs']),
            -float(r['score']['minimum_open_panel_coverage']),
            -float(r['score']['closure']['nontrivial_coverage']),
            float(r['score']['fidelity']['within_fiber_distortion_fraction']),
            r['mode'],
            r['specs'],
        ))
        best = candidates[0]
        specs_expanded = best['score']['expanded_specs']
        candidate = {
            'schema': 'co-qwen3-8b-content-typed-pooled160-candidate-v1',
            'candidate_name': f"CONTENT_TYPED_POOLED160_{best['mode']}_{len(specs_expanded)}F",
            'status': 'USEFUL_EXPLORATORY_CANDIDATE',
            'search_mode': best['mode'],
            'source_systems': len(rows),
            'source_panels': {
                name: {'sha256': sha256(paths[name]), 'n_systems': len(datasets[name]['rows'])}
                for name in paths
            },
            'feature_names': names,
            'standardizer': {'mu': mu.tolist(), 'sd': sd.tolist()},
            'selected_quantized_features': specs_expanded,
            'pooled_closure': best['score']['closure'],
            'pooled_fidelity': best['score']['fidelity'],
            'per_open_panel_metrics': best['score']['per_panel'],
            'minimum_open_panel_coverage': best['score']['minimum_open_panel_coverage'],
            'maximum_open_panel_distortion': best['score']['maximum_open_panel_distortion'],
            'descriptor_uses_baseline_outcome': False,
            'descriptor_uses_baseline_absolute_logprob': False,
            'feature_edges_fit_without_behavior': True,
            'feature_selection_used_behavioral_target': True,
            'opened_validation_12_and_13_are_training_data_after_refutation': True,
            'must_be_validated_on_fresh_data': True,
            'promotion_rule': {
                'n_collision_pairs': 0,
                'nontrivial_coverage_min': COVERAGE_MIN,
                'within_fiber_distortion_fraction_max': DISTORTION_MAX,
            },
            'scope_note': 'Exploratory pooled refinement of the typed content-effect Lambda proxy over 160 opened systems. It is not a scalar validation of full C_O.',
        }
        candidate_path = out / 'CONTENT_TYPED_REFINEMENT_14_CANDIDATE.json'
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + '\n')
        status = 'USEFUL_EXPLORATORY_CANDIDATE'
    else:
        candidate_path = None
        status = 'NO_USEFUL_CANDIDATE'

    result = {
        'schema': 'co-qwen3-8b-content-typed-pooled160-refinement-screen-v1',
        'status': status,
        'source_systems': len(rows),
        'panel_sizes': {name: len(datasets[name]['rows']) for name in datasets},
        'feature_dimension': int(X.shape[1]),
        'seed_13_specs': [
            {'feature_index': fi, 'feature_name': names[fi], 'bins_requested': bins}
            for fi, bins in SEED13_SPECS
        ],
        'searches': [{'mode': mode, 'n_useful_states': len(items)} for mode, items in searches],
        'candidate_name': candidate['candidate_name'] if candidate else None,
        'search_is_exploratory_and_target_informed': True,
        'opened_validation_12_and_13_are_no_longer_holdouts': True,
        'interpretation': 'The 32-system construction panel and both refuted 64-system validation panels are opened exploratory training data. Standardization and quantile edges are target-blind; feature selection uses the behavioral closure target. Any selected candidate requires a new independent validation panel.',
    }
    result_path = out / 'CONTENT_TYPED_REFINEMENT_14_RESULT.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')

    print('[CO-U CONTENT-TYPED REFINEMENT 14 COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[CONSTRUCTION 12]', len(datasets['construction_12']['rows']))
    print('[OPENED VALIDATION 12]', len(datasets['opened_validation_12']['rows']))
    print('[OPENED VALIDATION 13]', len(datasets['opened_validation_13']['rows']))
    print('[STATUS]', status)
    if candidate:
        c = candidate['pooled_closure']
        f = candidate['pooled_fidelity']
        print('[CANDIDATE]', candidate['candidate_name'])
        print('[SEARCH MODE]', candidate['search_mode'])
        print('[SELECTED FEATURES]', len(candidate['selected_quantized_features']))
        for s in candidate['selected_quantized_features']:
            print('[FEATURE]', names[int(s['feature_index'])], 'bins=' + str(s['bins_requested']))
        print('[FIBERS]', c['n_fibers'])
        print('[NONTRIVIAL COVERAGE]', f"{float(c['nontrivial_coverage']):.4f}")
        print('[MIN OPEN-PANEL COVERAGE]', f"{float(candidate['minimum_open_panel_coverage']):.4f}")
        print('[WITHIN-FIBER DISTORTION]', f"{float(f['within_fiber_distortion_fraction']):.4f}")
        print('[MAX OPEN-PANEL DISTORTION]', f"{float(candidate['maximum_open_panel_distortion']):.4f}")
        print('[COLLISION PAIRS]', c['n_collision_pairs'])
        print('[CANDIDATE FILE]', candidate_path)
    print('[RESULT]', result_path)


if __name__ == '__main__':
    main()
