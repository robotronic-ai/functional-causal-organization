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

BINS = (2, 3, 4, 5, 6, 8)
MAX_FEATURES = 8
BEAM_WIDTH = 48
COVERAGE_MIN = 0.20
DISTORTION_MAX = 0.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def family_indices(names: list[str]) -> dict[str, list[int]]:
    return {
        'SUBSTITUTE_ONLY': [i for i, n in enumerate(names) if n.startswith('SUBSTITUTE:')],
        'SUBSTITUTE_DONOR': [i for i, n in enumerate(names) if n.startswith('SUBSTITUTE:') and n.endswith('DONOR_MINUS_ORIGINAL')],
        'SUBSTITUTE_NONE': [i for i, n in enumerate(names) if n.startswith('SUBSTITUTE:') and n.endswith('NONE_MINUS_ORIGINAL')],
        'REVOKE_SUBSTITUTE': list(range(len(names))),
    }


def score_specs(Z: np.ndarray, specs: tuple[tuple[int, int], ...], edge_map: dict[tuple[int, int], list[float]], targets: list[str], units: list[int]) -> dict[str, Any]:
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


def state_key(sc: dict[str, Any], n_features: int) -> tuple[Any, ...]:
    c = sc['closure']
    f = sc['fidelity']
    useful = (
        int(c['n_collision_pairs']) == 0
        and float(c['nontrivial_coverage']) >= COVERAGE_MIN
        and float(f['within_fiber_distortion_fraction']) <= DISTORTION_MAX
    )
    return (
        0 if useful else 1,
        int(c['n_collision_pairs']),
        int(c['n_collision_fibers']),
        0 if float(c['nontrivial_coverage']) >= COVERAGE_MIN else 1,
        float(f['within_fiber_distortion_fraction']),
        -float(c['nontrivial_coverage']),
        int(c['n_fibers']),
        n_features,
    )


def panel_metrics(Z: np.ndarray, labels: list[str], targets: list[str], units: list[int], idxs: list[int]) -> dict[str, Any]:
    sub_labels = [labels[i] for i in idxs]
    sub_targets = [targets[i] for i in idxs]
    sub_units = [units[i] for i in idxs]
    return {
        'closure': closure_metrics(sub_labels, sub_targets, sub_units),
        'fidelity': fidelity_metrics(Z[idxs], sub_labels),
    }


def search_family(Z: np.ndarray, idxs: list[int], targets: list[str], units: list[int], edge_map: dict[tuple[int, int], list[float]]) -> list[dict[str, Any]]:
    beam: list[tuple[tuple[tuple[int, int], ...], dict[str, Any]]] = [(tuple(), score_specs(Z, tuple(), edge_map, targets, units))]
    useful_states: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], ...]] = {tuple()}

    for _depth in range(1, MAX_FEATURES + 1):
        expanded_states: list[tuple[tuple[tuple[int, int], ...], dict[str, Any]]] = []
        for specs, _old_sc in beam:
            used = {fi for fi, _bins in specs}
            for fi in idxs:
                if fi in used:
                    continue
                for bins in BINS:
                    if not edge_map.get((fi, bins)):
                        continue
                    nxt = tuple(sorted(specs + ((fi, bins),)))
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    sc = score_specs(Z, nxt, edge_map, targets, units)
                    expanded_states.append((nxt, sc))
                    c = sc['closure']
                    f = sc['fidelity']
                    if (
                        int(c['n_collision_pairs']) == 0
                        and float(c['nontrivial_coverage']) >= COVERAGE_MIN
                        and float(f['within_fiber_distortion_fraction']) <= DISTORTION_MAX
                    ):
                        useful_states.append({'specs': nxt, 'score': sc})
        if not expanded_states:
            break
        expanded_states.sort(key=lambda item: state_key(item[1], len(item[0])) + (item[0],))
        beam = expanded_states[:BEAM_WIDTH]
        if useful_states:
            # Keep one additional depth of exploration only if the first useful state is extremely close to the coverage floor.
            best_cov = max(float(x['score']['closure']['nontrivial_coverage']) for x in useful_states)
            if best_cov >= 0.30:
                break

    return useful_states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--construction-panel', required=True)
    ap.add_argument('--opened-validation-panel', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    construction_path = Path(args.construction_panel)
    validation_path = Path(args.opened_validation_panel)
    construction_sha = sha256(construction_path)
    validation_sha = sha256(validation_path)
    if construction_sha != '9b2b9ec96836d39383852ac666ddc2c4aa362473ebed7e9b52d24270e6a8d90a':
        raise ValueError(f'unexpected Construction 12 panel hash: {construction_sha}')
    if validation_sha != '09b7c65d30754f999fe1d93ed53ca2df4eba982ae1ac4c4ba3f740a86c8f4d40':
        raise ValueError(f'unexpected opened Validation 12 panel hash: {validation_sha}')
    construction = json.loads(construction_path.read_text())
    validation = json.loads(validation_path.read_text())
    for name, panel in (('construction', construction), ('opened_validation', validation)):
        if int(panel.get('n_skipped', 0)) != 0:
            raise ValueError(f'{name} panel contains technical skips')
        if not panel.get('rows'):
            raise ValueError(f'{name} panel has no rows')

    rows_a = list(construction['rows'])
    rows_b = list(validation['rows'])
    if len(rows_a) != 32 or len(rows_b) != 64:
        raise ValueError(f'unexpected source sizes: construction={len(rows_a)} validation={len(rows_b)}')
    rows = rows_a + rows_b
    source_tag = ['construction_12'] * len(rows_a) + ['opened_validation_12'] * len(rows_b)

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
        for bins in BINS:
            edge_map[(fi, bins)] = quantile_edges(Z[:, fi], bins)

    group_a = list(range(0, len(rows_a)))
    group_b = list(range(len(rows_a), len(rows)))
    all_candidates: list[dict[str, Any]] = []
    family_records: list[dict[str, Any]] = []

    for family, idxs in family_indices(names).items():
        useful_states = search_family(Z, idxs, targets, units, edge_map)
        family_records.append({'family': family, 'n_useful_states': len(useful_states)})
        for item in useful_states:
            specs = item['specs']
            sc = item['score']
            pa = panel_metrics(Z, sc['labels'], targets, units, group_a)
            pb = panel_metrics(Z, sc['labels'], targets, units, group_b)
            all_candidates.append({
                'family': family,
                'specs': specs,
                'score': sc,
                'construction_12': pa,
                'opened_validation_12': pb,
                'minimum_open_panel_coverage': min(
                    float(pa['closure']['nontrivial_coverage']),
                    float(pb['closure']['nontrivial_coverage']),
                ),
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if all_candidates:
        all_candidates.sort(key=lambda r: (
            -float(r['minimum_open_panel_coverage']),
            -float(r['score']['closure']['nontrivial_coverage']),
            float(r['score']['fidelity']['within_fiber_distortion_fraction']),
            len(r['specs']),
            str(r['family']),
            r['specs'],
        ))
        best = all_candidates[0]
        specs_expanded = best['score']['expanded_specs']
        status = 'USEFUL_EXPLORATORY_CANDIDATE'
        candidate = {
            'schema': 'co-qwen3-8b-content-typed-pooled-candidate-v1',
            'candidate_name': f"CONTENT_TYPED_POOLED96_{best['family']}_{len(specs_expanded)}F",
            'status': status,
            'family': best['family'],
            'source_panels': {
                'construction_12': {'sha256': sha256(construction_path), 'n_systems': len(rows_a)},
                'opened_validation_12': {'sha256': sha256(validation_path), 'n_systems': len(rows_b)},
            },
            'source_systems': len(rows),
            'feature_names': names,
            'standardizer': {'mu': mu.tolist(), 'sd': sd.tolist()},
            'selected_quantized_features': specs_expanded,
            'pooled_closure': best['score']['closure'],
            'pooled_fidelity': best['score']['fidelity'],
            'per_open_panel_metrics': {
                'construction_12': best['construction_12'],
                'opened_validation_12': best['opened_validation_12'],
            },
            'descriptor_uses_baseline_outcome': False,
            'descriptor_uses_baseline_absolute_logprob': False,
            'feature_edges_fit_without_behavior': True,
            'feature_selection_used_behavioral_target': True,
            'opened_validation_12_is_training_data_after_refutation': True,
            'must_be_validated_on_fresh_data': True,
            'promotion_rule': {
                'n_collision_pairs': 0,
                'nontrivial_coverage_min': COVERAGE_MIN,
                'within_fiber_distortion_fraction_max': DISTORTION_MAX,
            },
            'scope_note': 'Exploratory pooled refinement of the typed content-effect Lambda proxy over 96 opened systems. It is not a scalar validation of full C_O.',
        }
        candidate_path = out / 'CONTENT_TYPED_REFINEMENT_13_CANDIDATE.json'
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + '\n')
    else:
        status = 'NO_USEFUL_CANDIDATE'
        candidate = None
        candidate_path = None

    compact_records = []
    for r in sorted(all_candidates, key=lambda x: state_key(x['score'], len(x['specs'])) + (x['family'], x['specs']))[:100]:
        compact_records.append({
            'family': r['family'],
            'selected_features': [
                {
                    **s,
                    'feature_name': names[int(s['feature_index'])],
                }
                for s in r['score']['expanded_specs']
            ],
            'pooled_closure': r['score']['closure'],
            'pooled_fidelity': r['score']['fidelity'],
            'minimum_open_panel_coverage': r['minimum_open_panel_coverage'],
            'per_open_panel_metrics': r['construction_12'],
            'per_open_validation_metrics': r['opened_validation_12'],
        })

    result = {
        'schema': 'co-qwen3-8b-content-typed-pooled-refinement-screen-v1',
        'status': status,
        'source_systems': len(rows),
        'construction_12_systems': len(rows_a),
        'opened_validation_12_systems': len(rows_b),
        'feature_dimension': int(X.shape[1]),
        'search_bins': list(BINS),
        'max_features': MAX_FEATURES,
        'beam_width': BEAM_WIDTH,
        'families': family_records,
        'candidate_name': candidate['candidate_name'] if candidate else None,
        'search_is_exploratory_and_target_informed': True,
        'opened_validation_12_is_no_longer_a_holdout': True,
        'records': compact_records,
        'interpretation': 'Construction 12 and the refuted Validation 12 panel are treated as opened exploratory training data. Quantile edges and standardization are target-blind; beam-search feature selection uses the behavioral closure target.',
    }
    result_path = out / 'CONTENT_TYPED_REFINEMENT_13_RESULT.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')

    print('[CO-U CONTENT-TYPED REFINEMENT 13 COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[CONSTRUCTION 12]', len(rows_a))
    print('[OPENED VALIDATION 12]', len(rows_b))
    print('[STATUS]', status)
    if candidate:
        c = candidate['pooled_closure']
        f = candidate['pooled_fidelity']
        print('[CANDIDATE]', candidate['candidate_name'])
        print('[FAMILY]', candidate['family'])
        print('[SELECTED FEATURES]', len(candidate['selected_quantized_features']))
        for s in candidate['selected_quantized_features']:
            print('[FEATURE]', names[int(s['feature_index'])], 'bins=' + str(s['bins_requested']))
        print('[FIBERS]', c['n_fibers'])
        print('[NONTRIVIAL COVERAGE]', f"{float(c['nontrivial_coverage']):.4f}")
        print('[MIN OPEN-PANEL COVERAGE]', f"{min(float(candidate['per_open_panel_metrics']['construction_12']['closure']['nontrivial_coverage']), float(candidate['per_open_panel_metrics']['opened_validation_12']['closure']['nontrivial_coverage'])):.4f}")
        print('[WITHIN-FIBER DISTORTION]', f"{float(f['within_fiber_distortion_fraction']):.4f}")
        print('[COLLISION PAIRS]', c['n_collision_pairs'])
        print('[CANDIDATE FILE]', candidate_path)
    print('[RESULT]', result_path)


if __name__ == '__main__':
    main()
