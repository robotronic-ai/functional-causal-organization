#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_compression_fidelity import rich_vector
from cross_target_channel import load_rows, stable, target_vector
from fidelity_first_quotient import encode_rows

EPS = 1e-12


def _fidelity_metrics(X: np.ndarray, labels: np.ndarray, candidate: dict[str, Any]) -> dict[str, Any]:
    mu = np.asarray(candidate['rich_standardizer']['mu'], dtype=np.float64)
    sd = np.asarray(candidate['rich_standardizer']['sd'], dtype=np.float64)
    Z = (X - mu) / sd
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


def _closure_metrics(rows: list[dict[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    targets = [stable(target_vector(r)) for r in rows]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        groups[int(lab)].append(i)
    collision_fibers = 0
    collision_pairs = 0
    examples = []
    for lab, idxs in groups.items():
        counts: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            counts[targets[i]].append(int(rows[i]['unit']))
        if len(counts) <= 1:
            continue
        collision_fibers += 1
        n = len(idxs)
        same = sum(len(v) * (len(v) - 1) // 2 for v in counts.values())
        collision_pairs += n * (n - 1) // 2 - same
        if len(examples) < 8:
            examples.append({
                'fiber': int(lab),
                'members': [
                    {'target': json.loads(t), 'units': units}
                    for t, units in sorted(counts.items())
                ],
            })
    return {
        'n_collision_fibers': collision_fibers,
        'n_collision_pairs': collision_pairs,
        'collision_examples': examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    rows = load_rows([Path(args.panel)])
    candidate = json.loads(Path(args.candidate).read_text())
    labels = encode_rows(rows, candidate)
    X = np.stack([rich_vector(r) for r in rows])
    fidelity = _fidelity_metrics(X, labels, candidate)
    closure = _closure_metrics(rows, labels)

    distortion_max = float(candidate['quotient']['gate']['within_fiber_distortion_fraction_max'])
    coverage_min = float(candidate['quotient']['gate']['nontrivial_coverage_min'])
    collision_pairs = int(closure['n_collision_pairs'])
    coverage = float(fidelity['nontrivial_coverage'])
    distortion = float(fidelity['within_fiber_distortion_fraction'])

    if collision_pairs > 0:
        status = 'REFUTED'
        closure_status = 'REFUTED'
    elif coverage < coverage_min:
        status = 'NONPROMOTED_LOW_COVERAGE'
        closure_status = 'PASS'
    elif distortion > distortion_max:
        status = 'NONPROMOTED_FIDELITY_FAILURE'
        closure_status = 'PASS'
    else:
        status = 'USEFUL_VALIDATED'
        closure_status = 'PASS'

    result = {
        'schema': 'co-qwen3-8b-fidelity-first-validation-v1',
        'candidate_name': candidate['candidate_name'],
        'n_validation_systems': len(rows),
        'closure': closure_status,
        'promotion_status': status,
        'n_fibers': fidelity['n_fibers'],
        'n_nontrivial_fibers': fidelity['n_nontrivial_fibers'],
        'systems_in_nontrivial_fibers': fidelity['systems_in_nontrivial_fibers'],
        'nontrivial_coverage': coverage,
        'within_fiber_distortion_fraction': distortion,
        'rich_variance_explained_by_fibers': fidelity['rich_variance_explained_by_fibers'],
        'n_collision_fibers': closure['n_collision_fibers'],
        'n_collision_pairs': collision_pairs,
        'collision_examples': closure['collision_examples'],
        'validation_rule': {
            'collision_pairs_required': 0,
            'nontrivial_coverage_min': coverage_min,
            'within_fiber_distortion_fraction_max': distortion_max,
        },
        'candidate_refit_on_validation': False,
        'scope_note': 'Validation concerns the frozen cross-target Lambda/active-kernel proxy quotient only, not the full four-component C_O profile.',
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')

    print('[CO-U FIDELITY-FIRST VALIDATION SCORE]')
    print(json.dumps({
        'candidate_name': result['candidate_name'],
        'closure': result['closure'],
        'promotion_status': result['promotion_status'],
        'n_validation_systems': result['n_validation_systems'],
        'n_fibers': result['n_fibers'],
        'n_nontrivial_fibers': result['n_nontrivial_fibers'],
        'nontrivial_coverage': result['nontrivial_coverage'],
        'within_fiber_distortion_fraction': result['within_fiber_distortion_fraction'],
        'n_collision_fibers': result['n_collision_fibers'],
        'n_collision_pairs': result['n_collision_pairs'],
    }, indent=2, sort_keys=True))
    print('[RESULT]', out)


if __name__ == '__main__':
    main()
