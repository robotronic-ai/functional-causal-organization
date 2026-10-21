#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cross_target_channel import load_rows, stable, target_vector
from explore_fidelity_local_split import encode_rows

EPS = 1e-12


def fidelity_metrics(Z: np.ndarray, labels: list[str]) -> dict[str, Any]:
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


def closure_metrics(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    targets = [stable(target_vector(r)) for r in rows]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    collision_fibers = 0
    collision_pairs = 0
    examples = []
    for lab, idxs in groups.items():
        by_target: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            by_target[targets[i]].append(int(rows[i]['unit']))
        if len(by_target) <= 1:
            continue
        collision_fibers += 1
        n = len(idxs)
        same = sum(len(v) * (len(v) - 1) // 2 for v in by_target.values())
        collision_pairs += n * (n - 1) // 2 - same
        if len(examples) < 12:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    rows = load_rows([Path(args.panel)])
    candidate = json.loads(Path(args.candidate).read_text())
    labels, _X, Z = encode_rows(rows, candidate)
    fidelity = fidelity_metrics(Z, labels)
    closure = closure_metrics(rows, labels)

    rule = candidate['promotion_rule']
    cp = int(closure['n_collision_pairs'])
    cov = float(fidelity['nontrivial_coverage'])
    dist = float(fidelity['within_fiber_distortion_fraction'])
    if cp > 0:
        promotion = 'REFUTED'
        closure_status = 'REFUTED'
    elif cov < float(rule['nontrivial_coverage_min']):
        promotion = 'NONPROMOTED_LOW_COVERAGE'
        closure_status = 'PASS'
    elif dist > float(rule['within_fiber_distortion_fraction_max']):
        promotion = 'NONPROMOTED_FIDELITY_FAILURE'
        closure_status = 'PASS'
    else:
        promotion = 'USEFUL_VALIDATED'
        closure_status = 'PASS'

    result = {
        'schema': 'co-qwen3-8b-exploratory-co-validation-v1',
        'candidate_name': candidate['candidate_name'],
        'selection_mode': candidate['selection_mode'],
        'n_validation_systems': len(rows),
        'closure': closure_status,
        'promotion_status': promotion,
        'n_fibers': fidelity['n_fibers'],
        'n_nontrivial_fibers': fidelity['n_nontrivial_fibers'],
        'systems_in_nontrivial_fibers': fidelity['systems_in_nontrivial_fibers'],
        'nontrivial_coverage': cov,
        'within_fiber_distortion_fraction': dist,
        'rich_variance_explained_by_fibers': fidelity['rich_variance_explained_by_fibers'],
        'n_collision_fibers': closure['n_collision_fibers'],
        'n_collision_pairs': cp,
        'collision_examples': closure['collision_examples'],
        'candidate_refit_on_validation': False,
        'validation_rule': rule,
        'scope_note': 'Validation concerns the frozen exploratory operation-indexed cross-target organizational proxy. It is not a validation of a scalar full four-component C_O.',
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[CO EXPLORATORY VALIDATION 11 SCORE]')
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
