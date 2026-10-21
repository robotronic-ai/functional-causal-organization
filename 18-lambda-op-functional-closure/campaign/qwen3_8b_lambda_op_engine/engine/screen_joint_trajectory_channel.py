#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cross_target_channel import load_rows, stable, target_vector
from joint_trajectory_channel import collision_examples, collision_metrics, fit_candidate

FAMILIES = ('FULL', 'BY_OPERATION', 'SUBSTITUTE_BY_SOURCE', 'ALL_BY_OP_SOURCE')
K_VALUES = (2, 3, 4, 6, 8, 12)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rank(rec: dict) -> tuple:
    m = rec['metrics']
    closed = int(m['n_collision_pairs']) == 0
    coverage = float(m['nontrivial_coverage'])
    useful = closed and coverage >= 0.20
    complexity = int(rec['descriptor_length']) * int(rec['k_requested'])
    return (
        0 if useful else 1 if closed else 2,
        int(m['n_collision_pairs']),
        -coverage,
        complexity,
        int(m['n_fibers']),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    rows = load_rows([panel])
    targets = [stable(target_vector(r)) for r in rows]
    records = []
    fitted = {}
    for family in FAMILIES:
        for k in K_VALUES:
            candidate, descs = fit_candidate(rows, family, k)
            m = collision_metrics(descs, targets)
            key = candidate['candidate_name']
            fitted[key] = candidate
            records.append({
                'candidate_name': key,
                'family': family,
                'k_requested': k,
                'descriptor_length': len(descs[0]),
                'metrics': m,
                'collision_examples': collision_examples(rows, descs, targets),
            })
    records.sort(key=rank)
    best_rec = records[0]
    best = fitted[best_rec['candidate_name']]
    m = best_rec['metrics']
    useful = int(m['n_collision_pairs']) == 0 and float(m['nontrivial_coverage']) >= 0.20
    status = 'USEFUL_EXPLORATORY_CANDIDATE' if useful else 'NO_USEFUL_CANDIDATE'
    best.update({
        'status': status,
        'selection_used_behavioral_target': True,
        'feature_definition_used_behavioral_target': False,
        'source_panel_sha256': sha256(panel),
        'source_systems': len(rows),
        'exploratory_metrics': m,
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'must_be_validated_on_fresh_data': bool(useful),
        'representation': 'Quantitative joint typed trajectory code over source-target routing, preserving level and ordered horizon change.',
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'co-qwen3-8b-joint-typed-trajectory-screen-v1',
        'status': 'TARGET_INFORMED_EXPLORATORY_SCREEN',
        'source_panel': str(panel),
        'source_panel_sha256': sha256(panel),
        'n_systems': len(rows),
        'families': list(FAMILIES),
        'k_values': list(K_VALUES),
        'feature_definition_used_behavioral_target': False,
        'codebook_fit_used_behavioral_target': False,
        'selection_used_behavioral_target': True,
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'records': records,
    }
    (out / 'JOINT_TRAJECTORY_08_SCREEN_RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (out / 'JOINT_TRAJECTORY_08_BEST_CANDIDATE.json').write_text(json.dumps(best, indent=2, sort_keys=True) + '\n')

    print('[CO-U JOINT TYPED TRAJECTORY SCREEN COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[STATUS]', status)
    print('[CANDIDATE]', best['candidate_name'])
    print('[FAMILY]', best['family'])
    print('[K]', best['k_requested'])
    print('[DESCRIPTOR LENGTH]', best_rec['descriptor_length'])
    print('[FIBERS]', m['n_fibers'])
    print('[NONTRIVIAL FIBERS]', m['n_nontrivial_fibers'])
    print('[NONTRIVIAL COVERAGE]', f"{float(m['nontrivial_coverage']):.4f}")
    print('[COLLISION FIBERS]', m['n_collision_fibers'])
    print('[COLLISION PAIRS]', m['n_collision_pairs'])
    print('[BEST]', out / 'JOINT_TRAJECTORY_08_BEST_CANDIDATE.json')


if __name__ == '__main__':
    main()
