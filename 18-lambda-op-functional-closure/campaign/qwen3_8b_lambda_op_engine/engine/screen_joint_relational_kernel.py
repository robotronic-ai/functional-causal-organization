#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from joint_relational_kernel import FAMILIES, collision_examples, collision_metrics, descriptor, load_rows, stable, target_vector


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panels', nargs='+', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    paths = [Path(x) for x in args.panels]
    rows = load_rows(paths)
    targets = [stable(target_vector(r)) for r in rows]
    records = []
    for family in FAMILIES:
        descs = [descriptor(r, family) for r in rows]
        metrics = collision_metrics(descs, targets)
        records.append({
            'family': family,
            'descriptor_length': len(descs[0]),
            'metrics': metrics,
            'collision_examples': collision_examples(rows, descs, targets),
        })

    def rank(rec):
        m = rec['metrics']
        closed = int(m['n_collision_pairs']) == 0
        coverage = float(m['nontrivial_coverage'])
        useful = closed and coverage >= 0.20
        return (
            0 if useful else 1 if closed else 2,
            int(m['n_collision_pairs']),
            -coverage,
            int(rec['descriptor_length']),
            int(m['n_fibers']),
        )

    records.sort(key=rank)
    best = records[0]
    m = best['metrics']
    useful = int(m['n_collision_pairs']) == 0 and float(m['nontrivial_coverage']) >= 0.20
    status = 'USEFUL_EXPLORATORY_CANDIDATE' if useful else 'NO_USEFUL_CANDIDATE'

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'co-qwen3-8b-joint-relational-active-kernel-screen-v1',
        'status': 'TARGET_INFORMED_EXPLORATORY_SCREEN',
        'source_panels': [str(p) for p in paths],
        'source_panel_sha256': {p.name: sha256(p) for p in paths},
        'n_systems': len(rows),
        'families': list(FAMILIES),
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'records': records,
    }
    candidate = {
        'schema': 'co-qwen3-8b-joint-relational-active-kernel-candidate-v1',
        'candidate_name': best['family'],
        'family': best['family'],
        'descriptor_length': best['descriptor_length'],
        'status': status,
        'selection_used_behavioral_target': True,
        'feature_definition_used_behavioral_target': False,
        'representation': 'Target-blind relational topology of H1/H2/H3 active-kernel responses: address ranking, intervention winner, and temporal trend.',
        'source_panel_sha256': result['source_panel_sha256'],
        'source_systems': len(rows),
        'exploratory_metrics': m,
        'useful_candidate_rule': result['useful_candidate_rule'],
        'must_be_validated_on_fresh_data': bool(useful),
    }
    (out / 'JOINT_RELATIONAL_SCREEN_RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (out / 'JOINT_RELATIONAL_BEST_CANDIDATE.json').write_text(json.dumps(candidate, indent=2, sort_keys=True) + '\n')

    print('[CO-U JOINT RELATIONAL SCREEN COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[STATUS]', status)
    print('[CANDIDATE]', candidate['candidate_name'])
    print('[DESCRIPTOR LENGTH]', candidate['descriptor_length'])
    print('[FIBERS]', m['n_fibers'])
    print('[NONTRIVIAL FIBERS]', m['n_nontrivial_fibers'])
    print('[NONTRIVIAL COVERAGE]', f"{float(m['nontrivial_coverage']):.4f}")
    print('[COLLISION FIBERS]', m['n_collision_fibers'])
    print('[COLLISION PAIRS]', m['n_collision_pairs'])
    print('[BEST]', out / 'JOINT_RELATIONAL_BEST_CANDIDATE.json')


if __name__ == '__main__':
    main()
