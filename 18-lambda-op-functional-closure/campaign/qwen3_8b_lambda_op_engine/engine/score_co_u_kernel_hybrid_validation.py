#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from screen_co_u_causal_kernel import encode, load_panel, stable, target_vector
from co_u_causal_kernel_analysis import collision_metrics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collision_examples(rows: list[dict[str, Any]], descs: list[tuple[int, ...]], targets: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, d in enumerate(descs):
        groups[d].append(i)
    out = []
    for desc, idxs in groups.items():
        tg: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            tg[targets[i]].append(int(rows[i]['unit']))
        if len(tg) <= 1:
            continue
        out.append({
            'descriptor': list(desc),
            'units': [int(rows[i]['unit']) for i in idxs],
            'target_groups': [
                {'target': json.loads(t), 'units': units}
                for t, units in sorted(tg.items())
            ],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    candidate_path = Path(args.candidate)
    rows = load_panel(panel)
    candidate = json.loads(candidate_path.read_text())
    if not candidate.get('slice_models'):
        raise ValueError('candidate has no frozen slice models')

    descs = encode(rows, candidate)
    targets_raw = [target_vector(row) for row in rows]
    targets = [stable(x) for x in targets_raw]
    m = collision_metrics(descs, targets)
    closed = int(m['n_collision_pairs']) == 0
    useful = closed and float(m['nontrivial_coverage']) >= 0.20
    if useful:
        status = 'USEFUL_VALIDATED'
    elif closed:
        status = 'NONPROMOTED_LOW_COVERAGE'
    else:
        status = 'REFUTED'

    result = {
        'schema': 'co-qwen3-8b-active-causal-kernel-hybrid-validation-score-v1',
        'candidate_name': candidate.get('candidate_name'),
        'candidate_sha256': sha256(candidate_path),
        'candidate_origin': candidate.get('status'),
        'validation_panel_sha256': sha256(panel),
        'n_validation_systems': len(rows),
        'closure': 'PASS' if closed else 'REFUTED',
        'promotion_status': status,
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'metrics': m,
        'collision_examples': collision_examples(rows, descs, targets),
        'interpretation': 'Fresh validation of a target-informed exploratory active causal-kernel candidate. Frozen codebooks are applied without refitting.',
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[CO-U ACTIVE KERNEL HYBRID VALIDATION SCORE]')
    print(json.dumps({
        'candidate_name': result['candidate_name'],
        'closure': result['closure'],
        'promotion_status': result['promotion_status'],
        'n_validation_systems': result['n_validation_systems'],
        'n_fibers': m['n_fibers'],
        'n_nontrivial_fibers': m['n_nontrivial_fibers'],
        'nontrivial_coverage': m['nontrivial_coverage'],
        'n_collision_fibers': m['n_collision_fibers'],
        'n_collision_pairs': m['n_collision_pairs'],
    }, indent=2, sort_keys=True))
    print('[RESULT]', out)


if __name__ == '__main__':
    main()
