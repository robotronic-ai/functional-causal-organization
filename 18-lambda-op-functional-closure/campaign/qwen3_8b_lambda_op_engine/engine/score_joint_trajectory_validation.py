#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cross_target_channel import load_rows, stable, target_vector
from joint_trajectory_channel import collision_examples, collision_metrics, encode_rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    candidate_path = Path(args.candidate)
    rows = load_rows([panel])
    candidate = json.loads(candidate_path.read_text())
    descs = encode_rows(rows, candidate)
    targets = [stable(target_vector(r)) for r in rows]
    m = collision_metrics(descs, targets)
    closed = int(m['n_collision_pairs']) == 0
    coverage = float(m['nontrivial_coverage'])
    promotion = 'USEFUL_VALIDATED' if closed and coverage >= 0.20 else ('NONPROMOTED_LOW_COVERAGE' if closed else 'REFUTED')
    result = {
        'schema': 'co-qwen3-8b-joint-typed-trajectory-validation-score-v1',
        'candidate_name': candidate['candidate_name'],
        'candidate_sha256': sha256(candidate_path),
        'validation_panel_sha256': sha256(panel),
        'n_validation_systems': len(rows),
        **m,
        'closure': 'PASS' if closed else 'REFUTED',
        'promotion_status': promotion,
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'collision_examples': collision_examples(rows, descs, targets),
        'interpretation': 'Fresh validation of a frozen quantitative joint typed trajectory candidate. Codebooks and normalization are not refit on validation data.',
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[CO-U JOINT TYPED TRAJECTORY VALIDATION SCORE]')
    print(json.dumps({k: result[k] for k in (
        'candidate_name','closure','promotion_status','n_validation_systems','n_fibers',
        'n_nontrivial_fibers','nontrivial_coverage','n_collision_fibers','n_collision_pairs'
    )}, indent=2, sort_keys=True))
    print('[RESULT]', out)


if __name__ == '__main__':
    main()
