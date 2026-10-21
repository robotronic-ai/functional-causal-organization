#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from co_base_operator_profile import partition_quality
from score_quotient_resolution19_holdout import lambda_top1_key, target_key, collision_metrics

EXPECTED_CANDIDATE_SHA = 'ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc'
EXPECTED_HOLDOUT19_RESULT_SHA = '4fdeaf9e0145ecb5e99f0bb37a132382f07eb10f6f766e5caf41f973f69832e6'
PERMUTATIONS = 5000
PERMUTATION_SEED = 20020


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [lambda_top1_key(r) for r in rows]
    targets = [target_key(r) for r in rows]
    metrics = collision_metrics(labels, targets)
    quality = partition_quality(labels, targets, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
    admissible = (
        metrics['n_collision_pairs'] == 0
        and metrics['nontrivial_coverage'] >= 0.20
        and metrics['fiber_fraction'] <= 0.80
        and quality['sufficiency_loss_H_target_given_descriptor_over_H_target'] <= 1e-12
        and quality['ami_exceeds_permutation_95']
    )
    return {
        'metrics': metrics,
        'partition_quality': quality,
        'partition_labels': labels,
        'targets': targets,
        'admissible': bool(admissible),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--panel', required=True)
    ap.add_argument('--preregistration', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    candidate_path = Path(args.candidate)
    panel_path = Path(args.panel)
    prereg_path = Path(args.preregistration)
    if sha256(candidate_path) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError('Frozen Phase 18 candidate SHA mismatch')
    candidate = json.loads(candidate_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    if candidate.get('candidate_name') != 'Q18_RHO2_L1_R1__LAMBDA':
        raise RuntimeError('Unexpected candidate name')
    if candidate.get('family') != 'LAMBDA' or candidate.get('components') != ['Lambda']:
        raise RuntimeError('Replication 20 validates the Lambda-only Phase 18 candidate')
    if int(candidate['representation']['lambda_rank_depth']) != 1:
        raise RuntimeError('Frozen candidate Lambda rank depth changed')
    if candidate.get('holdout_refit_allowed') is not False:
        raise RuntimeError('Candidate unexpectedly allows refit')
    if prereg.get('holdout19_result_sha256') != EXPECTED_HOLDOUT19_RESULT_SHA:
        raise RuntimeError('Replication preregistration is not anchored to the passing Holdout 19 result')

    panel = json.loads(panel_path.read_text())
    if int(panel.get('n_skipped', 0)) != 0 or len(panel.get('rows', [])) != int(prereg['n']):
        raise RuntimeError('Replication panel incomplete or contains technical skips')

    scored = score_rows(list(panel['rows']))
    metrics = scored['metrics']
    quality = scored['partition_quality']
    if metrics['n_collision_pairs'] > 0:
        status = 'PROSPECTIVE_REPLICATION_REFUTED'
        closure = 'REFUTED_ON_REPLICATION'
        promotion = 'REPLICATION_REFUTED'
    elif scored['admissible']:
        status = 'PROSPECTIVE_REPLICATION_PASS'
        closure = 'NON_REFUTED_ON_SECOND_FRESH_PANEL'
        promotion = 'REPLICATED_PROSPECTIVE_CANDIDATE'
    else:
        status = 'PROSPECTIVE_REPLICATION_NOT_ADMISSIBLE'
        closure = 'NON_REFUTED_BUT_NOT_COMPACTLY_ADMISSIBLE_ON_REPLICATION'
        promotion = 'REPLICATION_NOT_PROMOTED'

    result = {
        'schema': 'co-qwen3-8b-quotient-resolution-20-replication-result-v1',
        'status': status,
        'candidate_name': candidate['candidate_name'],
        'family': 'LAMBDA',
        'generalization_level': 'OP',
        'operation_algebra_changed': False,
        'candidate_sha256': sha256(candidate_path),
        'panel_sha256': sha256(panel_path),
        'schedule_sha256': prereg['schedule_sha256'],
        'holdout19_result_sha256': prereg['holdout19_result_sha256'],
        'n_replication_systems': len(panel['rows']),
        'candidate_refit_on_replication': False,
        'grid_reopened_on_replication': False,
        'lambda_rank_depth': 1,
        'effective_replication_rule': 'Formula-17-compatible joint Lambda signature truncated to ordered top-1 for each OP x source x internal-intervention cell.',
        'metrics': metrics,
        'partition_quality': quality,
        'closure': closure,
        'promotion_status': promotion,
        'selection_rule': prereg['selection_rule'],
        'scope_note': 'Second prospective finite-battery replication of the same frozen OP-level Lambda quotient. A PASS supports reproducibility on the declared Qwen battery; it is not universal closure of full C_O or all Qwen systems.',
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / 'CO_QUOTIENT_RESOLUTION_20_REPLICATION_RESULT.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    summary_path = outdir / 'CO_QUOTIENT_RESOLUTION_20_REPLICATION_SUMMARY.md'
    summary_path.write_text(
        '# Qwen3-8B Quotient/Resolution 20 Prospective Replication\n\n'
        f'- Status: {status}\n'
        f'- Candidate: {candidate["candidate_name"]}\n'
        f'- Systems: {len(panel["rows"])}\n'
        f'- Fibers: {metrics["n_fibers"]}\n'
        f'- Nontrivial coverage: {metrics["nontrivial_coverage"]:.6f}\n'
        f'- Collision pairs: {metrics["n_collision_pairs"]}\n'
        f'- AMI: {quality["adjusted_mutual_information"]:.6f}\n'
        f'- Fixed-candidate permutation 95th percentile: {quality["ami_permutation_95"]:.6f}\n'
        f'- Sufficiency loss: {quality["sufficiency_loss_H_target_given_descriptor_over_H_target"]:.6f}\n\n'
        'No candidate refit or grid reopening is permitted.\n'
    )

    print('[CO QUOTIENT-RESOLUTION 20 REPLICATION SCORE]')
    print('[STATUS]', status)
    print('[CANDIDATE]', candidate['candidate_name'])
    print('[SYSTEMS]', len(panel['rows']))
    print('[FIBERS]', metrics['n_fibers'])
    print('[NONTRIVIAL COVERAGE]', f"{metrics['nontrivial_coverage']:.4f}")
    print('[COLLISION PAIRS]', metrics['n_collision_pairs'])
    print('[AMI]', f"{quality['adjusted_mutual_information']:.4f}")
    print('[AMI PERM95]', f"{quality['ami_permutation_95']:.4f}")
    print('[SUFFICIENCY LOSS]', f"{quality['sufficiency_loss_H_target_given_descriptor_over_H_target']:.6f}")
    print('[PROMOTION]', promotion)
    print('[RESULT]', result_path)
    print('[SUMMARY]', summary_path)


if __name__ == '__main__':
    main()
