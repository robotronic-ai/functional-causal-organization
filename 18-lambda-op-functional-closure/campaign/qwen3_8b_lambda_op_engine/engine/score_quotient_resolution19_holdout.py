#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from co_base_operator_profile import partition_quality
from formula_measurement17 import ADDRESSES, OPS, Q_INTERVENTIONS, stable

EXPECTED_CANDIDATE_SHA = 'ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc'
EXPECTED_DEVELOPMENT_GLOBAL_AMI_PERM95 = 0.4365622384138
PERMUTATIONS = 5000
PERMUTATION_SEED = 19019


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def lambda_top1_key(row: dict[str, Any]) -> str:
    cells = {(c['operation'], c['source_address']): c for c in row['cells']}
    parts = []
    for op in OPS:
        for source in ADDRESSES:
            rec = cells[(op, source)]['Lambda']
            for q_name in Q_INTERVENTIONS:
                joint = rec['joint_signature_by_intervention'][q_name]
                truncated = [list(role_rank)[:1] for role_rank in joint]
                parts.append(stable(truncated))
    return '|'.join(parts)


def target_key(row: dict[str, Any]) -> str:
    cells = {(c['operation'], c['source_address']): c for c in row['cells']}
    vec = [str(cells[(op, a)]['behavior']['outcome']) for op in OPS for a in ADDRESSES]
    return json.dumps(vec, separators=(',', ':'))


def collision_metrics(labels: list[str], targets: list[str]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    collision_pairs = 0
    collision_fibers = 0
    n_nontrivial = 0
    systems_nontrivial = 0
    for idxs in groups.values():
        if len(idxs) > 1:
            n_nontrivial += 1
            systems_nontrivial += len(idxs)
        counts: dict[str, int] = defaultdict(int)
        for i in idxs:
            counts[targets[i]] += 1
        cp = len(idxs) * (len(idxs) - 1) // 2 - sum(v * (v - 1) // 2 for v in counts.values())
        if cp:
            collision_fibers += 1
            collision_pairs += cp
    n = len(labels)
    return {
        'n_fibers': len(groups),
        'n_collision_fibers': collision_fibers,
        'n_collision_pairs': collision_pairs,
        'n_nontrivial_fibers': n_nontrivial,
        'systems_in_nontrivial_fibers': systems_nontrivial,
        'nontrivial_coverage': systems_nontrivial / n if n else 0.0,
        'fiber_fraction': len(groups) / n if n else 0.0,
    }


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [lambda_top1_key(r) for r in rows]
    targets = [target_key(r) for r in rows]
    metrics = collision_metrics(labels, targets)
    quality = partition_quality(labels, targets, permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
    quality['development_global_grid_max_ami_permutation_95'] = EXPECTED_DEVELOPMENT_GLOBAL_AMI_PERM95
    quality['ami_exceeds_development_global_grid_permutation_95_descriptive'] = bool(
        quality['adjusted_mutual_information'] > EXPECTED_DEVELOPMENT_GLOBAL_AMI_PERM95
    )
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
    if candidate.get('candidate_name') != 'Q18_RHO2_L1_R1__LAMBDA':
        raise RuntimeError('Unexpected candidate name')
    if candidate.get('family') != 'LAMBDA' or candidate.get('components') != ['Lambda']:
        raise RuntimeError('Holdout 19 only validates the frozen Lambda-only Phase 18 candidate')
    if int(candidate['representation']['lambda_rank_depth']) != 1:
        raise RuntimeError('Frozen candidate Lambda rank depth changed')
    if candidate.get('holdout_refit_allowed') is not False:
        raise RuntimeError('Candidate unexpectedly allows holdout refit')

    panel = json.loads(panel_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    if int(panel.get('n_skipped', 0)) != 0 or len(panel.get('rows', [])) != int(prereg['n']):
        raise RuntimeError('Holdout panel incomplete or contains technical skips')
    if sha256(Path(args.panel)) != prereg.get('panel_sha256_expected_after_run', sha256(Path(args.panel))):
        # The prereg does not know the future panel hash; this branch exists only if a caller adds one later.
        raise RuntimeError('Holdout panel SHA mismatch')

    scored = score_rows(list(panel['rows']))
    metrics = scored['metrics']
    quality = scored['partition_quality']
    if metrics['n_collision_pairs'] > 0:
        status = 'PROSPECTIVE_HOLDOUT_REFUTED'
        closure = 'REFUTED'
        promotion = 'REFUTED'
    elif scored['admissible']:
        status = 'PROSPECTIVE_HOLDOUT_PASS'
        closure = 'NON_REFUTED_ON_FRESH_HOLDOUT'
        promotion = 'USEFUL_PROSPECTIVE_CANDIDATE'
    else:
        status = 'PROSPECTIVE_HOLDOUT_NOT_ADMISSIBLE'
        closure = 'NON_REFUTED_BUT_NOT_COMPACTLY_ADMISSIBLE'
        promotion = 'NOT_PROMOTED'

    result = {
        'schema': 'co-qwen3-8b-quotient-resolution-19-holdout-result-v1',
        'status': status,
        'candidate_name': candidate['candidate_name'],
        'family': 'LAMBDA',
        'generalization_level': 'OP',
        'operation_algebra_changed': False,
        'candidate_sha256': sha256(candidate_path),
        'panel_sha256': sha256(panel_path),
        'schedule_sha256': prereg['schedule_sha256'],
        'n_validation_systems': len(panel['rows']),
        'candidate_refit_on_holdout': False,
        'grid_reopened_on_holdout': False,
        'lambda_rank_depth': 1,
        'effective_holdout_rule': 'Formula-17-compatible joint Lambda signature truncated to ordered top-1 for each OP x source x internal-intervention cell.',
        'metrics': metrics,
        'partition_quality': quality,
        'closure': closure,
        'promotion_status': promotion,
        'selection_rule': {
            'collision_pairs': 0,
            'nontrivial_coverage_min': 0.20,
            'fiber_fraction_max': 0.80,
            'sufficiency_loss_max': 0.0,
            'ami_above_single_fixed_candidate_permutation_95': True,
        },
        'development_multiplicity_threshold_usage': 'reported descriptively only; not a holdout selection threshold because the holdout evaluates one frozen candidate',
        'scope_note': 'Prospective finite-battery holdout of one frozen OP-level Lambda quotient. A PASS supports this candidate on the declared Qwen battery; it is not universal closure of full C_O or all Qwen systems.',
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_RESULT.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    summary_path = outdir / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_SUMMARY.md'
    summary_path.write_text(
        '# Qwen3-8B Quotient/Resolution 19 Prospective Holdout\n\n'
        f'- Status: {status}\n'
        f'- Candidate: {candidate["candidate_name"]}\n'
        f'- Systems: {len(panel["rows"])}\n'
        f'- Fibers: {metrics["n_fibers"]}\n'
        f'- Nontrivial coverage: {metrics["nontrivial_coverage"]:.6f}\n'
        f'- Collision pairs: {metrics["n_collision_pairs"]}\n'
        f'- AMI: {quality["adjusted_mutual_information"]:.6f}\n'
        f'- Fixed-candidate permutation 95th percentile: {quality["ami_permutation_95"]:.6f}\n'
        f'- Sufficiency loss: {quality["sufficiency_loss_H_target_given_descriptor_over_H_target"]:.6f}\n\n'
        'No holdout refit or grid reopening is permitted.\n'
    )

    print('[CO QUOTIENT-RESOLUTION 19 HOLDOUT SCORE]')
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
