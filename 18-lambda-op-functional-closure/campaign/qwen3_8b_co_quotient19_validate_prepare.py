#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import random
import shutil
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
SOURCE_CANDIDATE = WORK / 'exports/co_qwen3_8b_closure/CO_U_QUOTIENT_RESOLUTION_18/CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_QUOTIENT_RESOLUTION_HOLDOUT_19'
EXPECTED_CANDIDATE_SHA = 'ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc'
ADDRESSES = ('H1', 'H2', 'H3')
VALUE_SUPPORT = tuple(range(30, 90))
SEED = 202609291901
N = 32


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tuple_key(bindings: dict, donors: dict) -> tuple[int, ...]:
    return tuple([int(bindings[a]) for a in ADDRESSES] + [int(donors[a]) for a in ADDRESSES])


def existing_keys() -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    paths = (
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT/V427_HOLDOUT_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01/VALIDATION_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_02/VALIDATION_02_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_04/CO_U_CAUSAL_KERNEL_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_VALIDATION_05/VALIDATION_05_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CROSS_TARGET_CHANNEL_07/CROSS_TARGET_07_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_EXPLORATORY_VALIDATION_11/VALIDATION_11_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_12/CONTENT_TYPED_12_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_12/CONTENT_TYPED_VALIDATION_12_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json',
    )
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for row in data.get('rows', []):
            if 'bindings' in row and 'donor_values' in row:
                keys.add(tuple_key(row['bindings'], row['donor_values']))
    phase17_schedule = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_MEASUREMENT_17/CO_FORMULA_MEASUREMENT_17_SCHEDULE_SEALED.json'
    if phase17_schedule.is_file():
        for row in json.loads(phase17_schedule.read_text()):
            keys.add(tuple_key(row['bindings'], row['donor_values']))
    return keys


def make_schedule() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    seen = existing_keys()
    rows = []
    while len(rows) < N:
        bindings = rng.sample(list(VALUE_SUPPORT), len(ADDRESSES))
        donor_pool = [x for x in VALUE_SUPPORT if x not in bindings]
        donors = rng.sample(donor_pool, len(ADDRESSES))
        key = tuple(bindings + donors)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'unit': len(rows) + 1,
            'bindings': dict(zip(ADDRESSES, bindings)),
            'donor_values': dict(zip(ADDRESSES, donors)),
        })
    return rows


if not SOURCE_CANDIDATE.is_file():
    raise FileNotFoundError('Missing Phase 18 candidate. Run qwen3_8b_co_quotient18.py first.')
actual = sha256(SOURCE_CANDIDATE)
if actual != EXPECTED_CANDIDATE_SHA:
    raise RuntimeError(f'Phase 18 candidate SHA mismatch: expected {EXPECTED_CANDIDATE_SHA}, got {actual}')
candidate = json.loads(SOURCE_CANDIDATE.read_text())
if candidate.get('candidate_name') != 'Q18_RHO2_L1_R1__LAMBDA' or candidate.get('holdout_refit_allowed') is not False:
    raise RuntimeError('Unexpected or refittable Phase 18 candidate')

OUT.mkdir(parents=True, exist_ok=True)
frozen = OUT / 'FROZEN_CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json'
shutil.copy2(SOURCE_CANDIDATE, frozen)
schedule_path = OUT / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_SCHEDULE_SEALED.json'
schedule_path.write_text(json.dumps(make_schedule(), indent=2, sort_keys=True) + '\n')
prereg = {
    'schema': 'co-qwen3-8b-quotient-resolution-19-holdout-prereg-v1',
    'status': 'PROSPECTIVE_HOLDOUT_19_PREPARED',
    'seed': SEED,
    'n': N,
    'candidate_name': candidate['candidate_name'],
    'candidate_sha256': sha256(frozen),
    'schedule_sha256': sha256(schedule_path),
    'candidate_frozen_before_holdout': True,
    'candidate_refit_on_holdout': False,
    'grid_reopened_on_holdout': False,
    'generalization_level': 'OP',
    'operation_algebra': ['ID','KEEP','REVOKE','SUBSTITUTE'],
    'measured_component': 'Lambda',
    'raw_lambda_signature_depth': 2,
    'frozen_lambda_rank_depth': 1,
    'continuous_bins_inactive_for_selected_lambda_only_family': True,
    'r_rank_depth_inactive_for_selected_lambda_only_family': True,
    'functional_target': 'ordered 12-cell behavior vector: operation x source_address',
    'descriptor_construction_uses_behavioral_target': False,
    'selection_rule': {
        'collision_pairs': 0,
        'nontrivial_coverage_min': 0.20,
        'fiber_fraction_max': 0.80,
        'sufficiency_loss_max': 0.0,
        'ami_above_single_fixed_candidate_permutation_95': True,
    },
    'development_global_grid_max_ami_permutation_95': 0.4365622384138,
    'development_global_threshold_holdout_role': 'descriptive_only',
    'holdout_policy': 'One fresh holdout only for the frozen Phase 18 candidate. No representation refit, no grid reopening, no operation-algebra change.',
}
prereg_path = OUT / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_PREREGISTRATION.json'
prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO QUOTIENT-RESOLUTION 19 HOLDOUT PREPARED]')
print('[CANDIDATE]', frozen)
print('[CANDIDATE SHA256]', sha256(frozen))
print('[N]', N)
print('[SCHEDULE]', schedule_path)
print('[SCHEDULE SHA256]', sha256(schedule_path))
print('[PREREGISTRATION]', prereg_path)
