#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json'
CONTRACT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_CONTRACT_16/CO_FORMULA_CONTRACT_16_FROZEN.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_MEASUREMENT_17'
EXPECTED_SOURCE_SHA = '92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4'
EXPECTED_CONTRACT_SHA = 'c17c401ff51751b16c33426a65dda95d9659b15e4c34c907de565de7ef8d0d04'
N = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def row_key(row: dict) -> str:
    b = row['bindings']; d = row['donor_values']
    raw = '|'.join(str(int(b[a])) for a in ('H1','H2','H3')) + '|' + '|'.join(str(int(d[a])) for a in ('H1','H2','H3'))
    return hashlib.sha256(raw.encode()).hexdigest()


for path, expected, label in ((SOURCE, EXPECTED_SOURCE_SHA, 'Validation 13 source panel'), (CONTRACT, EXPECTED_CONTRACT_SHA, 'Formula Contract 16 freeze')):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {label}: {path}')
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'{label} SHA mismatch: expected {expected}, got {actual}')

panel = json.loads(SOURCE.read_text())
rows = sorted(panel['rows'], key=row_key)[:N]
schedule = []
for i, row in enumerate(rows, 1):
    schedule.append({
        'unit': i,
        'source_unit': int(row['unit']),
        'bindings': row['bindings'],
        'donor_values': row['donor_values'],
    })

OUT.mkdir(parents=True, exist_ok=True)
schedule_path = OUT / 'CO_FORMULA_MEASUREMENT_17_SCHEDULE_SEALED.json'
schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + '\n')
prereg = {
    'schema': 'co-qwen3-8b-formula-measurement-17-prereg-v1',
    'status': 'FORMULA_MEASUREMENT_17_PREPARED',
    'generalization_level': 'OP',
    'operation_algebra': ['ID','KEEP','REVOKE','SUBSTITUTE'],
    'source_panel': str(SOURCE),
    'source_panel_sha256': EXPECTED_SOURCE_SHA,
    'formula_contract_16_path': str(CONTRACT),
    'formula_contract_16_sha256': EXPECTED_CONTRACT_SHA,
    'selection': '12 rows with lexicographically smallest SHA256 of binding+donor tuple; behavior-blind',
    'n': N,
    'schedule_sha256': sha256(schedule_path),
    'measurement_uses_behavioral_target': False,
    'P_star_size': 3,
    'Delta_q': [0.5,0.3,0.2],
    'finite_numeric_support_rule': 'original, donor, then deterministic wrapped offsets until three unique values in 30..89',
    'Gamma_contexts_per_direction': 2,
    'R_q_interventions': ['CONTROL','ZERO_LOCAL','HEAD_ROTATE_LOCAL'],
    'R_auxiliary_contexts': 2,
    'R_return_contexts': 2,
    'Lambda_primary_object': 'joint ordered-top2 H1/H2/H3 signature by declared internal intervention',
    'fixed_partition_resolution': '1/16 for normalized continuous coordinates; exact categorical signatures otherwise',
    'resolution_selected_using_A_dagger': False,
    'candidate_family_count': 11,
    'candidate_families': ['GAMMA','DELTA','R','LAMBDA','GAMMA+DELTA','GAMMA+R','GAMMA+LAMBDA','DELTA+R','DELTA+LAMBDA','R+LAMBDA','FULL_CO_OP'],
    'development_selection_rule': {
        'collision_pairs': 0,
        'nontrivial_coverage_min': 0.20,
        'fiber_fraction_max': 0.80,
        'sufficiency_loss_max': 0.0,
        'ami_above_fixed_fiber_permutation_95': True,
        'coarsest_admissible_wins': True,
    },
    'holdout_policy': 'If and only if a development candidate is found, freeze it and open one fresh holdout. No development refit is allowed on holdout.',
}
prereg_path = OUT / 'CO_FORMULA_MEASUREMENT_17_PREREGISTRATION.json'
prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO FORMULA MEASUREMENT 17 PREPARED]')
print('[CONTRACT SHA256]', EXPECTED_CONTRACT_SHA)
print('[SOURCE SHA256]', EXPECTED_SOURCE_SHA)
print('[SELECTED SYSTEMS]', N)
print('[SCHEDULE]', schedule_path)
print('[PREREGISTRATION]', prereg_path)
