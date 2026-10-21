#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import random
import shutil
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_FIDELITY_FIRST_10/FIDELITY_FIRST_10_CANDIDATE.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FIDELITY_FIRST_VALIDATION_10'
ADDRESSES = ('H1', 'H2', 'H3')
VALUE_SUPPORT = tuple(range(30, 90))
SEED = 202609271910
N = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def existing_keys() -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    paths = (
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT/V427_HOLDOUT_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01/VALIDATION_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_02/VALIDATION_02_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_04/CO_U_CAUSAL_KERNEL_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_VALIDATION_05/VALIDATION_05_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/CO_U_CROSS_TARGET_CHANNEL_07/CROSS_TARGET_07_PANEL.json',
    )
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for row in data.get('rows', []):
            b = row['bindings']
            d = row['donor_values']
            keys.add(tuple([int(b[a]) for a in ADDRESSES] + [int(d[a]) for a in ADDRESSES]))
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


if not SOURCE.is_file():
    raise FileNotFoundError('Run qwen3_8b_fidelity_gate.py first.')
candidate = json.loads(SOURCE.read_text())
if candidate.get('status') != 'FIDELITY_QUALIFIED_EXPLORATORY_CANDIDATE':
    raise RuntimeError('The fidelity-first screen did not produce a validation-eligible candidate.')

OUT.mkdir(parents=True, exist_ok=True)
frozen = OUT / 'FROZEN_FIDELITY_FIRST_CANDIDATE.json'
shutil.copy2(SOURCE, frozen)
schedule = OUT / 'VALIDATION_10_SCHEDULE_SEALED.json'
schedule.write_text(json.dumps(make_schedule(), indent=2, sort_keys=True) + '\n')
prereg = {
    'schema': 'co-qwen3-8b-fidelity-first-validation-prereg-v1',
    'status': 'FRESH_VALIDATION_10_PREPARED',
    'seed': SEED,
    'n': N,
    'candidate_sha256': sha256(frozen),
    'schedule_sha256': sha256(schedule),
    'operation_algebra': ['ID', 'KEEP', 'REVOKE', 'SUBSTITUTE'],
    'source_addresses': ['H1', 'H2', 'H3'],
    'target_addresses': ['H1', 'H2', 'H3'],
    'experimental_intervention': 'ZERO_SUFFIX',
    'intervention_layers': [21, 22, 23],
    'horizons': [1, 2],
    'candidate_frozen_before_validation': True,
    'candidate_refit_on_validation': False,
    'behavioral_target_used_for_candidate_selection': False,
    'promotion_rule': 'zero collision pairs, nontrivial coverage >= 0.20, and within-fiber rich distortion <= 0.20',
    'scope': 'cross-target Lambda/active-kernel proxy only',
}
path = OUT / 'VALIDATION_10_PREREGISTRATION.json'
path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO-U FIDELITY-FIRST VALIDATION 10 PREPARED]')
print('[CANDIDATE]', frozen)
print('[SCHEDULE]', schedule)
print('[N]', N)
print('[PREREGISTRATION]', path)
