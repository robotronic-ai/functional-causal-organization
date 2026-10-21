#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import random
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CROSS_TARGET_CHANNEL_07'
ADDRESSES = ('H1', 'H2', 'H3')
VALUE_SUPPORT = tuple(range(30, 90))
SEED = 202609271607
N = 24


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


OUT.mkdir(parents=True, exist_ok=True)
schedule = OUT / 'CROSS_TARGET_07_SCHEDULE_SEALED.json'
schedule.write_text(json.dumps(make_schedule(), indent=2, sort_keys=True) + '\n')
prereg = {
    'schema': 'co-qwen3-8b-cross-target-active-channel-prereg-v1',
    'status': 'EXPLORATORY_CROSS_TARGET_CHANNEL_PREPARED',
    'seed': SEED,
    'n': N,
    'schedule_sha256': sha256(schedule),
    'operation_algebra': ['ID', 'KEEP', 'REVOKE', 'SUBSTITUTE'],
    'source_addresses': ['H1', 'H2', 'H3'],
    'target_addresses': ['H1', 'H2', 'H3'],
    'experimental_intervention': 'ZERO_SUFFIX',
    'intervention_layers': [21, 22, 23],
    'horizons': [1, 2],
    'functional_target': 'diagonal 12-cell behavior vector: operation x source address',
    'descriptor_construction_uses_behavior': False,
    'candidate_selection_is_exploratory_and_may_use_behavior': True,
    'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
    'scope': 'source-conditioned cross-target active internal response channel',
}
path = OUT / 'CROSS_TARGET_07_PREREGISTRATION.json'
path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO-U CROSS-TARGET CHANNEL PREPARED]')
print('[SCHEDULE]', schedule)
print('[N]', N)
print('[PREREGISTRATION]', path)
