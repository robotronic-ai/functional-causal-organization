#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import random
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_12'
ADDRESSES = ('H1', 'H2', 'H3')
VALUE_SUPPORT = tuple(range(30, 90))
SEED = 202609272212
N = 32


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
        WORK / 'exports/co_qwen3_8b_closure/CO_U_EXPLORATORY_VALIDATION_11/VALIDATION_11_PANEL.json',
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
schedule = OUT / 'CONTENT_TYPED_12_SCHEDULE_SEALED.json'
schedule.write_text(json.dumps(make_schedule(), indent=2, sort_keys=True) + '\n')
prereg = {
    'schema': 'co-qwen3-8b-content-typed-prereg-v1',
    'status': 'EXPLORATORY_CONTENT_TYPED_KERNEL_PREPARED',
    'seed': SEED,
    'n': N,
    'schedule_sha256': sha256(schedule),
    'operation_algebra': ['ID', 'KEEP', 'REVOKE', 'SUBSTITUTE'],
    'content_effect_operations': ['REVOKE', 'SUBSTITUTE'],
    'addresses': ['H1', 'H2', 'H3'],
    'experimental_interventions': ['ZERO_SUFFIX', 'HEAD_ROTATE'],
    'intervention_layers': [21, 22, 23],
    'functional_target': 'ordered 12-cell behavior vector: operation x address',
    'descriptor_uses_baseline_outcome': False,
    'descriptor_uses_baseline_absolute_logprob': False,
    'quantile_edges_fit_without_behavior': True,
    'feature_selection_is_exploratory_and_may_use_behavior': True,
    'useful_candidate_rule': 'zero collision pairs, nontrivial coverage >= 0.20, and within-fiber distortion <= 0.25',
    'scope': 'typed content-effect refinement of operation-indexed Lambda proxy',
}
path = OUT / 'CONTENT_TYPED_12_PREREGISTRATION.json'
path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO-U CONTENT-TYPED 12 PREPARED]')
print('[SCHEDULE]', schedule)
print('[N]', N)
print('[PREREGISTRATION]', path)
