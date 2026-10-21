#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import random
import shutil
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
SOURCE = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT'
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01'
CANDIDATE = SOURCE / 'EXPLORATORY_BEST_CANDIDATE.json'
VALUE_SUPPORT = tuple(range(30, 90))
ADDRESSES = ('H1', 'H2', 'H3')
SEED = 202609270801
N = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def make_schedule(seed: int, n: int):
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < n:
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

if not CANDIDATE.is_file():
    raise FileNotFoundError(f'Run qwen3_8b_lambda_op_explore.py first: {CANDIDATE}')
OUT.mkdir(parents=True, exist_ok=True)
candidate_copy = OUT / 'FROZEN_EXPLORATORY_CANDIDATE.json'
shutil.copy2(CANDIDATE, candidate_copy)
schedule = OUT / 'VALIDATION_SCHEDULE_SEALED.json'
schedule.write_text(json.dumps(make_schedule(SEED, N), indent=2, sort_keys=True) + '\n')
manifest = {
    'schema': 'co-qwen3-8b-lambda-op-candidate-validation-prereg-v1',
    'status': 'FROZEN_FOR_FRESH_VALIDATION',
    'candidate_sha256': sha256(candidate_copy),
    'schedule_sha256': sha256(schedule),
    'seed': SEED,
    'n': N,
    'selection_was_target_informed': True,
    'validation_must_not_modify_candidate': True,
}
path = OUT / 'VALIDATION_PREREGISTRATION.json'
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('[LAMBDA-OP VALIDATION PREPARED]')
print('[CANDIDATE]', candidate_copy)
print('[SCHEDULE]', schedule)
print('[N]', N)
print('[PREREGISTRATION]', path)
