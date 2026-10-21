#!/usr/bin/env python3
from __future__ import annotations
import os
import hashlib, json, random, shutil
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
SOURCE = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_REFINEMENT_02'
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_02'
CANDIDATE = SOURCE / 'REFINED_BEST_CANDIDATE.json'
VALUE_SUPPORT = tuple(range(30, 90))
ADDRESSES = ('H1', 'H2', 'H3')
SEED = 202609270902
N = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def existing_keys():
    keys = set()
    for path in (
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT/V427_HOLDOUT_PANEL.json',
        WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01/VALIDATION_PANEL.json',
    ):
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for row in data.get('rows', []):
            b = row['bindings']; d = row['donor_values']
            keys.add(tuple([b[a] for a in ADDRESSES] + [d[a] for a in ADDRESSES]))
    return keys


def make_schedule(seed: int, n: int):
    rng = random.Random(seed)
    rows, seen = [], existing_keys()
    while len(rows) < n:
        bindings = rng.sample(list(VALUE_SUPPORT), len(ADDRESSES))
        donor_pool = [x for x in VALUE_SUPPORT if x not in bindings]
        donors = rng.sample(donor_pool, len(ADDRESSES))
        key = tuple(bindings + donors)
        if key in seen:
            continue
        seen.add(key)
        rows.append({'unit': len(rows)+1, 'bindings': dict(zip(ADDRESSES, bindings)), 'donor_values': dict(zip(ADDRESSES, donors))})
    return rows

if not CANDIDATE.is_file():
    raise FileNotFoundError(f'Run qwen3_8b_lambda_op_refine.py first: {CANDIDATE}')
OUT.mkdir(parents=True, exist_ok=True)
frozen = OUT / 'FROZEN_REFINED_CANDIDATE.json'
shutil.copy2(CANDIDATE, frozen)
schedule = OUT / 'VALIDATION_02_SCHEDULE_SEALED.json'
schedule.write_text(json.dumps(make_schedule(SEED, N), indent=2, sort_keys=True) + '\n')
manifest = {
    'schema': 'co-qwen3-8b-lambda-op-validation-02-prereg-v1',
    'status': 'FROZEN_FOR_FRESH_VALIDATION_02',
    'candidate_sha256': sha256(frozen),
    'schedule_sha256': sha256(schedule),
    'seed': SEED,
    'n': N,
    'selection_was_target_informed': True,
    'validation_must_not_modify_candidate': True,
    'schedule_excludes_exact_rows_seen_in_current_and_validation_01': True,
}
path = OUT / 'VALIDATION_02_PREREGISTRATION.json'
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('[LAMBDA-OP VALIDATION 02 PREPARED]')
print('[CANDIDATE]', frozen)
print('[SCHEDULE]', schedule)
print('[N]', N)
print('[PREREGISTRATION]', path)
