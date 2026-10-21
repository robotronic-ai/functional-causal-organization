#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
MODEL = WORK / 'Qwen3-8B'
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT'
OPS = ('ID', 'KEEP', 'REVOKE', 'SUBSTITUTE')
ADDRESSES = ('H1', 'H2', 'H3')
VALUE_SUPPORT = tuple(range(30, 90))
SPLITS = {
    'construction': {'seed': 2026092642711, 'n': 48},
    'holdout': {'seed': 2026092642712, 'n': 96},
}
SCHEMA = 'co-qwen3-8b-lambda-op-v42.7-current'


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def main():
    if not MODEL.is_dir():
        raise FileNotFoundError(f'Model directory not found: {MODEL}')
    if not (MODEL / 'config.json').is_file():
        raise FileNotFoundError(f'Model config not found: {MODEL / "config.json"}')

    OUT.mkdir(parents=True, exist_ok=True)
    schedules = {}
    for split, cfg in SPLITS.items():
        path = OUT / f'{split.upper()}_SCHEDULE_SEALED.json'
        path.write_text(json.dumps(make_schedule(cfg['seed'], cfg['n']), indent=2, sort_keys=True) + '\n')
        schedules[split] = {
            'path': str(path),
            'file': path.name,
            'sha256': sha256(path),
            'seed': cfg['seed'],
            'n': cfg['n'],
        }

    preregistration = {
        'schema': SCHEMA + '-preregistration-v2',
        'status': 'FROZEN_BEFORE_MODEL_EXECUTION',
        'model': 'Qwen3-8B',
        'model_dir': str(MODEL),
        'operation_algebra': list(OPS),
        'addresses': list(ADDRESSES),
        'primary_layers': [21, 22, 23],
        'reference_layers': [30, 31, 32],
        'descriptor': {
            'name': 'LAMBDA_OP_OPERATION_ADDRESS_EFFECT',
            'domain': 'operation x address',
            'definition': (
                'Keep the complete 4 x 3 operation-address grid. For each non-ID cell, compare target-blind '
                'internal corridor features against ID at the same address. Construction-only KEEP-vs-ID '
                '95th-percentile thresholds define QUIET, ACTIVE_UP, ACTIVE_DOWN and ACTIVE_FLAT states.'
            ),
            'behavior_used_for_descriptor': False,
        },
        'behavioral_target': (
            'Ordered 4 x 3 vector of categorical outcomes ORIGINAL, NONE or DONOR under the same frozen '
            'operation-address grid.'
        ),
        'closure_rule': (
            'PASS iff the behavioral target is constant on every frozen Lambda-op descriptor fiber. '
            'PASS is NONTRIVIAL when at least one fiber contains multiple holdout systems; otherwise '
            'TRIVIAL_INJECTIVE.'
        ),
        'schedules': schedules,
        'guardrail': (
            'Construction contains no behavioral readout. Freeze the descriptor from construction before '
            'executing the sealed behavioral holdout. Do not alter the descriptor or thresholds after holdout.'
        ),
    }
    prereg_path = OUT / 'V427_PREREGISTRATION.json'
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + '\n')

    env = {
        'schema': SCHEMA + '-prepare-result-v2',
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'torch_version': package_version('torch'),
        'transformers_version': package_version('transformers'),
        'accelerate_version': package_version('accelerate'),
        'model_dir': str(MODEL),
        'output_dir': str(OUT),
        'preregistration_sha256': sha256(prereg_path),
        'prepared': True,
    }
    result_path = OUT / 'PREPARE_RESULT.json'
    result_path.write_text(json.dumps(env, indent=2, sort_keys=True) + '\n')

    print('[LAMBDA-OP PREPARE COMPLETE]')
    print('[OUTPUT]', OUT)
    print('[PREREGISTRATION]', prereg_path)
    print('[CONSTRUCTION N]', SPLITS['construction']['n'])
    print('[HOLDOUT N]', SPLITS['holdout']['n'])
    print('[PREREGISTRATION SHA256]', env['preregistration_sha256'])


if __name__ == '__main__':
    main()
