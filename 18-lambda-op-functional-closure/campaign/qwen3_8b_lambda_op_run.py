#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
MODEL = WORK / 'Qwen3-8B'
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT'
ENGINE_ROOT = HERE / 'qwen3_8b_lambda_op_engine'
ENGINE = ENGINE_ROOT / 'engine'
DATA = ENGINE_ROOT / 'data'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def call(*parts):
    cmd = [str(x) for x in parts]
    print('[RUN]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    prereg_path = OUT / 'V427_PREREGISTRATION.json'
    if not prereg_path.is_file():
        raise FileNotFoundError('Run qwen3_8b_lambda_op_prepare.py first.')
    prereg = json.loads(prereg_path.read_text())

    DATA.mkdir(parents=True, exist_ok=True)
    engine_manifest = {
        'schema': prereg['schema'],
        'operation_algebra': prereg['operation_algebra'],
        'schedules': {},
    }
    for stage in ('construction', 'holdout'):
        info = prereg['schedules'][stage]
        src = Path(info['path'])
        if sha256(src) != info['sha256']:
            raise RuntimeError(f'Sealed {stage} schedule SHA256 mismatch')
        dst = DATA / info['file']
        shutil.copy2(src, dst)
        engine_manifest['schedules'][stage] = {
            'file': info['file'],
            'sha256': info['sha256'],
            'seed': info['seed'],
            'n': info['n'],
        }
    (DATA / 'v427_schedule_manifest.json').write_text(json.dumps(engine_manifest, indent=2, sort_keys=True) + '\n')

    construction = OUT / 'V427_CONSTRUCTION_PANEL.json'
    freeze = OUT / 'V427_DESCRIPTOR_FREEZE.json'
    holdout = OUT / 'V427_HOLDOUT_PANEL.json'

    call(
        sys.executable,
        ENGINE / 'run_v427_panel.py',
        '--stage', 'construction',
        '--model-dir', MODEL,
        '--output-dir', OUT,
        '--device', 'cuda',
        '--dtype', 'bfloat16',
    )

    call(
        sys.executable,
        ENGINE / 'freeze_lambda_op_descriptor.py',
        '--construction', construction,
        '--output', freeze,
    )

    call(
        sys.executable,
        ENGINE / 'run_v427_panel.py',
        '--stage', 'holdout',
        '--model-dir', MODEL,
        '--output-dir', OUT,
        '--device', 'cuda',
        '--dtype', 'bfloat16',
    )

    result = {
        'schema': 'co-qwen3-8b-lambda-op-v42.7-run-result-v2',
        'construction_panel': str(construction),
        'construction_sha256': sha256(construction),
        'descriptor_freeze': str(freeze),
        'descriptor_freeze_sha256': sha256(freeze),
        'holdout_panel': str(holdout),
        'holdout_sha256': sha256(holdout),
        'run_complete': True,
    }
    path = OUT / 'RUN_RESULT.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[LAMBDA-OP RUN COMPLETE]')
    print('[DESCRIPTOR FREEZE]', freeze)
    print('[HOLDOUT]', holdout)
    print('[RESULT]', path)


if __name__ == '__main__':
    main()
