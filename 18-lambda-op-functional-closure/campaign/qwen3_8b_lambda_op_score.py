#!/usr/bin/env python3
from __future__ import annotations
import os

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine' / 'engine'


def main():
    holdout = OUT / 'V427_HOLDOUT_PANEL.json'
    freeze = OUT / 'V427_DESCRIPTOR_FREEZE.json'
    prereg = OUT / 'V427_PREREGISTRATION.json'
    result = OUT / 'QWEN3_8B_LAMBDA_OP_CLOSURE_RESULT.json'
    for path in (holdout, freeze, prereg):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}. Run prepare then run first.')

    subprocess.run([
        sys.executable,
        str(ENGINE / 'score_lambda_op_closure.py'),
        '--holdout', str(holdout),
        '--freeze', str(freeze),
        '--preregistration', str(prereg),
        '--output', str(result),
    ], check=True)


if __name__ == '__main__':
    main()
