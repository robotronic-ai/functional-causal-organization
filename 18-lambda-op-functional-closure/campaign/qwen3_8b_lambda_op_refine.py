#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess, sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
CURRENT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT/V427_HOLDOUT_PANEL.json'
VALID1 = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01/VALIDATION_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_REFINEMENT_02'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/refine_lambda_op_candidate.py'

for p in (CURRENT, VALID1):
    if not p.is_file():
        raise FileNotFoundError(f'Missing required panel: {p}')
subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(CURRENT), '--panel', str(VALID1), '--output-dir', str(OUT)
], check=True)
