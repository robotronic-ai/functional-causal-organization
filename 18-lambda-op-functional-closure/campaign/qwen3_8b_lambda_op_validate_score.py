#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess, sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01'
PANEL = OUT / 'VALIDATION_PANEL.json'
CANDIDATE = OUT / 'FROZEN_EXPLORATORY_CANDIDATE.json'
RESULT = OUT / 'QWEN3_8B_LAMBDA_OP_CANDIDATE_VALIDATION_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_candidate_validation.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_lambda_op_validate_run.py first.')
subprocess.run([
    sys.executable, str(ENGINE), '--panel', str(PANEL), '--candidate', str(CANDIDATE),
    '--output', str(RESULT)
], check=True)
