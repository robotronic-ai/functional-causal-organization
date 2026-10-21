#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13'
PANEL = OUT / 'CONTENT_TYPED_VALIDATION_13_PANEL.json'
CANDIDATE = OUT / 'FROZEN_CONTENT_TYPED_REFINEMENT_13_CANDIDATE.json'
RESULT = OUT / 'CONTENT_TYPED_VALIDATION_13_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_content_typed_validation13.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_content13_validate_run.py first.')
if not CANDIDATE.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_content13_validate_prepare.py first.')
subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
