#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_12'
PANEL = OUT / 'CONTENT_TYPED_VALIDATION_12_PANEL.json'
CANDIDATE = OUT / 'FROZEN_CONTENT_TYPED_12_CANDIDATE.json'
RESULT = OUT / 'CONTENT_TYPED_VALIDATION_12_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_content_typed_validation.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_content12_validate_run.py first.')
if not CANDIDATE.is_file():
    raise FileNotFoundError('Frozen content-typed candidate is missing.')
subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
