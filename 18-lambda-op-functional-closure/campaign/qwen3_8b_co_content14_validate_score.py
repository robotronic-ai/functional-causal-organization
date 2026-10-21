#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_14'
PANEL = OUT / 'CONTENT_TYPED_VALIDATION_14_PANEL.json'
CANDIDATE = OUT / 'FROZEN_CONTENT_TYPED_REFINEMENT_14_CANDIDATE.json'
RESULT = OUT / 'CONTENT_TYPED_VALIDATION_14_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_content_typed_validation14.py'

for path, label in ((PANEL, 'Validation 14 panel'), (CANDIDATE, 'frozen Refinement 14 candidate')):
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
