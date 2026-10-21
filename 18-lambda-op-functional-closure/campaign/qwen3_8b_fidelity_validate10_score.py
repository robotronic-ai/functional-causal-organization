#!/usr/bin/env python3
from __future__ import annotations
import os

import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FIDELITY_FIRST_VALIDATION_10'
PANEL = OUT / 'VALIDATION_10_PANEL.json'
CANDIDATE = OUT / 'FROZEN_FIDELITY_FIRST_CANDIDATE.json'
RESULT = OUT / 'FIDELITY_FIRST_VALIDATION_10_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_fidelity_first_validation.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_fidelity_validate10_run.py first.')
if not CANDIDATE.is_file():
    raise FileNotFoundError('Frozen fidelity-first candidate is missing.')

subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
