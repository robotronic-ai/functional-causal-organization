#!/usr/bin/env python3
from __future__ import annotations
import os

import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_EXPLORATORY_VALIDATION_11'
PANEL = OUT / 'VALIDATION_11_PANEL.json'
CANDIDATE = OUT / 'FROZEN_CO_EXPLORATORY_CANDIDATE_11.json'
RESULT = OUT / 'CO_EXPLORATORY_VALIDATION_11_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_exploratory_co_validation.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_explore11_validate_run.py first.')
if not CANDIDATE.is_file():
    raise FileNotFoundError('Frozen exploratory candidate is missing.')

subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
