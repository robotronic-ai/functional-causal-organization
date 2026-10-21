#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_QUOTIENT_RESOLUTION_HOLDOUT_19'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_quotient_resolution19_holdout.py'
CANDIDATE = OUT / 'FROZEN_CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json'
PANEL = OUT / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_PANEL.json'
PREREG = OUT / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_PREREGISTRATION.json'

for path in (CANDIDATE, PANEL, PREREG):
    if not path.is_file():
        raise FileNotFoundError(path)
subprocess.run([
    sys.executable, str(ENGINE),
    '--candidate', str(CANDIDATE),
    '--panel', str(PANEL),
    '--preregistration', str(PREREG),
    '--output-dir', str(OUT),
], check=True)
