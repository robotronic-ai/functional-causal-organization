#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
MODEL = WORK / 'Qwen3-8B'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_QUOTIENT_RESOLUTION_HOLDOUT_19'
SCHEDULE = OUT / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_SCHEDULE_SEALED.json'
PANEL = OUT / 'CO_QUOTIENT_RESOLUTION_19_HOLDOUT_PANEL.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/run_lambda_quotient19_holdout.py'

if not SCHEDULE.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_quotient19_validate_prepare.py first.')
subprocess.run([
    sys.executable, str(ENGINE),
    '--schedule', str(SCHEDULE),
    '--model-dir', str(MODEL),
    '--output', str(PANEL),
    '--device', 'cuda',
    '--dtype', 'bfloat16',
], check=True)
print('[CO QUOTIENT-RESOLUTION 19 HOLDOUT RUN COMPLETE]')
print('[PANEL]', PANEL)
