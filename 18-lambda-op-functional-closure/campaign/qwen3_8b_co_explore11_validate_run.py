#!/usr/bin/env python3
from __future__ import annotations
import os

import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_EXPLORATORY_VALIDATION_11'
SCHEDULE = OUT / 'VALIDATION_11_SCHEDULE_SEALED.json'
PANEL = OUT / 'VALIDATION_11_PANEL.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/run_cross_target_channel.py'
MODEL = WORK / 'Qwen3-8B'

if not SCHEDULE.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_explore11_validate_prepare.py first.')

subprocess.run([
    sys.executable, str(ENGINE),
    '--schedule', str(SCHEDULE),
    '--model-dir', str(MODEL),
    '--output', str(PANEL),
    '--device', 'cuda',
    '--dtype', 'bfloat16',
], check=True)
print('[CO EXPLORATORY VALIDATION 11 RUN COMPLETE]')
print('[PANEL]', PANEL)
