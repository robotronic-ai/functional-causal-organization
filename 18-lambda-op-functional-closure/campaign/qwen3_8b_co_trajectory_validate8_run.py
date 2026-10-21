#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
MODEL = WORK / 'Qwen3-8B'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_JOINT_TRAJECTORY_VALIDATION_08'
SCHEDULE = OUT / 'VALIDATION_08_SCHEDULE_SEALED.json'
PANEL = OUT / 'VALIDATION_08_PANEL.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/run_cross_target_channel.py'

if not SCHEDULE.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_trajectory_validate8_prepare.py first.')
subprocess.run([
    sys.executable, str(ENGINE),
    '--schedule', str(SCHEDULE),
    '--model-dir', str(MODEL),
    '--output', str(PANEL),
    '--device', 'cuda',
    '--dtype', 'bfloat16',
], check=True)
print('[CO-U JOINT TYPED TRAJECTORY VALIDATION 08 RUN COMPLETE]')
print('[PANEL]', PANEL)
