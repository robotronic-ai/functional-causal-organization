#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_JOINT_TRAJECTORY_VALIDATION_08'
PANEL = OUT / 'VALIDATION_08_PANEL.json'
CANDIDATE = OUT / 'FROZEN_JOINT_TRAJECTORY_CANDIDATE.json'
RESULT = OUT / 'JOINT_TRAJECTORY_VALIDATION_08_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_joint_trajectory_validation.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_trajectory_validate8_run.py first.')
subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
