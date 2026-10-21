#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess, sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/screen_co_u_candidate.py'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_OPERATION_INDEXED_SCREEN_03'
PANELS = [
    WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT/V427_HOLDOUT_PANEL.json',
    WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_01/VALIDATION_PANEL.json',
    WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_VALIDATION_02/VALIDATION_02_PANEL.json',
]
for path in PANELS:
    if not path.is_file():
        raise FileNotFoundError(path)
cmd = [sys.executable, str(ENGINE)]
for path in PANELS:
    cmd.extend(['--panel', str(path)])
cmd.extend(['--output-dir', str(OUT)])
subprocess.run(cmd, check=True)
