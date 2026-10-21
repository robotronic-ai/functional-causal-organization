#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess, sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/LAMBDA_OP_V427_CURRENT'
HOLDOUT = OUT / 'V427_HOLDOUT_PANEL.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/explore_lambda_op_candidate.py'

if not HOLDOUT.is_file():
    raise FileNotFoundError(f'Existing Qwen holdout not found: {HOLDOUT}')
subprocess.run([sys.executable, str(ENGINE), '--holdout', str(HOLDOUT), '--output-dir', str(OUT)], check=True)
