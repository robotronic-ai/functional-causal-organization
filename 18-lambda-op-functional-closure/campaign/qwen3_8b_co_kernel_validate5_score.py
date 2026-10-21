#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_VALIDATION_05'
PANEL = OUT / 'VALIDATION_05_PANEL.json'
CANDIDATE = OUT / 'FROZEN_ACTIVE_KERNEL_CANDIDATE.json'
RESULT = OUT / 'CO_U_CAUSAL_KERNEL_VALIDATION_05_RESULT.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_co_u_kernel_hybrid_validation.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_kernel_validate5_run.py first.')
if not CANDIDATE.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_kernel_validate5_prepare.py first.')
subprocess.run([
    sys.executable,
    str(ENGINE),
    '--panel', str(PANEL),
    '--candidate', str(CANDIDATE),
    '--output', str(RESULT),
], check=True)
