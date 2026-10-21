#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_04'
PANEL = OUT / 'CO_U_CAUSAL_KERNEL_PANEL.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/screen_co_u_causal_kernel.py'

if not PANEL.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_kernel_run.py first.')
subprocess.run([
    sys.executable,
    str(ENGINE),
    '--panel', str(PANEL),
    '--output-dir', str(OUT),
], check=True)
