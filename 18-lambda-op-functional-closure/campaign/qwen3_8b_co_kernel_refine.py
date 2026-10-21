#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_04/CO_U_CAUSAL_KERNEL_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_REFINEMENT_05'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/refine_co_u_causal_kernel.py'

if not SOURCE.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_kernel_run.py first.')
subprocess.run([
    sys.executable,
    str(ENGINE),
    '--panel', str(SOURCE),
    '--output-dir', str(OUT),
], check=True)
