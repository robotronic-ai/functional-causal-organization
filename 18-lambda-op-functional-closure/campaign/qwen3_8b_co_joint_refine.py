#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
PANEL_04 = WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_04/CO_U_CAUSAL_KERNEL_PANEL.json'
PANEL_05 = WORK / 'exports/co_qwen3_8b_closure/CO_U_CAUSAL_KERNEL_VALIDATION_05/VALIDATION_05_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_JOINT_RELATIONAL_REFINEMENT_06'
SCRIPT = HERE / 'qwen3_8b_lambda_op_engine/engine/screen_joint_relational_kernel.py'

for p in (PANEL_04, PANEL_05):
    if not p.is_file():
        raise FileNotFoundError(f'Missing active-kernel panel: {p}')

subprocess.run([
    sys.executable, str(SCRIPT),
    '--panels', str(PANEL_04), str(PANEL_05),
    '--output-dir', str(OUT),
], check=True)
