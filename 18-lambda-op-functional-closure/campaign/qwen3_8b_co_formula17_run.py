#!/usr/bin/env python3
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
ENGINE = WORK / 'qwen3_8b_lambda_op_engine/engine'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_MEASUREMENT_17'
subprocess.run([
    sys.executable,
    str(ENGINE / 'run_formula_measurement17.py'),
    '--schedule', str(OUT / 'CO_FORMULA_MEASUREMENT_17_SCHEDULE_SEALED.json'),
    '--model-dir', str(WORK / 'Qwen3-8B'),
    '--output', str(OUT / 'CO_FORMULA_MEASUREMENT_17_PANEL.json'),
    '--device', 'cuda',
    '--dtype', 'bfloat16',
], check=True, cwd=str(ENGINE))
print('[CO FORMULA MEASUREMENT 17 RUN COMPLETE]')
print('[PANEL]', OUT / 'CO_FORMULA_MEASUREMENT_17_PANEL.json')
