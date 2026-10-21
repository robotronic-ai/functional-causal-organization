#!/usr/bin/env python3
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
ENGINE = WORK / 'qwen3_8b_lambda_op_engine/engine'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_MEASUREMENT_17'
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json'
subprocess.run([
    sys.executable,
    str(ENGINE / 'score_formula_measurement17.py'),
    '--measurement-panel', str(OUT / 'CO_FORMULA_MEASUREMENT_17_PANEL.json'),
    '--source-content-panel', str(SOURCE),
    '--output-dir', str(OUT),
], check=True, cwd=str(ENGINE))
