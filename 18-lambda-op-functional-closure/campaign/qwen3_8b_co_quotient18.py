#!/usr/bin/env python3
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
ENGINE = WORK / 'qwen3_8b_lambda_op_engine/engine/screen_quotient_resolution18.py'
MEASUREMENT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_MEASUREMENT_17/CO_FORMULA_MEASUREMENT_17_PANEL.json'
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_QUOTIENT_RESOLUTION_18'

if not MEASUREMENT.is_file():
    raise FileNotFoundError('Missing Formula Measurement 17 panel. Run Phase 17 first.')
if not SOURCE.is_file():
    raise FileNotFoundError('Missing Content-Typed Validation 13 source panel.')

subprocess.run([
    sys.executable, str(ENGINE),
    '--measurement-panel', str(MEASUREMENT),
    '--source-content-panel', str(SOURCE),
    '--output-dir', str(OUT),
], check=True)
