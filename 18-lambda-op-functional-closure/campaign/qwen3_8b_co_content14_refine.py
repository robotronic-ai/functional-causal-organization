#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
CONSTRUCTION = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_12/CONTENT_TYPED_12_PANEL.json'
VALIDATION12 = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_12/CONTENT_TYPED_VALIDATION_12_PANEL.json'
VALIDATION13 = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_REFINEMENT_14'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/refine_content_typed_pooled160.py'

for path, label in ((CONSTRUCTION, 'Construction 12 panel'), (VALIDATION12, 'Validation 12 panel'), (VALIDATION13, 'Validation 13 panel')):
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
subprocess.run([
    sys.executable, str(ENGINE),
    '--construction-panel', str(CONSTRUCTION),
    '--opened-validation12-panel', str(VALIDATION12),
    '--opened-validation13-panel', str(VALIDATION13),
    '--output-dir', str(OUT),
], check=True)
