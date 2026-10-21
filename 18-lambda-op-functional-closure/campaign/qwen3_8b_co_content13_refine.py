#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
CONSTRUCTION = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_12/CONTENT_TYPED_12_PANEL.json'
OPENED_VALIDATION = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_12/CONTENT_TYPED_VALIDATION_12_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_REFINEMENT_13'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/refine_content_typed_pooled.py'

for path, label in ((CONSTRUCTION, 'Construction 12 panel'), (OPENED_VALIDATION, 'Validation 12 panel')):
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
subprocess.run([
    sys.executable, str(ENGINE),
    '--construction-panel', str(CONSTRUCTION),
    '--opened-validation-panel', str(OPENED_VALIDATION),
    '--output-dir', str(OUT),
], check=True)
