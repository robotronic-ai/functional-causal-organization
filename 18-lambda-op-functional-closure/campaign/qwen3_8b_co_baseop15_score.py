#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_BASE_OPERATOR_15'
ROUTING = OUT / 'CO_BASE_OPERATOR_15_ROUTING_PANEL.json'
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/diagnose_co_base_operator.py'

if not ROUTING.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_baseop15_run.py first.')
if not SOURCE.is_file():
    raise FileNotFoundError(f'Missing opened Validation 13 panel: {SOURCE}')
subprocess.run([
    sys.executable, str(ENGINE),
    '--routing-panel', str(ROUTING),
    '--source-content-panel', str(SOURCE),
    '--output-dir', str(OUT),
], check=True)
