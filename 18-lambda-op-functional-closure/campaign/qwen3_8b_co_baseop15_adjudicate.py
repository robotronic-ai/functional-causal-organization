#!/usr/bin/env python3
from __future__ import annotations
import os
import hashlib
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
RESULT = WORK / 'exports/co_qwen3_8b_closure/CO_U_BASE_OPERATOR_15/CO_BASE_OPERATOR_15_RESULT.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_BASE_OPERATOR_15'
EXPECTED_SHA = 'c240454260413c9ddf720c67ee8d04766ba5fb65b87e3a29a90b37fb848da01a'
ENGINE = WORK / 'qwen3_8b_lambda_op_engine/engine/adjudicate_co_base_operator15.py'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

if not RESULT.is_file():
    raise FileNotFoundError(f'Missing Phase 15 result: {RESULT}')
actual = sha256(RESULT)
if actual != EXPECTED_SHA:
    raise RuntimeError(f'Phase 15 result SHA mismatch: expected {EXPECTED_SHA}, got {actual}')
subprocess.run([
    sys.executable, str(ENGINE), '--result', str(RESULT), '--output-dir', str(OUT)
], check=True)
