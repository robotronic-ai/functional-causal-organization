#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_METROLOGY_09'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/score_metrology_repeatability.py'
FIDELITY = OUT / 'COMPRESSION_FIDELITY_09_RESULT.json'
RESULT = OUT / 'METROLOGY_09_RESULT.json'
REPEATS = [OUT / f'METROLOGY_09_REPEAT_{i}.json' for i in range(1, 4)]

for p in [FIDELITY, *REPEATS]:
    if not p.is_file():
        raise FileNotFoundError(f'Missing required audit artifact: {p}')

cmd = [sys.executable, str(ENGINE)]
for p in REPEATS:
    cmd.extend(['--repeat', str(p)])
cmd.extend(['--fidelity', str(FIDELITY), '--output', str(RESULT)])
subprocess.run(cmd, check=True)
