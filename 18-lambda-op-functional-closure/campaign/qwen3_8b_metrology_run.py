#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
MODEL = WORK / 'Qwen3-8B'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_METROLOGY_09'
SCHEDULE = OUT / 'METROLOGY_09_REPEAT_SCHEDULE_SEALED.json'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/run_cross_target_channel.py'

if not SCHEDULE.is_file():
    raise FileNotFoundError('Run qwen3_8b_metrology_prepare.py first.')

for rep in range(1, 4):
    panel = OUT / f'METROLOGY_09_REPEAT_{rep}.json'
    print(f'[METROLOGY REPEAT] {rep}/3')
    subprocess.run([
        sys.executable,
        str(ENGINE),
        '--schedule', str(SCHEDULE),
        '--model-dir', str(MODEL),
        '--output', str(panel),
        '--device', 'cuda',
        '--dtype', 'bfloat16',
    ], check=True)
print('[CO-U METROLOGY 09 REPEATS COMPLETE]')
