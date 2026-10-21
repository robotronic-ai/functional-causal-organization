#!/usr/bin/env python3
from __future__ import annotations
import os

import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
PANEL = WORK / 'exports/co_qwen3_8b_closure/CO_U_CROSS_TARGET_CHANNEL_07/CROSS_TARGET_07_PANEL.json'
FIDELITY = WORK / 'exports/co_qwen3_8b_closure/CO_U_FIDELITY_FIRST_10/FIDELITY_FIRST_10_CANDIDATE.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_EXPLORATORY_CANDIDATE_11'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/explore_fidelity_local_split.py'

if not PANEL.is_file():
    raise FileNotFoundError('CROSS_TARGET_07_PANEL.json is required.')
if not FIDELITY.is_file():
    raise FileNotFoundError('Run qwen3_8b_fidelity_gate.py first.')

subprocess.run([
    sys.executable, str(ENGINE),
    '--panel', str(PANEL),
    '--fidelity-candidate', str(FIDELITY),
    '--output-dir', str(OUT),
], check=True)
