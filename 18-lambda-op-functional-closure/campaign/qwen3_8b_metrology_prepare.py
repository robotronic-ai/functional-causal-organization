#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
HERE = Path(__file__).resolve().parent
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CROSS_TARGET_CHANNEL_07/CROSS_TARGET_07_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_METROLOGY_09'
ENGINE = HERE / 'qwen3_8b_lambda_op_engine/engine/audit_compression_fidelity.py'
N = 6
REPEATS = 3
EXPECTED_SOURCE_SHA256 = 'a9bde0fa2529186701054b451cdfe4d00086453018226ccef668791be1b79970'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def target(row: dict) -> tuple[str, ...]:
    cells = {(str(c['operation']), str(c['source_address'])): c for c in row['cells']}
    ops = ('ID', 'KEEP', 'REVOKE', 'SUBSTITUTE')
    addresses = ('H1', 'H2', 'H3')
    return tuple(str(cells[(op, a)]['behavior']['outcome']) for op in ops for a in addresses)


if not SOURCE.is_file():
    raise FileNotFoundError('Run the Cross-Target 07 campaign first; CROSS_TARGET_07_PANEL.json is required.')

source_hash = sha256(SOURCE)
if source_hash != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(f'Cross-Target 07 panel hash mismatch: {source_hash}')
data = json.loads(SOURCE.read_text())
if int(data.get('n_skipped', 0)) != 0:
    raise RuntimeError('Source panel contains technical skips.')
rows = sorted(data['rows'], key=lambda r: int(r['unit']))
selected = []
seen_targets = set()
for row in rows:
    t = target(row)
    if t not in seen_targets:
        selected.append(row)
        seen_targets.add(t)
    if len(selected) >= N:
        break
if len(selected) < N:
    used = {int(r['unit']) for r in selected}
    for row in rows:
        if int(row['unit']) in used:
            continue
        selected.append(row)
        if len(selected) >= N:
            break
if len(selected) != N:
    raise RuntimeError(f'Could not select {N} systems from source panel.')

OUT.mkdir(parents=True, exist_ok=True)
schedule = [
    {
        'unit': int(r['unit']),
        'bindings': {str(k): int(v) for k, v in r['bindings'].items()},
        'donor_values': {str(k): int(v) for k, v in r['donor_values'].items()},
    }
    for r in selected
]
schedule_path = OUT / 'METROLOGY_09_REPEAT_SCHEDULE_SEALED.json'
schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + '\n')

fidelity = OUT / 'COMPRESSION_FIDELITY_09_RESULT.json'
subprocess.run([
    sys.executable, str(ENGINE), '--panel', str(SOURCE), '--output', str(fidelity)
], check=True)

prereg = {
    'schema': 'co-qwen3-8b-metrology-prereg-v1',
    'status': 'METROLOGY_09_PREPARED',
    'source_panel': str(SOURCE),
    'source_panel_sha256': source_hash,
    'repeat_schedule_sha256': sha256(schedule_path),
    'selected_units': [int(r['unit']) for r in selected],
    'n_repeated_systems': N,
    'repeat_count': REPEATS,
    'dtype': 'bfloat16',
    'fresh_process_model_reload_each_repeat': True,
    'repeatability_targets': [
        '12-cell behavioral outcome vector',
        'continuous cross-target internal response coordinates',
        'declared target-blind routing descriptors',
    ],
    'compression_fidelity_scope': 'current cross-target Lambda/active-kernel proxy only',
    'full_co_scalarization_claimed': False,
    'full_co_missing_certified_components': ['Gamma', 'Delta', 'R'],
    'engineering_repeatability_gate': {
        'clean': 'behavior target stability = 1, ALL_FLOW_H2_FORWARD descriptor stability = 1, and p95 noise/signal <= 0.05',
        'caution': 'behavior target stability = 1, descriptor stability >= 0.95, and p95 noise/signal <= 0.20',
        'otherwise': 'REPEATABILITY_UNRESOLVED',
    },
    'known_limit': 'No float32 reference run is required; systematic bfloat16 bias is not estimated by this audit.',
}
prereg_path = OUT / 'METROLOGY_09_PREREGISTRATION.json'
prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO-U METROLOGY 09 PREPARED]')
print('[SOURCE]', SOURCE)
print('[SELECTED UNITS]', ','.join(str(x) for x in prereg['selected_units']))
print('[REPEATS]', REPEATS)
print('[SCHEDULE]', schedule_path)
print('[FIDELITY]', fidelity)
print('[PREREGISTRATION]', prereg_path)
