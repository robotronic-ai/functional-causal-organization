#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
SOURCE = WORK / 'exports/co_qwen3_8b_closure/CO_U_CONTENT_TYPED_VALIDATION_13/CONTENT_TYPED_VALIDATION_13_PANEL.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_BASE_OPERATOR_15'
EXPECTED_SOURCE_SHA = '92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4'
N = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def row_key(row: dict) -> str:
    b = row['bindings']; d = row['donor_values']
    raw = '|'.join(str(int(b[a])) for a in ('H1','H2','H3')) + '|' + '|'.join(str(int(d[a])) for a in ('H1','H2','H3'))
    return hashlib.sha256(raw.encode()).hexdigest()


if not SOURCE.is_file():
    raise FileNotFoundError(f'Missing opened Validation 13 panel: {SOURCE}')
actual = sha256(SOURCE)
if actual != EXPECTED_SOURCE_SHA:
    raise RuntimeError(f'Validation 13 panel SHA mismatch: expected {EXPECTED_SOURCE_SHA}, got {actual}')
panel = json.loads(SOURCE.read_text())
rows = sorted(panel['rows'], key=row_key)[:N]
schedule = []
for i, row in enumerate(rows, 1):
    schedule.append({
        'unit': i,
        'source_unit': int(row['unit']),
        'bindings': row['bindings'],
        'donor_values': row['donor_values'],
    })
OUT.mkdir(parents=True, exist_ok=True)
schedule_path = OUT / 'CO_BASE_OPERATOR_15_SCHEDULE_SEALED.json'
schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + '\n')
prereg = {
    'schema': 'co-qwen3-8b-base-operator-15-prereg-v1',
    'status': 'EXPLORATORY_BASE_OPERATOR_COMPONENT_AUDIT_PREPARED',
    'source_panel': str(SOURCE),
    'source_panel_sha256': actual,
    'selection': '16 rows with lexicographically smallest SHA256 of binding+donor tuple; behavior-blind',
    'n': N,
    'schedule_sha256': sha256(schedule_path),
    'operation_algebra': ['ID','KEEP','REVOKE','SUBSTITUTE'],
    'generalization_level': 'OP',
    'generalization_level_changed': False,
    'components': ['Gamma','Delta','R','Lambda'],
    'global_scalarization_allowed': False,
    'experimental_interventions': ['ZERO_LOCAL','HEAD_ROTATE_LOCAL'],
    'experimental_interventions_are_not_system_operations': True,
    'intervention_layers': [21,22,23],
    'horizons': [1,2,3,4],
    'diagnostic_status_vocabulary': ['LOCALIZED','INTERACTIVE','DIFFUSE','UNRESOLVED'],
    'lambda_source': 'reuse existing typed content-effect proxy from opened Validation 13 panel',
    'purpose': 'compare component estimator definitions at fixed OP level before testing ROLE/PARAM_OP/ALGEBRA or numerical compression as a separate axis',
}
prereg_path = OUT / 'CO_BASE_OPERATOR_15_PREREGISTRATION.json'
prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + '\n')
print('[CO BASE-OP 15 PREPARED]')
print('[SOURCE SHA256]', actual)
print('[SOURCE SYSTEMS]', len(panel['rows']))
print('[SELECTED SYSTEMS]', N)
print('[SCHEDULE]', schedule_path)
print('[PREREGISTRATION]', prereg_path)
