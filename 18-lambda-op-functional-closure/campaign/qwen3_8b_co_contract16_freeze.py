#!/usr/bin/env python3
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

WORK = Path(os.environ.get('FCO_WORKDIR', '.')).resolve()  # working directory holding Qwen3-8B/ and the campaign files
BASE = WORK / 'exports/co_qwen3_8b_closure/CO_U_BASE_OPERATOR_15'
SOURCE = BASE / 'CO_BASE_OPERATOR_15_RESULT.json'
ADJUDICATION = BASE / 'CO_BASE_OPERATOR_15_ADJUDICATION.json'
OUT = WORK / 'exports/co_qwen3_8b_closure/CO_U_FORMULA_CONTRACT_16'
EXPECTED_SOURCE_SHA = 'c240454260413c9ddf720c67ee8d04766ba5fb65b87e3a29a90b37fb848da01a'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


if not SOURCE.is_file():
    raise FileNotFoundError(f'Missing Phase 15 result: {SOURCE}')
actual = sha256(SOURCE)
if actual != EXPECTED_SOURCE_SHA:
    raise RuntimeError(f'Phase 15 result SHA mismatch: expected {EXPECTED_SOURCE_SHA}, got {actual}')
if not ADJUDICATION.is_file():
    raise FileNotFoundError('Run qwen3_8b_co_baseop15_adjudicate.py first.')
adj = json.loads(ADJUDICATION.read_text())
if adj.get('scientific_attribution_status') != 'UNRESOLVED':
    raise RuntimeError('Unexpected Phase 15 adjudication status.')

contract = {
    'schema': 'co-qwen3-8b-formula-contract-16-v1',
    'status': 'FROZEN_READY_FOR_MEASUREMENT_IMPLEMENTATION',
    'source_phase15_result_sha256': actual,
    'source_phase15_adjudication_sha256': sha256(ADJUDICATION),
    'system_operation_algebra': ['ID', 'KEEP', 'REVOKE', 'SUBSTITUTE'],
    'system_operation_algebra_changed': False,
    'generalization_level': 'OP',
    'generalization_level_changed': False,
    'design_space': {
        'A': 'system-operation intervention algebra; OP remains fixed in this contract phase',
        'Q': 'declared quotient / causal symmetries; no target-informed quotient is introduced',
        'rho': 'macro roles H1/H2/H3 plus prospectively declared categorical or ordinal readout resolution',
        'principle': 'algebra richness and quotient strength are independent design axes',
    },
    'Gamma': {
        'contract': 'J_q(A->B)=E_b I(A;B_prime|do(B=b)); normalize directional terms at the same macro grain; arithmetic-mean directional aggregator under the declared G-WEAK/marginal-independence semantics; minimum over admissible cuts',
        'phase15_estimator_promoted': False,
        'repository_source': '04-gamma-delta-characterization/docs/gamma-axiom-characterization.md',
    },
    'Delta': {
        'contract': 'primary profile (kappa_q,delta_q), with kappa_q=H(P)/log2(|P*|), delta_q=I(P;Y)/H(P), Delta_cap=kappa_q*delta_q=I(P;Y)/log2(|P*|)',
        'phase15_estimator_promoted': False,
        'repository_source': '04-gamma-delta-characterization/docs/delta-decomposition.md',
    },
    'R': {
        'contract': 'source quotient C must be built without the return channel; K_eff=H(C|U); B_reentry=I(C;Y_ret|U); R=B_reentry/K_eff; require temporal causal control and label-duplication invariance',
        'phase15_cycle_estimator_promoted': False,
        'source_quotient_rule_preference': 'threshold-free ordinal source signature before return measurement',
        'repository_source': '06-transformer-measurement/docs/reentry-measurement-protocol.md',
    },
    'Lambda': {
        'contract': 'typed content-to-role response must preserve joint cross-role structure when it is load-bearing; scalar norms and marginal-only channels are not sufficient in general',
        'phase15_content_typed_proxy_promoted_as_full_Lambda': False,
        'repository_source': '15-causal-construct-validity-protocol/docs/organizational-closure-repair-chain.md',
    },
    'selection': {
        'measurement_construction_uses_A_dagger': False,
        'evaluation_metrics': [
            'sufficiency_loss_H_target_given_descriptor_over_H_target',
            'adjusted_mutual_information',
            'fixed-fiber permutation control',
            'fiber count and partition entropy',
            'nontrivial coverage',
        ],
        'coarseness_rule': 'among prospectively admissible candidates choose the coarsest partition',
        'single_holdout_after_development_selection': True,
        'report_number_of_candidate_contracts_tested': True,
    },
    'phase15_pilot_partition_promoted': False,
    'next_stage': 'FORMULA_MEASUREMENT_17_IMPLEMENTATION',
}
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / 'CO_FORMULA_CONTRACT_16_FROZEN.json'
out.write_text(json.dumps(contract, indent=2, sort_keys=True) + '\n')
print('[CO FORMULA CONTRACT 16 FROZEN]')
print('[STATUS]', contract['status'])
print('[GENERALIZATION LEVEL]', contract['generalization_level'])
print('[PHASE 15 PARTITION PROMOTED]', contract['phase15_pilot_partition_promoted'])
print('[OUTPUT]', out)
print('[SHA256]', sha256(out))
