#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/CO_COMPONENT_DIAGNOSTICS.json'


def gamma_two_role(forward: float, backward: float) -> float:
    return min(float(forward), float(backward))


def main() -> None:
    # Gamma normalization witness: one logical role is represented by two
    # coherent physical coordinates. Logical aggregation preserves capacity;
    # naive coordinate averaging halves it.
    gamma_original = gamma_two_role(1.0, 1.0)
    replicated_edges = [0.5, 0.5]
    gamma_macro = gamma_two_role(sum(replicated_edges), sum(replicated_edges))
    gamma_naive_physical = gamma_two_role(float(np.mean(replicated_edges)), float(np.mean(replicated_edges)))

    # Delta composition witness from the supplied methodological note.
    delta_cap = 1.0 / 3.0
    pi_role = 0.0
    delta_product = delta_cap * pi_role
    delta_mean = 0.5 * (delta_cap + pi_role)
    delta_min = min(delta_cap, pi_role)

    # R witness: a two-role swap has return only after an even number of steps.
    T = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    Tk = np.eye(2)
    r = []
    for _ in range(4):
        Tk = Tk @ T
        r.append(float(np.mean(np.diag(Tk))))

    # Lambda typed-binding witness: same scalar amount, different relation.
    lambda_a = [('C1', 'R1'), ('C2', 'R2')]
    lambda_b = [('C1', 'R2'), ('C2', 'R1')]
    lambda_scalar_a = len(lambda_a)
    lambda_scalar_b = len(lambda_b)

    # ID -> OP witness: ID slice is identical, SUBSTITUTE slice differs.
    id_a = ('ORIGINAL', 'ORIGINAL')
    id_b = ('ORIGINAL', 'ORIGINAL')
    op_a = {'ID': id_a, 'SUBSTITUTE': ('DONOR', 'DONOR')}
    op_b = {'ID': id_b, 'SUBSTITUTE': ('DONOR', 'ORIGINAL')}

    result = {
        'schema': 'co-synthetic-component-diagnostics-v1',
        'status': 'PASS',
        'purpose': 'finite witnesses for component-definition and generalization diagnostics; not Qwen measurements',
        'witnesses': {
            'Gamma.normalization_rule': {
                'diagnostic': 'LOCALIZED',
                'gamma_original': gamma_original,
                'gamma_macro_normalized_replication': gamma_macro,
                'gamma_naive_physical_replication': gamma_naive_physical,
                'pass': gamma_original == gamma_macro and gamma_naive_physical == 0.5,
            },
            'Delta.composition_rule': {
                'diagnostic': 'LOCALIZED',
                'delta_cap': delta_cap,
                'pi_role': pi_role,
                'product': delta_product,
                'minimum': delta_min,
                'normalized_mean': delta_mean,
                'pass': delta_product == 0.0 and delta_min == 0.0 and abs(delta_mean - (1.0 / 6.0)) < 1e-12,
            },
            'R.horizon_k': {
                'diagnostic': 'LOCALIZED',
                'r_cycle_profile_k1_to_k4': r,
                'pass': np.allclose(r, [0.0, 1.0, 0.0, 1.0]),
            },
            'Lambda.typed_binding': {
                'diagnostic': 'LOCALIZED',
                'same_scalar_amount': lambda_scalar_a == lambda_scalar_b,
                'typed_relation_equal': lambda_a == lambda_b,
                'pass': lambda_scalar_a == lambda_scalar_b and lambda_a != lambda_b,
            },
            'ID_to_OP_generalization': {
                'diagnostic': 'LOCALIZED',
                'id_equal': id_a == id_b,
                'op_equal': op_a == op_b,
                'pass': id_a == id_b and op_a != op_b,
            },
        },
        'diagnostic_status_vocabulary': ['LOCALIZED', 'INTERACTIVE', 'DIFFUSE', 'UNRESOLVED'],
        'generalization_ladder_frozen_for_later_test': ['ID', 'OP', 'ROLE', 'PARAM_OP', 'ALGEBRA', 'ORBIT_QUOTIENT'],
        'phase_rule': 'component definition is tested at fixed OP before changing generalization level',
    }
    if not all(x['pass'] for x in result['witnesses'].values()):
        raise SystemExit('synthetic component diagnostic failed')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[CO SYNTHETIC COMPONENT DIAGNOSTICS PASS]')
    print('[RESULT]', OUT)


if __name__ == '__main__':
    main()
