#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def useful(rec: dict[str, Any]) -> bool:
    if rec.get('status') != 'COMPACT_CLOSED_QUOTIENT_FOUND':
        return False
    m = rec.get('metrics', {})
    q = rec.get('partition_quality', {})
    return (
        int(m.get('n_collision_pairs', 1)) == 0
        and float(m.get('nontrivial_coverage', 0.0)) >= 0.20
        and float(q.get('sufficiency_loss_H_target_given_descriptor_over_H_target', 1.0)) <= 1e-12
        and bool(q.get('ami_exceeds_permutation_95', False))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    result_path = Path(args.result)
    data = json.loads(result_path.read_text())
    if data.get('schema') != 'co-qwen3-8b-base-operator-component-audit-v1':
        raise RuntimeError(f'unexpected schema: {data.get("schema")!r}')
    if int(data.get('n_systems', 0)) != 16:
        raise RuntimeError(f'expected 16 systems, got {data.get("n_systems")!r}')

    profiles = data['component_profiles']
    cycle_profiles = []
    persistence_profiles = []
    for row in profiles:
        cycle = []
        persistence = []
        for op in row['R']['per_operation']:
            cycle.extend(round(float(x), 12) for x in op['r_cycle_candidate'])
            persistence.extend(round(float(x), 12) for x in op['persistence_control_profile'])
        cycle_profiles.append(tuple(cycle))
        persistence_profiles.append(tuple(persistence))

    pair_sets = Counter(tuple(x.get('separated_by_primary_components', [])) for x in data.get('heterogeneous_pair_attribution', []))
    screens = data['family_screens']
    primary_names = ('GAMMA_PROFILE', 'DELTA_FACTORS_PRODUCT', 'R_CYCLE_PROFILE', 'LAMBDA_TYPED')
    primary_useful = [name for name in primary_names if useful(screens[name])]
    admissible = []
    for name, rec in screens.items():
        if useful(rec):
            m = rec['metrics']
            q = rec['partition_quality']
            admissible.append({
                'family': name,
                'n_fibers': int(m['n_fibers']),
                'nontrivial_coverage': float(m['nontrivial_coverage']),
                'adjusted_mutual_information': float(q['adjusted_mutual_information']),
                'ami_permutation_95': float(q['ami_permutation_95']),
                'ami_permutation_p_ge': float(q['ami_permutation_p_ge']),
                'selected_feature_count': int(rec.get('selected_feature_count', 0)),
            })
    admissible.sort(key=lambda x: (x['n_fibers'], x['selected_feature_count'], -x['nontrivial_coverage']))

    r_cycle = screens['R_CYCLE_PROFILE']
    r_full = screens['R_CYCLE_PLUS_PERSISTENCE']
    r_full_features = [x.get('feature_name', '') for x in r_full.get('selected_features', [])]
    r_closed_only_with_persistence = useful(r_full) and all(':persistence:' in x for x in r_full_features)

    adjudication = {
        'schema': 'co-qwen3-8b-base-operator-15-adjudication-v1',
        'source_result_sha256': sha256(result_path),
        'raw_diagnostic_status': data['diagnostic_attribution']['status'],
        'scientific_attribution_status': 'UNRESOLVED',
        'localized_measurement_issue': {
            'component': 'R',
            'status': 'LOCALIZED_ESTIMATOR_DEGENERACY',
            'unique_cycle_profiles_across_systems': len(set(cycle_profiles)),
            'unique_persistence_profiles_across_systems': len(set(persistence_profiles)),
            'cycle_screen_metrics': r_cycle['metrics'],
            'cycle_partition_quality': r_cycle['partition_quality'],
            'cycle_plus_persistence_closed_using_persistence_only': bool(r_closed_only_with_persistence),
            'cycle_plus_persistence_selected_features': r_full_features,
            'interpretation': 'The algebraic role-cycle estimator is constant across the pilot and contributes no system discrimination. Any closure of the combined R screen is carried by the separately reported persistence control, not by cycle return.'
        },
        'component_contract_audit': {
            'Gamma': {
                'phase15_estimator': profiles[0]['Gamma']['estimator'],
                'status': 'FORMULA_CONTRACT_MISMATCH',
                'required_contract': 'Directional interventional mutual information per declared cut, arithmetic-mean directional aggregator under the declared G-WEAK/marginal-independence semantics, minimum over admissible cuts.',
            },
            'Delta': {
                'phase15_estimator': profiles[0]['Delta']['estimator'],
                'status': 'FORMULA_CONTRACT_MISMATCH',
                'required_contract': 'Primary profile (kappa_q, delta_q) with Delta_cap = kappa_q * delta_q = I(P;Y)/log2(|P*|); do not replace this by effective-rank times off-diagonal mass.',
            },
            'R': {
                'phase15_estimator': profiles[0]['R']['estimator'],
                'status': 'DEGENERATE_AND_NOT_CONSTRUCT_VALIDATED',
                'required_contract': 'Build source quotient C without consulting the return channel; report K_eff = H(C|U), B_reentry = I(C;Y_ret|U), and R = B_reentry/K_eff with causal controls.',
            },
            'Lambda': {
                'phase15_estimator': profiles[0]['Lambda']['estimator'],
                'status': 'PARTIAL_TYPED_PROXY',
                'required_contract': 'Retain typed content-to-role identity and joint cross-role response structure; scalar norms and marginal-only signatures are insufficient in general.',
            },
        },
        'pilot_identifiability': {
            'n_heterogeneous_pairs': len(data.get('heterogeneous_pair_attribution', [])),
            'pair_primary_component_separation_patterns': [
                {'components': list(k), 'count': int(v)} for k, v in sorted(pair_sets.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            'primary_compact_closed_families': primary_useful,
            'interpretation': 'Because multiple target-informed single-component searches separate the same heterogeneous pairs on only 16 systems, the raw DIFFUSE label does not identify a distributed causal mechanism.'
        },
        'best_pilot_partition_by_declared_coarseness_rule': admissible[0] if admissible else None,
        'pilot_partition_promotion_allowed': False,
        'promotion_blockers': [
            'pilot feature search is target-informed',
            'n=16 is a development pilot, not a prospective holdout',
            'Gamma and Delta do not yet implement the manuscript formula contracts',
            'R cycle estimator is degenerate',
            'Lambda remains a partial typed proxy rather than the full joint typed channel',
        ],
        'design_decision': {
            'operation_level': 'OP remains fixed',
            'intervention_algebra_A': 'do not enrich to compositions yet',
            'quotient_symmetry_Q': 'do not tune from behavioral targets',
            'resolution_rho': 'freeze from declared macro roles / threshold-free categorical or ordinal readouts before target opening',
            'next_stage': 'FORMULA_CONTRACT_16',
        },
        'source_paths': {
            'gamma_contract': '04-gamma-delta-characterization/docs/gamma-axiom-characterization.md',
            'delta_contract': '04-gamma-delta-characterization/docs/delta-decomposition.md',
            'r_contract': '06-transformer-measurement/docs/reentry-measurement-protocol.md',
            'lambda_joint_repair': '15-causal-construct-validity-protocol/docs/organizational-closure-repair-chain.md',
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / 'CO_BASE_OPERATOR_15_ADJUDICATION.json'
    out.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + '\n')
    summary = outdir / 'CO_BASE_OPERATOR_15_ADJUDICATION.md'
    best = adjudication['best_pilot_partition_by_declared_coarseness_rule']
    summary.write_text(
        '# Qwen3-8B Base-Operator 15 Adjudication\n\n'
        f"- Raw scorer status: {adjudication['raw_diagnostic_status']}\n"
        f"- Scientific attribution status: {adjudication['scientific_attribution_status']}\n"
        f"- R cycle profiles across 16 systems: {len(set(cycle_profiles))}\n"
        f"- Heterogeneous target pairs: {len(data.get('heterogeneous_pair_attribution', []))}\n"
        f"- Primary compact closed families: {', '.join(primary_useful) if primary_useful else 'none'}\n"
        + (f"- Coarsest admissible pilot partition: {best['family']} ({best['n_fibers']} fibers, coverage {best['nontrivial_coverage']:.4f}, AMI {best['adjusted_mutual_information']:.4f})\n" if best else '')
        + '\nThe raw DIFFUSE label is retained as the software output, but it is not promoted to a causal attribution. The component contracts must be corrected before any Qwen C_O^op closure claim.\n'
    )
    print('[CO BASE-OP 15 ADJUDICATION COMPLETE]')
    print('[RAW STATUS]', adjudication['raw_diagnostic_status'])
    print('[SCIENTIFIC STATUS]', adjudication['scientific_attribution_status'])
    print('[R UNIQUE CYCLE PROFILES]', len(set(cycle_profiles)))
    if best:
        print('[COARSEST PILOT PARTITION]', best['family'])
        print('[PILOT FIBERS]', best['n_fibers'])
        print('[PILOT COVERAGE]', f"{best['nontrivial_coverage']:.4f}")
        print('[PILOT AMI]', f"{best['adjusted_mutual_information']:.4f}")
    print('[RESULT]', out)
    print('[SUMMARY]', summary)


if __name__ == '__main__':
    main()
