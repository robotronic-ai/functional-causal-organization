#!/usr/bin/env python3
from __future__ import annotations
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    syn = json.loads((ROOT / 'synthetic_closure/results/LAMBDA_OP_CLOSURE_2_3_4_6_8.json').read_text())['results']
    for bits in ('2_bit', '3_bit', '4_bit', '6_bit', '8_bit'):
        check(syn[bits]['closure'] == 'PASS', f'{bits} closure mismatch')
    check(syn['3_bit']['coverage'] == 'EXHAUSTIVE', '3-bit coverage mismatch')
    for bits in ('4_bit', '6_bit', '8_bit'):
        check(syn[bits]['coverage'] == 'DECLARED_TRANSPORT_ENVELOPE', f'{bits} coverage mismatch')

    st = json.loads((ROOT / 'CAMPAIGN_STATUS.json').read_text())
    check(st['schema'] == 'co-functional-closure-campaign-status-v32', 'status schema mismatch')
    q = st['qwen3_8b']
    check(q['current_phase'] == 'QUOTIENT_RESOLUTION_REPLICATION_20_READY', 'current phase mismatch')

    v13 = q['content_typed_validation_13']
    check(v13['closure'] == 'REFUTED', 'validation 13 closure mismatch')
    check(v13['n_collision_pairs'] == 15, 'validation 13 collision mismatch')
    check(v13['n_collision_fibers'] == 6, 'validation 13 collision fiber mismatch')
    check(abs(v13['nontrivial_coverage'] - 0.421875) < 1e-12, 'validation 13 coverage mismatch')
    check(abs(v13['within_fiber_distortion_fraction'] - 0.1600208821732915) < 1e-12, 'validation 13 distortion mismatch')
    check(v13['validation_panel_sha256'] == '92d9a79472c05cf091f823ed5e970542926106bbf13dd7e1ce16737fa5e0dff4', 'validation 13 panel hash mismatch')

    r13 = q['content_typed_refinement_13']
    check(r13['status'] == 'USEFUL_EXPLORATORY_CANDIDATE', 'refinement 13 status mismatch')
    check(r13['candidate_name'] == 'CONTENT_TYPED_POOLED96_REVOKE_SUBSTITUTE_3F', 'refinement 13 candidate mismatch')
    check(r13['selected_feature_count'] == 3, 'refinement 13 feature count mismatch')
    check(r13['n_collision_pairs'] == 0, 'refinement 13 collision mismatch')

    r14 = q['content_typed_refinement_14']
    check(r14['status'] == 'SUSPENDED_BY_COMPONENT_AUDIT_15', 'refinement 14 status mismatch')
    check(r14['opened_source_systems'] == 160, 'refinement 14 source count mismatch')
    check(r14['feature_dimension'] == 24, 'refinement 14 dimension mismatch')
    check(r14['seed_feature_count'] == 3, 'refinement 14 seed mismatch')
    check(r14['max_features'] == 8 and r14['beam_width'] == 72, 'refinement 14 search configuration mismatch')

    base15 = q['co_base_operator_15']
    check(base15['status'] == 'EXECUTED_ADJUDICATED_UNRESOLVED', 'base operator 15 status mismatch')
    check(base15['selected_systems'] == 16, 'base operator 15 system count mismatch')
    check(base15['generalization_level'] == 'OP', 'base operator 15 generalization mismatch')
    check(base15['generalization_level_changed'] is False, 'base operator 15 changed generalization unexpectedly')
    check(base15['operation_algebra'] == ['ID', 'KEEP', 'REVOKE', 'SUBSTITUTE'], 'base operator 15 operation algebra mismatch')
    check(base15['global_scalar_co_constructed'] is False, 'base operator 15 scalarization mismatch')
    check(base15['horizons'] == [1, 2, 3, 4], 'base operator 15 horizon mismatch')
    check(base15['raw_diagnostic_status'] == 'DIFFUSE', 'base operator 15 raw diagnostic mismatch')
    check(base15['scientific_attribution_status'] == 'UNRESOLVED', 'base operator 15 scientific adjudication mismatch')
    check(base15['r_unique_cycle_profiles'] == 1, 'base operator 15 R degeneracy mismatch')
    check(base15['coarsest_pilot_partition'] == 'DELTA+LAMBDA', 'base operator 15 pilot partition mismatch')
    check(base15['pilot_partition_promotion_allowed'] is False, 'base operator 15 promotion guard mismatch')

    contract16 = q['formula_contract_16']
    check(contract16['status'] == 'FROZEN', 'formula contract 16 status mismatch')
    check(contract16['freeze_sha256'] == 'c17c401ff51751b16c33426a65dda95d9659b15e4c34c907de565de7ef8d0d04', 'formula contract 16 freeze hash mismatch')
    check(contract16['generalization_level'] == 'OP', 'formula contract 16 generalization mismatch')
    check(contract16['ami_is_evaluation_only'] is True, 'formula contract 16 AMI contract mismatch')

    fm17 = q['formula_measurement_17']
    check(fm17['status'] == 'EXECUTED_NO_ADMISSIBLE_FIXED_RHO_PARTITION', 'formula measurement 17 status mismatch')
    check(fm17['development_systems'] == 12 and fm17['n_systems'] == 12, 'formula measurement 17 system count mismatch')
    check(fm17['P_star_size'] == 3, 'formula measurement 17 support mismatch')
    check(fm17['candidate_family_count'] == 11, 'formula measurement 17 family count mismatch')
    check(fm17['raw_measurement_uses_behavioral_target'] is False, 'formula measurement 17 target leakage')
    check(fm17['holdout_refit_allowed'] is False, 'formula measurement 17 refit guard mismatch')
    check(fm17['measurement_panel_sha256'] == 'ea9e74daa3e7f681d8efbbd8c7fa8b3afe77cb891857383cf455db872d511561', 'formula measurement 17 panel hash mismatch')
    check(fm17['failure_mode'] == 'QUASI_INJECTIVE_AT_FIXED_RHO16', 'formula measurement 17 failure mode mismatch')
    check(fm17['full_co_op_fibers'] == 12 and fm17['full_co_op_nontrivial_coverage'] == 0.0, 'formula measurement 17 full profile compactness mismatch')
    check(fm17['r_nontrivial_source_quotient_cells'] == 21 and fm17['r_total_cells'] == 144, 'formula measurement 17 R construct mismatch')

    q18 = q['quotient_resolution_18']
    check(q18['status'] == 'USEFUL_EXPLORATORY_CANDIDATE', 'quotient resolution 18 status mismatch')
    check(q18['generalization_level'] == 'OP' and q18['operation_algebra_changed'] is False, 'quotient resolution 18 algebra mismatch')
    check(q18['representation_grid_count'] == 6 and q18['total_declared_screens'] == 66, 'quotient resolution 18 grid mismatch')
    check(q18['candidate_name'] == 'Q18_RHO2_L1_R1__LAMBDA', 'quotient resolution 18 candidate mismatch')
    check(q18['candidate_family'] == 'LAMBDA', 'quotient resolution 18 family mismatch')
    check(q18['candidate_sha256'] == 'ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc', 'quotient resolution 18 candidate hash mismatch')
    check(q18['n_collision_pairs'] == 0 and abs(q18['nontrivial_coverage'] - 2/3) < 1e-12, 'quotient resolution 18 closure metrics mismatch')
    check(abs(q18['adjusted_mutual_information'] - 0.8533973610411538) < 1e-12, 'quotient resolution 18 AMI mismatch')
    check(q18['ami_exceeds_global_grid_permutation_95'] is True, 'quotient resolution 18 global permutation mismatch')
    q19 = q['quotient_resolution_holdout_19']
    check(q19['status'] == 'PROSPECTIVE_HOLDOUT_PASS', 'quotient resolution holdout 19 status mismatch')
    check(q19['promotion_status'] == 'USEFUL_PROSPECTIVE_CANDIDATE', 'quotient resolution holdout 19 promotion mismatch')
    check(q19['n_validation_systems'] == 32, 'quotient resolution holdout 19 size mismatch')
    check(q19['candidate_refit_on_holdout'] is False and q19['grid_reopened_on_holdout'] is False, 'quotient resolution holdout 19 freeze guard mismatch')
    check(q19['n_fibers'] == 7 and q19['n_collision_pairs'] == 0, 'quotient resolution holdout 19 closure mismatch')
    check(abs(q19['nontrivial_coverage'] - 0.90625) < 1e-12, 'quotient resolution holdout 19 coverage mismatch')
    check(abs(q19['adjusted_mutual_information'] - 0.8060739596361114) < 1e-12, 'quotient resolution holdout 19 AMI mismatch')
    check(q19['panel_sha256'] == 'ae22a9aeaf78dc6c320efa495e1a6bdd4626257f1abc9fb281b88002913155b5', 'quotient resolution holdout 19 panel hash mismatch')
    check(q19['result_sha256'] == '4fdeaf9e0145ecb5e99f0bb37a132382f07eb10f6f766e5caf41f973f69832e6', 'quotient resolution holdout 19 result hash mismatch')

    q20 = q['quotient_resolution_replication_20']
    check(q20['status'] == 'READY_FOR_SECOND_FRESH_REPLICATION', 'quotient resolution replication 20 status mismatch')
    check(q20['n_replication_systems'] == 64 and q20['seed'] == 202609292001, 'quotient resolution replication 20 schedule mismatch')
    check(q20['candidate_refit_on_replication'] is False and q20['grid_reopened_on_replication'] is False, 'quotient resolution replication 20 freeze guard mismatch')
    check(q20['candidate_sha256'] == 'ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc', 'quotient resolution replication 20 candidate mismatch')

    compdiag = json.loads((ROOT / 'synthetic_closure/results/CO_COMPONENT_DIAGNOSTICS.json').read_text())
    check(compdiag['status'] == 'PASS', 'synthetic component diagnostics mismatch')
    for name, witness in compdiag['witnesses'].items():
        check(witness['pass'] is True, f'synthetic component witness failed: {name}')

    result13 = json.loads((ROOT / 'qwen3_8b_results/CONTENT_TYPED_VALIDATION_13_RESULT.json').read_text())
    check(result13['candidate_name'] == 'CONTENT_TYPED_POOLED96_REVOKE_SUBSTITUTE_3F', 'result 13 candidate mismatch')
    check(result13['candidate_refit_on_validation'] is False, 'result 13 refit mismatch')
    check(result13['n_collision_pairs'] == 15, 'result 13 pair mismatch')
    check(abs(result13['within_fiber_distortion_fraction'] - 0.1600208821732915) < 1e-12, 'result 13 distortion mismatch')

    result15 = json.loads((ROOT / 'qwen3_8b_results/CO_BASE_OPERATOR_15_RESULT.json').read_text())
    check(result15['diagnostic_attribution']['status'] == 'DIFFUSE', 'result 15 raw status mismatch')
    check(result15['n_systems'] == 16, 'result 15 system count mismatch')
    check(result15['routing_panel_sha256'] == 'fdbcd9d2adb5c1eea4a1bb066f050981ff98f770299c225e278685a496786177', 'result 15 routing hash mismatch')
    adj15 = json.loads((ROOT / 'qwen3_8b_results/CO_BASE_OPERATOR_15_ADJUDICATION.json').read_text())
    check(adj15['raw_diagnostic_status'] == 'DIFFUSE', 'adjudication raw status mismatch')
    check(adj15['scientific_attribution_status'] == 'UNRESOLVED', 'adjudication scientific status mismatch')
    check(adj15['localized_measurement_issue']['unique_cycle_profiles_across_systems'] == 1, 'adjudication R unique profile mismatch')
    check(adj15['best_pilot_partition_by_declared_coarseness_rule']['family'] == 'DELTA+LAMBDA', 'adjudication coarsest pilot mismatch')

    result17 = json.loads((ROOT / 'qwen3_8b_results/CO_FORMULA_MEASUREMENT_17_RESULT.json').read_text())
    check(result17['status'] == 'NO_ADMISSIBLE_FIXED_RHO_PARTITION', 'formula 17 result status mismatch')
    check(result17['measurement_panel_sha256'] == 'ea9e74daa3e7f681d8efbbd8c7fa8b3afe77cb891857383cf455db872d511561', 'formula 17 result hash mismatch')
    check(result17['R_construct_check']['n_nontrivial_source_quotient_cells'] == 21, 'formula 17 R nontrivial mismatch')
    check(result17['R_construct_check']['negative_temporal_storage_control_all_pass'] is True, 'formula 17 R temporal control mismatch')
    check(result17['R_construct_check']['label_duplication_invariance_all_pass'] is True, 'formula 17 duplication control mismatch')
    check(result17['family_screens']['FULL_CO_OP']['metrics']['n_fibers'] == 12, 'formula 17 full profile fiber mismatch')
    check(result17['family_screens']['FULL_CO_OP']['metrics']['nontrivial_coverage'] == 0.0, 'formula 17 full profile coverage mismatch')

    result19 = json.loads((ROOT / 'qwen3_8b_results/CO_QUOTIENT_RESOLUTION_19_HOLDOUT_RESULT.json').read_text())
    check(result19['status'] == 'PROSPECTIVE_HOLDOUT_PASS', 'quotient 19 result status mismatch')
    check(result19['candidate_sha256'] == 'ca2405d2d96556fed56ec476c9c5ccdd1c7994144b979e7dd94be508a7b417cc', 'quotient 19 candidate hash mismatch')
    check(result19['panel_sha256'] == 'ae22a9aeaf78dc6c320efa495e1a6bdd4626257f1abc9fb281b88002913155b5', 'quotient 19 panel hash mismatch')
    check(result19['metrics']['n_collision_pairs'] == 0 and result19['metrics']['n_fibers'] == 7, 'quotient 19 closure mismatch')
    check(abs(result19['metrics']['nontrivial_coverage'] - 0.90625) < 1e-12, 'quotient 19 coverage mismatch')
    check(abs(result19['partition_quality']['adjusted_mutual_information'] - 0.8060739596361114) < 1e-12, 'quotient 19 AMI mismatch')
    check(result19['candidate_refit_on_holdout'] is False and result19['grid_reopened_on_holdout'] is False, 'quotient 19 freeze guard mismatch')

    result18 = json.loads((ROOT / 'qwen3_8b_results/CO_QUOTIENT_RESOLUTION_18_RESULT.json').read_text())
    candidate18 = json.loads((ROOT / 'qwen3_8b_results/CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json').read_text())
    check(result18['status'] == 'USEFUL_EXPLORATORY_CANDIDATE', 'quotient 18 result status mismatch')
    check(result18['selected_candidate']['candidate_name'] == 'Q18_RHO2_L1_R1__LAMBDA', 'quotient 18 selected candidate mismatch')
    check(abs(result18['global_max_ami_permutation_95'] - 0.4365622384138) < 1e-12, 'quotient 18 global threshold mismatch')
    check(candidate18['metrics']['n_collision_pairs'] == 0, 'quotient 18 candidate collision mismatch')
    check(abs(candidate18['metrics']['nontrivial_coverage'] - 2/3) < 1e-12, 'quotient 18 candidate coverage mismatch')
    check(abs(candidate18['partition_quality']['adjusted_mutual_information'] - 0.8533973610411538) < 1e-12, 'quotient 18 candidate AMI mismatch')
    check(candidate18['holdout_refit_allowed'] is False, 'quotient 18 refit guard mismatch')

    required = [
        'CONTENT_TYPED_VALIDATION_13_RESULT.md',
        'CONTENT_TYPED_REFINEMENT_14_PROTOCOL.md',
        'qwen3_8b_results/CONTENT_TYPED_VALIDATION_13_RESULT.json',
        'qwen3_8b_results/CO_BASE_OPERATOR_15_READINESS.json',
        'qwen3_8b_co_content14_refine.py',
        'qwen3_8b_co_content14_validate_prepare.py',
        'qwen3_8b_co_content14_validate_run.py',
        'qwen3_8b_co_content14_validate_score.py',
        'qwen3_8b_lambda_op_engine/engine/refine_content_typed_pooled160.py',
        'qwen3_8b_lambda_op_engine/engine/score_content_typed_validation14.py',
        'qwen3_8b_lambda_op_engine/engine/content_typed_kernel.py',
        'qwen3_8b_lambda_op_engine/engine/run_content_typed_kernel.py',
        'CO_BASE_OPERATOR_15_PROTOCOL.md',
        'CO_BASE_OPERATOR_15_RESULT.md',
        'CO_BASE_OPERATOR_15_ADJUDICATION.md',
        'FORMULA_CONTRACT_16_PROTOCOL.md',
        'FORMULA_MEASUREMENT_17_PLAN.md',
        'GENERALIZATION_AXES_16_PLAN.md',
        'qwen3_8b_results/CO_BASE_OPERATOR_15_RESULT.json',
        'qwen3_8b_results/CO_BASE_OPERATOR_15_ADJUDICATION.json',
        'qwen3_8b_co_baseop15_adjudicate.py',
        'qwen3_8b_co_contract16_freeze.py',
        'qwen3_8b_lambda_op_engine/engine/adjudicate_co_base_operator15.py',
        'synthetic_closure/scripts/run_co_component_diagnostics.py',
        'synthetic_closure/results/CO_COMPONENT_DIAGNOSTICS.json',
        'qwen3_8b_co_baseop15_prepare.py',
        'qwen3_8b_co_baseop15_run.py',
        'qwen3_8b_co_baseop15_score.py',
        'qwen3_8b_lambda_op_engine/engine/co_base_operator_profile.py',
        'qwen3_8b_lambda_op_engine/engine/run_co_base_operator_profile.py',
        'qwen3_8b_lambda_op_engine/engine/diagnose_co_base_operator.py',
        'FORMULA_MEASUREMENT_17_PROTOCOL.md',
        'qwen3_8b_co_formula17_prepare.py',
        'qwen3_8b_co_formula17_run.py',
        'qwen3_8b_co_formula17_score.py',
        'qwen3_8b_lambda_op_engine/engine/formula_measurement17.py',
        'qwen3_8b_lambda_op_engine/engine/run_formula_measurement17.py',
        'qwen3_8b_lambda_op_engine/engine/score_formula_measurement17.py',
        'FORMULA_MEASUREMENT_17_RESULT.md',
        'qwen3_8b_results/CO_FORMULA_MEASUREMENT_17_RESULT.json',
        'QUOTIENT_RESOLUTION_18_PROTOCOL.md',
        'qwen3_8b_co_quotient18.py',
        'qwen3_8b_lambda_op_engine/engine/screen_quotient_resolution18.py',
        'QUOTIENT_RESOLUTION_18_RESULT.md',
        'qwen3_8b_results/CO_QUOTIENT_RESOLUTION_18_RESULT.json',
        'qwen3_8b_results/CO_QUOTIENT_RESOLUTION_18_CANDIDATE.json',
        'QUOTIENT_RESOLUTION_HOLDOUT_19_PROTOCOL.md',
        'qwen3_8b_co_quotient19_validate_prepare.py',
        'qwen3_8b_co_quotient19_validate_run.py',
        'qwen3_8b_co_quotient19_validate_score.py',
        'qwen3_8b_lambda_op_engine/engine/run_lambda_quotient19_holdout.py',
        'qwen3_8b_lambda_op_engine/engine/score_quotient_resolution19_holdout.py',
        'QUOTIENT_RESOLUTION_HOLDOUT_19_RESULT.md',
        'qwen3_8b_results/CO_QUOTIENT_RESOLUTION_19_HOLDOUT_RESULT.json',
        'QUOTIENT_RESOLUTION_REPLICATION_20_PROTOCOL.md',
        'qwen3_8b_co_quotient20_replication_prepare.py',
        'qwen3_8b_co_quotient20_replication_run.py',
        'qwen3_8b_co_quotient20_replication_score.py',
        'qwen3_8b_lambda_op_engine/engine/run_lambda_quotient20_replication.py',
        'qwen3_8b_lambda_op_engine/engine/score_quotient_resolution20_replication.py',
    ]
    for rel in required:
        check((ROOT / rel).is_file(), f'missing {rel}')

    sys.path.insert(0, str(ROOT / 'qwen3_8b_lambda_op_engine/engine'))
    from content_typed_kernel import feature_names, quantile_edges
    from refine_content_typed_pooled160 import beam_search
    import numpy as np

    check(len(feature_names()) == 24, 'content feature count mismatch')
    from co_base_operator_profile import compact_search, control_clause_spans
    from formula_measurement17 import entropy_bits, mutual_information_bits, tagged_role_mixture, reentry_metrics, family_components
    import math
    check(abs(mutual_information_bits([1/3]*3, [[1,0,0],[0,1,0],[0,0,1]]) - math.log2(3)) < 1e-12, 'formula17 perfect MI mismatch')
    check(mutual_information_bits([1/3]*3, [[.2,.3,.5]]*3) < 1e-12, 'formula17 independent MI mismatch')
    mix17 = tagged_role_mixture({'H1':[.25]*4,'H2':[.1,.2,.3,.4]}, ['H1','H2'])
    check(abs(sum(mix17)-1.0) < 1e-12, 'formula17 tagged mixture mismatch')
    r17 = reentry_metrics(['C','Z','H'], [[['V0','V1']],[['V0','V1']],[['V2','V1']]], {'RET0':[[.8,.1,.05,.05],[.8,.1,.05,.05],[.1,.1,.7,.1]], 'RET1':[[.7,.2,.05,.05],[.7,.2,.05,.05],[.1,.2,.6,.1]]})
    check(r17['label_duplication_invariance'] is True, 'formula17 duplication invariance mismatch')
    check(len(family_components()) == 11, 'formula17 family count selftest mismatch')

    from screen_quotient_resolution18 import REPRESENTATIONS, qbin, _truncate_rank_signature, _reentry_at_rank
    check(len(REPRESENTATIONS) == 6, 'quotient18 representation count mismatch')
    check(qbin(0.51, 4) == '2', 'quotient18 continuous bin mismatch')
    check(_truncate_rank_signature([['V0','V1'],['V2','V1']], 1) == [['V0'],['V2']], 'quotient18 rank truncation mismatch')
    rec18 = {
        'source_signatures': [[['V0','V1']],[['V0','V2']],[['V2','V1']]],
        'source_class_ids': [0,1,2],
        'return_contexts': [
            {'conditionals_by_class': [[.8,.1,.05,.05],[.7,.2,.05,.05],[.1,.1,.7,.1]]},
            {'conditionals_by_class': [[.7,.2,.05,.05],[.6,.3,.05,.05],[.1,.2,.6,.1]]},
        ],
    }
    key18, metric18 = _reentry_at_rank(rec18, 1, 4)
    check(metric18['n_source_classes'] == 2, 'quotient18 R rank quotient mismatch')
    check(metric18['K_eff_bits'] > 0.0 and metric18['R'] is not None, 'quotient18 R recomputation mismatch')

    from score_quotient_resolution19_holdout import score_rows as score_q19_rows
    from formula_measurement17 import OPS as OPS19, ADDRESSES as ADDRESSES19, Q_INTERVENTIONS as Q19
    def _q19_row(unit, cls, target):
        cells = []
        for op in OPS19:
            for addr in ADDRESSES19:
                byq = {}
                for qname in Q19:
                    joint = [['V0','V1'],['V0','V1'],['V0','V1']]
                    if cls and op == 'SUBSTITUTE' and addr == 'H3' and qname == 'HEAD_ROTATE_LOCAL':
                        joint = [['V1','V0'],['V0','V1'],['V0','V1']]
                    byq[qname] = joint
                outcome = 'DONOR' if target and op == 'SUBSTITUTE' and addr == 'H3' else 'ORIGINAL'
                cells.append({'operation': op, 'source_address': addr, 'behavior': {'outcome': outcome}, 'Lambda': {'joint_signature_by_intervention': byq}})
        return {'unit': unit, 'cells': cells}
    q19_fixture = [_q19_row(i+1, 0, 0) for i in range(4)] + [_q19_row(i+5, 1, 1) for i in range(4)]
    q19_good = score_q19_rows(q19_fixture)
    check(q19_good['admissible'] is True and q19_good['metrics']['n_collision_pairs'] == 0, 'quotient19 pass fixture mismatch')
    q19_fixture[1]['cells'][-1]['behavior']['outcome'] = 'DONOR'
    q19_bad = score_q19_rows(q19_fixture)
    check(q19_bad['admissible'] is False and q19_bad['metrics']['n_collision_pairs'] > 0, 'quotient19 refutation fixture mismatch')


    from score_quotient_resolution20_replication import score_rows as score_q20_rows
    q20_good = score_q20_rows([_q19_row(i+1, 0, 0) for i in range(8)] + [_q19_row(i+9, 1, 1) for i in range(8)])
    check(q20_good['admissible'] is True and q20_good['metrics']['n_collision_pairs'] == 0, 'quotient20 pass fixture mismatch')
    q20_fixture = [_q19_row(i+1, 0, 0) for i in range(8)] + [_q19_row(i+9, 1, 1) for i in range(8)]
    q20_fixture[2]['cells'][-1]['behavior']['outcome'] = 'DONOR'
    q20_bad = score_q20_rows(q20_fixture)
    check(q20_bad['admissible'] is False and q20_bad['metrics']['n_collision_pairs'] > 0, 'quotient20 refutation fixture mismatch')

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=False):
            return ''.join(chr(int(x)) for x in ids)

    fake = _FakeTok()
    for text in (
        'CONTROL_MAP=H1:KEEP,H2:REVOKE,H3:KEEP. Apply each control only to its named handle.',
        'CONTROL_MAP=H2:REVOKE,H1:KEEP,H3:KEEP. Apply each control only to its named handle.',
        'CONTROL_MAP=H3:SUBSTITUTE=55,H1:KEEP,H2:KEEP. Apply each control only to its named handle.',
    ):
        ids = [ord(ch) for ch in text]
        spans = control_clause_spans(fake, ids)
        check(set(spans) == {'H1','H2','H3'}, 'source-dependent clause parser keys mismatch')
        for address, (a,b) in spans.items():
            check(address in fake.decode(ids[a:b]), f'clause parser lost {address}')

    Xc = np.asarray([[-1.0, 0.0], [-0.9, 0.1], [-1.1, -0.1], [1.0, 0.0], [0.9, 0.1], [1.1, -0.1]], dtype=np.float64)
    tc = [json.dumps('A')] * 3 + [json.dumps('B')] * 3
    cfound = compact_search(Xc, tc, max_features=2)
    check(cfound['status'] == 'COMPACT_CLOSED_QUOTIENT_FOUND', 'base operator compact search fixture failed')
    check(cfound['metrics']['n_collision_pairs'] == 0, 'base operator compact search collision mismatch')
    check('partition_quality' in cfound, 'base operator partition quality missing')
    check(cfound['partition_quality']['adjusted_mutual_information'] > 0.0, 'base operator AMI fixture mismatch')
    check(cfound['partition_quality']['sufficiency_loss_H_target_given_descriptor_over_H_target'] <= 1e-12, 'base operator sufficiency loss mismatch')

    # Semantic search fixture: one target-aligned coordinate must yield a closed, compressive quotient.
    Z = np.zeros((12, 24), dtype=np.float64)
    Z[:6, 0] = -1.0
    Z[6:, 0] = 1.0
    targets = [json.dumps('A')] * 6 + [json.dumps('B')] * 6
    units = list(range(1, 13))
    panels = {'a': list(range(0, 4)) + list(range(6, 8)), 'b': list(range(4, 6)) + list(range(8, 12))}
    edge_map = {(0, 2): quantile_edges(Z[:, 0], 2)}
    found = beam_search(Z, [0], (2,), edge_map, targets, units, panels, tuple(), 1)
    check(bool(found), 'refinement 14 semantic search fixture failed')
    check(found[0]['score']['closure']['n_collision_pairs'] == 0, 'refinement 14 fixture collision mismatch')

    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))

    banned = ('A5' + 'V08', 'histo' + 'rical', 'pre' + 'vious', 'old ' + 'campaign', 're' + 'run', 'LEGACY_ID_' + 'ROLE_ONLY', 'T' + '65', 'T' + '74', 'T' + '78')
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.md', '.json', '.py'}:
            t = p.read_text(errors='ignore')
            for token in banned:
                check(token not in t, f'forbidden term {token!r} in {p.relative_to(ROOT)}')
        check('__pycache__' not in p.parts, f'bytecode cache {p}')
        check(p.suffix != '.pyc', f'bytecode file {p}')
    print('PACKAGE_VERIFICATION_PASS')


if __name__ == '__main__':
    main()
