#!/usr/bin/env python3
from pathlib import Path
import json, hashlib, sys

ROOT=Path(__file__).resolve().parents[1]
EMPTY="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"

def load(rel): return json.loads((ROOT/rel).read_text())

def main():
    c0=load('data/REAL_C0_STAGE_A_CANONICAL_SUMMARY.json')
    assert c0['states']==201
    assert c0['candidate_count']==0
    assert c0['candidate_list_sha256']==EMPTY
    assert c0['stage_b_instantiated'] is False

    rd=load('data/RICH_DELTA_STAGE_A_RESULT_v3.15.63.3.json')
    assert rd['state_count']==134
    assert rd['pair_search']['total_unordered_pairs']==8911
    assert rd['pair_search']['anchor_rejected_pairs']==8911
    assert rd['pair_search']['candidate_count']==0
    assert rd['pair_search']['candidate_list_sha256']==EMPTY
    assert rd['pair_search']['full_trace_compared_pairs']==0
    assert rd['numerical_equivalence']['tolerance']==1e-7
    assert rd['screening_representation']['rich_schema']['raw_feature_dim']==13056
    assert rd['screening_representation']['rich_schema']['rich_feature_dim']==52224
    assert rd['screening_representation']['reduction']=='NONE'
    assert rd['grouping']['lambda_used'] is False
    assert rd['grouping']['task_outcome_used'] is False

    lc=load('data/RICH_DELTA_LAYER_COMPLETE_STAGE_A_RESULT_v3.15.66.json')
    assert lc['state_count']==1586
    assert lc['cell_schedule']['layers']==12
    assert lc['cell_schedule']['cells_per_lineage']==793
    assert lc['pair_search']['total_unordered_pairs']==1256905
    assert lc['pair_search']['anchor_rejected_pairs']==1256905
    assert lc['pair_search']['candidate_count']==0
    assert lc['pair_search']['candidate_list_sha256']==EMPTY
    assert lc['numerical_equivalence']['tolerance']==1e-7
    assert lc['grouping']['lambda_used'] is False

    rb=load('data/LAMBDA_REBIND_WITNESS_RESULT_v3.16.0.json')
    assert rb['baseline_typed_channel']['comparison_count']==262656
    assert rb['baseline_typed_channel']['mismatch_count']==0
    assert rb['baseline_typed_channel']['exact_equal'] is True
    assert rb['baseline_typed_channel']['relink_sha256']==rb['baseline_typed_channel']['lock_sha256']
    assert rb['late_rebind']['RELINK']['lambda_gate_normalized']==1.0
    assert rb['late_rebind']['LOCK']['lambda_gate_normalized']==0.0
    assert rb['late_rebind']['RELINK']['task_accuracy']==1.0
    assert abs(rb['late_rebind']['LOCK']['task_accuracy']-1/24)<1e-15
    for mode in ('RELINK','LOCK'):
        assert all(v==1.0 for v in rb['controls'][mode].values())
    assert rb['status']=='LAMBDA-CONSTRUCTIVE-EMPIRICAL-WITNESS-ESTABLISHED'

    print('LAMBDA-EMPIRICAL-CLEAN-CAMPAIGN-VERIFY-PASS')

if __name__=='__main__': main()
