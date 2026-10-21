#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

import qwen_v427_runtime as rt
import runtime as base
from co_base_operator_profile import control_clause_spans
from formula_measurement17 import ADDRESSES, OPS, Q_INTERVENTIONS, numeric_support
from run_co_base_operator_profile import apply_local_intervention
from run_formula_measurement17 import _build_control, _role_distributions


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bfloat16')
    args = ap.parse_args()

    schedule_path = Path(args.schedule)
    schedule = json.loads(schedule_path.read_text())
    model, tok = rt.load_model(Path(args.model_dir), args.device, args.dtype)
    rows = []
    skips = []
    started = time.perf_counter()

    for i, episode in enumerate(schedule, 1):
        try:
            bindings = {str(k): int(v) for k, v in episode['bindings'].items()}
            donors = {str(k): int(v) for k, v in episode['donor_values'].items()}
            supports = {a: numeric_support(bindings[a], donors[a]) for a in ADDRESSES}
            presentation = rt.presentation(bindings)
            cells = []

            for operation in OPS:
                for source in ADDRESSES:
                    control_cache, prefix_ids, suffix, history = _build_control(
                        model, tok, bindings, donors, operation, source, args.device
                    )
                    instruction = rt.operation_instruction(operation, source, donors[source])
                    behavior = rt.probe_candidate_scores(
                        model,
                        tok,
                        control_cache,
                        presentation,
                        instruction,
                        source,
                        bindings[source],
                        donors[source],
                        args.device,
                    )

                    spans = control_clause_spans(tok, suffix)
                    rel_start, rel_end = spans[source]
                    abs_start = len(prefix_ids) + rel_start
                    abs_end = len(prefix_ids) + rel_end
                    lambda_by_q = {}
                    for q_name in Q_INTERVENTIONS:
                        if q_name == 'CONTROL':
                            qcache = base.clone_cache(control_cache)
                        else:
                            qcache = apply_local_intervention(control_cache, abs_start, abs_end, q_name)
                        role_dists = _role_distributions(
                            model, tok, qcache, history, supports, ADDRESSES, args.device
                        )
                        lambda_by_q[q_name] = {
                            'joint_signature': [role_dists[a]['ordered_top2'] for a in ADDRESSES],
                            'marginal_probability_vectors': {
                                a: role_dists[a]['probability_vector'] for a in ADDRESSES
                            },
                        }
                        del qcache

                    cells.append({
                        'operation': operation,
                        'source_address': source,
                        'behavior': {
                            'outcome': behavior['outcome'],
                            'logprob': behavior['logprob'],
                            'token_counts': behavior['token_counts'],
                        },
                        'Lambda': {
                            'interventions': list(Q_INTERVENTIONS),
                            'joint_signature_by_intervention': {
                                q: lambda_by_q[q]['joint_signature'] for q in Q_INTERVENTIONS
                            },
                            'marginal_probability_vectors_by_intervention': {
                                q: lambda_by_q[q]['marginal_probability_vectors'] for q in Q_INTERVENTIONS
                            },
                            'joint_signature_definition': 'Formula-17-compatible ordered top-2 content labels for H1/H2/H3 retained jointly; holdout scorer freezes top-1 truncation.',
                        },
                    })
                    del control_cache

            rows.append({
                'unit': int(episode['unit']),
                'bindings': bindings,
                'donor_values': donors,
                'finite_value_supports': supports,
                'cells': cells,
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            skips.append({'unit': int(episode.get('unit', -1)), 'error': repr(exc)})

        elapsed = (time.perf_counter() - started) / 60.0
        print(f'[q19-lambda] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} elapsed={elapsed:.1f}m')

    result = {
        'schema': 'co-qwen3-8b-quotient-resolution-19-holdout-panel-v1',
        'stage': 'prospective_fixed_lambda_quotient_holdout',
        'schedule_file': schedule_path.name,
        'schedule_sha256': sha256(schedule_path),
        'model_dir': str(Path(args.model_dir)),
        'device': args.device,
        'dtype': args.dtype,
        'operation_algebra': list(OPS),
        'generalization_level': 'OP',
        'q_interventions': list(Q_INTERVENTIONS),
        'measured_component': 'Lambda',
        'raw_lambda_signature_depth': 2,
        'frozen_candidate_lambda_rank_depth': 1,
        'functional_target': 'ordered 12-cell behavior vector: operation x source_address',
        'descriptor_uses_behavioral_target': False,
        'behavior_is_co_measured_only_for_prospective_evaluation': True,
        'system_operations_unchanged': True,
        'experimental_interventions_are_not_system_operations': True,
        'n_kept': len(rows),
        'n_skipped': len(skips),
        'rows': rows,
        'skips': skips,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[CO QUOTIENT-RESOLUTION 19 LAMBDA HOLDOUT PANEL COMPLETE]', out)
    print('[SHA256]', sha256(out))
    if skips:
        raise SystemExit(f'Technical skips occurred: {len(skips)}')


if __name__ == '__main__':
    main()
