#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from screen_co_u_causal_kernel import (
    ADDRESSES,
    encode,
    fit_family,
    load_panel,
    stable,
    target_vector,
)
from co_u_causal_kernel_analysis import collision_metrics

K_GRID = (2, 3, 4, 5, 6, 8, 10, 12)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def merge_candidate(base: dict[str, Any], extras: list[dict[str, Any]], name: str) -> dict[str, Any]:
    models = list(base['slice_models'])
    slices = list(base['slices'])
    components = [{
        'kind': 'operation_indexed_base',
        'candidate_name': base['candidate_name'],
        'k_requested': int(base['k_requested']),
        'slices': base['slices'],
    }]
    for extra in extras:
        models.extend(extra['slice_models'])
        slices.extend(extra['slices'])
        components.append({
            'kind': 'substitute_address_refinement',
            'candidate_name': extra['candidate_name'],
            'k_requested': int(extra['k_requested']),
            'slices': extra['slices'],
        })
    return {
        'schema': 'co-qwen3-8b-active-causal-kernel-hybrid-candidate-v1',
        'candidate_name': name,
        'family': 'HYBRID_OPERATION_INDEXED_PLUS_SUBSTITUTE_ADDRESS',
        'slice_models': models,
        'slices': slices,
        'components': components,
        'quantizer': 'deterministic standardized k-means fitted without behavioral targets',
        'representation': 'Operation-indexed active causal-kernel codes plus address-resolved SUBSTITUTE refinement.',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    panel = Path(args.panel)
    rows = load_panel(panel)
    targets = [stable(target_vector(row)) for row in rows]

    bases: dict[int, dict[str, Any]] = {}
    base_codes: dict[int, list[tuple[int, ...]]] = {}
    for k in K_GRID:
        c = fit_family(rows, 'K3_ALL_PER_OPERATION', [('ID',), ('KEEP',), ('REVOKE',), ('SUBSTITUTE',)], k)
        bases[k] = c
        base_codes[k] = encode(rows, c)

    extras: dict[tuple[str, int], dict[str, Any]] = {}
    extra_codes: dict[tuple[str, int], list[tuple[int, ...]]] = {}
    for address in ADDRESSES:
        for k in K_GRID:
            c = fit_family(rows, f'KX_SUBSTITUTE_{address}', [('SUBSTITUTE', address)], k)
            extras[(address, k)] = c
            extra_codes[(address, k)] = encode(rows, c)

    records: list[dict[str, Any]] = []
    for base_k in K_GRID:
        base = bases[base_k]
        bcodes = base_codes[base_k]
        choices: list[tuple[tuple[str, int], ...]] = [tuple()]
        choices.extend(((address, k),) for address in ADDRESSES for k in K_GRID)
        for a1, a2 in itertools.combinations(ADDRESSES, 2):
            choices.extend(((a1, k1), (a2, k2)) for k1 in K_GRID for k2 in K_GRID)

        for choice in choices:
            extras_chosen = [extras[key] for key in choice]
            descs = []
            for i in range(len(rows)):
                d = tuple(bcodes[i])
                for key in choice:
                    d += tuple(extra_codes[key][i])
                descs.append(d)
            metrics = collision_metrics(descs, targets)
            total_centers = 4 * int(base_k) + sum(int(k) for _, k in choice)
            max_k = max([int(base_k)] + [int(k) for _, k in choice])
            name_parts = [f'K3K{base_k}'] + [f'SUB_{a}_K{k}' for a, k in choice]
            candidate = merge_candidate(base, extras_chosen, 'HYBRID_' + '__'.join(name_parts))
            candidate.update({
                'base_k': int(base_k),
                'extra_specs': [{'address': a, 'k': int(k)} for a, k in choice],
                'total_centers': int(total_centers),
                'max_k': int(max_k),
            })
            records.append({'candidate': candidate, 'metrics': metrics})

    def rank(rec: dict[str, Any]):
        m = rec['metrics']
        c = rec['candidate']
        closed = int(m['n_collision_pairs']) == 0
        coverage = float(m['nontrivial_coverage'])
        useful = closed and coverage >= 0.20
        return (
            0 if useful else 1 if closed else 2,
            int(m['n_collision_pairs']),
            -coverage,
            int(c['total_centers']),
            int(c['max_k']),
            len(c['extra_specs']),
            c['candidate_name'],
        )

    records.sort(key=rank)
    best = records[0]
    m = best['metrics']
    useful = int(m['n_collision_pairs']) == 0 and float(m['nontrivial_coverage']) >= 0.20
    best_candidate = dict(best['candidate'])
    best_candidate.update({
        'status': 'USEFUL_EXPLORATORY_CANDIDATE' if useful else 'NO_USEFUL_CANDIDATE',
        'selection_used_behavioral_target': True,
        'fit_of_codebooks_used_behavioral_target': False,
        'source_panel_sha256': sha256(panel),
        'source_systems': len(rows),
        'exploratory_metrics': m,
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'must_be_validated_on_fresh_data': bool(useful),
        'scope_note': 'Target-informed exploratory refinement of the active causal-kernel proxy; not a direct Gamma_u, Delta_u, or R_u measurement.',
    })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    search_path = out / 'CO_U_CAUSAL_KERNEL_REFINEMENT_RESULT.json'
    candidate_path = out / 'CO_U_CAUSAL_KERNEL_REFINED_CANDIDATE.json'
    compact_records = [
        {
            'candidate_name': rec['candidate']['candidate_name'],
            'base_k': rec['candidate']['base_k'],
            'extra_specs': rec['candidate']['extra_specs'],
            'total_centers': rec['candidate']['total_centers'],
            'max_k': rec['candidate']['max_k'],
            'metrics': rec['metrics'],
        }
        for rec in records
    ]
    search_path.write_text(json.dumps({
        'schema': 'co-qwen3-8b-active-causal-kernel-hybrid-search-v1',
        'status': 'TARGET_INFORMED_EXPLORATORY_REFINEMENT',
        'source_panel': str(panel),
        'source_panel_sha256': sha256(panel),
        'n_systems': len(rows),
        'k_grid': list(K_GRID),
        'search_space': 'K3 operation-indexed base plus zero, one, or two address-resolved SUBSTITUTE refinements',
        'useful_candidate_rule': 'zero collision pairs and nontrivial coverage >= 0.20',
        'n_candidates': len(compact_records),
        'candidates': compact_records,
    }, indent=2, sort_keys=True) + '\n')
    candidate_path.write_text(json.dumps(best_candidate, indent=2, sort_keys=True) + '\n')

    print('[CO-U ACTIVE KERNEL REFINEMENT COMPLETE]')
    print('[SOURCE SYSTEMS]', len(rows))
    print('[STATUS]', best_candidate['status'])
    print('[CANDIDATE]', best_candidate['candidate_name'])
    print('[BASE K]', best_candidate['base_k'])
    print('[EXTRAS]', ','.join(f"{x['address']}:K{x['k']}" for x in best_candidate['extra_specs']) or 'NONE')
    print('[FIBERS]', m['n_fibers'])
    print('[NONTRIVIAL FIBERS]', m['n_nontrivial_fibers'])
    print('[NONTRIVIAL COVERAGE]', f"{float(m['nontrivial_coverage']):.4f}")
    print('[COLLISION FIBERS]', m['n_collision_fibers'])
    print('[COLLISION PAIRS]', m['n_collision_pairs'])
    print('[BEST]', candidate_path)


if __name__ == '__main__':
    main()
