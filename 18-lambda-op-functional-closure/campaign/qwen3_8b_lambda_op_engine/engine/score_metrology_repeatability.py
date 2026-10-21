#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cross_target_channel import ADDRESSES, FAMILIES, HORIZONS, METRICS, OPS, descriptor, target_vector


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def row_map(panel: dict) -> dict[int, dict]:
    return {int(r['unit']): r for r in panel['rows']}


def cell_map(row: dict) -> dict[tuple[str, str], dict]:
    return {(str(c['operation']), str(c['source_address'])): c for c in row['cells']}


def flatten_internal(row: dict) -> dict[tuple, float]:
    cells = cell_map(row)
    out = {}
    for op in OPS:
        for source in ADDRESSES:
            cell = cells[(op, source)]
            for target in ADDRESSES:
                recs = cell['cross_target_kernel'][target]['ZERO_SUFFIX']
                by_h = {int(r['horizon']): r for r in recs}
                for horizon in HORIZONS:
                    for metric in METRICS:
                        out[(op, source, target, horizon, metric)] = float(by_h[horizon][metric])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', action='append', required=True)
    ap.add_argument('--fidelity', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    panels = [json.loads(Path(p).read_text()) for p in args.repeat]
    if len(panels) < 3:
        raise ValueError('at least three repeat panels are required')
    for p in panels:
        if int(p.get('n_skipped', 0)) != 0:
            raise ValueError('repeat panel contains technical skips')

    maps = [row_map(p) for p in panels]
    units = sorted(maps[0])
    if any(sorted(m) != units for m in maps[1:]):
        raise ValueError('repeat panels do not contain identical unit IDs')

    relative_ranges = []
    absolute_ranges = []
    exact_equal_values = 0
    n_values = 0
    target_stable = 0
    target_total = 0
    behavior_cell_stable = 0
    behavior_cell_total = 0

    # coordinate -> per-unit within-repeat SD and first-repeat values for between-system SD
    coord_within: dict[tuple, list[float]] = defaultdict(list)
    coord_rep1: dict[tuple, list[float]] = defaultdict(list)

    desc_stable = {family: 0 for family in FAMILIES}

    for unit in units:
        rows = [m[unit] for m in maps]
        targets = [target_vector(r) for r in rows]
        target_total += 1
        if all(t == targets[0] for t in targets[1:]):
            target_stable += 1

        cell_maps = [cell_map(r) for r in rows]
        for op in OPS:
            for source in ADDRESSES:
                outcomes = [cm[(op, source)]['behavior']['outcome'] for cm in cell_maps]
                behavior_cell_total += 1
                if all(x == outcomes[0] for x in outcomes[1:]):
                    behavior_cell_stable += 1

        flats = [flatten_internal(r) for r in rows]
        if any(set(f) != set(flats[0]) for f in flats[1:]):
            raise ValueError('internal coordinate mismatch between repeats')
        for coord in sorted(flats[0]):
            vals = np.asarray([f[coord] for f in flats], dtype=np.float64)
            rg = float(vals.max() - vals.min())
            mean = float(np.mean(np.abs(vals)))
            relative_ranges.append(rg / max(mean, 1e-12))
            absolute_ranges.append(rg)
            exact_equal_values += int(np.all(vals == vals[0]))
            n_values += 1
            coord_within[coord].append(float(np.std(vals)))
            coord_rep1[coord].append(float(vals[0]))

        for family in FAMILIES:
            ds = [descriptor(r, family) for r in rows]
            if all(d == ds[0] for d in ds[1:]):
                desc_stable[family] += 1

    noise_signal_ratios = []
    for coord in sorted(coord_within):
        within = float(np.sqrt(np.mean(np.square(coord_within[coord]))))
        between = float(np.std(np.asarray(coord_rep1[coord], dtype=np.float64)))
        if between > 1e-12:
            noise_signal_ratios.append(within / between)

    descriptor_stability = {family: desc_stable[family] / len(units) for family in FAMILIES}
    selected_stability = descriptor_stability.get('ALL_FLOW_H2_FORWARD', 0.0)
    outcome_stability = target_stable / target_total if target_total else 0.0
    p95_ratio = percentile(noise_signal_ratios, 0.95)
    if outcome_stability == 1.0 and selected_stability == 1.0 and p95_ratio <= 0.05:
        status = 'REPEATABILITY_CLEAN'
    elif outcome_stability == 1.0 and selected_stability >= 0.95 and p95_ratio <= 0.20:
        status = 'REPEATABILITY_CAUTION'
    else:
        status = 'REPEATABILITY_UNRESOLVED'

    fidelity = json.loads(Path(args.fidelity).read_text())
    result = {
        'schema': 'co-qwen3-8b-metrology-audit-v1',
        'status': status,
        'status_is_engineering_heuristic_not_formal_theorem': True,
        'repeat_count': len(panels),
        'n_systems_repeated': len(units),
        'dtype': panels[0].get('dtype'),
        'fresh_process_model_reload_each_repeat': True,
        'behavior': {
            'target_vector_stability_fraction': outcome_stability,
            'cell_outcome_stability_fraction': behavior_cell_stable / behavior_cell_total if behavior_cell_total else 0.0,
        },
        'continuous_internal_measurements': {
            'n_values_compared': n_values,
            'exact_equality_fraction': exact_equal_values / n_values if n_values else 0.0,
            'relative_range_median': percentile(relative_ranges, 0.50),
            'relative_range_p95': percentile(relative_ranges, 0.95),
            'relative_range_max': max(relative_ranges) if relative_ranges else 0.0,
            'absolute_range_p95': percentile(absolute_ranges, 0.95),
            'noise_to_between_system_signal_ratio_median': percentile(noise_signal_ratios, 0.50),
            'noise_to_between_system_signal_ratio_p95': p95_ratio,
            'noise_to_between_system_signal_ratio_max': max(noise_signal_ratios) if noise_signal_ratios else 0.0,
        },
        'descriptor_stability_fraction': descriptor_stability,
        'compression_fidelity': fidelity,
        'limitations': [
            'This audit estimates repeatability in the current bfloat16 CUDA stack; it does not estimate systematic bias relative to a float32 reference.',
            'The compression-fidelity audit covers the current cross-target Lambda/active-kernel proxy, not certified Gamma, Delta, R, or a scalar C_O.',
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('[CO-U METROLOGY 09 SCORE]')
    print('[STATUS]', status)
    print('[REPEATED SYSTEMS]', len(units))
    print('[TARGET STABILITY]', f'{outcome_stability:.4f}')
    print('[SELECTED DESCRIPTOR STABILITY]', f'{selected_stability:.4f}')
    print('[NOISE/SIGNAL P95]', f'{p95_ratio:.6g}')
    print('[INTERNAL RELATIVE RANGE P95]', f'{percentile(relative_ranges, 0.95):.6g}')
    print('[RESULT]', out)


if __name__ == '__main__':
    main()
