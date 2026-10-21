#!/usr/bin/env python3
"""GPU time budget of a joint C_O / rivals acquisition (CPU, standard library only).

One acquisition row = one planned independent realization carrying same-state numerical Gamma, Delta, R, Lambda0, the four M0 covariates,
the four influence-rival covariates and both primary targets (FCA_E1, PCI_T_DS_L27). This script does not run a model and does not decide
any scientific parameter. It converts measured stage durations (copied below with their provenance) and design parameters given on the
command line into a row count and a GPU time estimate, for each of the acquisition scenarios of the table at the end.

Measured stage durations, one operator run each, RTX 5060 Ti, Qwen3-8B (seconds):
  gauge     1912.6  bf16 gauge cube, u26 (certified bf16 scripts; REPRODUCE.md)
  r_greedy   928.5  bf16 R, greedy transition, u26
  r_sampled 6005.0  bf16 R, sampled transition with the 10 replicas, u26
  role_t1   2062.0  bf16 role identity t to t+1, u26 (needed for Lambda0)
  f16_gauge 2184.6  t347 float16 gauge stage, u26 (operator terminal output)
  f16_core  1120.5  t347 float16 core stage, u26 (Gamma frozen rule, Delta, R, Lambda0 stages as chained there; no sampled R)
  e1_6      1127.8  FCA_E1, 6 instances, one arm (bf16 calibration V2, ZERO arm, total_instance_seconds); range over arms 1083-1188
  e1_20     1650.0  FCA_E1, 20 instances, one arm (T353 bf16 pilot, total_instance_seconds 1630-1675 over arms)
  pcit      127.0   PCI_T_DS_L27, 6 episodes, one arm (T353 bf16 pilot, total_site_seconds 126-127 over arms)
Not measured: all four C_O stages on a modified (intervened) state; reference-moment acquisition; any model other than Qwen3-8B.
The estimate assumes the C_O stages cost on a modified state what they cost on the unmodified one; this is an assumption, not a measurement.

Usage:
  python -I -u t366_acquisition_budget_cpu.py --selftest
  python -I -u t366_acquisition_budget_cpu.py --out T366_ACQUISITION_BUDGET.json [--rows-per-predictor 3] [--min-families 5] [--hours-per-night 10]
"""
import argparse
import json
import math
from pathlib import Path

STAGE_SECONDS = dict(gauge=1912.6, r_greedy=928.5, r_sampled=6005.0, role_t1=2062.0, f16_gauge=2184.6, f16_core=1120.5, e1_6=1127.8, e1_20=1650.0, pcit=127.0)
N_M1, N_MR = 8, 8  # predictors without the intercept: M0 (4) + C_O (4), M0 (4) + influence (4)


def rows_needed(rows_per_predictor, min_families, n_pred=N_M1):
    """Smallest total row count such that every outer fold of 5 family-held-out folds keeps rows_per_predictor * (n_pred + 1) training rows
    when the folds are of equal size (4/5 of the rows are in training)."""
    train = rows_per_predictor * (n_pred + 1)
    total = math.ceil(train * 5 / 4)
    return max(total, min_families)


COMPONENTS = {
    'FULL_ROW_BF16': ('gauge', 'r_greedy', 'r_sampled', 'role_t1'),
    'ROW_WITHOUT_R_SAMPLED_BF16': ('gauge', 'r_greedy', 'role_t1'),
    'ROW_GAUGE_AND_R_GREEDY_BF16': ('gauge', 'r_greedy'),
    'ROW_FLOAT16_T347_CHAIN': ('f16_gauge', 'f16_core'),
}


def row_seconds(e1_key, comps):
    return sum(STAGE_SECONDS[c] for c in comps) + STAGE_SECONDS[e1_key] + STAGE_SECONDS['pcit']


def scenarios(rows_per_predictor, min_families, hours_per_night):
    out = []
    for name, comps in COMPONENTS.items():
        for e1 in ('e1_6', 'e1_20'):
            if e1 == 'e1_20' and name != 'FULL_ROW_BF16':
                continue
            for rpp in sorted({2, rows_per_predictor}):
                n = rows_needed(rpp, min_families)
                s = row_seconds(e1, comps)
                out.append(dict(scenario=name + ('' if e1 == 'e1_6' else '_E1_20'), rows_per_predictor=rpp, rows=n, seconds_per_row=s,
                                gpu_hours=round(n * s / 3600, 1), nights=math.ceil(n * s / 3600 / hours_per_night), components=list(comps) + [e1, 'pcit']))
    return out


def selftest():
    assert rows_needed(3, 5) == 34, rows_needed(3, 5)
    assert rows_needed(2, 5) == 23
    assert abs(row_seconds('e1_6', COMPONENTS['FULL_ROW_BF16']) - (1912.6 + 928.5 + 6005.0 + 2062.0 + 1127.8 + 127.0)) < 1e-9
    sc = scenarios(3, 5, 10)
    assert sc[0]['rows'] == 23 and sc[1]['rows'] == 34 and all(s['gpu_hours'] > 0 for s in sc)
    print('[t366] selftest OK (row counts 34 and 23, stage sums)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('T366_ACQUISITION_BUDGET.json'))
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--rows-per-predictor', type=int, default=3)
    ap.add_argument('--min-families', type=int, default=5)
    ap.add_argument('--hours-per-night', type=float, default=10.0)
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    sc = scenarios(a.rows_per_predictor, a.min_families, a.hours_per_night)
    res = dict(schema='T366_ACQUISITION_BUDGET_V1', stage_seconds_measured=STAGE_SECONDS,
               not_measured=['C_O stages on a modified state', 'reference-moment acquisition', 'any model other than Qwen3-8B'],
               parameters=dict(rows_per_predictor=a.rows_per_predictor, min_families=a.min_families, hours_per_night=a.hours_per_night,
                               predictors_M1=N_M1, predictors_MR=N_MR),
               scenarios=sc, gpu_invoked=False,
               independence_note='The row count is a lower bound on independent realizations. Interventions on one set of Qwen3-8B weights are not shown to be independent systems.')
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True), encoding='utf-8')
    print(f"{'scenario':28s} {'rows/pred':>9s} {'rows':>5s} {'s/row':>8s} {'GPU h':>7s} {'nights':>6s}")
    for s in sc:
        print(f"{s['scenario']:28s} {s['rows_per_predictor']:9d} {s['rows']:5d} {s['seconds_per_row']:8.0f} {s['gpu_hours']:7.1f} {s['nights']:6d}")
    print('[t366] written', a.out)


if __name__ == '__main__':
    main()
