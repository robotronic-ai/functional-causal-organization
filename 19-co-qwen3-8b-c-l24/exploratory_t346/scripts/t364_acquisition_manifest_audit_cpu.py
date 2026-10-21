#!/usr/bin/env python3
"""Structural audit of an arm-level acquisition manifest for the C_O versus rivals test (CPU, standard library only).

It checks the manifest against the contract of the T363 handoff: one row per planned independent realization, precommitted family
assignments before any target label, same-arm numerical Gamma / Delta / R / Lambda0, the four M0 covariates, the four influence-rival
covariates (MR) and the two primary targets (FCA_E1, PCI_T_DS_L27) measured on the same row.
It does not fit any model, compute any R2, or decide any closure cell: the answer is a readiness table for the four contrasts
D0_j = R2(M1) - R2(M0) and D4_j = R2(M1) - R2(MR), j in {FCA_E1, PCI_T_DS_L27}, plus a list of what is missing.

Usage:
  python -I -u t364_acquisition_manifest_audit_cpu.py --selftest
  python -I -u t364_acquisition_manifest_audit_cpu.py --manifest T364_ACQUISITION_MANIFEST_DRAFT.json --out T364_MANIFEST_AUDIT.json
Audit parameters (not scientific criteria; to be frozen by the arbiter): --min-families (default 5, from the handoff), --rows-per-predictor (default 3).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CO = ('gamma', 'delta', 'r', 'lambda0')
M0 = ('intervention_rms', 'output_information', 'passive_effective_memory', 'preintervention_independent_access_score')
MR = ('influence_frobenius_normalized', 'influence_spectral_norm', 'influence_stable_rank', 'influence_direction_asymmetry')
TARGETS = ('fca_e1', 'pci_t_ds_l27')
FAMILY_STATUS = ('PROPOSED_NOT_PRECOMMITTED', 'PRECOMMITTED')
CO_STATUS = ('NOT_MEASURED', 'MEASURED')


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def row_problems(r):
    p = []
    for k in ('realization_id', 'family_id', 'family_status', 'physical_model', 'boundary', 'regime', 'intervention', 'co', 'm0', 'mr', 'targets'):
        if k not in r:
            p.append('missing key ' + k)
    if p:
        return p
    if r['family_status'] not in FAMILY_STATUS:
        p.append('family_status not in ' + str(FAMILY_STATUS))
    for k in CO:
        c = r['co'].get(k)
        if not isinstance(c, dict) or c.get('status') not in CO_STATUS or 'value' not in c or 'contract_id' not in c:
            p.append('co.' + k + ' malformed')
        elif c['status'] == 'MEASURED' and (not num(c['value']) or not c['contract_id']):
            p.append('co.' + k + ' MEASURED without a finite value and a contract id')
        elif c['status'] == 'NOT_MEASURED' and c['value'] is not None:
            p.append('co.' + k + ' NOT_MEASURED but has a value')
    for grp, names in (('m0', M0), ('mr', MR)):
        for k in names:
            if k not in r[grp] or not (r[grp][k] is None or num(r[grp][k])):
                p.append(grp + '.' + k + ' must be null or a finite number')
    for t in TARGETS:
        v = r['targets'].get(t)
        if not isinstance(v, dict) or 'value' not in v or 'n_units' not in v or 'measured_at' not in v:
            p.append('targets.' + t + ' malformed')
        elif v['value'] is not None and not num(v['value']):
            p.append('targets.' + t + '.value must be null or finite')
    return p


def has(r, group, names):
    return all(num(r[group].get(k)) for k in names)


def co_ok(r):
    return all(r['co'][k]['status'] == 'MEASURED' for k in CO)


def target_ok(r, t):
    return num(r['targets'][t]['value'])


def leakage(r):
    """The family must be precommitted, and committed before the first target measurement of the row."""
    if r['family_status'] != 'PRECOMMITTED':
        return 'family not precommitted'
    t0 = r.get('family_committed_at')
    if not t0:
        return 'no family_committed_at'
    for t in TARGETS:
        m = r['targets'][t]['measured_at']
        if m and m <= t0:
            return 'target ' + t + ' measured before the family commitment'
    return None


def audit(man, min_families=5, rows_per_predictor=3):
    rows = man['rows']
    probs = {r.get('realization_id', '?'): row_problems(r) for r in rows}
    bad = {k: v for k, v in probs.items() if v}
    out = dict(schema='T364_MANIFEST_AUDIT_V1', n_rows=len(rows), n_rows_with_schema_problems=len(bad), schema_problems=bad,
               families_declared=sorted({r.get('family_id') for r in rows if 'family_id' in r}), contrasts={}, verdict='STRUCTURAL_READINESS_ONLY_NO_PREDICTIVE_RESULT')
    if bad:
        out['status'] = 'MANIFEST_NOT_WELL_FORMED'
        return out
    ids = [r['realization_id'] for r in rows]
    if len(set(ids)) != len(ids):
        out['status'] = 'DUPLICATE_REALIZATION_IDS'
        return out
    regimes = {(r['regime'].get('dtype'), r['regime'].get('quantization'), r['boundary'].get('roles_active'), r['boundary'].get('context_role'),
                r['boundary'].get('donor_convention')) for r in rows}
    out['distinct_regime_boundary_donor_conventions'] = len(regimes)
    out['mixed_regimes_must_not_be_pooled'] = len(regimes) > 1
    out['n_controls'] = sum(1 for r in rows if r.get('is_control'))
    out['rows_complete'] = dict(
        M0=sum(has(r, 'm0', M0) for r in rows), CO=sum(co_ok(r) for r in rows), MR=sum(has(r, 'mr', MR) for r in rows),
        fca_e1=sum(target_ok(r, 'fca_e1') for r in rows), pci_t_ds_l27=sum(target_ok(r, 'pci_t_ds_l27') for r in rows))
    for t in TARGETS:
        usable = [r for r in rows if has(r, 'm0', M0) and co_ok(r) and has(r, 'mr', MR) and target_ok(r, t)]
        fams = sorted({r['family_id'] for r in usable})
        leaks = {r['realization_id']: leakage(r) for r in usable if leakage(r)}
        p_max = len(M0) + max(len(CO), len(MR))
        need_rows = rows_per_predictor * (p_max + 1)
        train_rows_min = math.floor(len(usable) * 4 / 5) if usable else 0
        reasons = []
        if not usable:
            reasons.append('no row has M0, the four C_O coordinates, the four MR covariates and the target together')
        if len(fams) < min_families:
            reasons.append(f'{len(fams)} independent families, at least {min_families} required for 5 family-held-out folds')
        if usable and train_rows_min < need_rows:
            reasons.append(f'about {train_rows_min} training rows per fold, audit parameter asks {need_rows} for {p_max} predictors')
        if leaks:
            reasons.append(f'{len(leaks)} usable rows fail the precommitment check')
        ready = not reasons
        for name in ('D0', 'D4'):
            out['contrasts'][f'{name}_{t}'] = dict(status='STRUCTURALLY_COMPUTABLE' if ready else 'NOT_COMPUTABLE', usable_rows=len(usable), families=fams,
                                                    reasons=reasons, precommitment_failures=leaks)
    cp = [r for r in rows if co_ok(r)]
    out['closure'] = dict(rows_with_numeric_profile=len(cp), cell_rule='NOT_EVALUATED_HERE: the frozen T341 cell and eta rule must be applied by the closure test itself',
                          closure_contract_present=bool(man.get('closure_contract')),
                          status='NOT_READY' if not cp or not man.get('closure_contract') else 'PROFILES_AND_CONTRACT_PRESENT_CELL_RULE_NOT_CHECKED')
    out['status'] = 'READY_FOR_ALL_FOUR_CONTRASTS' if all(v['status'] == 'STRUCTURALLY_COMPUTABLE' for v in out['contrasts'].values()) else 'NOT_READY'
    return out


def synthetic(n_fam, per, with_pci=True, precommit=True):
    rows = []
    for f in range(n_fam):
        for a in range(per):
            v = 0.1 * (f + 1) + 0.01 * a
            rows.append(dict(
                realization_id=f'F{f}_A{a}', family_id=f'F{f}', family_status='PRECOMMITTED' if precommit else 'PROPOSED_NOT_PRECOMMITTED',
                family_committed_at='2026-01-01T00:00:00', physical_model='toy', boundary=dict(roles_active=11, context_role='ctx', donor_convention='d'),
                regime=dict(dtype='bfloat16', quantization='none'), intervention=dict(kind='K', site='s', dose=a), is_control=False,
                co={k: dict(contract_id='c', value=v, status='MEASURED') for k in CO}, m0={k: v for k in M0}, mr={k: v for k in MR},
                targets=dict(fca_e1=dict(value=v, n_units=20, measured_at='2026-02-01T00:00:00'),
                             pci_t_ds_l27=dict(value=v if with_pci else None, n_units=6, measured_at='2026-02-01T00:00:00'))))
    return dict(rows=rows)


def selftest():
    a = audit(synthetic(6, 6))
    assert a['status'] == 'READY_FOR_ALL_FOUR_CONTRASTS', a['contrasts']
    a = audit(synthetic(4, 9))
    assert a['status'] == 'NOT_READY' and a['contrasts']['D0_fca_e1']['status'] == 'NOT_COMPUTABLE'
    a = audit(synthetic(6, 6, with_pci=False))
    assert a['contrasts']['D4_pci_t_ds_l27']['status'] == 'NOT_COMPUTABLE' and a['contrasts']['D4_fca_e1']['status'] == 'STRUCTURALLY_COMPUTABLE'
    a = audit(synthetic(6, 6, precommit=False))
    assert a['contrasts']['D0_fca_e1']['status'] == 'NOT_COMPUTABLE' and a['contrasts']['D0_fca_e1']['precommitment_failures']
    m = synthetic(6, 6)
    m['rows'][0]['co']['gamma'] = dict(contract_id='c', value=None, status='MEASURED')
    assert audit(m)['status'] == 'MANIFEST_NOT_WELL_FORMED'
    m = synthetic(6, 6)
    m['rows'][1]['realization_id'] = m['rows'][0]['realization_id']
    assert audit(m)['status'] == 'DUPLICATE_REALIZATION_IDS'
    m = synthetic(6, 6)
    m['rows'][0]['targets']['fca_e1']['measured_at'] = '2025-12-31T00:00:00'
    assert audit(m)['contrasts']['D0_fca_e1']['precommitment_failures']
    print('[t364] selftest OK (complete table, too few families, missing target, no precommitment, malformed row, duplicate ids, label before commitment)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--out', type=Path, default=Path('T364_MANIFEST_AUDIT.json'))
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--min-families', type=int, default=5)
    ap.add_argument('--rows-per-predictor', type=int, default=3)
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    if not a.manifest:
        ap.error('--manifest is required')
    if not a.manifest.is_file():
        raise SystemExit('Manifest missing: ' + str(a.manifest) + ' ; restore it from the backup')
    man = json.loads(a.manifest.read_text(encoding='utf-8'))
    res = audit(man, a.min_families, a.rows_per_predictor)
    res['manifest_sha256'] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    res['gpu_invoked'] = False
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('status', 'n_rows', 'verdict') if k in res}, indent=2))
    for k, v in res.get('contrasts', {}).items():
        print(f"  {k}: {v['status']} (usable rows {v['usable_rows']}, families {len(v['families'])})")
        for why in v['reasons']:
            print('     -', why)
    print('  closure:', res.get('closure', {}).get('status'))
    print('[t364] written', a.out)


if __name__ == '__main__':
    main()
