#!/usr/bin/env python3
"""Independent T321 certificate audit for the sole required world, u26.

This verifier audits every cut, gauge, source/target direction, witness gap,
Bernoulli(1/2)^12 weight and entropy lower bound in a T321 report. If the S9
FULL12 NPZ is accessible, it additionally reconstructs each scalar gap from
raw observations and checks the S20 shared-assignment role screen.

No u25 gate, no q_bal, no universal gauge decoder, no claims about other C_O
components. A report-only PASS verifies internal arithmetic, not raw evidence.

Run from quantum/:
  python -I -u T322_VERIFY_U26_GAMMA_CERT_CPU.py --selftest
  python -I -u T322_VERIFY_U26_GAMMA_CERT_CPU.py
  python -I -u T322_VERIFY_U26_GAMMA_CERT_CPU.py --require-raw
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
RESULT_REL = Path('FAST_GPU_RESULTS/T321_GAMMA_C_L24_GAP_WORLD26_CPU/RESULT.json')
RAW_REL = Path('FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU/S9_FULL12_GAUGE_CUBES_u26.npz')
T320_REL = Path('FAST_GPU_RESULTS/T320_S20_SHARED_CONFIG_WORLD26_CPU/RESULT.json')
OUT_REL = Path('FAST_GPU_RESULTS/T322_GAMMA_U26_INDEPENDENT_AUDIT_CPU/RESULT.json')
GAUGES = ('IDENTITY_0', 'UPDOWN_SCALE_0', 'UPDOWN_SCALE_1', 'UPDOWN_SCALE_2',
          'VO_DIAG_0', 'VO_DIAG_1', 'VO_DIAG_2',
          'QKNORM_RECIP_0', 'QKNORM_RECIP_1', 'QKNORM_RECIP_2')
RETAINED = tuple(range(11))
CONTEXT = (11,)
N_CUTS = (1 << (len(RETAINED)-1))-1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, what, *, atol=2e-12):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), what + ': non-finite')
    require(math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=2e-12),
            f'{what}: actual={actual}, expected={expected}')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fd:
        for part in iter(lambda: fd.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def h2(k, n):
    p = k / n
    return 0.0 if k == 0 or k == n else -p*math.log2(p) - (1-p)*math.log2(1-p)


def offsets(indices):
    v = np.arange(1 << len(indices), dtype=np.int64)
    o = np.zeros(len(v), dtype=np.int64)
    for j, bit in enumerate(indices):
        o |= ((v >> j) & 1) << bit
    return o


def screen_raw(cubes, delta):
    masks = np.arange(4096, dtype=np.int64)
    kept = []
    for bit in range(12):
        lo = masks[(masks & (1 << bit)) == 0]
        hi = lo | (1 << bit)
        # The witness assignment must be shared by all ten gauges;
        # the reader is allowed to differ among them.
        observed = np.abs(cubes[:, hi] - cubes[:, lo]) > delta[None, None, :] + 1e-15
        if np.all(np.any(observed, axis=-1), axis=0).any():
            kept.append(bit)
    return kept


def verify(report, *, raw=None, raw_sha=None, t320=None):
    require(report['world'] == 26, 'Only world u26 is a certification gate')
    require(report['status'] == 'NONZERO_CERTIFIED_ON_C_L24_FINITE_DONOR_INTERFACE', 'T321 did not pass')
    require(report['schema'] == 'T321_C_L24_GAMMA_GAP_WITNESS_V1', 'Unexpected report schema')
    require(report['certificate_method'] == 'scalar_sorted_gap_implies_disconnected_local_delta_graph', 'Unexpected method')
    require(report['q'] == 'q_donor=independent_Bernoulli_0.5_on_all_12_donor_sites', 'Incorrect intervention law')
    require(report['q_bal_used'] is False, 'q_bal is forbidden in this audit')
    require(report['exhaustive_search_for_each_cut'] is True, 'Incomplete search flag')
    require(report['t320_status'] == 'S20_SHARED_ASSIGNMENT_MATCH', 'Role screen not qualified')
    require(tuple(report['retained_indices']) == RETAINED, 'Unexpected retained role set')
    require(tuple(report['context_indices']) == CONTEXT, 'Unexpected context role')
    require(report['n_cuts'] == N_CUTS, 'Wrong declared cut count')
    require(tuple(report['gauge_names']) == GAUGES, 'Unexpected gauge ordering')
    require(len(report['per_gauge']) == len(GAUGES), 'Wrong gauge count')
    role_names = [report['roles'][str(i)] for i in range(12)]
    delta = np.array([report['absolute_delta'][s] for s in role_names], dtype=np.float64)
    require(np.isfinite(delta).all() and (delta >= 0).all(), 'Invalid delta')
    if raw is not None:
        require(raw.shape == (10, 4096, 12), 'Wrong raw shape')
        require(raw_sha == report['source_sha256'], 'NPZ does not match the uploaded report')
        require(np.isfinite(raw).all(), 'Raw contains non-finite values')
        near(np.max(np.abs(raw[1:] - raw[:1]) - delta[None, None, :]), 0.0,
             'Reader absolute delta check', atol=2e-10)
        require(screen_raw(raw, delta) == list(RETAINED), 'Raw role screen does not match S20')
    if t320 is not None:
        require(t320['world'] == 26 and t320['status'] == 'S20_SHARED_ASSIGNMENT_MATCH', 'T320 status differs')
        require(t320['source_sha256'] == report['source_sha256'], 'T320 and T321 hashes differ')
        require(t320['retained_indices'] == list(RETAINED), 'T320 and T321 role sets differ')
    low_by_gauge = []
    all_margins = []
    verifications = 0
    for gi, gr in enumerate(report['per_gauge']):
        require(gr['gauge'] == GAUGES[gi], 'Wrong gauge in per-gauge record')
        require(gr['certified_cuts'] == N_CUTS and gr['total_cuts'] == N_CUTS, 'Incomplete cut certificate')
        require(gr['all_cuts_certified_by_gap'] is True and gr['uncertified_cuts'] == [], 'Uncertified cut')
        require(len(gr['cuts']) == N_CUTS, 'Cut record count mismatch')
        bounds = []
        for cm, cut in enumerate(gr['cuts'], start=1):
            a = tuple(r for j,r in enumerate(RETAINED) if (cm >> j) & 1)
            b = tuple(r for r in RETAINED if r not in a)
            require(a and b, 'Empty side in a cut')
            require(cut['A'] == list(a) and cut['B'] == list(b), f'Wrong canonical cut #{cm}')
            require(cut['witness_found'] is True, f'No witness for {GAUGES[gi]} cut #{cm}')
            w = cut['witness']
            src = tuple(w['source_roles']); tgt = tuple(w['target_roles'])
            require((src == a and tgt == b) or (src == b and tgt == a), 'Witness not across the declared cut')
            require(w['reader_role_index'] in tgt, 'Reader is not in the target side')
            k = int(w['left_count']); n = int(w['n_source_assignments'])
            require(n == 2**len(src) and 0 < k < n, 'Invalid source partition size')
            require(int(w['source_assignment_count']) == n, 'Bad source count')
            require(int(w['context_mask']) in (0, 1<<11), 'Wrong context mask')
            require(int(w['target_fixed_mask']) & ~sum(1<<j for j in tgt) == 0, 'Fixed mask outside target')
            fixed = int(w['target_fixed_mask']) | int(w['context_mask'])
            require(int(w['source_zero_mask']) == fixed, 'Source-zero mask mismatch')
            require(fixed & sum(1<<j for j in src) == 0, 'Source bits should be zero in base mask')
            r = int(w['reader_role_index'])
            near(w['threshold_delta'], delta[r], 'Witness delta')
            margin = float(w['gap']) - float(delta[r])
            require(margin > 1e-12, 'Unresolved scalar gap')
            all_margins.append(margin)
            ent = h2(k, n)
            prob = 2.0**(-len(CONTEXT)-len(tgt))
            direction_lb = prob * ent
            gamma_cut_lb = direction_lb/(2*min(len(a),len(b)))
            near(w['entropy_lb_bits'], ent, 'Binary entropy')
            near(w['local_context_and_fixed_probability'], prob, 'q_donor fixed assignment probability')
            near(w['directional_J_lb_bits'], direction_lb, 'J lower bound')
            near(cut['gamma_cut_lower_bound'], gamma_cut_lb, 'Gamma cut lower bound')
            if raw is not None:
                vals = np.sort(raw[gi, fixed | offsets(src), r])
                gap = float(vals[k]-vals[k-1])
                near(gap, w['gap'], 'Witness scalar sorted gap', atol=2e-10)
                require(gap > delta[r]+1e-12, 'Raw gap fails absolute delta')
            bounds.append(gamma_cut_lb)
            verifications += 1
        lg = min(bounds)
        near(gr['gamma_lower_bound'], lg, f'{GAUGES[gi]} min lower bound')
        low_by_gauge.append(lg)
    overall = min(low_by_gauge)
    require(overall > 0, 'Nonpositive global bound')
    near(report['gamma_global_lower_bound'], overall, 'Global Gamma bound')
    require(report['all_gauges_pass'] is True, 'Unexpected top-level status')
    return {
        'status': 'RAW_WITNESSES_VERIFIED' if raw is not None else 'REPORT_ARITHMETIC_VERIFIED_RAW_PENDING',
        'world': 26,
        'scope': 'C_L24 finite donor interface; frozen L25 readers; single layer; absolute Q1 delta',
        'q': 'q_donor=Bernoulli(0.5)^12',
        'q_bal_used': False,
        'gauge_count': len(GAUGES),
        'cuts_per_gauge': N_CUTS,
        'witness_records_checked': verifications,
        'retained_roles': list(RETAINED),
        'context_roles': list(CONTEXT),
        'gamma_lower_bound': overall,
        'per_gauge_lower_bounds': {name: lb for name,lb in zip(GAUGES, low_by_gauge)},
        'minimum_scalar_gap_margin_over_all_witnesses': min(all_margins),
        'source_sha256': report['source_sha256'],
        'raw_evidence_checked': raw is not None,
        'limitations': [
            'One certified world, u26; no u25 replication is required.',
            'Certification concerns the declared finite C_L24 donor intervention contract only.',
            'Report-only mode checks witness consistency and lower-bound arithmetic, not the source NPZ.',
            'A certified scalar gap proves finite-graph separation but does not estimate exact Gamma.',
        ],
    }


def selftest():
    near(h2(1,2), 1.0, 'Binary one-bit entropy')
    near(h2(1,4), 0.8112781244591328, 'Binary nonuniform split entropy')
    require(list(offsets((0,2,10))) == [0,1,4,5,1024,1025,1028,1029], 'Offset bit order')
    x = np.zeros((10,4096,12), dtype=np.float64)
    y = screen_raw(x, np.zeros(12))
    require(y == [], 'Null-effect site retained')
    x[:, np.arange(4096) & 1 == 1, 1] = 2.0
    require(screen_raw(x, np.full(12, 0.1)) == [0], 'Shared-assignment site not detected')
    print('[T322] SELFTEST PASS: weights, entropy, bit ordering and shared-assignment role screen')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--report', type=Path, default=None)
    ap.add_argument('--raw', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--require-raw', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return
    report_path = args.report or args.root/RESULT_REL
    report = json.loads(report_path.read_text(encoding='utf-8'))
    raw_path = args.raw or args.root/RAW_REL
    raw = None
    raw_digest = None
    if raw_path.is_file():
        raw_digest = sha(raw_path)
        with np.load(raw_path, allow_pickle=False) as z:
            names = list(map(str,z['replica_names']))
            readers = list(map(str,z['readers']))
            sites = list(map(str,z['mask_bits']))
            require(tuple(names) == GAUGES, 'NPZ gauge names mismatch')
            require(len(set(sites)) == len(readers) == 12 and set(readers) == set(sites), 'Reader/site mismatch')
            require(sites == [report['roles'][str(i)] for i in range(12)], 'NPZ site ordering mismatch')
            raw = np.asarray(z['cubes'][:,:, [readers.index(s) for s in sites]], dtype=np.float64)
    require(not args.require_raw or raw is not None, 'Raw FULL12 NPZ is required')
    t320_path = args.root / T320_REL
    t320 = json.loads(t320_path.read_text(encoding='utf-8')) if t320_path.is_file() else None
    out = verify(report,raw=raw,raw_sha=raw_digest,t320=t320)
    path = args.out or args.root/OUT_REL
    if path.exists() and not args.overwrite:
        raise FileExistsError('Refusing overwrite; pass --overwrite or --out to choose another path: ' + str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print('[T322]', out['status'], 'world', 26, 'gauges', out['gauge_count'],
          'verified_witnesses', out['witness_records_checked'],
          'gamma_lb', out['gamma_lower_bound'], 'RESULT', path, flush=True)

if __name__ == '__main__':
    main()
