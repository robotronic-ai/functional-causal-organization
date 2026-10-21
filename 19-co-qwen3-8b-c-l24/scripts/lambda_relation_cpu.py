#!/usr/bin/env python3
"""Lambda_O lower relation on the C_L24 contract (CPU, no GPU), from the gauge cube(s).

V45.1 section 7.3: Lambda_O is the class, modulo admissible joint recodings, of the oriented content -> role relation
induced by interventional responses; a simultaneous permutation of content and role labels leaves it unchanged, a
one-sided permutation can change it.
Declared instantiation on C_L24:
  contents z_i : the frozen donor distinction at retained L24 role i (its two frozen donor values), i = 1..11 ;
                 distinct contents act on disjoint sites, so every pair of contents is resolved ;
  roles k      : the 12 L25 sites read by the frozen readers (destination side) ;
  cell (z_i, k) RESOLVED_POSITIVE when there exists a configuration x of the other donor bits such that, in EVERY one of
                the 10 Q1 replicas, |Y_{g,k}(x^i) - Y_{g,k}(x)| > delta_k (delta_k = maximal deviation of the 9
                recodings from identity, the 'more than delta' rule of C_L24), the same x and the same reader k in
                every replica ; a 2 delta_k reading is reported as a stricter secondary variant ;
  no negative cell is asserted: an unresolved cell is not evidence that content z_i does not reach role k.
Primary claim: Lambda_O != 0 when the lower relation is non-empty. Every admissible completion of the unresolved cells
contains the lower relation, so no completion can make Lambda_O identically zero.
Descriptive only (not a property of Lambda_O): degree multisets, distinct row supports and one-sided permutation
sensitivity of the LABELED LOWER relation ; without negative cells a completion may fill every unresolved cell, so
selectivity of Lambda_O itself is not claimed. When the gauge cube at t+1 is present, the same relation is reported at
the next active position.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u lambda_relation_cpu.py --world 26
Writes FAST_GPU_RESULTS/CLAUDE_LAMBDA_RELATION_C_L24_WORLD{w}_CPU/RESULT.json.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
T0 = ('FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU', 'S9_FULL12_GAUGE_CUBES_u{w}.npz')
T1 = ('FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU', 'S28_T1_GAUGE_CUBES_u{w}.npz')
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_LAMBDA_RELATION_C_L24_WORLD{w}_CPU'
EPS = 1e-15

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load(root, spec, w, required=True):
    d = root / spec[0].format(w=w); n = d / spec[1].format(w=w)
    if not (d / 'RESULT.json').is_file() or not n.is_file():
        if required:
            raise SystemExit(f'File missing: {d / "RESULT.json"} or {n} ; restore it from the backup')
        return None
    if sha(n) != json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))['raw_sha256']:
        raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    with np.load(n, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits = [str(x) for x in z['mask_bits']]
    assert cubes.shape == (10, 4096, 12)
    return cubes[:, :, [rn.index(b) for b in bits]], bits, sha(n)

def retained(cubes, tol):
    keep = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS).any(2)
        keep.append(bool(hit.all(0).any()))
    return keep

def relation(cubes, contents, factor=1.0):
    """L[j, k]: content contents[j] reaches reader k (exists x, forall g, |Delta| > factor * delta_k) ; W = witness counts."""
    tol = np.abs(cubes[1:] - cubes[0][None]).max((0, 1))
    L = np.zeros((len(contents), 12), bool); W = np.zeros((len(contents), 12), np.int64)
    for j, i in enumerate(contents):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > factor * tol[None, None, :] + EPS).all(0)
        L[j] = hit.any(0); W[j] = hit.sum(0)
    return L, W, tol

def invariants(L):
    rows = [tuple(np.flatnonzero(r)) for r in L]
    return {'cells_resolved_positive': int(L.sum()), 'cells_unresolved': int(L.size - L.sum()), 'non_empty': bool(L.any()),
            'descriptive_lower_relation': {
                'row_degree_multiset': sorted(int(x) for x in L.sum(1)),
                'column_degree_multiset': sorted(int(x) for x in L.sum(0)),
                'distinct_row_supports': len(set(rows)),
                'one_sided_permutation_changes_labeled_lower_relation': len(set(rows)) > 1}}

def selftest():
    rng = np.random.default_rng(6)
    X = np.array([[(m >> i) & 1 for i in range(12)] for m in range(4096)], float)
    W = np.zeros((12, 12)); W[0, 3] = 1; W[1, 3] = 1; W[1, 4] = 1; W[2, 5] = 1; W[11, 11] = 0
    c = np.stack([X @ W + rng.normal(0, 0.002, (4096, 12)) * (g > 0) for g in range(10)])
    L, _, _ = relation(c, [0, 1, 2])
    assert [list(np.flatnonzero(r)) for r in L] == [[3], [3, 4], [5]]
    inv = invariants(L)
    assert inv['non_empty'] and inv['cells_resolved_positive'] == 4
    assert inv['descriptive_lower_relation']['distinct_row_supports'] == 3
    L2, _, _ = relation(c, [0], factor=1000.0)
    assert not L2.any() and not invariants(L2)['non_empty']
    print('[lambda] selftest OK (lower relation, non-emptiness, descriptive invariants, stricter factor)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    c0, bits, h0 = load(a.root, T0, a.world)
    tol0 = np.abs(c0[1:] - c0[0][None]).max((0, 1))
    keep = retained(c0, tol0); S = [i for i in range(12) if keep[i]]; C = [i for i in range(12) if not keep[i]]
    res = {'world': a.world, 'cube_t_sha256': h0, 'contents': [bits[i] for i in S], 'context': [bits[i] for i in C],
           'destination_roles': bits, 'definition': __doc__.split('Declared instantiation on C_L24:')[1].split('Reported:')[0].strip()}
    for label, factor in (('delta', 1.0), ('two_delta', 2.0)):
        L, W, tol = relation(c0, S, factor)
        res[f'relation_t_{label}'] = {bits[i]: [bits[k] for k in np.flatnonzero(L[j])] for j, i in enumerate(S)}
        res[f'witness_counts_t_{label}'] = {bits[i]: {bits[k]: int(W[j, k]) for k in np.flatnonzero(L[j])} for j, i in enumerate(S)}
        res[f'invariants_t_{label}'] = invariants(L)
    t1 = load(a.root, T1, a.world, required=False)
    if t1 is not None:
        c1, bits1, h1 = t1
        assert bits1 == bits
        res['cube_t1_sha256'] = h1
        for label, factor in (('delta', 1.0), ('two_delta', 2.0)):
            L0, _, _ = relation(c0, S, factor); L1, W1, _ = relation(c1, S, factor)
            res[f'relation_t1_{label}'] = {bits[i]: [bits[k] for k in np.flatnonzero(L1[j])] for j, i in enumerate(S)}
            res[f'witness_counts_t1_{label}'] = {bits[i]: {bits[k]: int(W1[j, k]) for k in np.flatnonzero(L1[j])} for j, i in enumerate(S)}
            res[f'invariants_t1_{label}'] = invariants(L1)
            res[f'cells_conserved_t_to_t1_{label}'] = int((L0 & L1).sum())
            res[f'cells_t_{label}'] = int(L0.sum()); res[f'cells_t1_{label}'] = int(L1.sum())
    inv = res['invariants_t_delta']; inv2 = res['invariants_t_two_delta']
    res['status'] = ('LAMBDA_O_NONZERO_LOWER_RELATION_NONEMPTY__AWAITING_INDEPENDENT_CERTIFICATE'
                     if inv['non_empty'] else 'LAMBDA_O_LOWER_RELATION_EMPTY__NO_CLAIM')
    res['status_two_delta_variant'] = 'NON_EMPTY' if inv2['non_empty'] else 'EMPTY_NO_CLAIM'
    out = a.root / OUTDIR.format(w=a.world)
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    for i in S:
        print(f"[lambda] {bits[i]:20s} -> {res['relation_t_delta'][bits[i]]}", flush=True)
    print(f"[lambda] invariants (delta): {inv}", flush=True)
    print(f"[lambda] invariants (2 delta): {inv2}", flush=True)
    print('[lambda] STATUS', res['status'], '; 2 delta variant:', res['status_two_delta_variant'], '; RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
