#!/usr/bin/env python3
"""Role identity t -> t+1 (CPU): compares the causal signature of the C_L24 donor battery at t and at t+1 (u26).

Inputs: the gauge cube at t (acquire_gauge_cube_gpu.py) and at t+1 (role_identity_t1_gpu.py), SHA-256 checked.
Pre-registered comparison (declared before reading the t+1 cube):
  resolution: delta_abs per reader, recalibrated separately at t and at t+1 (maximal deviation of the 9 recodings) ;
  (a) frozen inclusion rule (exists x, forall g, exists r): set of retained roles at t+1 versus at t ;
  (b) per-reader signature: for each reader r, replica g and context c, the partition of the 2^m role configurations
      into scalar delta-components of Y_{g,r} (split at sorted gaps larger than delta_r), on the roles and context of t ;
      compared at t and t+1 up to relabelling (exact equality) and by normalized mutual information ;
  (c) informative readers: readers whose partition has at least two classes, at t and at t+1.
Strong sufficient test: (a) equal and (b) equal for every reader, replica and context. A failure of the strong test
is reported with the graded measures; it is not by itself a refutation of the functional role identity.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u role_identity_t1_cpu.py --world 26
Writes FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_CPU/RESULT.json.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
T0DIR, T0NPZ = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU', 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
T1DIR, T1NPZ = 'FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU', 'S28_T1_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_CPU'
EPS = 1e-15

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load(root, d, n, w):
    d = root / d.format(w=w); n = d / n.format(w=w)
    if not (d / 'RESULT.json').is_file() or not n.is_file():
        raise SystemExit(f'File missing: {d / "RESULT.json"} or {n} ; restore it from the backup')
    if sha(n) != json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))['raw_sha256']:
        raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    with np.load(n, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits = [str(x) for x in z['mask_bits']]
        names = [str(x) for x in z['replica_names']]
    assert cubes.shape == (10, 4096, 12) and names[0] == 'IDENTITY_0'
    return cubes[:, :, [rn.index(b) for b in bits]], bits, names, sha(n)

def masks(bits):
    return np.array([sum(1 << bits[i] for i in range(len(bits)) if (x >> i) & 1) for x in range(1 << len(bits))], dtype=np.int64)

def delta_abs(cubes):
    return np.abs(cubes[1:] - cubes[0][None]).max((0, 1))

def retained(cubes, tol):
    keep = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS).any(2)
        keep.append(bool(hit.all(0).any()))
    return keep

def scalar_partition(y, tol):
    """Scalar delta-components: sort, split at gaps > tol ; canonical labels (order of first occurrence)."""
    order = np.argsort(y, kind='stable')
    cut = np.r_[0, np.cumsum(np.diff(y[order]) > tol + EPS)]
    lab = np.empty(len(y), np.int64); lab[order] = cut
    _, first = np.unique(lab, return_index=True)
    remap = {old: new for new, old in enumerate(lab[np.sort(first)])}
    return np.array([remap[v] for v in lab])

def nmi(a, b):
    n = len(a)
    _, ia = np.unique(a, return_inverse=True); _, ib = np.unique(b, return_inverse=True)
    J = np.zeros((ia.max() + 1, ib.max() + 1)); np.add.at(J, (ia, ib), 1); J /= n
    pa, pb = J.sum(1), J.sum(0)
    H = lambda p: -(p[p > 0] * np.log2(p[p > 0])).sum()
    nz = J > 0
    I = (J[nz] * np.log2(J[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    d = max(H(pa), H(pb))
    return 1.0 if d == 0 else float(I / d)

def compare(c0, c1, S, C):
    t0, t1 = delta_abs(c0), delta_abs(c1)
    cm, xm = masks(C), masks(S)
    eq = np.zeros((10, 12, len(cm)), bool); nm = np.zeros((10, 12, len(cm))); k0 = np.zeros_like(nm); k1 = np.zeros_like(nm)
    for g in range(10):
        for r in range(12):
            for ci, c in enumerate(cm):
                p0 = scalar_partition(c0[g, c + xm, r], t0[r]); p1 = scalar_partition(c1[g, c + xm, r], t1[r])
                eq[g, r, ci] = np.array_equal(p0, p1); nm[g, r, ci] = nmi(p0, p1)
                k0[g, r, ci] = p0.max() + 1; k1[g, r, ci] = p1.max() + 1
    return t0, t1, eq, nm, k0, k1

def selftest():
    rng = np.random.default_rng(5)
    X = np.array([[(m >> i) & 1 for i in range(12)] for m in range(4096)], float)
    W = np.eye(12); W[11, 11] = 0
    c0 = np.stack([X @ W + rng.normal(0, 0.01, (4096, 12)) * (g > 0) for g in range(10)])
    c1 = 2.0 * c0 + 0.3
    keep = retained(c0, delta_abs(c0)); S = [i for i in range(12) if keep[i]]; C = [i for i in range(12) if not keep[i]]
    assert S == list(range(11)) and C == [11]
    _, _, eq, nm, _, _ = compare(c0, c1, S[:4], C + S[4:])
    assert eq.all() and np.allclose(nm, 1.0)
    c2 = c0.copy(); c2[:, :, 3] = c0[:, :, 3] * 0
    _, _, eq2, _, _, k1 = compare(c0, c2, S[:4], C + S[4:])
    assert not eq2[:, 3].any() and (k1[:, 3] == 1).all() and eq2[:, :3].all()
    assert abs(nmi(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) - 1) < 1e-12 and nmi(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])) < 1e-12
    print('[role-t1-cpu] selftest OK (relabelling-invariant partitions, NMI, change detected on one reader only)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    c0, bits, names, s0 = load(a.root, T0DIR, T0NPZ, a.world)
    c1, bits1, names1, s1 = load(a.root, T1DIR, T1NPZ, a.world)
    assert bits == bits1 and names == names1
    k0 = retained(c0, delta_abs(c0)); k1 = retained(c1, delta_abs(c1))
    S = [i for i in range(12) if k0[i]]; C = [i for i in range(12) if not k0[i]]
    t0, t1, eq, nm, n0, n1 = compare(c0, c1, S, C)
    inf0 = sorted({bits[r] for r in range(12) if (n0[:, r] >= 2).all()}); inf1 = sorted({bits[r] for r in range(12) if (n1[:, r] >= 2).all()})
    strong = bool(k0 == k1 and eq.all())
    res = {'world': a.world, 'cube_t_sha256': s0, 'cube_t1_sha256': s1,
           'retained_roles_t': [bits[i] for i in S], 'retained_roles_t1': [bits[i] for i in range(12) if k1[i]],
           'retained_sets_equal': k0 == k1, 'context_t': [bits[i] for i in C],
           'delta_abs_t': dict(zip(bits, t0.tolist())), 'delta_abs_t1': dict(zip(bits, t1.tolist())),
           'partition_equal_fraction': float(eq.mean()),
           'partition_equal_per_reader': {bits[r]: float(eq[:, r].mean()) for r in range(12)},
           'nmi_min_per_reader': {bits[r]: float(nm[:, r].min()) for r in range(12)},
           'classes_t_range_per_reader': {bits[r]: [int(n0[:, r].min()), int(n0[:, r].max())] for r in range(12)},
           'classes_t1_range_per_reader': {bits[r]: [int(n1[:, r].min()), int(n1[:, r].max())] for r in range(12)},
           'informative_readers_t': inf0, 'informative_readers_t1': inf1,
           'strong_sufficient_test_passed': strong,
           'status': 'ROLE_IDENTITY_T_T1_STRONG_TEST_PASSED' if strong else 'ROLE_IDENTITY_T_T1_STRONG_TEST_NOT_PASSED__SEE_GRADED_MEASURES'}
    out = a.root / OUTDIR.format(w=a.world)
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"[role-t1-cpu] retained roles equal: {res['retained_sets_equal']} ; partitions equal: {100 * eq.mean():.1f}% ; "
          f"informative readers t {inf0} / t+1 {inf1}", flush=True)
    for r in range(12):
        print(f"[role-t1-cpu]   {bits[r]:20s} equal {100 * eq[:, r].mean():5.1f}% ; NMI min {nm[:, r].min():.3f} ; "
              f"classes t {res['classes_t_range_per_reader'][bits[r]]} t+1 {res['classes_t1_range_per_reader'][bits[r]]}", flush=True)
    print('[role-t1-cpu] STATUS', res['status'], '; RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
