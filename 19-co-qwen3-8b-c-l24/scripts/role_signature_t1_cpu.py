#!/usr/bin/env python3
"""Per-role causal signature t -> t+1 (CPU), from the gauge cubes at t and at t+1 (u26). No GPU.

For each role i (mask bit of a site retained at t), replica g, context c and reader k, the role depends on reader k when
the scalar delta-partition of reader k (split at sorted gaps larger than delta_k, delta recalibrated at each time) changes
under the flip of bit i for at least one configuration of the other roles. The signature of role i is the boolean array
over (replica, context, reader). Robust signature of role i: the readers on which it depends in every replica and context.
Role i passes the per-role identity test when its robust signature at t+1 equals its robust signature at t and is
non-empty; strict equality over every (replica, context, reader) is reported as well.
Effect signature (no closure): role i acts on reader k when there exists a configuration x_{-i} such that, in EVERY
replica g, |Y_{g,k}(x^i) - Y_{g,k}(x)| > delta_k (the frozen C_L24 inclusion criterion, read reader by reader; delta
recalibrated at each time). Role i passes the effect-signature test when its set of acted-on readers is the same at t and
t+1 and is non-empty.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u role_signature_t1_cpu.py --world 26
Writes FAST_GPU_RESULTS/CLAUDE_S28_ROLE_SIGNATURE_T1_WORLD{w}_CPU/RESULT.json.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
T0DIR, T0NPZ = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU', 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
T1DIR, T1NPZ = 'FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU', 'S28_T1_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_S28_ROLE_SIGNATURE_T1_WORLD{w}_CPU'
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
    assert cubes.shape == (10, 4096, 12)
    return cubes[:, :, [rn.index(b) for b in bits]], bits, sha(n)

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

def scalar_labels(y, tol):
    order = np.argsort(y, kind='stable')
    cut = np.r_[0, np.cumsum(np.diff(y[order]) > tol + EPS)]
    lab = np.empty(len(y), np.int64); lab[order] = cut
    return lab

def signature(cubes, S, C):
    """sig[i, g, c, k]: reader k's delta-partition depends on role S[i] (replica g, context c)."""
    tol = delta_abs(cubes); cm, xm = masks(C), masks(S); m = len(S)
    sig = np.zeros((m, 10, len(cm), 12), bool)
    for g in range(10):
        for ci, c in enumerate(cm):
            for k in range(12):
                lab = scalar_labels(cubes[g, c + xm, k], tol[k])
                for i in range(m):
                    flip = np.arange(1 << m) ^ (1 << i)
                    sig[i, g, ci, k] = bool((lab != lab[flip]).any())
    return sig

def effect_signature(cubes, S):
    """eff[i, k]: exists x_{-i}, forall g, |Delta_i Y_{g,k}(x)| > delta_k."""
    tol = delta_abs(cubes); eff = np.zeros((len(S), 12), bool); wit = np.zeros((len(S), 12), np.int64)
    for j, i in enumerate(S):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS).all(0)
        eff[j] = hit.any(0); wit[j] = hit.sum(0)
    return eff, wit

def selftest():
    rng = np.random.default_rng(2)
    X = np.array([[(m >> i) & 1 for i in range(12)] for m in range(4096)], float)
    W = np.zeros((12, 12)); W[0, 0] = 1; W[1, 1] = 1; W[2, 0] = 1; W[3, 3] = 1
    c0 = np.stack([X @ W + rng.normal(0, 0.002, (4096, 12)) * (g > 0) for g in range(10)])
    W1 = W.copy(); W1[2, 0] = 0; W1[2, 2] = 1
    c1 = np.stack([X @ W1 + rng.normal(0, 0.002, (4096, 12)) * (g > 0) for g in range(10)])
    S, C = [0, 1, 2, 3], list(range(4, 12))
    s0, s1 = signature(c0, S, C), signature(c1, S, C)
    rob = lambda s: [sorted(k for k in range(12) if s[i][:, :, k].all()) for i in range(4)]
    assert rob(s0) == [[0], [1], [0], [3]] and rob(s1) == [[0], [1], [2], [3]], (rob(s0), rob(s1))
    e0, _ = effect_signature(c0, S); e1, _ = effect_signature(c1, S)
    assert [sorted(np.flatnonzero(e0[i])) for i in range(4)] == [[0], [1], [0], [3]]
    assert [sorted(np.flatnonzero(e1[i])) for i in range(4)] == [[0], [1], [2], [3]]
    print('[role-sig] selftest OK (closure and effect signatures: unchanged roles pass, rewired role fails)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    c0, bits, h0 = load(a.root, T0DIR, T0NPZ, a.world)
    c1, bits1, h1 = load(a.root, T1DIR, T1NPZ, a.world)
    assert bits == bits1
    k0 = retained(c0, delta_abs(c0))
    S = [i for i in range(12) if k0[i]]; C = [i for i in range(12) if not k0[i]]
    s0, s1 = signature(c0, S, C), signature(c1, S, C)
    per = {}
    for j, i in enumerate(S):
        eq = np.array_equal(s0[j], s1[j]); nontriv = bool(s0[j].any() and s1[j].any())
        r0 = sorted({bits[k] for k in range(12) if s0[j][:, :, k].all()}); r1 = sorted({bits[k] for k in range(12) if s1[j][:, :, k].all()})
        per[bits[i]] = {'strict_signature_equal_all_replicas_contexts': bool(eq), 'strict_non_trivial': nontriv,
                        'agreement_fraction': float((s0[j] == s1[j]).mean()),
                        'robust_dependent_readers_t': r0, 'robust_dependent_readers_t1': r1,
                        'per_role_identity_test_passed': bool(r0 == r1 and len(r0) > 0)}
        print(f"[role-sig] {bits[i]:20s} robust readers t {r0} ; t+1 {r1} ; strict equal {eq!s:5s} ; agreement "
              f"{per[bits[i]]['agreement_fraction']:.3f}", flush=True)
    e0, w0 = effect_signature(c0, S); e1, w1 = effect_signature(c1, S)
    eff = {}
    for j, i in enumerate(S):
        r0 = [bits[k] for k in range(12) if e0[j, k]]; r1 = [bits[k] for k in range(12) if e1[j, k]]
        uni = set(r0) | set(r1)
        eff[bits[i]] = {'acted_on_readers_t': r0, 'acted_on_readers_t1': r1,
                        'witness_configurations_t': {bits[k]: int(w0[j, k]) for k in range(12) if e0[j, k]},
                        'witness_configurations_t1': {bits[k]: int(w1[j, k]) for k in range(12) if e1[j, k]},
                        'jaccard': float(len(set(r0) & set(r1)) / len(uni)) if uni else 1.0,
                        'effect_signature_test_passed': bool(r0 == r1 and len(r0) > 0)}
        print(f"[role-sig] effect {bits[i]:20s} t {r0} ; t+1 {r1} ; passed {eff[bits[i]]['effect_signature_test_passed']}", flush=True)
    res = {'world': a.world, 'cube_t_sha256': h0, 'cube_t1_sha256': h1, 'roles_t': [bits[i] for i in S], 'context_t': [bits[i] for i in C],
           'test': 'per-role causal signature on the 12 frozen L25 readers, delta recalibrated at t and at t+1, 10 replicas x contexts',
           'per_role': per, 'roles_passing_per_role_identity': [r for r, v in per.items() if v['per_role_identity_test_passed']],
           'effect_signature_test': 'exists x, forall g, |Delta_i Y_{g,k}(x)| > delta_k, reader by reader (frozen C_L24 inclusion criterion)',
           'effect_per_role': eff, 'roles_passing_effect_signature': [r for r, v in eff.items() if v['effect_signature_test_passed']]}
    out = a.root / OUTDIR.format(w=a.world)
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('[role-sig] roles passing the closure-signature test:', res['roles_passing_per_role_identity'], flush=True)
    print('[role-sig] roles passing the effect-signature test:', res['roles_passing_effect_signature'], flush=True)
    print('[role-sig] RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
