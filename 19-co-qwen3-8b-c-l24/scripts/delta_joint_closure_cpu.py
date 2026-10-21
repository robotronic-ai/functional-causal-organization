#!/usr/bin/env python3
"""Delta (CPU, V45.1 section 7.2) on the C_L24 contract, from the already acquired u26 gauge cube (no GPU).

C_L24-Delta contract (declared in this script, before any computation):
  individual, sites, donors, frozen L25 readers, Q1 gauges (10), delta_abs primary / delta_contrast control:
  identical to C_L24 (Gamma) ;
  roles = sites retained by the frozen C_L24 rule (exists x, forall g, exists r) ; absorbed sites = conditioned context ;
  P = donor configuration on the retained roles, q_donor uniform (2^-|N| per context) ;
  Y_S = readout of the L25 readers of the roles in S ; classes of P at resolution delta = connected components of the
        relation 'all readers of S within delta' (joint closure; computed separately per gauge and per context) ;
  coalition game v(S) = I_q(P ; Y_S | context) = mean over contexts of H(classes of P under Y_S) (Y deterministic) ;
        v is monotone (fewer readers => coarser partition) ;
  P* = classes of P under Y_N (all roles) ; Delta_cap = v(N) / log2|P*| (mean over contexts of each ratio) ;
  phi_r = exact Shapley value of the game v (2^|N| coalitions) ; shares s_r = phi_r / v(N) ;
  Pi_H = (2^{H(s)} - 1)/(m - 1), Pi_S = (1/sum s^2 - 1)/(m - 1), m = |N| ;
  Delta_H = Delta_cap * Pi_H, Delta_S = Delta_cap * Pi_S ; reported per gauge, minimum over the 10 gauges.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u delta_joint_closure_cpu.py --world 26 [--resume]
Writes FAST_GPU_RESULTS/CLAUDE_S23_DELTA_C_L24_WORLD{w}_CPU/{RESULT.json, S23_GAME_TABLE_u{w}.npz}.
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S9DIR = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
S9NPZ = 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_S23_DELTA_C_L24_WORLD{w}_CPU'
EPS = 1e-15

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def masks(bits):
    return np.array([sum(1 << bits[i] for i in range(len(bits)) if (x >> i) & 1) for x in range(1 << len(bits))], dtype=np.int64)

def deltas(cubes):
    dev = cubes[1:] - cubes[0][None]
    return np.abs(dev).max((0, 1)), (dev.max(1) - dev.min(1)).max(0)

def retained_sites(cubes, tol):
    """Frozen C_L24 rule: exists x_{-i}, forall g, exists r : |Delta_i Y_{g,r}(x)| > delta_r."""
    keep = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS).any(2)
        keep.append(bool(hit.all(0).any()))
    return keep

def entropy_of_components(adj):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(adj)
    if adj.sum() == n:
        return math.log2(n), n
    k, lab = connected_components(csr_matrix(adj), directed=False)
    p = np.bincount(lab) / n
    return float(-(p * np.log2(p)).sum()), int(k)

def game_values(Yc, tol):
    """Yc: (n_arms, m) readers of the retained roles for one context; returns v[mask] and the number of classes for N."""
    n, m = Yc.shape
    base = [np.abs(Yc[:, None, r] - Yc[None, :, r]) <= tol[r] + EPS for r in range(m)]
    v = np.zeros(1 << m); v[0] = 0.0
    full_classes = 1
    stack = [(0, -1, np.ones((n, n), bool))]                       # depth-first: adj(S u {r}) = adj(S) & A_r
    while stack:
        mask, last, adj = stack.pop()
        for r in range(last + 1, m):
            a2 = adj & base[r]; m2 = mask | (1 << r)
            v[m2], k = entropy_of_components(a2)
            if m2 == (1 << m) - 1:
                full_classes = k
            stack.append((m2, r, a2))
    return v, full_classes

def shapley(v, m):
    phi = np.zeros(m)
    fact = [math.factorial(i) for i in range(m + 1)]
    pop = np.array([bin(s).count('1') for s in range(1 << m)])
    for r in range(m):
        bit = 1 << r
        S = np.array([s for s in range(1 << m) if not s & bit])
        w = np.array([fact[pop[s]] * fact[m - pop[s] - 1] / fact[m] for s in S])
        phi[r] = float((w * (v[S | bit] - v[S])).sum())
    return phi

def participation(phi, vN, m):
    if vN <= 0:
        return 0.0, 0.0, np.zeros(m)
    s = np.clip(phi / vN, 0, None); s = s / s.sum()
    nz = s[s > 0]
    H = float(-(nz * np.log2(nz)).sum())
    return (2 ** H - 1) / (m - 1), (1 / float((s ** 2).sum()) - 1) / (m - 1), s

def delta_for_gauge(cube, S, ctx, tol):
    cm = masks(ctx); xm = masks(S); m = len(S)
    vs, caps = [], []
    for c in cm:
        Yc = cube[c + xm][:, S]
        v, k = game_values(Yc, tol[S])
        vs.append(v); caps.append(v[-1] / math.log2(k) if k > 1 else 0.0)
    v = np.mean(vs, 0)
    phi = shapley(v, m)
    pH, pS, s = participation(phi, v[-1], m)
    dcap = float(np.mean(caps))
    return {'v_N_bits': float(v[-1]), 'delta_cap': dcap, 'Pi_H': pH, 'Pi_S': pS, 'Delta_H': dcap * pH, 'Delta_S': dcap * pS,
            'shapley_bits': phi.tolist(), 'shares': s.tolist(), 'shapley_efficiency_residual': float(phi.sum() - v[-1])}, v

def selftest():
    rng = np.random.default_rng(11)
    X = np.array([[(x >> i) & 1 for i in range(6)] for x in range(64)], float)
    W = np.eye(6) + 0.5 * (np.ones((6, 6)) - np.eye(6))
    Y = X @ W + rng.normal(0, 1e-3, (64, 6))
    v, k = game_values(Y, np.full(6, 0.05))
    phi = shapley(v, 6)
    assert abs(phi.sum() - v[-1]) < 1e-9 and k == 64 and abs(v[-1] - 6) < 1e-9
    pH, pS, _ = participation(phi, v[-1], 6)
    assert pH > 0.95 and pS > 0.95, (pH, pS)
    Y2 = np.zeros((64, 6)); Y2[:, 0] = X @ (2.0 ** np.arange(6))           # a single reader carries all the information
    v2, _ = game_values(Y2, np.full(6, 0.05)); phi2 = shapley(v2, 6)
    pH2, pS2, _ = participation(phi2, v2[-1], 6)
    assert pH2 < 1e-9 and pS2 < 1e-9, (pH2, pS2)
    for a in range(64):                                                    # monotonicity
        for r in range(6):
            assert v[a | (1 << r)] >= v[a] - 1e-12
    print(f'[delta] selftest OK (efficient Shapley; symmetric roles -> Pi ~ {pH:.3f}; single informative role -> Pi = 0; v monotone)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    d9 = a.root / S9DIR.format(w=a.world); npz = d9 / S9NPZ.format(w=a.world)
    if not (d9 / 'RESULT.json').is_file():
        raise SystemExit(f'Gauge cube missing: {d9 / "RESULT.json"} ; restore it from the backup')
    r9 = json.loads((d9 / 'RESULT.json').read_text(encoding='utf-8'))
    if not npz.is_file() or sha(npz) != r9['raw_sha256']:
        raise SystemExit(f'NPZ {npz} missing or SHA differs from RESULT.json')
    with np.load(npz, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits = [str(x) for x in z['mask_bits']]
        names = [str(x) for x in z['replica_names']]
    cubes = cubes[:, :, [rn.index(b) for b in bits]]
    assert cubes.shape == (10, 4096, 12) and names[0] == 'IDENTITY_0'
    tols = dict(zip(('delta_abs', 'delta_contraste'), deltas(cubes)))
    label = {'delta_abs': 'delta_abs', 'delta_contraste': 'delta_contrast'}
    keep = retained_sites(cubes, tols['delta_abs'])
    S = [i for i in range(12) if keep[i]]; ctx = [i for i in range(12) if not keep[i]]
    print(f'[delta] u{a.world} retained roles ({len(S)}): {[bits[i] for i in S]} ; context: {[bits[i] for i in ctx]}', flush=True)
    out = a.root / OUTDIR.format(w=a.world); ck = out / 'S23_CHECKPOINT.json'
    if out.exists() and not a.resume:
        raise SystemExit('Output folder already exists: rerun with --resume, or move it: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    done = json.loads(ck.read_text(encoding='utf-8')) if (a.resume and ck.exists()) else {}
    games = {}
    t0 = time.time()
    for dname, tol in tols.items():
        for g in range(len(cubes)):
            key = f'{dname}|{names[g]}'
            if key in done:
                games[key] = np.array(done[key]['_v']); continue
            rec, v = delta_for_gauge(cubes[g], S, ctx, tol)
            rec['_v'] = v.tolist(); done[key] = rec; games[key] = v
            tmp = ck.with_suffix('.tmp'); tmp.write_text(json.dumps(done), encoding='utf-8'); os.replace(tmp, ck)
            print(f"[delta] {label[dname]:16s} {names[g]:15s} v(N) = {rec['v_N_bits']:.4f} bit ; Delta_cap = {rec['delta_cap']:.4f} ; "
                  f"Pi_H = {rec['Pi_H']:.4f} ; Pi_S = {rec['Pi_S']:.4f} ; Delta_H = {rec['Delta_H']:.4f} ; Delta_S = {rec['Delta_S']:.4f} ; "
                  f"{time.time() - t0:.0f} s", flush=True)
    table = out / f'S23_GAME_TABLE_u{a.world}.npz'
    with open(table, 'wb') as f:
        np.savez_compressed(f, **{k.replace('|', '__'): v for k, v in games.items()}, roles=np.array([bits[i] for i in S]),
                            context=np.array([bits[i] for i in ctx]), gauges=np.array(names))
    res = {'world': a.world, 'npz_sha256': sha(npz), 'contract': 'C_L24-Delta (see the script header)', 'roles': [bits[i] for i in S],
           'context': [bits[i] for i in ctx], 'q_law': 'q_donor uniform over the configurations of the retained roles, per context',
           'm': len(S), 'results': {}, 'table_npz': table.name, 'table_sha256': sha(table)}
    for dname in tols:
        per = {n: {k: v for k, v in done[f'{dname}|{n}'].items() if k != '_v'} for n in names}
        res['results'][label[dname]] = {'per_gauge': per,
                                        'min_10_gauges': {q: min(p[q] for p in per.values()) for q in ('delta_cap', 'Pi_H', 'Pi_S', 'Delta_H', 'Delta_S')},
                                        'max_10_gauges': {q: max(p[q] for p in per.values()) for q in ('delta_cap', 'Pi_H', 'Pi_S', 'Delta_H', 'Delta_S')}}
    pm = res['results']['delta_abs']['min_10_gauges']
    res['status'] = ('DELTA_POSITIVE_ON_C_L24__ALL_GAUGES__AWAITING_INDEPENDENT_CERTIFICATE'
                     if pm['Delta_H'] > 1e-12 and pm['Delta_S'] > 1e-12 else 'DELTA_NOT_POSITIVE_ON_C_L24')
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[delta] STATUS: {res['status']} ; min over 10 gauges (delta_abs): Delta_cap {pm['delta_cap']:.4f}, Delta_H {pm['Delta_H']:.4f}, "
          f"Delta_S {pm['Delta_S']:.4f}", flush=True)
    print('[delta] RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
