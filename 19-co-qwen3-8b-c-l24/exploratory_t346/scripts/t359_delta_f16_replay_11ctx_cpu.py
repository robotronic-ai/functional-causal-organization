#!/usr/bin/env python3
"""CPU-only replay of the float16 Delta game from the raw u26 gauge cube (no GPU, no model).

Purpose: separate the effect of the role boundary from the effect of the numerical regime.
  Variant A: 12 active roles, no context (what T347 computed; replay used as a consistency check).
  Variant B: 11 active roles + context role 'heads[24,25,26,27]' (the historical bf16 boundary), same float16 cube,
             same float16 tolerances, same q_donor law (uniform over donor configurations of the active roles, per context).
Per gauge and tolerance it reports v(N), the number of classes of P* (operational denominator of Delta_cap),
Delta_cap, Pi_H, Pi_S, Delta_H, Delta_S and the exact Shapley values. Nothing here certifies anything.

Usage (from quantum/):
  PY -I -u t359_delta_f16_replay_11ctx_cpu.py --selftest
  PY -I -u t359_delta_f16_replay_11ctx_cpu.py --world 26 --resume [--compare FAST_GPU_RESULTS/T347_S23_DELTA_C_L24_F16_WORLD26_CPU/RESULT.json]
Input : FAST_GPU_RESULTS/T347_S9_FULL12_GAUGE_CUBE_F16_WORLD{w}_GPU/{RESULT.json, S9_FULL12_GAUGE_CUBES_u{w}.npz}
Output: FAST_GPU_RESULTS/T359_DELTA_F16_REPLAY_11CTX_WORLD{w}_CPU/{RESULT.json, CHECKPOINT.json}
Duration: not measured on the real cube. On a synthetic cube in the build sandbox, one gauge took 60 s per tolerance for variant A and about 18 s per
tolerance for variant B (single CPU thread); extrapolated to 10 gauges that is roughly 20 min (A) plus 6 min (B), real data may differ. --resume is supported.
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S9DIR = 'FAST_GPU_RESULTS/T347_S9_FULL12_GAUGE_CUBE_F16_WORLD{w}_GPU'
S9NPZ = 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/T359_DELTA_F16_REPLAY_11CTX_WORLD{w}_CPU'
CONTEXT_ROLE = 'heads[24,25,26,27]'
MODES = ('delta_abs', 'delta_contraste')
EPS = 1e-15


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def masks(bits):
    return np.array([sum(1 << bits[i] for i in range(len(bits)) if (x >> i) & 1) for x in range(1 << len(bits))], dtype=np.int64)


def tolerances(cubes):
    dev = cubes[1:] - cubes[0][None]
    return {'delta_abs': np.abs(dev).max((0, 1)), 'delta_contraste': (dev.max(1) - dev.min(1)).max(0)}


def components(adj):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(adj)
    if adj.sum() == n:
        return math.log2(n), n
    k, lab = connected_components(csr_matrix(adj), directed=False)
    p = np.bincount(lab) / n
    return float(-(p * np.log2(p)).sum()), int(k)


def game_values(Yc, tol):
    n, m = Yc.shape
    base = [np.abs(Yc[:, None, r] - Yc[None, :, r]) <= tol[r] + EPS for r in range(m)]
    v = np.zeros(1 << m)
    classes_full = 1
    stack = [(0, -1, np.ones((n, n), bool))]
    while stack:
        mask, last, adj = stack.pop()
        for r in range(last + 1, m):
            a2 = adj & base[r]
            m2 = mask | (1 << r)
            v[m2], k = components(a2)
            if m2 == (1 << m) - 1:
                classes_full = k
            stack.append((m2, r, a2))
    return v, classes_full


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
    s = np.clip(phi / vN, 0, None)
    s = s / s.sum()
    nz = s[s > 0]
    H = float(-(nz * np.log2(nz)).sum())
    return (2 ** H - 1) / (m - 1), (1 / float((s ** 2).sum()) - 1) / (m - 1), s


def delta_game(cube, S, ctx, tol):
    """cube: (4096, 12) readers for one gauge; S active roles, ctx context roles (indices)."""
    cm = masks(ctx)
    xm = masks(S)
    m = len(S)
    vs, caps, ks = [], [], []
    for c in cm:
        Yc = cube[c + xm][:, S]
        v, k = game_values(Yc, tol[S])
        vs.append(v)
        ks.append(int(k))
        caps.append(v[-1] / math.log2(k) if k > 1 else 0.0)
    v = np.mean(vs, 0)
    phi = shapley(v, m)
    pH, pS, s = participation(phi, v[-1], m)
    dcap = float(np.mean(caps))
    return {'v_N_bits': float(v[-1]), 'classes_Pstar_per_context': ks, 'delta_cap': dcap, 'Pi_H': pH, 'Pi_S': pS,
            'Delta_H': dcap * pH, 'Delta_S': dcap * pS, 'shapley_bits': phi.tolist(), 'shares': s.tolist(),
            'shapley_efficiency_residual': float(phi.sum() - v[-1]), 'min_marginal': float(min(
                (v[[s_ | (1 << r) for s_ in range(1 << m) if not s_ & (1 << r)]] - v[[s_ for s_ in range(1 << m) if not s_ & (1 << r)]]).min()
                for r in range(m)))}


def selftest():
    rng = np.random.default_rng(7)
    m_all = 7
    X = np.array([[(x >> i) & 1 for i in range(m_all)] for x in range(1 << m_all)], float)
    W = np.eye(m_all) + 0.3 * (np.ones((m_all, m_all)) - np.eye(m_all))
    Y = X @ W + rng.normal(0, 1e-3, (1 << m_all, m_all))
    tol = np.full(m_all, 0.05)
    S = [0, 1, 2, 3, 4, 5]
    ctx = [6]
    rec = delta_game(Y, S, ctx, tol)
    # brute force of the same definition
    vs = []
    for c in (0, 1 << 6):
        idx = [c + sum(1 << S[i] for i in range(6) if (x >> i) & 1) for x in range(64)]
        Yc = Y[idx][:, S]
        v = np.zeros(64)
        for mask in range(1, 64):
            rs = [r for r in range(6) if (mask >> r) & 1]
            adj = np.ones((64, 64), bool)
            for r in rs:
                adj &= np.abs(Yc[:, None, r] - Yc[None, :, r]) <= tol[r] + EPS
            seen = -np.ones(64, int)
            k = 0
            for i in range(64):
                if seen[i] < 0:
                    st = [i]
                    seen[i] = k
                    while st:
                        a = st.pop()
                        for b in np.nonzero(adj[a])[0]:
                            if seen[b] < 0:
                                seen[b] = k
                                st.append(b)
                    k += 1
            p = np.bincount(seen) / 64
            v[mask] = float(-(p * np.log2(p)).sum())
        vs.append(v)
    vb = np.mean(vs, 0)
    phib = shapley(vb, 6)
    assert abs(rec['v_N_bits'] - vb[-1]) < 1e-9 and np.allclose(rec['shapley_bits'], phib, atol=1e-9)
    assert abs(rec['shapley_efficiency_residual']) < 1e-9 and rec['min_marginal'] >= -1e-12
    assert rec['classes_Pstar_per_context'] == [64, 64] and abs(rec['delta_cap'] - 1.0) < 1e-9
    # 7 active roles, no context, equals the 6+1 boundary only when the 7th role is informative; here it just must run
    rec7 = delta_game(Y, list(range(7)), [], tol)
    assert abs(rec7['v_N_bits'] - 7.0) < 1e-9
    print('[t359] selftest OK (replay = brute force of the same definition on a 6+1 toy; efficiency; monotone; P* classes reported)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--skip-a', action='store_true', help='skip the 12-role variant (variant B only)')
    ap.add_argument('--compare', type=Path, default=None, help='RESULT.json of the T347 12-role Delta to compare variant A with')
    ap.add_argument('--max-gauges', type=int, default=10)
    ap.add_argument('--cube', type=Path, default=None, help='explicit path of the raw cube NPZ')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    d9 = a.root / S9DIR.format(w=a.world)
    npz = a.cube or (d9 / S9NPZ.format(w=a.world))
    if not npz.is_file():
        raise SystemExit(f'Raw gauge cube missing: {npz} ; restore it from the backup')
    raw_sha = sha(npz)
    rj = d9 / 'RESULT.json'
    if rj.is_file():
        expect = json.loads(rj.read_text(encoding='utf-8')).get('raw_sha256')
        if expect and expect != raw_sha:
            raise SystemExit(f'Cube SHA-256 {raw_sha} differs from RESULT.json raw_sha256 {expect}')
    with np.load(npz, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64)
        rn = [str(x) for x in z['readers']]
        bits = [str(x) for x in z['mask_bits']]
        names = [str(x) for x in z['replica_names']]
    cubes = cubes[:, :, [rn.index(b) for b in bits]]
    assert cubes.shape == (10, 4096, 12) and names[0] == 'IDENTITY_0'
    if CONTEXT_ROLE not in bits:
        raise SystemExit('Context role not in the mask bits: ' + CONTEXT_ROLE)
    ci = bits.index(CONTEXT_ROLE)
    variants = {'B_11_active_plus_context': ([i for i in range(12) if i != ci], [ci])}
    if not a.skip_a:
        variants = {'A_12_active': (list(range(12)), []), **variants}
    tols = tolerances(cubes)
    out = a.root / OUTDIR.format(w=a.world)
    ck = out / 'CHECKPOINT.json'
    if out.exists() and not a.resume:
        raise SystemExit('Output folder already exists: rerun with --resume, or move it: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    done = json.loads(ck.read_text(encoding='utf-8')) if (a.resume and ck.exists()) else {}
    t0 = time.time()
    for vn, (S, ctx) in variants.items():
        for dn in MODES:
            for g in range(min(a.max_gauges, len(cubes))):
                key = f'{vn}|{dn}|{names[g]}'
                if key in done:
                    continue
                rec = delta_game(cubes[g], S, ctx, tols[dn])
                done[key] = rec
                tmp = ck.with_suffix('.tmp')
                tmp.write_text(json.dumps(done), encoding='utf-8')
                os.replace(tmp, ck)
                print(f"[t359] {vn:26s} {dn:16s} {names[g]:15s} v(N) = {rec['v_N_bits']:.4f} ; classes P* = {rec['classes_Pstar_per_context']} ; "
                      f"cap = {rec['delta_cap']:.4f} ; Pi_H = {rec['Pi_H']:.4f} ; Pi_S = {rec['Pi_S']:.4f} ; Delta_H = {rec['Delta_H']:.4f} ; "
                      f"Delta_S = {rec['Delta_S']:.4f} ; {time.time() - t0:.0f} s", flush=True)
    res = {'schema': 'T359_DELTA_F16_REPLAY_11CTX_V1', 'world': a.world, 'cube': str(npz), 'cube_sha256': raw_sha,
           'context_role': CONTEXT_ROLE, 'tolerances': {k: v.tolist() for k, v in tols.items()}, 'readers': bits,
           'q_law': 'q_donor uniform over donor configurations of the active roles, per context', 'variants': {},
           'status': 'REPLAY_DESCRIPTIVE_NOT_A_CERTIFICATE', 'gpu_invoked': False}
    for vn, (S, ctx) in variants.items():
        res['variants'][vn] = {'active': [bits[i] for i in S], 'context': [bits[i] for i in ctx], 'results': {}}
        for dn in MODES:
            per = {names[g]: {k: v for k, v in done[f'{vn}|{dn}|{names[g]}'].items()} for g in range(min(a.max_gauges, len(cubes)))}
            res['variants'][vn]['results'][dn] = {
                'per_gauge': per,
                'min_over_gauges': {q: min(p[q] for p in per.values()) for q in ('delta_cap', 'Pi_H', 'Pi_S', 'Delta_H', 'Delta_S')},
                'max_over_gauges': {q: max(p[q] for p in per.values()) for q in ('delta_cap', 'Pi_H', 'Pi_S', 'Delta_H', 'Delta_S')}}
    if a.compare and 'A_12_active' in res['variants']:
        ref = json.loads(a.compare.read_text(encoding='utf-8'))
        cmp = {}
        for dn, lab in (('delta_abs', 'delta_abs'), ('delta_contraste', 'delta_contrast')):
            d = 0.0
            for gname, rec in res['variants']['A_12_active']['results'][dn]['per_gauge'].items():
                r0 = ref['results'][lab]['per_gauge'][gname]
                d = max(d, abs(rec['Pi_H'] - r0['Pi_H']), abs(rec['Pi_S'] - r0['Pi_S']), abs(rec['delta_cap'] - r0['delta_cap']))
            cmp[dn] = {'max_abs_difference_vs_T347_result': d}
        res['variant_A_vs_T347_result'] = cmp
        print('[t359] variant A versus T347 RESULT.json:', cmp, flush=True)
    with open(out / 'RESULT.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=float)
    for vn in res['variants']:
        for dn in MODES:
            m = res['variants'][vn]['results'][dn]['min_over_gauges']
            print(f"[t359] SUMMARY {vn} {dn}: min over gauges cap {m['delta_cap']:.4f} Pi_H {m['Pi_H']:.4f} Pi_S {m['Pi_S']:.4f} "
                  f"Delta_H {m['Delta_H']:.4f} Delta_S {m['Delta_S']:.4f}", flush=True)
    print('[t359] RESULT', out / 'RESULT.json')


if __name__ == '__main__':
    main()
