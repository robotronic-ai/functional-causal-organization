#!/usr/bin/env python3
"""Gamma (CPU) on the C_L24 contract: frozen inclusion rule and the complete cut x context x gauge table.

Frozen inclusion rule: site i is retained if there exists a configuration x of the 11 other sites such that, for EVERY
gauge g, at least one of the 12 L25 readers r (r may depend on g) satisfies |Delta_i Y_{g,r}(x)| > delta_r; otherwise it
is absorbed and conditioned as context (exists x, forall g, exists r).
Reported as secondary variants, without replacing the frozen rule: (forall g, exists x, exists r) and (exists x, exists r, forall g).
Gamma (q_donor = independent Bernoulli(0.5) per site, uniform over donor configurations):
  for each gauge separately, each context and each cut A|B of the retained sites,
  J(A->B) reads the readers of the sites of B (destination side), J(B->A) those of A; pairwise comparisons at delta,
  local closure (within the gauge, the context and the assignment of the fixed side);
  cut value = (mean_c J(A->B) + mean_c J(B->A)) / (2 min(|A|,|B|)) ; Gamma = minimum over all cuts.
delta_abs (primary) and delta_contrast (control), over the 4096 arms of the full gauge cube.
All cuts are computed for every gauge (no early stop); the complete table is written as NPZ for independent
certification. Checkpoint per (delta, gauge).
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u gamma_frozen_rule_cpu.py --world 26 [--resume]
Writes FAST_GPU_RESULTS/CLAUDE_S22_FROZEN_RULE_GAMMA_WORLD{w}_CPU/{RESULT.json, S22_CUT_TABLE_u{w}.npz}.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S9DIR = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
S9NPZ = 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_S22_FROZEN_RULE_GAMMA_WORLD{w}_CPU'
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

def role_screen(cubes, tol):
    """For each site: frozen rule (exists x, forall g, exists r) and two secondary variants."""
    out = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS
        hx = hit.any(2)
        good = hx.all(0)
        out.append({'retained_frozen_rule': bool(good.any()), 'witness_configurations': int(good.sum()),
                    'variant_forall_g_exists_x': bool(hx.any(1).all()),
                    'variant_same_x_same_reader': bool(hit.all(0).any()),
                    'witness_per_gauge': hx.any(1).tolist()})
    return out

def entropy_components(Y, tol):
    n, m, r = Y.shape
    if m == 1:
        return np.zeros(n)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    out = np.zeros(n); step = max(1, (1 << 22) // (m * m))
    for s in range(0, n, step):
        y = Y[s:s + step]
        close = np.ones((len(y), m, m), bool)
        for k in range(r):
            close &= np.abs(y[:, :, None, k] - y[:, None, :, k]) <= tol[k] + EPS
        for i, c in enumerate(close):
            if c.sum() == m:
                out[s + i] = np.log2(m); continue
            lab = connected_components(csr_matrix(c), directed=False)[1]
            _, cnt = np.unique(lab, return_counts=True); p = cnt / m
            out[s + i] = -(p * np.log2(p)).sum()
    return out

def cut_list(S):
    k = len(S)
    return [([S[i] for i in range(k) if (am >> i) & 1], [S[i] for i in range(k) if not (am >> i) & 1]) for am in range(1, 1 << (k - 1))]

def cut_table(cube, S, ctx, tol):
    """Returns (n_cuts,) values averaged over contexts, (n_cuts, n_ctx) values per context, J A->B, J B->A."""
    cm = masks(ctx); cuts = cut_list(S)
    val = np.zeros(len(cuts)); per = np.zeros((len(cuts), len(cm))); jab = np.zeros(len(cuts)); jba = np.zeros(len(cuts))
    for ci, (A, B) in enumerate(cuts):
        js = []
        for V, F in ((A, B), (B, A)):
            idx = cm[:, None, None] + masks(F)[None, :, None] + masks(V)[None, None, :]
            Y = cube[idx.reshape(-1, idx.shape[2])][:, :, F]
            js.append(entropy_components(Y, tol[F]).reshape(len(cm), -1).mean(1))
        den = 2 * min(len(A), len(B))
        per[ci] = (js[0] + js[1]) / den; val[ci] = per[ci].mean(); jab[ci] = js[0].mean(); jba[ci] = js[1].mean()
    return val, per, jab, jba

def selftest():
    rng = np.random.default_rng(9)
    X = np.array([[(m >> i) & 1 for i in range(12)] for m in range(4096)], float)
    W = np.eye(12)
    for i in range(11):
        W[i, i + 1] = 0.6
    W[11, :] = 0; W[:, 11] = 0
    cubes = np.stack([X @ W + rng.normal(0, 0.01, 12) for _ in range(3)])
    da, _ = deltas(cubes)
    scr = role_screen(cubes, da)
    assert all(s['retained_frozen_rule'] for s in scr[:11]) and not scr[11]['retained_frozen_rule']
    S = [1, 2, 3, 4]; ctx = [b for b in range(12) if b not in S]
    val, per, jab, jba = cut_table(cubes[0], S, ctx, da)
    assert len(val) == 7 and val.min() > 0.2
    W2 = W.copy(); W2[2, 3] = 0
    v2, _, _, _ = cut_table(X @ W2, S, ctx, da)
    assert v2.min() < 0.13, v2.min()
    print('[gamma] selftest OK (frozen rule: site without effect absorbed; table of the 7 cuts; broken chain -> weak cut)')

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
    screen = role_screen(cubes, tols['delta_abs'])
    S = [i for i in range(12) if screen[i]['retained_frozen_rule']]; ctx = [i for i in range(12) if i not in S]
    print(f"[gamma] u{a.world} frozen rule: retained {[bits[i] for i in S]} ; absorbed {[bits[i] for i in ctx]}", flush=True)
    print(f"[gamma] variant (forall g, exists x): retained {[bits[i] for i in range(12) if screen[i]['variant_forall_g_exists_x']]}", flush=True)
    print(f"[gamma] variant (same x and same reader): retained {[bits[i] for i in range(12) if screen[i]['variant_same_x_same_reader']]}", flush=True)
    cuts = cut_list(S)
    print(f'[gamma] {len(S)} retained sites -> {len(cuts)} cuts x {len(masks(ctx))} contexts x 10 gauges x 2 deltas', flush=True)
    out = a.root / OUTDIR.format(w=a.world); ck = out / 'S22_CHECKPOINT.npz'
    if out.exists() and not a.resume:
        raise SystemExit('Output folder already exists: rerun with --resume, or move it: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    if a.resume and ck.exists():
        with np.load(ck, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
    t0 = time.time()
    for dname, tol in tols.items():
        for g in range(len(cubes)):
            key = f'{dname}__{g}'
            if f'{key}__val' in data:
                continue
            val, per, jab, jba = cut_table(cubes[g], S, ctx, tol)
            data.update({f'{key}__val': val, f'{key}__per': per, f'{key}__jab': jab, f'{key}__jba': jba})
            tmp = ck.with_suffix('.tmp.npz')
            with open(tmp, 'wb') as f:
                np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, ck)
            k = int(np.argmin(val))
            print(f'[gamma] {label[dname]:16s} {names[g]:15s} Gamma = {val[k]:.6f} ; critical cut A = {[bits[i] for i in cuts[k][0]]} ; '
                  f'{time.time() - t0:.0f} s', flush=True)
    table = out / f'S22_CUT_TABLE_u{a.world}.npz'
    arrays = {f'{d}_{part}': np.stack([data[f'{d}__{g}__{part}'] for g in range(len(cubes))]) for d in tols for part in ('val', 'per', 'jab', 'jba')}
    with open(table, 'wb') as f:
        np.savez_compressed(f, **arrays, gauges=np.array(names), retained_sites=np.array([bits[i] for i in S]),
                            context_sites=np.array([bits[i] for i in ctx]),
                            cut_A_masks=np.array([sum(1 << S.index(i) for i in A) for A, _ in cuts], dtype=np.int64))
    res = {'world': a.world, 'npz_sha256': sha(npz), 'q_law': 'q_donor(x) = 2^-12 over the 4096 donor configurations (no q_bal, no reweighting)',
           'rule': 'C_L24: there exists x_{-i} such that for every gauge there exists a reader r with |Delta_i Y| > delta_abs_r',
           'retained_sites': [bits[i] for i in S], 'context_sites': [bits[i] for i in ctx],
           'screen': {bits[i]: screen[i] for i in range(12)}, 'cuts': len(cuts),
           'cut_readout': 'J(A->B) on the readers of B, J(B->A) on the readers of A', 'results': {},
           'table_npz': table.name, 'table_sha256': sha(table)}
    for d in tols:
        v = arrays[f'{d}_val']; k = v.argmin(1)
        res['results'][label[d]] = {'gamma_per_gauge': dict(zip(names, v.min(1).tolist())), 'gamma_min_10_gauges': float(v.min()),
                                    'all_gauges_positive': bool((v.min(1) > 1e-12).all()),
                                    'critical_cut_per_gauge': {names[g]: [bits[i] for i in cuts[k[g]][0]] for g in range(len(names))},
                                    'min_per_context_10_gauges': float(arrays[f'{d}_per'].min())}
    prim = res['results']['delta_abs']
    res['status'] = ('GAMMA_POSITIVE_ON_FROZEN_RULE_MACROSTATE__ALL_GAUGES__AWAITING_INDEPENDENT_CERTIFICATE'
                     if prim['all_gauges_positive'] else 'GAMMA_NOT_POSITIVE_ON_FROZEN_RULE_MACROSTATE')
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[gamma] STATUS: {res['status']} ; Gamma min (delta_abs, 10 gauges) = {prim['gamma_min_10_gauges']:.6f} ; "
          f"delta_contrast control = {res['results']['delta_contrast']['gamma_min_10_gauges']:.6f}", flush=True)
    print('[gamma] RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
