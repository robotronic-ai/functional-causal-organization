#!/usr/bin/env python3
"""Raw witness extract for one role (CPU): return bound (sampled acquisition) and effect signature t -> t+1 (gauge cubes).

Inputs (FAST_GPU_RESULTS/, SHA-256 checked against their RESULT.json):
  CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU/S26_R_SAMPLED_RAW_u{w}.npz      sampled return acquisition
  CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU/S9_FULL12_GAUGE_CUBES_u{w}.npz  gauge cube at t
  CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU/S28_T1_GAUGE_CUBES_u{w}.npz     gauge cube at t+1
Output (FAST_GPU_RESULTS/ROLE_WITNESS_EXTRACT_{role}_u{w}/):
  return_pairs.csv     one row per (replica, donor pair x0 -> x1 = x0 with the role bit set): interval of P(A | x0) and of
                       P(A | x1) (A = return event of the selected reference token, worst-case mass outside V) and the
                       minimal binary Jensen-Shannon divergence in bits ; B_r per replica = mean over the 2048 pairs ;
  effect_witnesses.csv one row per (time, reader, x0) witness of the effect signature: the 10 raw differences
                       Y_{g,k}(x1) - Y_{g,k}(x0) and the margin min_g |difference| - delta_k ;
  summary.json         definitions, deltas, reference token and its ball, B_r per replica, witness counts per reader at
                       t and t+1, SHA-256 of the sources and of the two CSV files ;
  and a zip of the folder next to it, ready to send.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u extract_role_witnesses_cpu.py --world 26 --role mlp0
"""
import argparse, csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S26 = ('FAST_GPU_RESULTS/CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU', 'S26_R_SAMPLED_RAW_u{w}.npz')
T0 = ('FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU', 'S9_FULL12_GAUGE_CUBES_u{w}.npz')
T1 = ('FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU', 'S28_T1_GAUGE_CUBES_u{w}.npz')
OUTDIR = 'FAST_GPU_RESULTS/ROLE_WITNESS_EXTRACT_{role}_u{w}'
EPS = 1e-15

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def checked(root, spec, w):
    d = root / spec[0].format(w=w); n = d / spec[1].format(w=w)
    if not (d / 'RESULT.json').is_file() or not n.is_file():
        raise SystemExit(f'File missing: {d / "RESULT.json"} or {n} ; restore it from the backup')
    if sha(n) != json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))['raw_sha256']:
        raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    return n

def hbin(p):
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, -p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
        b = np.where(p < 1, -(1 - p) * np.log1p(-np.where(p < 1, p, 0.0)) / np.log(2), 0.0)
    return a + b

def js_bin(p, q):
    return np.maximum(hbin(0.5 * (p + q)) - 0.5 * (hbin(p) + hbin(q)), 0.0)

def js_min_box(l1, u1, l2, u2):
    lo = np.where(u1 < l2, u1, np.where(u2 < l1, u2, 0.0))
    hi = np.where(u1 < l2, l2, np.where(u2 < l1, l1, 0.0))
    gap = np.where(u1 < l2, l2 - u1, np.where(u2 < l1, l1 - u2, 0.0))
    return np.where(gap > 0, js_bin(lo, hi), 0.0)

def pairs_for_bit(i):
    b = np.array([m for m in range(4096) if not (m >> i) & 1])
    return b, b | (1 << i)

def event_interval(pV, tail, ball):
    s_in = pV[..., ball].sum(-1); s_out = pV[..., ~ball].sum(-1)
    small = s_out if s_in.mean() > 0.5 else s_in
    return small, small + tail

def return_extract(lpV, tail, prel, d_t1, i, s):
    """Same construction as r_return_bound_cpu.py (primary bound): best reference token by the minimum over replicas."""
    pV = np.exp(lpV); b0, b1 = pairs_for_bit(i); best = None
    for rho in range(prel.shape[0]):
        ball = prel[rho, :, s] <= d_t1[s]
        if ball.all():
            continue
        lo, hi = event_interval(pV, tail, ball)
        js = js_min_box(lo[:, b0], hi[:, b0], lo[:, b1], hi[:, b1])
        B = js.mean(1)
        if best is None or B.min() > best[0].min():
            best = (B, rho, ball, lo, hi, js)
    return best, b0, b1

def effect_extract(cubes, i):
    tol = np.abs(cubes[1:] - cubes[0][None]).max((0, 1))
    b0, b1 = pairs_for_bit(i)
    diff = cubes[:, b1] - cubes[:, b0]
    hit = (np.abs(diff) > tol[None, None, :] + EPS).all(0)
    margin = (np.abs(diff) - tol[None, None, :]).min(0)
    return tol, b0, b1, diff, hit, margin

def load_cube(n):
    with np.load(n, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits = [str(x) for x in z['mask_bits']]
    return cubes[:, :, [rn.index(b) for b in bits]], bits

def selftest():
    rng = np.random.default_rng(4)
    G, n = 3, 4096
    p = np.tile(np.array([0.999, 0.0006, 0.0004]), (G, n, 1)); on = (np.arange(n) >> 5) & 1
    p[:, on == 1, 1] += 0.0003; p[:, on == 1, 0] -= 0.0003
    prel = np.full((3, 3, 12), 0.01); prel[0, 1, 2] = prel[1, 0, 2] = 0.9; prel[1, 2, 2] = prel[2, 1, 2] = 0.9
    best, b0, b1 = return_extract(np.log(p), np.zeros((G, n)), prel, np.full(12, 0.05), 5, 2)
    assert best[0].min() > 1e-7 and best[5].shape == (G, 2048)
    X = np.array([[(m >> k) & 1 for k in range(12)] for m in range(n)], float)
    W = np.zeros((12, 12)); W[5, 3] = 1.0; W[2, 2] = 1.0
    c = np.stack([X @ W + rng.normal(0, 0.001, (n, 12)) * (g > 0) for g in range(10)])
    tol, _, _, diff, hit, margin = effect_extract(c, 5)
    assert hit[:, 3].all() and not hit[:, 2].any() and (margin[:, 3] > 0.9).all()
    print('[witness-extract] selftest OK (return pairs, effect witnesses, margins)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--role', default='mlp0')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    w = a.world
    n26, n0, n1 = checked(a.root, S26, w), checked(a.root, T0, w), checked(a.root, T1, w)
    with np.load(n26, allow_pickle=False) as z:
        bits = [str(x) for x in z['mask_bits']]; sites = [str(x) for x in z['sites']]; V = [int(t) for t in z['V']]
        lpV = z['lp_V'].astype(np.float64); tail = z['mass_outside_V'].astype(np.float64)
        cg = [str(x) for x in z['channel_gauges']]; d_t1 = z['delta_t1'].astype(np.float64)
        prel = z['token_pair_rel_t1'].astype(np.float64)
    if a.role not in bits:
        raise SystemExit(f'Unknown role {a.role}; mask bits: {bits}')
    i = bits.index(a.role); s = sites.index(a.role)
    (B, rho, ball, lo, hi, js), b0, b1 = return_extract(lpV, tail, prel, d_t1, i, s)
    out = a.root / OUTDIR.format(role=a.role.replace('[', '').replace(']', '').replace(',', '-'), w=w)
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'return_pairs.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['replica', 'x0', 'x1', 'P_A_x0_low', 'P_A_x0_high', 'P_A_x1_low', 'P_A_x1_high', 'js_min_bits'])
        for g in range(len(cg)):
            for j in range(len(b0)):
                wr.writerow([cg[g], int(b0[j]), int(b1[j]), repr(float(lo[g, b0[j]])), repr(float(hi[g, b0[j]])),
                             repr(float(lo[g, b1[j]])), repr(float(hi[g, b1[j]])), repr(float(js[g, j]))])
    c0, bits0 = load_cube(n0); c1, bits1 = load_cube(n1)
    assert bits0 == bits == bits1
    eff = {}
    with open(out / 'effect_witnesses.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        names = None
        with np.load(n0, allow_pickle=False) as z:
            names = [str(x) for x in z['replica_names']]
        wr.writerow(['time', 'reader', 'x0', 'x1', 'margin_min_over_replicas'] + [f'diff_{g}' for g in names])
        for label, cubes in (('t', c0), ('t+1', c1)):
            tol, e0, e1, diff, hit, margin = effect_extract(cubes, i)
            eff[label] = {'delta_abs_per_reader': dict(zip(bits, tol.tolist())),
                          'witness_count_per_reader': {bits[k]: int(hit[:, k].sum()) for k in range(12)},
                          'acted_on_readers': [bits[k] for k in range(12) if hit[:, k].any()],
                          'smallest_margin_per_reader': {bits[k]: float(margin[hit[:, k], k].min()) for k in range(12) if hit[:, k].any()}}
            for k in range(12):
                for j in np.flatnonzero(hit[:, k]):
                    wr.writerow([label, bits[k], int(e0[j]), int(e1[j]), repr(float(margin[j, k]))] + [repr(float(v)) for v in diff[:, j, k]])
    summary = {'world': w, 'role': a.role, 'mask_bit': i, 'L24_site_index_in_sampled_npz': s,
               'sources_sha256': {n26.name: sha(n26), n0.name: sha(n0), n1.name: sha(n1)},
               'return': {'definition': 'A = {role write at t+1 within delta_t1 of the write reached through the reference token}; '
                                        'P(A|x) in [low, high], high - low = mass outside V; js_min = minimal binary JS over the two intervals; '
                                        'B_r(replica) = mean of js_min over the 2048 pairs (x0 with the role bit 0, x1 = x0 with the bit set)',
                          'reference_token': int(V[rho]), 'ball_tokens': [V[t] for t in np.flatnonzero(ball)],
                          'delta_t1': float(d_t1[s]), 'B_r_per_replica': dict(zip(cg, B.tolist())), 'B_r_min': float(B.min()),
                          'R_state_lower_bound': float(B.min() / 11.0), 'mass_outside_V_max': float(tail.max())},
               'effect_signature': {'definition': 'witness of reader k: |Y_{g,k}(x1) - Y_{g,k}(x0)| > delta_k for EVERY replica g; '
                                                  'delta_k = max over the 9 recodings and the 4096 masks of |Y_g,k - Y_identity,k|, per time',
                                    **eff, 'identical_acted_on_sets': eff['t']['acted_on_readers'] == eff['t+1']['acted_on_readers']},
               'files_sha256': {'return_pairs.csv': sha(out / 'return_pairs.csv'), 'effect_witnesses.csv': sha(out / 'effect_witnesses.csv')}}
    with open(out / 'summary.json', 'x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[witness-extract] {a.role}: B_r min over replicas = {B.min():.6e} bit (reference token {V[rho]}) ; "
          f"acted-on readers t {len(eff['t']['acted_on_readers'])}, t+1 {len(eff['t+1']['acted_on_readers'])} ; "
          f"identical: {summary['effect_signature']['identical_acted_on_sets']}", flush=True)
    zp = out.with_suffix('.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f'{out.name}/{f.name}')
    print('[witness-extract] written', out, 'and', zp, flush=True)

if __name__ == '__main__':
    main()
