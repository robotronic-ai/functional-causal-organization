#!/usr/bin/env python3
"""R, greedy token channel (CPU) on the C_L24 contract (u26), from the greedy return acquisition (no GPU).

Reads S24_R_RAW_u{w}.npz (SHA checked against its RESULT.json) and the gauge cube (mask bit names, roles retained by
the frozen C_L24 rule, context = absorbed sites).
Computes, under q_donor uniform over the 4096 masks:
  1. greedy (argmax) channel: I(P ; token | context), number of distinct tokens, B_greedy ;
  2. margins: log p(token 1) - log p(token 2) per mask (min, quantiles, mask closest to a flip) ;
  3. sensitivity per mask bit: mean and maximal change of the margin when the bit is flipped.
The exact full-vocabulary sampled channel is computed by r_sampled_return_gpu.py.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u r_token_channel_cpu.py --world 26
Writes FAST_GPU_RESULTS/CLAUDE_S25_R_TOKEN_CHANNEL_WORLD{w}_CPU/RESULT.json.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S24DIR = 'FAST_GPU_RESULTS/CLAUDE_S24_R_TOKEN_RETURN_WORLD{w}_GPU'
S24NPZ = 'S24_R_RAW_u{w}.npz'
S9DIR = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
S9NPZ = 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_S25_R_TOKEN_CHANNEL_WORLD{w}_CPU'
EPS = 1e-15

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def retained_sites(cubes, tol):
    """Frozen C_L24 rule: exists x_{-i}, forall g, exists r : |Delta_i Y_{g,r}(x)| > delta_r."""
    keep = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS).any(2)
        keep.append(bool(hit.all(0).any()))
    return keep

def H(p, axis=-1):
    p = np.clip(p, 0, None)
    return -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(axis)

def cond_mi(P, ctx_of):
    """I(X ; T | C) in bits, X uniform ; P[x, t] = p(t | x) ; ctx_of[x] = context index."""
    tot = 0.0; ctxs = np.unique(ctx_of)
    for c in ctxs:
        Pc = P[ctx_of == c]
        tot += (H(Pc.mean(0)) - H(Pc).mean()) / len(ctxs)
    return float(max(tot, 0.0))

def selftest():
    ctx = np.arange(4096) >> 11
    flat = np.tile(np.array([0.7, 0.2, 0.1]), (4096, 1))
    assert cond_mi(flat, ctx) < 1e-12
    dep = flat.copy(); dep[np.arange(4096) & 1 == 1] = [0.2, 0.7, 0.1]
    assert cond_mi(dep, ctx) > 0.1
    one = np.zeros((4096, 2)); one[np.arange(4096) & 4 > 0, 1] = 1; one[:, 0] = 1 - one[:, 1]
    assert abs(cond_mi(one, ctx) - 1.0) < 1e-12
    print('[r-channel] selftest OK (conditional mutual information)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    w = a.world
    d24 = a.root / S24DIR.format(w=w); n24 = d24 / S24NPZ.format(w=w)
    d9 = a.root / S9DIR.format(w=w); n9 = d9 / S9NPZ.format(w=w)
    for d, n in ((d24, n24), (d9, n9)):
        if not (d / 'RESULT.json').is_file() or not n.is_file():
            raise SystemExit(f'File missing: {d / "RESULT.json"} or {n} ; restore it from the backup')
        if sha(n) != json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))['raw_sha256']:
            raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    with np.load(n9, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits = [str(x) for x in z['mask_bits']]
    cubes = cubes[:, :, [rn.index(b) for b in bits]]
    tol = np.abs(cubes[1:] - cubes[0][None]).max((0, 1))
    keep = retained_sites(cubes, tol)
    S = [i for i in range(12) if keep[i]]; C = [i for i in range(12) if not keep[i]]
    with np.load(n24, allow_pickle=False) as z:
        tokens = z['tokens'].astype(np.int64); ids = z['top_ids'].astype(np.int64); lp = z['top_logprobs'].astype(np.float64)
        m12 = z['mask12'].astype(np.int64); clean = int(z['clean_token'])
    assert np.array_equal(m12, np.arange(4096)) and ids.shape == (4096, 20)
    assert np.array_equal(ids[:, 0], tokens)
    ctx_of = np.zeros(4096, np.int64)
    for k, i in enumerate(C):
        ctx_of |= ((np.arange(4096) >> i) & 1) << k
    print(f'[r-channel] u{w} roles ({len(S)}): {[bits[i] for i in S]} ; context: {[bits[i] for i in C]}', flush=True)

    utok = np.unique(tokens)
    greedy = np.zeros((4096, len(utok)))
    greedy[np.arange(4096), np.searchsorted(utok, tokens)] = 1.0
    I_greedy = cond_mi(greedy, ctx_of)
    margin = lp[:, 0] - lp[:, 1]
    xm = int(np.argmin(margin))
    sens = {}
    for i in range(12):
        b = np.array([m for m in range(4096) if not (m >> i) & 1])
        dm = np.abs(margin[b] - margin[b | (1 << i)])
        sens[bits[i]] = {'role': i in S, 'margin_change_mean_nats': float(dm.mean()), 'margin_change_max_nats': float(dm.max())}
    second, cnt2 = np.unique(ids[:, 1], return_counts=True)
    res = {'world': w, 'greedy_npz_sha256': sha(n24), 'gauge_cube_npz_sha256': sha(n9),
           'roles': [bits[i] for i in S], 'context': [bits[i] for i in C], 'q_law': 'q_donor uniform over the 4096 masks',
           'greedy': {'distinct_tokens': sorted(set(tokens.tolist())), 'clean_token': clean,
                      'I_P_token_given_context_bits': I_greedy, 'B_greedy_zero': bool(I_greedy == 0.0 and len(utok) == 1),
                      'census': 'exhaustive over the 4096 masks'},
           'margins_nats': {'min': float(margin.min()), 'q01': float(np.quantile(margin, 0.01)), 'median': float(np.median(margin)),
                            'max': float(margin.max()), 'closest_mask': xm,
                            'closest_mask_bits': [bits[i] for i in range(12) if (xm >> i) & 1],
                            'p_token1_min': float(np.exp(lp[:, 0]).min()), 'p_token2_max': float(np.exp(lp[:, 1]).max())},
           'second_token': {str(int(t)): int(c) for t, c in zip(second, cnt2)},
           'sensitivity_per_mask_bit': sens,
           'R_computed': False}
    out = a.root / OUTDIR.format(w=w)
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"[r-channel] greedy: tokens {res['greedy']['distinct_tokens']} ; I(P;token|C) = {I_greedy:.3g} bit", flush=True)
    print(f"[r-channel] margin log p1 - log p2: min {margin.min():.3f} nats, median {np.median(margin):.3f} ; p(token 1) min "
          f"{np.exp(lp[:, 0]).min():.6f} ; second token: {res['second_token']}", flush=True)
    for k, v in sens.items():
        print(f"[r-channel]   {k:20s} role={v['role']!s:5s} margin change mean {v['margin_change_mean_nats']:.3g} "
              f"max {v['margin_change_max_nats']:.3g} nats", flush=True)
    print('[r-channel] RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
