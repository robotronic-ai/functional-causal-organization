#!/usr/bin/env python3
"""Raw witnesses of the Lambda_O lower relation on C_L24 (CPU), from the gauge cubes at t and t+1.

For every RESOLVED_POSITIVE cell (content z_i, reader k) of lambda_relation_cpu.py (rule: exists x, for all 10 replicas g,
|Y_{g,k}(x^i) - Y_{g,k}(x)| > factor * delta_k ; factor 1 = the 'more than delta' rule of C_L24, primary ; factor 2 =
stricter secondary reading), one witness row: the configuration x (bit i = 0) and x^i (bit i = 1), the 10 raw
scores Y_{g,k}(x) and Y_{g,k}(x^i), the 10 differences, delta_k and the margin min_g |difference| - factor * delta_k. The witness
kept is the one with the largest margin ; the number of witnesses is also given. delta_k is reported with the replica and
mask where it is attained, so it can be checked from the cube alone.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u extract_lambda_witnesses_cpu.py --world 26 [--factor 2]
Writes FAST_GPU_RESULTS/LAMBDA_WITNESS_EXTRACT_C_L24_u{w}[_2delta]/ (lambda_witnesses.csv, delta_per_reader.csv, summary.json) and
a zip of that folder.
"""
import argparse, csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
CUBES = {'t': ('FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU', 'S9_FULL12_GAUGE_CUBES_u{w}.npz'),
         't+1': ('FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU', 'S28_T1_GAUGE_CUBES_u{w}.npz')}
OUTDIR = 'FAST_GPU_RESULTS/LAMBDA_WITNESS_EXTRACT_C_L24_u{w}'
EPS = 1e-15

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load(root, spec, w):
    d = root / spec[0].format(w=w); n = d / spec[1].format(w=w)
    if not (d / 'RESULT.json').is_file() or not n.is_file():
        raise SystemExit(f'File missing: {d / "RESULT.json"} or {n} ; restore it from the backup')
    h = sha(n)
    if h != json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))['raw_sha256']:
        raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    with np.load(n, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits = [str(x) for x in z['mask_bits']]
        gauges = [str(x) for x in z['gauges']] if 'gauges' in z.files else [f'replica{g}' for g in range(cubes.shape[0])]
    assert cubes.shape == (10, 4096, 12)
    return cubes[:, :, [rn.index(b) for b in bits]], bits, gauges, h

def deltas(cubes):
    dev = np.abs(cubes[1:] - cubes[0][None])
    tol = dev.max((0, 1)); where = []
    for k in range(cubes.shape[2]):
        g, m = np.unravel_index(int(np.argmax(dev[:, :, k])), dev.shape[:2])
        where.append((int(g) + 1, int(m)))
    return tol, where

def retained(cubes, tol):
    keep = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        hit = (np.abs(cubes[:, base | (1 << i)] - cubes[:, base]) > tol[None, None, :] + EPS).any(2)
        keep.append(bool(hit.all(0).any()))
    return keep

def witnesses(cubes, tol, contents, factor=1.0):
    rows = []
    for i in contents:
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        d = cubes[:, base | (1 << i)] - cubes[:, base]
        margin = np.abs(d).min(0) - factor * tol[None, :]
        ok = (np.abs(d) > factor * tol[None, None, :] + EPS).all(0)
        for k in range(12):
            if not ok[:, k].any():
                continue
            j = int(np.flatnonzero(ok[:, k])[np.argmax(margin[ok[:, k], k])])
            x0 = int(base[j]); x1 = x0 | (1 << i)
            rows.append(dict(content=i, reader=k, n_witnesses=int(ok[:, k].sum()), x=x0, x_i=x1,
                             y_x=cubes[:, x0, k].tolist(), y_xi=cubes[:, x1, k].tolist(), diff=d[:, j, k].tolist(),
                             delta=float(tol[k]), margin=float(margin[j, k])))
    return rows

def selftest():
    rng = np.random.default_rng(8)
    X = np.array([[(m >> i) & 1 for i in range(12)] for m in range(4096)], float)
    W = np.zeros((12, 12)); W[0, 3] = 1; W[1, 3] = 1; W[1, 4] = 1; W[2, 5] = 1
    c = np.stack([X @ W + rng.normal(0, 0.002, (4096, 12)) * (g > 0) for g in range(10)])
    tol, where = deltas(c)
    rows = witnesses(c, tol, [0, 1, 2])
    assert sorted((r['content'], r['reader']) for r in rows) == [(0, 3), (1, 3), (1, 4), (2, 5)]
    for r in rows:
        dd = np.array(r['y_xi']) - np.array(r['y_x'])
        assert np.allclose(dd, r['diff']) and r['margin'] > 0 and r['n_witnesses'] == 2048
        assert abs(np.abs(dd).min() - r['delta'] - r['margin']) < 1e-12
    g, m = where[3]
    assert abs(abs(c[g, m, 3] - c[0, m, 3]) - tol[3]) < 1e-15
    print('[lambda-witness] selftest OK (cells, raw scores, differences, margins, delta location)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--factor', type=float, default=1.0, choices=(1.0, 2.0), help='1: more than delta (primary) ; 2: more than 2 delta')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    out = a.root / (OUTDIR.format(w=a.world) + ('' if a.factor == 1.0 else '_2delta'))
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    data = {lab: load(a.root, spec, a.world) for lab, spec in CUBES.items()}
    bits = data['t'][1]; assert data['t+1'][1] == bits
    tol0, _ = deltas(data['t'][0]); keep = retained(data['t'][0], tol0); S = [i for i in range(12) if keep[i]]
    out.mkdir(parents=True)
    summary = {'world': a.world, 'factor': a.factor, 'rule': f'exists x, for all 10 replicas, |Y(x^i) - Y(x)| > {a.factor:g} delta_k',
               'contents': [bits[i] for i in S], 'readers': bits, 'source_sha256': {}, 'cells': {}, 'delta': {}}
    with open(out / 'lambda_witnesses.csv', 'w', newline='', encoding='utf-8') as fw, \
         open(out / 'delta_per_reader.csv', 'w', newline='', encoding='utf-8') as fd:
        ww = csv.writer(fw); wd = csv.writer(fd)
        gz = data['t'][2]
        ww.writerow(['time', 'content', 'reader', 'n_witnesses', 'x_mask', 'x_i_mask', 'delta_k', 'margin_over_threshold']
                    + [f'y_x_{g}' for g in gz] + [f'y_xi_{g}' for g in gz] + [f'diff_{g}' for g in gz])
        wd.writerow(['time', 'reader', 'delta_k', 'attained_replica', 'attained_mask'])
        for lab, (cubes, _, gauges, h) in data.items():
            summary['source_sha256'][lab] = h
            tol, where = deltas(cubes)
            for k in range(12):
                wd.writerow([lab, bits[k], repr(float(tol[k])), gauges[where[k][0]], where[k][1]])
            summary['delta'][lab] = {bits[k]: float(tol[k]) for k in range(12)}
            rows = witnesses(cubes, tol, S, a.factor)
            summary['cells'][lab] = {'resolved_positive': len(rows),
                                     'min_margin_over_threshold': min(r['margin'] for r in rows) if rows else None,
                                     'relation': {bits[i]: [bits[r['reader']] for r in rows if r['content'] == i] for i in S}}
            for r in rows:
                ww.writerow([lab, bits[r['content']], bits[r['reader']], r['n_witnesses'], r['x'], r['x_i'], repr(r['delta']),
                             repr(r['margin'])] + [repr(v) for v in r['y_x'] + r['y_xi'] + r['diff']])
    summary['csv_sha256'] = {n: sha(out / n) for n in ('lambda_witnesses.csv', 'delta_per_reader.csv')}
    with open(out / 'summary.json', 'x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for n in ('lambda_witnesses.csv', 'delta_per_reader.csv', 'summary.json'):
            z.write(out / n, f'{out.name}/{n}')
    for lab in data:
        c = summary['cells'][lab]
        print(f"[lambda-witness] {lab}: {c['resolved_positive']} cells, min margin over threshold = {c['min_margin_over_threshold']}", flush=True)
    print('[lambda-witness] wrote', out, 'and', out.with_suffix('.zip'))

if __name__ == '__main__':
    main()
