#!/usr/bin/env python3
"""Replay of the restricted Lambda0 lower relation (L1 anchor runs at L23 and L24) from the per-replica raw role writes (CPU).

Inputs (from quantum/): NUIT_T127/T135/ck_L1_L23/ and ck_L1_L24/ (10 replica files rep{r}_{KIND}_{s}.pt and meta.json), and
the recorded outputs NUIT_T127/T135/V_T135_L1_L23.json and V_T135_L1_L24.json.
Each replica file holds w[(t, content)][role] (the write of the role at layer a, context t, residual replaced by the content
vector) and xin[content] (the content vector). Definitions (ROLELAMBDA_v1):
  rel(a, b)   = rms(a - b) / (0.5 (rms(a) + rms(b)))
  delta_w     = max over every state (t, content, role) and every pair of replicas of rel (orbit diameter of the writes)
  delta_x     = max over every content and every pair of replicas of rel (orbit diameter of the content vectors)
  e_g(z, r)   = max_t rel(w_g[(t, z)][r], w_g[(t, REF)][r])
  cell (z, r) RESOLVED_POSITIVE if e_g(z, r) > 2 delta_w in every replica g ; otherwise UNRESOLVED (never negative)
  contents z, z' resolved distinct if rel(xin_g[z], xin_g[z']) > 2 delta_x in every replica g
Claim checked: the lower relation is non-empty at each layer with all content pairs resolved, so every admissible completion
of the unresolved cells is non-empty and Lambda_O is not identically zero on this restricted domain.
The rel formula, the thresholds and the verdicts are recomputed here with numpy only; the recorded JSON is used only for the
final comparison. Files are loaded with torch.load(weights_only=True).
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u lambda0_l23_replay_cpu.py
Writes FAST_GPU_RESULTS/LAMBDA0_L23_REPLAY_CPU/RESULT.json.
"""
import argparse, hashlib, itertools, json, re
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
RUNS = (('L23', 'NUIT_T127/T135/ck_L1_L23', 'NUIT_T127/T135/V_T135_L1_L23.json'),
        ('L24', 'NUIT_T127/T135/ck_L1_L24', 'NUIT_T127/T135/V_T135_L1_L24.json'))
OUTDIR = 'FAST_GPU_RESULTS/LAMBDA0_L23_REPLAY_CPU'
REF = '__REF__'
TOL = 1e-9

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / (0.5 * (np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(b ** 2))) + 1e-30))

def load_states(d):
    import torch
    files = sorted(d.glob('rep*.pt'), key=lambda p: int(re.match(r'rep(\d+)_', p.name).group(1)))
    if len(files) != 10:
        raise SystemExit(f'{d}: expected 10 replica files, found {len(files)} ; restore them from the backup')
    states = []
    for f in files:
        s = torch.load(f, weights_only=True, map_location='cpu')
        states.append(dict(w={k: {r: v.double().numpy() for r, v in dd.items()} for k, dd in s['w'].items()},
                           xin={k: v.double().numpy() for k, v in s['xin'].items()}))
    return states, {f.name: sha(f) for f in files}

def recompute(states, contents, roles):
    R = len(states); names = [REF] + contents
    keys = [(t, z, r) for (t, z), dd in states[0]['w'].items() for r in dd]
    dw = max(rel(states[g]['w'][(t, z)][r], states[q]['w'][(t, z)][r]) for (t, z, r) in keys for g, q in itertools.combinations(range(R), 2))
    dx = max(rel(states[g]['xin'][z], states[q]['xin'][z]) for z in names for g, q in itertools.combinations(range(R), 2))
    dw = float(np.nextafter(dw, np.inf)); dx = float(np.nextafter(dx, np.inf))
    T = max(t for t, _ in states[0]['w']) + 1
    cells = {}
    for z in contents:
        for r in roles:
            es = [max(rel(s['w'][(t, z)][r], s['w'][(t, REF)][r]) for t in range(T)) for s in states]
            pos = all(e > 2 * dw for e in es)
            cells[f'{z}->{r}'] = dict(status='RESOLVED_POSITIVE' if pos else 'UNRESOLVED',
                                      effect_over_delta=[min(es) / dw, max(es) / dw],
                                      min_margin_over_2delta=min(es) / dw - 2.0)
    pairs = {f'{a}|{b}': ('RESOLVED_DISTINCT' if all(rel(s['xin'][a], s['xin'][b]) > 2 * dx for s in states)
                          else 'CONTENT_PAIR_UNRESOLVED') for a, b in itertools.combinations(names, 2)}
    lower = {z: [r for r in roles if cells[f'{z}->{r}']['status'] == 'RESOLVED_POSITIVE'] for z in contents}
    return dict(delta_w=dw, delta_x=dx, contexts=T, cells=cells, content_pairs=pairs, lower_relation=lower)

def compare(mine, rec):
    diffs = []
    for k in ('delta_w', 'delta_x'):
        if abs(mine[k] - rec[k]) > TOL * max(1.0, abs(rec[k])):
            diffs.append(f'{k}: {mine[k]} vs {rec[k]}')
    for c, v in rec['cells'].items():
        m = mine['cells'][c]
        if m['status'] != v['status']:
            diffs.append(f'{c}: status {m["status"]} vs {v["status"]}')
        if any(abs(x - y) > TOL * max(1.0, abs(y)) for x, y in zip(m['effect_over_delta'], v['effect_over_delta'])):
            diffs.append(f'{c}: band {m["effect_over_delta"]} vs {v["effect_over_delta"]}')
    if mine['content_pairs'] != rec['content_pairs']:
        diffs.append('content_pairs differ')
    if mine['lower_relation'] != rec['lower_relation']:
        diffs.append('lower_relation differs')
    return diffs

def selftest():
    rng = np.random.default_rng(3)
    base = {('w', t, z, r): rng.normal(size=64) for t in range(2) for z in (REF, 'a', 'b') for r in ('r1', 'r2')}
    xb = {z: rng.normal(size=32) for z in (REF, 'a', 'b')}
    states = []
    for g in range(10):
        n = (lambda: rng.normal(0, 1e-3, 64)) if g else (lambda: 0.0)
        w = {(t, z): {r: base[('w', t, z, r)] + n() for r in ('r1', 'r2')} for t in range(2) for z in (REF, 'a', 'b')}
        w[(0, 'a')]['r1'] = w[(0, REF)]['r1'] * 1.5
        for t in range(2):
            w[(t, 'b')]['r1'] = w[(t, REF)]['r1'] + 1e-4; w[(t, 'b')]['r2'] = w[(t, REF)]['r2'] + 1e-4
            w[(t, 'a')]['r2'] = w[(t, REF)]['r2'] + 1e-4
        states.append(dict(w=w, xin={z: xb[z] + rng.normal(0, 1e-3, 32) for z in xb}))
    out = recompute(states, ['a', 'b'], ['r1', 'r2'])
    assert out['lower_relation'] == {'a': ['r1'], 'b': []}, out['lower_relation']
    assert all(v == 'RESOLVED_DISTINCT' for v in out['content_pairs'].values())
    print('[replay] selftest OK (rel, orbit diameters, 2 delta rule, no negative status)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    res = {'definition': __doc__.split('Definitions (ROLELAMBDA_v1):')[1].split('Usage')[0].strip(), 'layers': {}}
    ok = True
    for lab, ck, js in RUNS:
        d = a.root / ck; jf = a.root / js
        if not d.is_dir() or not jf.is_file():
            raise SystemExit(f'File missing: {d} or {jf} ; restore it from the backup')
        rec_all = json.loads(jf.read_text(encoding='utf-8')); rec = rec_all['result']
        meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        states, shas = load_states(d)
        mine = recompute(states, rec['contents'], rec['roles'])
        diffs = compare(mine, rec)
        if meta['signature'] != rec['signature']:
            diffs.append(f'checkpoint signature {meta["signature"]} differs from recorded {rec["signature"]}')
        non_empty = any(mine['lower_relation'].values())
        all_pairs = all(v == 'RESOLVED_DISTINCT' for v in mine['content_pairs'].values())
        res['layers'][lab] = dict(recorded_json=js, recorded_json_sha256=sha(jf), checkpoint_signature=meta['signature'],
                                  replica_sha256=shas, roles=rec['roles'], contents=rec['contents'], **mine,
                                  lower_relation_non_empty=non_empty, all_content_pairs_resolved=all_pairs,
                                  matches_recorded=not diffs, differences=diffs)
        ok &= (not diffs)
        print(f'[replay] {lab}: delta_w={mine["delta_w"]:.6g} lower={mine["lower_relation"]} match={not diffs}', flush=True)
        for c, v in mine['cells'].items():
            print(f'[replay]   {c:45s} {v["status"]:18s} band={v["effect_over_delta"][0]:.4f}..{v["effect_over_delta"][1]:.4f}', flush=True)
    L = res['layers']
    nz = all(L[k]['lower_relation_non_empty'] and L[k]['all_content_pairs_resolved'] for k in L)
    res['status'] = ('LAMBDA0_RESTRICTED_LOWER_RELATION_NONEMPTY_REPLAYED_FROM_RAW_WRITES' if ok and nz else
                     'REPLAY_MISMATCH' if not ok else 'LOWER_RELATION_EMPTY_NO_CLAIM')
    out = a.root / OUTDIR
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('[replay] STATUS', res['status'], '; RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
