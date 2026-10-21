#!/usr/bin/env python3
"""Adapter (CPU): converts the sampled-return acquisition (r_sampled_return_gpu.py) into the T330 export schema (S26_U26_FULLSOFTMAX_EXPORT.npz).

Inputs (FAST_GPU_RESULTS/CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU/):
  RESULT.json, S26_R_SAMPLED_RAW_u{w}.npz (SHA checked), S26_T1_<gauge>.npz for the 10 replicas.
Output (FAST_GPU_RESULTS/S26_U26_FULLSOFTMAX/):
  S26_U26_FULLSOFTMAX_EXPORT.npz with keys world, fixed_v_before_analysis, gauge_names, token_ids, logp_v, tail_mass,
  return_scores, delta_t1 ; EXPORT_PROVENANCE.json (definitions and SHA-256 of every source file).
Definitions (declared here, before any analysis of the donor channel):
  role r (r = 0..10) = mask bit r ; bit 11 (heads[24-27]) is the context ;
  logp_v[g, x, v] = natural log of the full-softmax probability of token V[v] under replica g and donor mask x (float64) ;
  tail_mass[g, x] = probability mass outside V, summed directly over the vocabulary (float64) ;
  return_scores[g, r, v] = || W_g,r(V[v]) - W_g,r(c_g) ||_2 / || W_g,r(c_g) ||_2, where W_g,r(tau) is the residual-space
    write of the L24 site of role r at t+1 when tau is emitted (clean prefix, replica g) and c_g is the clean next token
    of replica g: a fixed scalar readout of the same role at t+1, independent of the donor probabilities ;
  delta_t1[g, r] = max over the 9 recodings g' and over V of |return_scores[g', r, v] - return_scores[identity, r, v]|,
    the same value for all 10 rows.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u export_t330_schema_cpu.py --world 26
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S26DIR = 'FAST_GPU_RESULTS/CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU'
S26NPZ = 'S26_R_SAMPLED_RAW_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/S26_U26_FULLSOFTMAX'
OUTNPZ = 'S26_U26_FULLSOFTMAX_EXPORT.npz'
GAUGES = ('IDENTITY_0', 'UPDOWN_SCALE_0', 'UPDOWN_SCALE_1', 'UPDOWN_SCALE_2', 'VO_DIAG_0', 'VO_DIAG_1', 'VO_DIAG_2',
          'QKNORM_RECIP_0', 'QKNORM_RECIP_1', 'QKNORM_RECIP_2')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def return_scores(W1, sites, bits, V, clean):
    """W1[g]: (|V|, 12, D) writes at t+1 ; returns (G, 11, |V|) relative distances to the clean next-token write."""
    G = len(W1); out = np.zeros((G, 11, len(V)))
    for g in range(G):
        j0 = V.index(clean[g])
        for r in range(11):
            s = sites.index(bits[r])
            ref = W1[g][j0, s].astype(np.float64)
            out[g, r] = np.linalg.norm(W1[g][:, s].astype(np.float64) - ref[None], axis=1) / np.linalg.norm(ref)
    return out

def build(root, w):
    d = root / S26DIR.format(w=w); n = d / S26NPZ.format(w=w)
    if not (d / 'RESULT.json').is_file() or not n.is_file():
        raise SystemExit(f'Sampled acquisition missing: {d} ; restore it from the backup or run it first')
    res = json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))
    if sha(n) != res['raw_sha256']:
        raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    if res['status'] not in ('PASS_R_SAMPLED_RETURN_ACQUISITION', 'PASS_S26_R_SAMPLED_RAW_ACQUISITION'):
        raise SystemExit('Sampled acquisition not passed (a PASS status implies a valid lemma check): ' + res['status'])
    with np.load(n, allow_pickle=False) as z:
        cg = tuple(str(x) for x in z['channel_gauges']); bits = [str(x) for x in z['mask_bits']]
        sites = [str(x) for x in z['sites']]; V = [int(t) for t in z['V']]
        lp = z['lp_V'].astype(np.float64); tail = z['mass_outside_V'].astype(np.float64)
        W1id = z['W1_L24_t1_identity']
    if cg != GAUGES:
        raise SystemExit(f'The token channel must cover the 10 replicas in the contract order; found {cg}')
    W1 = []; clean = []; src = {n.name: sha(n)}
    for g in GAUGES:
        f = d / f'S26_T1_{g}.npz'
        if not f.is_file():
            raise SystemExit(f'Per-replica file missing: {f} ; restore it from the backup')
        with np.load(f, allow_pickle=False) as z:
            if [int(t) for t in z['V']] != V:
                raise SystemExit(f'Token set of {f.name} differs from V')
            W1.append(z['W1'].astype(np.float32)); clean.append(int(z['clean']))
        src[f.name] = sha(f)
    if not np.array_equal(W1[0], W1id):
        raise SystemExit('Identity t+1 writes differ between the per-replica file and the final NPZ')
    err = float(np.abs(np.exp(lp).sum(2) + tail - 1).max())
    if err > 2e-9:
        raise SystemExit(f'Full-softmax normalization error {err:.3e} > 2e-9')
    sc = return_scores(W1, sites, bits, V, clean)
    delta = np.abs(sc[1:] - sc[0][None]).max((0, 2))
    return dict(world=np.array(w), fixed_v_before_analysis=np.array(True), gauge_names=np.array(GAUGES),
                token_ids=np.array(V, np.int64), logp_v=lp, tail_mass=tail, return_scores=sc,
                delta_t1=np.tile(delta, (10, 1))), src, bits, clean, err

def selftest():
    rng = np.random.default_rng(0)
    V = [5, 19, 21]; sites = [f's{i}' for i in range(12)]; bits = sites[::-1]
    W = [rng.standard_normal((3, 12, 16)).astype(np.float32) for _ in range(10)]
    sc = return_scores(W, sites, bits, V, [19] * 10)
    assert sc.shape == (10, 11, 3) and np.all(sc[:, :, 1] == 0)
    s = sites.index(bits[4])
    assert abs(sc[3, 4, 2] - np.linalg.norm(W[3][2, s] - W[3][1, s]) / np.linalg.norm(W[3][1, s])) < 1e-6
    print('[export-T330] selftest OK (role -> site mapping, relative distance to the clean next-token write)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    arrs, src, bits, clean, err = build(a.root, a.world)
    out = a.root / OUTDIR; out.mkdir(parents=True, exist_ok=True)
    f = out / OUTNPZ
    if f.exists():
        raise SystemExit('Export already exists: move it to rebuild: ' + str(f))
    with open(f, 'xb') as fh:
        np.savez_compressed(fh, **arrs)
    prov = {'export_sha256': sha(f), 'source_files_sha256': src, 'roles_mask_bits_0_10': bits[:11], 'context_bit_11': bits[11],
            'clean_next_token_per_gauge': dict(zip(GAUGES, clean)), 'max_normalization_error': err,
            'return_scores': 'relative L2 distance of the role-r L24 write at t+1 to the clean next-token write, per replica',
            'delta_t1': 'max over the 9 recodings and over V of the deviation of return_scores from identity; same for all rows',
            'token_pair_js_bits': 'not exported', 'tail_signed_posneg': 'not exported'}
    (out / 'EXPORT_PROVENANCE.json').write_text(json.dumps(prov, indent=2), encoding='utf-8')
    print(f"[export-T330] written {f} ; |V| = {arrs['token_ids'].size} ; max tail {arrs['tail_mass'].max():.3e} ; "
          f"delta_t1 = {np.round(arrs['delta_t1'][0], 4).tolist()}", flush=True)

if __name__ == '__main__':
    main()
