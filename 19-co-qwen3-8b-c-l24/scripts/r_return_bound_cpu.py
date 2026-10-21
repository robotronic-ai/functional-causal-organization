#!/usr/bin/env python3
"""R (CPU) on the C_L24-R contract (u26), from the sampled return acquisition (no GPU).

Contract (declared before any computation):
  transition T ~ Categorical(softmax(logits)), temperature 1, full vocabulary ;
  q_donor: 12 independent Bernoulli(0.5) bits ; roles = sites retained by the frozen C_L24 rule, context = absorbed sites ;
  lemma: L24(t+1) = f(token) (arm cache + tau == clean cache + tau, checked by the acquisition) ;
  state of role r at t+1: L24 write of site r ; resolution delta_t1[r] = maximal deviation of the 9 gauges from identity ;
  return event A_{r,rho} = { write of role r at t+1 within delta_t1[r] of the write of the reference token rho }, rho in V,
    a deterministic function of the state of role r at t+1 ;
  B_r = I(P_r ; A_{r,rho} | P_{-r}, C) = mean over the 2048 x_{-r} of JS_2(P(A | x), P(A | x^r)) ;
    the mass outside V (membership in A unknown) is allocated in the worst case: P(A | x) lies in an interval whose width
    is the mass outside V, and the minimal JS over the two intervals is taken (exact lower bound, JS being convex) ;
    B_r >= max over rho of this bound, minimum over the replicas ;
  witness (rule exists x, forall g): pair (x, x^r) whose two intervals are separated by more than delta_A(r, rho) under
    every replica, delta_A = maximal deviation of the gauges from identity on P(A | x) ;
  K_r = I(P_r ; T | P_{-r}, C): bracketed by the fixed grouping (V + one class outside V) and the mass outside V ;
  R_tok_r >= B_r / K_r_sup ; R_state_r >= B_r / 11 bits (H(P | C) = 11 bits bounds any state information about P).
Usage (from quantum/, after r_sampled_return_gpu.py):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u r_return_bound_cpu.py --world 26
Writes FAST_GPU_RESULTS/CLAUDE_S27_R_RETURN_BOUND_WORLD{w}_CPU/RESULT.json.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
S26DIR = 'FAST_GPU_RESULTS/CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU'
S26NPZ = 'S26_R_SAMPLED_RAW_u{w}.npz'
S9DIR = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
S9NPZ = 'S9_FULL12_GAUGE_CUBES_u{w}.npz'
OUTDIR = 'FAST_GPU_RESULTS/CLAUDE_S27_R_RETURN_BOUND_WORLD{w}_CPU'
EPS = 1e-15
H_P_GIVEN_C = 11.0
NUM_FLOOR_BITS = 1e-12
PASS_SAMPLED = ('PASS_R_SAMPLED_RETURN_ACQUISITION', 'PASS_S26_R_SAMPLED_RAW_ACQUISITION')  # a PASS status implies a valid lemma check

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

def hbin(p):
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, -p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
        b = np.where(p < 1, -(1 - p) * np.log1p(-np.where(p < 1, p, 0.0)) / np.log(2), 0.0)
    return a + b

def js_bin(p, q):
    return np.maximum(hbin(0.5 * (p + q)) - 0.5 * (hbin(p) + hbin(q)), 0.0)

def js_min_box(l1, u1, l2, u2):
    """min JS_2(p, q) for p in [l1, u1], q in [l2, u2] (JS convex, zero on the diagonal)."""
    lo = np.where(u1 < l2, u1, np.where(u2 < l1, u2, 0.0))
    hi = np.where(u1 < l2, l2, np.where(u2 < l1, l1, 0.0))
    gap = np.where(u1 < l2, l2 - u1, np.where(u2 < l1, l1 - u2, 0.0))
    return np.where(gap > 0, js_bin(lo, hi), 0.0), gap

def js_cat(P, Q):
    M = 0.5 * (P + Q)
    def kl(A):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(A > 0, A * np.log2(np.where(A > 0, A, 1.0) / np.where(M > 0, M, 1.0)), 0.0).sum(-1)
    return np.maximum(0.5 * (kl(P) + kl(Q)), 0.0)

def pairs_for_bit(i):
    b = np.array([m for m in range(4096) if not (m >> i) & 1])
    return b, b | (1 << i)

def event_interval(pV, tail, ball):
    """P(A|x) in [s, s + tail]; works on the smaller-probability side (JS is invariant under complement)."""
    s_in = pV[..., ball].sum(-1); s_out = pV[..., ~ball].sum(-1)
    small = s_out if s_in.mean() > 0.5 else s_in
    return small, small + tail

def role_bound(lpV, tail, ball, i):
    """For one role (bit i) and one ball: lower bound of B per replica, gauge deviation delta_A, witnesses exists x forall g."""
    pV = np.exp(lpV)
    lo, hi = event_interval(pV, tail, ball)
    b0, b1 = pairs_for_bit(i)
    js, gap = js_min_box(lo[:, b0], hi[:, b0], lo[:, b1], hi[:, b1])
    B_per_g = js.mean(1)
    dA = float(np.abs(lo[1:] - lo[0][None]).max()) if len(lo) > 1 else np.nan
    if len(lo) > 1:
        W = (gap > dA).all(0)
        B_rob = (js * W[None]).mean(1)
        return B_per_g, dA, int(W.sum()), B_rob
    return B_per_g, dA, None, None

def role_bound_per_gauge(lpV, tail, ball, i):
    """Variant (forall g, exists x): witnesses chosen separately in each replica; returns min over replicas of B and of the witness count."""
    pV = np.exp(lpV)
    lo, hi = event_interval(pV, tail, ball)
    b0, b1 = pairs_for_bit(i)
    js, gap = js_min_box(lo[:, b0], hi[:, b0], lo[:, b1], hi[:, b1])
    dA = float(np.abs(lo[1:] - lo[0][None]).max())
    Wg = gap > dA
    return float((js * Wg).mean(1).min()), int(Wg.sum(1).min())

def token_channel_bounds(lpV, tail, i):
    pV = np.exp(lpV)
    Pc = np.concatenate([pV, tail[..., None]], -1)
    b0, b1 = pairs_for_bit(i)
    lo = js_cat(Pc[:, b0], Pc[:, b1]).mean(1)
    up = lo + 0.5 * (tail[:, b0] + tail[:, b1]).mean(1)
    return lo, up

def selftest():
    rng = np.random.default_rng(0)
    for _ in range(200):
        l1, l2 = rng.random(2) * 0.01; w1, w2 = rng.random(2) * 0.002
        m, _ = js_min_box(np.array(l1), np.array(l1 + w1), np.array(l2), np.array(l2 + w2))
        for _ in range(20):
            p = l1 + rng.random() * w1; q = l2 + rng.random() * w2
            assert float(m) <= float(js_bin(np.array(p), np.array(q))) + 1e-18
    assert abs(float(js_bin(np.array(0.0), np.array(1.0))) - 1.0) < 1e-12
    G, n = 3, 4096
    lp = np.log(np.tile(np.array([0.99, 0.006, 0.003, 0.001]), (G, n, 1)))
    on = (np.arange(n) >> 5) & 1
    lp[:, on == 1, 1] = np.log(0.009); lp[:, on == 1, 0] = np.log(0.987)
    lp += 1e-9 * rng.standard_normal(lp.shape)
    tail = np.full((G, n), 1e-9)
    ball19 = np.array([True, False, True, True])
    B, dA, wit, Brob = role_bound(lp, tail, ball19, 5)
    assert B.min() > 1e-6 and wit == 2048 and Brob.min() > 1e-6
    B2, _, wit2, _ = role_bound(lp, tail, np.array([True, True, True, True]), 5)
    assert B2.max() == 0.0 and wit2 == 0
    B3, _, wit3, _ = role_bound(lp, tail, ball19, 4)
    assert wit3 == 0
    _, _, wit4, _ = role_bound(lp, np.full((G, n), 0.01), ball19, 5)
    assert wit4 == 0
    klo, kup = token_channel_bounds(lp, tail, 5)
    assert np.all(klo <= kup) and np.all(B <= kup + 1e-15)
    print('[r-bound] selftest OK (minimal JS over intervals, witnesses exists x forall g, worst-case mass outside V, B <= K)')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    w = a.world
    d26 = a.root / S26DIR.format(w=w); n26 = d26 / S26NPZ.format(w=w)
    d9 = a.root / S9DIR.format(w=w); n9 = d9 / S9NPZ.format(w=w)
    for d, n in ((d26, n26), (d9, n9)):
        if not (d / 'RESULT.json').is_file() or not n.is_file():
            raise SystemExit(f'File missing: {d / "RESULT.json"} or {n} ; restore it from the backup')
        if sha(n) != json.loads((d / 'RESULT.json').read_text(encoding='utf-8'))['raw_sha256']:
            raise SystemExit(f'SHA of {n} differs from its RESULT.json')
    r26 = json.loads((d26 / 'RESULT.json').read_text(encoding='utf-8'))
    if r26['status'] not in PASS_SAMPLED:
        raise SystemExit('Sampled return acquisition not conforming: ' + r26['status'])
    lemma_valid = True
    with np.load(n9, allow_pickle=False) as z:
        cubes = z['cubes'].astype(np.float64); rn = [str(x) for x in z['readers']]; bits9 = [str(x) for x in z['mask_bits']]
    cubes = cubes[:, :, [rn.index(b) for b in bits9]]
    keep = retained_sites(cubes, np.abs(cubes[1:] - cubes[0][None]).max((0, 1)))
    with np.load(n26, allow_pickle=False) as z:
        bits = [str(x) for x in z['mask_bits']]; sites = [str(x) for x in z['sites']]; V = z['V'].astype(np.int64)
        lpV = z['lp_V'].astype(np.float64); tail = z['mass_outside_V'].astype(np.float64)
        cg = [str(x) for x in z['channel_gauges']]; infos = [json.loads(str(x)) for x in z['channel_info_json']]
        d_t1 = z['delta_t1'].astype(np.float64); prel = z['token_pair_rel_t1'].astype(np.float64)
    assert bits == bits9 and lpV.shape[1:] == (4096, len(V)) and cg[0] == 'IDENTITY_0'
    roles = [bits[i] for i in range(12) if keep[i]]; ctx = [bits[i] for i in range(12) if not keep[i]]
    gauges_ok = len(cg) == 10
    print(f'[r-bound] u{w} roles ({len(roles)}): {roles} ; context: {ctx} ; channel replicas: {len(cg)} ; |V| = {len(V)} ; '
          f'mass outside V max {tail.max():.3e}', flush=True)
    j0 = int(np.argmax(np.exp(lpV[0]).mean(0)))
    per_role = {}
    for name in roles:
        i = bits.index(name); s = sites.index(name)
        best = None
        best_pg = (0.0, 0, None)
        best_raw = (0.0, None)
        for rho in range(len(V)):
            ball = prel[rho, :, s] <= d_t1[s]
            if ball.all():
                continue
            B_g, dA, wit, B_rob = role_bound(lpV, tail, ball, i)
            cand = dict(rho=int(V[rho]), ball_size=int(ball.sum()), B_lb_min_g=float(B_g.min()), delta_A=dA,
                        witnesses=wit, B_rob_min_g=float(B_rob.min()) if B_rob is not None else None)
            if cand['B_lb_min_g'] > best_raw[0]:
                best_raw = (cand['B_lb_min_g'], int(V[rho]))
            if B_rob is not None:
                cand['B_rob_per_gauge_witnesses_min_g'], cand['witnesses_per_gauge_min'] = role_bound_per_gauge(lpV, tail, ball, i)
                if cand['B_rob_per_gauge_witnesses_min_g'] > best_pg[0]:
                    best_pg = (cand['B_rob_per_gauge_witnesses_min_g'], cand['witnesses_per_gauge_min'], int(V[rho]))
            key = (cand['B_rob_min_g'] if cand['B_rob_min_g'] is not None else -1.0, cand['B_lb_min_g'])
            if best is None or key > best[0]:
                best = (key, cand)
        k_lo, k_up = token_channel_bounds(lpV, tail, i)
        rec = {'mask_bit': i, 'L24_site': s, 'delta_t1': float(d_t1[s]),
               'tokens_of_V_separated_from_clean_token_at_t1': int((prel[j0, :, s] > d_t1[s]).sum()),
               'K_r_inf_min_g': float(k_lo.min()), 'K_r_sup_max_g': float(k_up.max())}
        if best is None:
            rec.update({'event': None, 'B_r_inf': 0.0, 'witnesses': 0, 'R_tok_inf': 0.0, 'R_state_inf': 0.0})
        else:
            c = best[1]; Bv = c['B_rob_min_g'] if gauges_ok else c['B_lb_min_g']; kmax = float(k_up.max())
            rec.update({'event': c, 'B_r_inf': Bv, 'witnesses': c['witnesses'],
                        'R_tok_inf': float(Bv / kmax) if kmax > 0 else 0.0, 'R_state_inf': float(Bv / H_P_GIVEN_C)})
        rec.update({'B_r_inf_primary': best_raw[0], 'rho_primary': best_raw[1], 'R_state_inf_primary': best_raw[0] / H_P_GIVEN_C,
                    'R_tok_inf_primary': best_raw[0] / float(k_up.max()) if float(k_up.max()) > 0 else 0.0})
        rec.update({'B_r_inf_per_gauge_witnesses': best_pg[0], 'witnesses_per_gauge_min': best_pg[1], 'rho_per_gauge_witnesses': best_pg[2],
                    'R_state_inf_per_gauge_witnesses': best_pg[0] / H_P_GIVEN_C})
        per_role[name] = rec
        print(f"[r-bound] {name:20s} B_r >= {rec['B_r_inf']:.3e} bit ; witnesses {rec['witnesses']} ; K_r in [{rec['K_r_inf_min_g']:.3e}, "
              f"{rec['K_r_sup_max_g']:.3e}] ; R_tok >= {rec['R_tok_inf']:.3e} ; R_state >= {rec['R_state_inf']:.3e}", flush=True)
    ctx_bits = [bits.index(c) for c in ctx]
    I_ctx = {g: (inf[f'I_X_token_given_bit{ctx_bits[0]}'] if len(ctx_bits) == 1 else None) for g, inf in zip(cg, infos)}
    pos = [r for r, v in per_role.items() if v['B_r_inf'] > 0 and (v['witnesses'] or 0) > 0]
    pos_primary = [r for r, v in per_role.items() if v['B_r_inf_primary'] > NUM_FLOOR_BITS]
    status_primary = ('R_POSITIVE_PRIMARY_BF16_AS_RUN_ON_C_L24_R__AWAITING_INDEPENDENT_CERTIFICATE' if gauges_ok and lemma_valid and pos_primary
                      else 'R_NO_POSITIVE_PRIMARY_RETURN_ON_C_L24_R')
    status = ('R_POSITIVE_REPORTED_ON_C_L24_R__AWAITING_INDEPENDENT_CERTIFICATE' if gauges_ok and lemma_valid and pos
              else ('R_NOT_RESOLVED_WITHOUT_GAUGE_CHANNEL' if not gauges_ok else 'R_NO_GAUGE_ROBUST_RETURN_ON_C_L24_R'))
    res = {'world': w, 'sampled_npz_sha256': sha(n26), 'gauge_cube_npz_sha256': sha(n9), 'contract': 'C_L24-R (see the script header)',
           'roles': roles, 'context': ctx, 'V': V.tolist(), 'mass_outside_V_max': float(tail.max()),
           'I_P_token_given_context_bits_per_replica': I_ctx,
           'lemma_valid': lemma_valid, 'per_role': per_role, 'roles_with_witnessed_return': pos,
           'B_max_inf': max((v['B_r_inf'] for v in per_role.values()), default=0.0),
           'B_max_inf_per_gauge_witnesses': max((v['B_r_inf_per_gauge_witnesses'] for v in per_role.values()), default=0.0),
           'R_state_max_inf_per_gauge_witnesses': max((v['R_state_inf_per_gauge_witnesses'] for v in per_role.values()), default=0.0),
           'witness_rules': {'B_r_inf': 'exists x, forall g (same background pair in the 10 replicas)',
                             'B_r_inf_per_gauge_witnesses': 'forall g, exists x (witness pairs chosen within each replica); minimum over replicas'},
           'R_state_max_inf': max((v['R_state_inf'] for v in per_role.values()), default=0.0),
           'v45_1_conformance': {
               'numerator': 'B_r = I(P_r ; f_r(T) | P_-r, C) <= I(P ; N_r(t+1) | C) (chain rule): same-role return, V45.1 8.3',
               'neutralisation': 'independent-noise kernel functional (each counterfactual world draws its own token), V45.1 8.3 primary target',
               'bypass': 'C_bypass for the same-role return measured with the token held fixed: bit for bit zero (lemma checks), V45.1 11.1',
               'horizon_k0': 1,
               'K_eff_bound': 'K_eff(q) = H_q(Z* | U) <= H(P | C) = 11 bits on the declared source alphabet ; same source, grain and q as B (V45.1 11.3, 12.4)',
               'K_eff_positive': 'K_eff >= 1 bit on C_L24 (Delta game: v(N) >= 1 bit), so R is defined (not NA)',
               'role_identity_t_to_t1': 'NOT_TESTED: L24 site r at t+1 identified with L24 site r at t by physical label; V45.1 8.1 requires an interventional bisimulation',
           },
           'primary_contract': 'Qwen3-8B bf16 as run: probabilities exact values of the declared system; bound = minimum over the 10 replicas, '
                               'worst-case allocation of the mass outside V, float64 post-processing floor NUM_FLOOR_BITS',
           'num_floor_bits': NUM_FLOOR_BITS, 'roles_with_primary_return': pos_primary,
           'B_max_inf_primary': max((v['B_r_inf_primary'] for v in per_role.values()), default=0.0),
           'R_state_max_inf_primary': max((v['R_state_inf_primary'] for v in per_role.values()), default=0.0),
           'status_delta_A_robustness': status,
           'status': status_primary}
    out = a.root / OUTDIR.format(w=w)
    if out.exists():
        raise SystemExit('Output folder already exists: move it to rerun: ' + str(out))
    out.mkdir(parents=True)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"[r-bound] roles with witnessed return: {pos} ; B_max >= {res['B_max_inf']:.3e} bit ; R_state_max >= {res['R_state_max_inf']:.3e}", flush=True)
    print('[r-bound] STATUS (primary)', status_primary, '; delta_A robustness', status, '; RESULT', out / 'RESULT.json')

if __name__ == '__main__':
    main()
