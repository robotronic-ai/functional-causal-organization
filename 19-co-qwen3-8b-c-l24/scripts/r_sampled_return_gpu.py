#!/usr/bin/env python3
"""R, sampled transition (GPU): return L24(t) -> sampled token -> L24(t+1) on the C_L24-R contract, world u26.

Transition: T_t ~ Categorical(softmax(logits_t)), temperature 1, full vocabulary, no truncation.
All token-law computations are done in float64 over the entire vocabulary, from the model logits.
V = tokens present in the top-20 of every mask of the greedy acquisition, plus the 64 most probable tokens of the clean
distribution (identity gauge); fixed before the acquisition.
Stage 1 (10 replicas; --no-gauge-channel: identity only): for the 4096 masks, full forward at time t;
  per mask: log p(tau | x) for tau in V, mass outside V (direct sum of the probabilities outside V), entropy H(p_x),
  argmax token, L25 scores (bit for bit equal to the gauge cube of the same replica);
  per replica: sums of the laws p_x over the full vocabulary (all masks, and per value of each mask bit), hence
  I(X ; token) and I(X ; token | bit b) for every bit b, exact over the full vocabulary.
Stage 2 (10 replicas): writes of the 12 L24 sites at t+1 for each tau in V, from the clean cache.
Lemma (identity): on 8 masks, arm cache + tau == clean cache + tau, bit for bit, for every tau in V.
No value of R is computed here.
Run (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u r_sampled_return_gpu.py --preflight
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u r_sampled_return_gpu.py --gpu [--no-gauge-channel]
  (resume: --gpu --resume ; an interrupted replica restarts from its beginning)
Writes FAST_GPU_RESULTS/CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU/{RESULT.json, S26_R_SAMPLED_RAW_u{w}.npz}.
"""
import argparse, contextlib, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
NAME = 'CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU'
S24DIR = 'FAST_GPU_RESULTS/CLAUDE_S24_R_TOKEN_RETURN_WORLD{w}_GPU'
S24NPZ = 'S24_R_RAW_u{w}.npz'
S9FULL = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
PIN_T292 = '64955123f179ff1fc9d798be0d7954a689ce1fadc933c372e14e4a4839423bc3'
N_ARMS = 4096
CHUNK = 256
N_LEMMA = 8
N_REPLAY = 16
N_TOP_CLEAN = 64
LN2 = 0.6931471805599453
MASK12 = np.arange(4096, dtype=np.int64)

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def import_pin(p, name, digest):
    need(Path(p).is_file() and sha(p) == digest, 'Pinned preflight script missing or SHA mismatch: ' + str(p))
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

GAUGES = [('UPDOWN_SCALE', s) for s in (0, 1, 2)] + [('VO_DIAG', s) for s in (0, 1, 2)] + [('QKNORM_RECIP', s) for s in (0, 1, 2)]

def gauge_factors(kind, seed, layer, nkv, hd, dff, log_std=0.3):
    """Factors of apply_gauge_ (qwen_local_transition) for `layer`."""
    import torch
    g = torch.Generator().manual_seed(1009 * seed + len(kind))
    if kind == 'UPDOWN_SCALE':
        return {'d': torch.exp(log_std * torch.randn(dff, generator=g, dtype=torch.float64)).numpy()}
    if kind == 'VO_DIAG':
        return {'o': [torch.exp(log_std * torch.randn(hd, generator=g, dtype=torch.float64)).numpy() for _ in range(nkv)]}
    return {}

def transport(values, cand, fac, hd, group):
    """Coordinates of a donor value in the recoded replica."""
    out = []
    for arr, sl in zip(values, cand['slices']):
        a = np.asarray(arr, dtype=np.float64)
        if cand['kind'] == 'o' and 'o' in fac:
            a = a * fac['o'][(sl.start // hd) // group]
        elif cand['kind'] == 'd' and 'd' in fac:
            a = a * fac['d'][sl]
        out.append(a.astype(np.float32))
    return out

def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).mean()) / (0.5 * (np.sqrt((a ** 2).mean()) + np.sqrt((b ** 2).mean())) + 1e-30))

def fixed_group(top_ids):
    """Tokens present in the top-k of every mask."""
    return sorted(set.intersection(*[set(int(t) for t in r) for r in top_ids]))

def chan_row(logits, Vt, notV):
    """Full law in float64: log p on V, mass outside V (direct sum), entropy in bits, law p."""
    import torch
    l = logits.double()
    lp = l - torch.logsumexp(l, -1)
    p = torch.exp(lp)
    return p, lp[Vt], float(p[notV].sum()), float(torch.special.entr(p).sum()) / LN2

def acc_add(acc, p, m):
    acc[0] += p
    for b in range(12):
        acc[1 + 2 * b + ((m >> b) & 1)] += p

def acc_info(acc, Hx_mean, n):
    """I(X ; token) and I(X ; token | bit b) in bits, X uniform over n masks (full vocabulary)."""
    import torch
    Hd = lambda v: float(torch.special.entr(v).sum()) / LN2
    out = {'I_X_token': Hd(acc[0] / n) - Hx_mean}
    for b in range(12):
        out[f'I_X_token_given_bit{b}'] = 0.5 * (Hd(acc[1 + 2 * b] / (n / 2)) + Hd(acc[2 + 2 * b] / (n / 2))) - Hx_mean
    return out

def selftest():
    import torch
    nkv, hd, dff = 8, 128, 12288
    f1 = gauge_factors('VO_DIAG', 1, 24, nkv, hd, dff)
    cand = {'kind': 'o', 'slices': [slice(h * hd, (h + 1) * hd) for h in (28, 29, 30, 31)]}
    v = [np.ones(hd, np.float32) for _ in range(4)]
    need(all(np.allclose(x, f1['o'][7]) for x in transport(v, cand, f1, hd, 4)), 'Transport VO_DIAG')
    ids = np.tile(np.arange(20), (6, 1)); ids[2, 5] = 77
    need(fixed_group(ids) == [t for t in range(20) if t != 5], 'Fixed grouping')
    g = torch.Generator().manual_seed(0)
    W = 50; n = 4096
    logits = torch.randn(n, W, generator=g, dtype=torch.float64) * 3
    logits[torch.arange(n) & 8 > 0, 3] += 1.0
    logits[torch.arange(n) & 2048 > 0, 7] += 0.5
    Vt = torch.tensor([0, 3, 7]); notV = torch.ones(W, dtype=torch.bool); notV[Vt] = False
    acc = torch.zeros(25, W, dtype=torch.float64); Hs = []; P = []
    for m in range(n):
        pr, lpv, tail, Hx = chan_row(logits[m], Vt, notV)
        need(abs(tail - float(pr[notV].sum())) < 1e-15 and abs(float(lpv[1]) - float(torch.log(pr[3]))) < 1e-12, 'Channel row')
        acc_add(acc, pr, m); Hs.append(Hx); P.append(pr.numpy())
    info = acc_info(acc, float(np.mean(Hs)), n)
    P = np.array(P)
    def H(q): q = q[q > 0]; return float(-(q * np.log2(q)).sum())
    direct = H(P.mean(0)) - np.mean([H(r) for r in P])
    need(abs(info['I_X_token'] - direct) < 1e-10, 'I(X;token) par accumulation')
    c = (np.arange(n) >> 11) & 1
    direct_c = 0.5 * sum(H(P[c == k].mean(0)) for k in (0, 1)) - np.mean([H(r) for r in P])
    need(abs(info['I_X_token_given_bit11'] - direct_c) < 1e-10, 'I(X;token|bit) par accumulation')
    need(info['I_X_token_given_bit3'] < info['I_X_token'] and info['I_X_token_given_bit3'] > 0, 'Conditionnement')
    need(rel(np.ones(4), np.ones(4)) == 0.0, 'rel')
    print('[r-sampled] selftest OK (transport, fixed grouping, float64 channel row, exact I(X;token) and I(X;token|bit))')

def run_gpu(root, out, T292, pre, resume, world, gauge_channel):
    T290, prior, oldbit, oldscore = pre
    T287, olds, _, _, _ = prior
    T286, old_obj, prior2, prior_audit, donor_meta, oldrawpath = olds
    T284, T276, old, t276raw = old_obj
    T275, hist, folder, report, contract = old
    T258, T256, T230, T234, clean, names, _ = hist
    T = T230.import_ref(root); T219 = T.import_t219(root); D0V = T219.import_d0(root)
    src = root / 'CLAUDE_CO_Q2L1'; eng = root / 'TURN203/REFERENCE_ENGINE_COMPLETE'
    modeldir = root / 'Qwen3-8B'
    schedule = root / 'direct_co_op_scripts_v42/DIRECT_CO_OP_DEVELOPMENT_SCHEDULE_34.json'
    D0V.verify_sources(src, eng, schedule, modeldir)
    import torch
    need(torch.cuda.is_available(), 'CUDA unavailable')
    sys.path.insert(0, str(src))
    import gamma_local_qwen as G, role_quotient as RQ, qwen_local_transition as Q
    need(RQ.VERSION == 'ROLEQ_v3.2_T126_CONFORMANT', 'Unexpected ROLEQ version')
    need([tuple(x) for x in Q.GAUGE_SET] == GAUGES, 'Unexpected Q1 gauge set')
    base, rt, engine = G._imports(str(eng))
    worlds = tuple(sorted(set(T286.D0) | {world}))
    need(world in T276.EVAL and world not in T286.D0, f'World {world} outside the evaluation worlds or inside D0')
    cases = {c['key']: c for c in G.schedule_cases(T276.read(schedule), worlds, ('REVOKE', 'SUBSTITUTE'), ('H1', 'H2'))}

    full = None
    pf = root / S9FULL.format(w=world)
    if (pf / 'RESULT.json').is_file():
        rf = json.loads((pf / 'RESULT.json').read_text(encoding='utf-8'))
        npf = pf / f'S9_FULL12_GAUGE_CUBES_u{world}.npz'
        need(npf.is_file() and sha(npf) == rf['raw_sha256'], 'Gauge cube NPZ missing or SHA differs from its RESULT.json')
        with np.load(npf, allow_pickle=False) as z:
            need(z['cubes'].shape[1:] == (4096, 12), 'Gauge cube: shape (replicas, 4096, 12) expected')
            order = np.argsort(z['mask12'].astype(np.int64)) if 'mask12' in z.files else MASK12
            if 'mask12' in z.files:
                need(np.array_equal(z['mask12'].astype(np.int64)[order], MASK12), 'Gauge cube: incomplete masks')
            full = {str(n): z['cubes'][i][order].copy() for i, n in enumerate(z['replica_names'])}
            bits = [str(x) for x in z['mask_bits']]
        print(f'[r-sampled] gauge cube u{world} present: bit for bit check of the L25 scores per replica', flush=True)
    need(full is not None, f'Gauge cube u{world} missing: restore it from the backup ' + str(pf))
    d24 = root / S24DIR.format(w=world); n24 = d24 / S24NPZ.format(w=world)
    need((d24 / 'RESULT.json').is_file() and n24.is_file(), 'Greedy acquisition outputs missing: restore them from the backup ' + str(d24))
    r24 = json.loads((d24 / 'RESULT.json').read_text(encoding='utf-8'))
    need(sha(n24) == r24['raw_sha256'] and r24['status'] in ('PASS_R_GREEDY_RETURN_ACQUISITION', 'PASS_S24_R_C_L24_RAW_ACQUISITION'),
         'Greedy NPZ: SHA mismatch or acquisition not passed (a PASS status implies a valid lemma check)')
    with np.load(n24, allow_pickle=False) as z:
        need(np.array_equal(z['mask12'], MASK12), 'Greedy acquisition: masks')
        ids24 = z['top_ids'].astype(np.int64); lp24 = z['top_logprobs'].astype(np.float32); tok24 = z['tokens'].astype(np.int64)
        clean24 = int(z['clean_token'])
    U = fixed_group(ids24)
    print(f'[r-sampled] U (top-20 common to the 4096 masks) = {U}', flush=True)
    t0 = time.monotonic()
    print('[r-sampled] Loading Qwen', flush=True)
    model, tok = G.load_model(argparse.Namespace(model_dir=str(modeldir), device='cuda', dtype='bfloat16'), rt)
    model.eval()
    nq, nkv, hd, group = Q.geom(model)
    dff = Q.decoder(model).layers[0].mlp.down_proj.weight.shape[1]
    reg = RQ.candidate_registry(model)
    act = T275.site_registry(reg, T286.NINE)
    background = T275.site_registry(reg, T286.BACKGROUND)
    all12 = T275.site_registry(reg, T286.ALL12)
    donors, orig, dirs, valid = T276.load_frozen_arrays(folder, act, T275)
    need(orig.shape[0] == 12 and dirs.shape[0] == 12, 'Frozen readers: 12 expected')
    TI = list(T286.TARGET_INDICES); OTHER = [k for k in range(12) if k not in TI]
    l25 = Q.decoder(model).layers[25]
    clean_by_u = {u: int(clean[list(T256.ALL_UNITS).index(u)]) for u in worlds}
    with np.load(t276raw, allow_pickle=False) as z:
        reference = z['future_L25_physical_writes'].copy()

    def prep_of(u):
        preps, _ = T230.get_unit_preps(u, model, tok, G, engine, rt, RQ, base, cases)
        pre_, source_tok = preps['SUBSTITUTE|H1']
        cache, token, _ = T234.first_forward(model, base, RQ, pre_, source_tok, [], act[:2], T230)
        need(int(token) == clean_by_u[u], f'Unexpected active token u{u}')
        return RQ.prepare_window(model, base, cache, [int(token)], 24, 'cuda')

    def scores12(prep, edits):
        captured = {}
        def grab(kind):
            def hook(mod, args): captured[kind] = args[0][0, -1, :].detach().clone()
            return hook
        hs = [l25.self_attn.o_proj.register_forward_pre_hook(grab('o')),
              l25.mlp.down_proj.register_forward_pre_hook(grab('d'))]
        try:
            RQ.run_window(model, prep, 24, 'cuda', edits=edits)
        finally:
            for h in hs: h.remove()
        w = np.stack([RQ.cand_write(model, 25, c, captured).numpy() for c in all12]).astype(np.float32)
        s = np.zeros(12)
        s[TI] = np.einsum('jd,jd->j', (w[TI] - orig[TI]).astype(np.float64), dirs[TI].astype(np.float64))
        s[OTHER] = np.einsum('jd,jd->j', (w[OTHER] - orig[OTHER]).astype(np.float64), dirs[OTHER].astype(np.float64))
        return s, w

    values_bg = []
    with torch.inference_mode():
        for u in T286.D0:
            run = RQ.run_window(model, prep_of(u), 24, 'cuda', capture=True)
            inp = {'o': run['o'], 'd': run['d']}
            values_bg.append([[np.asarray(x.detach().float().cpu().numpy(), dtype=np.float32).copy() for x in RQ.cand_values(inp, c)] for c in background])
    pairs = T286.freeze_background_pair(values_bg)
    need([{'site': e['site'], 'donor_units': e['donor_units'], 'squared_rms_contrast': e['squared_rms_contrast']} for e in pairs]
         == donor_meta['variable_background'], 'D0 background pairs differ from the frozen pairs')

    def edits_for(mask12, fac):
        ed = []
        for j, c in enumerate(act):
            v = donors[j]['values'][(mask12 >> j) & 1]
            ed.extend(RQ.edits_for(c, [torch.from_numpy(a).to('cuda') for a in transport(v, c, fac, hd, group)]))
        for j, (c, pair) in enumerate(zip(background, pairs)):
            v = pair['values'][(mask12 >> (9 + j)) & 1]
            ed.extend(RQ.edits_for(c, [torch.from_numpy(a).to('cuda') for a in transport(v, c, fac, hd, group)]))
        return ed

    def replay_historical(prep):
        wi = list(T276.EVAL).index(world)
        for j, m9 in enumerate(T286.HISTORICAL):
            _, w = scores12(prep, T275.build_edits(act, donors, m9, RQ, torch))
            need(np.array_equal(w.view(np.uint32), reference[wi, j].view(np.uint32)), f'Drift of frozen arm u{world} m9={m9}')
        print(f'[r-sampled] u{world}: 10/10 frozen historical arms bit for bit (writes of the 12 sites)', flush=True)

    L24 = Q.decoder(model).layers[24]

    def forward_t(prep, token, edits):
        """Full forward at time t with the L24 writes replaced; returns the cache after t, the logits and the L25 scores."""
        cache = base.clone_cache(prep['cache']); captured = {}
        def patch(kind):
            def f(mod, args):
                sel = [(sl, v) for k, sl, v in (edits or []) if k == kind]
                if not sel:
                    return None
                x = args[0].clone()
                for sl, v in sel:
                    x[:, -1, sl] = v.to(x.dtype)
                return (x,) + tuple(args[1:])
            return f
        def grab(kind):
            def f(mod, args): captured[kind] = args[0][0, -1, :].detach().clone()
            return f
        hs = [L24.self_attn.o_proj.register_forward_pre_hook(patch('o')), L24.mlp.down_proj.register_forward_pre_hook(patch('d')),
              l25.self_attn.o_proj.register_forward_pre_hook(grab('o')), l25.mlp.down_proj.register_forward_pre_hook(grab('d'))]
        try:
            res_ = model(input_ids=torch.tensor([[int(token)]], device='cuda'),
                         attention_mask=torch.ones((1, prep['n'] + 1), dtype=torch.long, device='cuda'),
                         past_key_values=cache, use_cache=True)
        finally:
            for h in hs: h.remove()
        need(int(base.cache_seq_length(cache)) == prep['n'] + 1, 'Inconsistent cache at t')
        w = np.stack([RQ.cand_write(model, 25, c, captured).numpy() for c in all12]).astype(np.float32)
        s = np.zeros(12)
        s[TI] = np.einsum('jd,jd->j', (w[TI] - orig[TI]).astype(np.float64), dirs[TI].astype(np.float64))
        s[OTHER] = np.einsum('jd,jd->j', (w[OTHER] - orig[OTHER]).astype(np.float64), dirs[OTHER].astype(np.float64))
        return cache, res_.logits[0, -1, :].float(), s

    def forward_t1(prep, cache_after_t, token):
        """Forward at t+1 (token imposed) from a cache after t; writes of the 12 L24 sites at t+1."""
        cache = base.clone_cache(cache_after_t); cap = {}
        def grab(kind):
            def f(mod, args): cap[kind] = args[0][0, -1, :].detach().clone()
            return f
        hs = [L24.self_attn.o_proj.register_forward_pre_hook(grab('o')), L24.mlp.down_proj.register_forward_pre_hook(grab('d'))]
        try:
            model(input_ids=torch.tensor([[int(token)]], device='cuda'),
                  attention_mask=torch.ones((1, prep['n'] + 2), dtype=torch.long, device='cuda'),
                  past_key_values=cache, use_cache=True)
        finally:
            for h in hs: h.remove()
        need(int(base.cache_seq_length(cache)) == prep['n'] + 2, 'Inconsistent cache at t+1')
        return np.stack([RQ.cand_write(model, 24, c, cap).numpy() for c in all12]).astype(np.float32)

    replicas = [('IDENTITY', 0)] + GAUGES
    token = clean_by_u[world]
    vocab = None

    def rep_file(kind_, key):
        return out / f'S26_{kind_}_{key}.npz'

    def load_rep(kind_, key):
        f = rep_file(kind_, key)
        if resume and f.is_file():
            with np.load(f, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}
        return None

    def save_rep(kind_, key, **arrs):
        f = rep_file(kind_, key); tmp = f.with_suffix('.tmp.npz')
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, **arrs); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, f)

    with torch.inference_mode():
        prep_id = prep_of(world)
        replay_historical(prep_id)
        _, logits0, _ = forward_t(prep_id, token, None)
        vocab = int(logits0.shape[-1])
        top_clean = torch.topk(logits0.double(), N_TOP_CLEAN).indices.cpu().tolist()
        V = sorted(set(U) | set(int(t) for t in top_clean))
        Vt = torch.tensor(V, dtype=torch.long, device='cuda')
        notV = torch.ones(vocab, dtype=torch.bool, device='cuda'); notV[Vt] = False
        print(f'[r-sampled] V = {len(V)} tokens (U plus clean top-{N_TOP_CLEAN}) ; vocabulary {vocab}', flush=True)

        rng = np.random.default_rng(0)
        probe = rng.choice(N_ARMS, size=N_REPLAY, replace=False)
        eq_s = eq_t = 0; dlp = 0.0
        for m in probe:
            _, logits, sf = forward_t(prep_id, token, edits_for(int(m), {}))
            _, lpv, _, _ = chan_row(logits, Vt, notV)
            lpv = dict(zip(V, lpv.cpu().numpy()))
            eq_s += int(np.array_equal(sf, full['IDENTITY_0'][m])); eq_t += int(int(logits.argmax()) == tok24[m])
            dlp = max(dlp, max(abs(float(lpv[int(t)]) - float(v_)) for t, v_ in zip(ids24[m], lp24[m]) if int(t) in lpv))
        print(f'[r-sampled] identity replay ({N_REPLAY} masks): L25 scores == gauge cube: {eq_s}/{N_REPLAY} ; argmax == greedy run: {eq_t}/{N_REPLAY} ; '
              f'max deviation log p float64 / float32 greedy run: {dlp:.2e}', flush=True)
        need(eq_s == N_REPLAY and eq_t == N_REPLAY and dlp < 1e-4, 'Identity replay not conforming')

        stage1 = replicas if gauge_channel else replicas[:1]
        S1 = {}
        for kind, seed in stage1:
            key = f'{kind}_{seed}'
            got = load_rep('CANAL', key)
            if got is not None and np.array_equal(got['V'], np.array(V)):
                S1[key] = got; continue
            fac = {} if kind == 'IDENTITY' else gauge_factors(kind, seed, 24, nkv, hd, dff)
            cm = contextlib.nullcontext() if kind == 'IDENTITY' else Q.Gauged(model, 24, kind, seed)
            lpV = np.zeros((N_ARMS, len(V))); tail = np.zeros(N_ARMS); Hx = np.zeros(N_ARMS)
            tk = np.zeros(N_ARMS, np.int64); sc = np.zeros((N_ARMS, 12))
            acc = torch.zeros(25, vocab, dtype=torch.float64, device='cuda')
            with cm:
                prep = prep_id if kind == 'IDENTITY' else prep_of(world)
                for m in range(N_ARMS):
                    _, logits, s_ = forward_t(prep, token, edits_for(int(MASK12[m]), fac))
                    pr, lpv, tail[m], Hx[m] = chan_row(logits, Vt, notV)
                    acc_add(acc, pr, int(MASK12[m]))
                    lpV[m] = lpv.cpu().numpy(); tk[m] = int(logits.argmax()); sc[m] = s_
                    if (m + 1) % CHUNK == 0:
                        print(f'[r-sampled] stage 1 {key} {m + 1}/{N_ARMS} ; {time.monotonic() - t0:.0f} s', flush=True)
            info = acc_info(acc, float(Hx.mean()), N_ARMS); del acc
            S1[key] = dict(V=np.array(V), lpV=lpV, tail=tail, Hx=Hx, tok=tk, sc=sc, info=np.array(json.dumps(info)))
            save_rep('CANAL', key, **S1[key])
            print(f"[r-sampled] {key}: I(X;token) = {info['I_X_token']:.4e} bit (full vocabulary) ; mass outside V max {tail.max():.3e} ; "
                  f"distinct argmax {sorted(set(tk.tolist()))}", flush=True)
        cube_eq = {k: bool(np.array_equal(v_['sc'], full[k])) for k, v_ in S1.items()}
        print(f'[r-sampled] L25 scores == gauge cube, per replica: {cube_eq}', flush=True)

        W1 = {}; clean_by_gauge = {}
        for kind, seed in replicas:
            key = f'{kind}_{seed}'
            got = load_rep('T1', key)
            if got is not None and np.array_equal(got['V'], np.array(V)):
                W1[key] = got['W1']; clean_by_gauge[key] = int(got['clean']); continue
            cm = contextlib.nullcontext() if kind == 'IDENTITY' else Q.Gauged(model, 24, kind, seed)
            with cm:
                prep = prep_id if kind == 'IDENTITY' else prep_of(world)
                cg, lg, _ = forward_t(prep, token, None)
                clean_by_gauge[key] = int(lg.argmax())
                W1[key] = np.stack([forward_t1(prep, cg, t_) for t_ in V])
                del cg
            save_rep('T1', key, V=np.array(V), W1=W1[key], clean=np.array(clean_by_gauge[key]))
            print(f'[r-sampled] stage 2 {key}: L24(t+1) for {len(V)} tokens ; {time.monotonic() - t0:.0f} s', flush=True)

        margin24 = lp24[:, 0] - lp24[:, 1]
        lem_masks = np.unique(np.r_[int(np.argmin(margin24)), rng.choice(N_ARMS, size=N_LEMMA - 1, replace=False)]).astype(int)
        lem_ok = lem_n = 0
        for m in lem_masks:
            cp, _, _ = forward_t(prep_id, token, edits_for(int(m), {}))
            for j, t_ in enumerate(V):
                lem_ok += int(np.array_equal(forward_t1(prep_id, cp, t_), W1['IDENTITY_0'][j])); lem_n += 1
            del cp
        print(f'[r-sampled] lemma: arm cache + tau == clean cache + tau: {lem_ok}/{lem_n} ({len(lem_masks)} masks x {len(V)} tokens)', flush=True)

    names_g = [f'{k}_{s_}' for k, s_ in replicas]
    W1s = np.stack([W1[k] for k in names_g])
    relg = np.array([[[rel(W1s[g, t, r], W1s[0, t, r]) for r in range(12)] for t in range(len(V))] for g in range(len(names_g))])
    d_t1 = relg[1:].max((0, 1))
    pair_rel = np.array([[[rel(W1s[0, i_, r], W1s[0, j_, r]) for r in range(12)] for j_ in range(len(V))] for i_ in range(len(V))])
    sep = pair_rel > d_t1[None, None, :]
    names_c = list(S1)
    final = out / f'S26_R_SAMPLED_RAW_u{world}.npz'
    if final.exists():
        final.unlink()
    with open(final, 'xb') as f:
        np.savez_compressed(f, mask12=MASK12, mask_bits=np.array(bits), V=np.array(V, np.int64), U=np.array(U, np.int64),
                            channel_gauges=np.array(names_c),
                            lp_V=np.stack([S1[k]['lpV'] for k in names_c]), mass_outside_V=np.stack([S1[k]['tail'] for k in names_c]),
                            H_x_bits=np.stack([S1[k]['Hx'] for k in names_c]), tokens_argmax=np.stack([S1[k]['tok'] for k in names_c]),
                            channel_info_json=np.array([str(S1[k]['info']) for k in names_c]),
                            W1_L24_t1_identity=W1s[0], W1_rel_to_identity=relg, gauges=np.array(names_g), sites=np.asarray(T286.ALL12),
                            delta_t1=d_t1, token_pair_rel_t1=pair_rel, token_separation=sep)
    lemma_ok = bool(lem_n > 0 and lem_ok == lem_n)
    infos = {k: json.loads(str(S1[k]['info'])) for k in names_c}
    res = {'status': 'PASS_R_SAMPLED_RETURN_ACQUISITION' if lemma_ok and all(cube_eq.values()) else 'R_SAMPLED_CHECK_FAILED',
           'world': world, 'clean_token_greedy_run': clean24, 'transition': 'Categorical(softmax(logits)), temperature 1, full vocabulary', 'vocabulary': vocab,
           'V': V, 'channel_under_gauges': bool(gauge_channel), 'active_token_t': token, 'clean_token_per_gauge': clean_by_gauge,
           'L25_scores_equal_gauge_cube_per_replica': cube_eq,
           'I_X_token_bits_per_replica': {k: v_['I_X_token'] for k, v_ in infos.items()},
           'I_X_token_given_bit_per_replica': {k: {bits[b]: v_[f'I_X_token_given_bit{b}'] for b in range(12)} for k, v_ in infos.items()},
           'mass_outside_V_max': {k: float(S1[k]['tail'].max()) for k in names_c},
           'distinct_argmax': {k: sorted(set(S1[k]['tok'].tolist())) for k in names_c},
           'delta_L24_t1_per_site': dict(zip(T286.ALL12, d_t1.tolist())),
           'token_pairs_of_V_separated_at_L24_t1_per_site': {T286.ALL12[r]: int(sep[:, :, r].sum() // 2) for r in range(12)},
           'lemma': {'tests': lem_n, 'equal': lem_ok, 'valid': lemma_ok},
           'raw_sha256': sha(final), 'gpu_seconds': round(time.monotonic() - t0, 1), 'R_computed': False}
    with open(out / 'RESULT.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('[r-sampled] I(X;token) per replica:', {k: f"{v_:.4e}" for k, v_ in res['I_X_token_bits_per_replica'].items()}, flush=True)
    print('[r-sampled] token pairs of V separated at L24(t+1) per site:', res['token_pairs_of_V_separated_at_L24_t1_per_site'], flush=True)
    print('[r-sampled] FINAL', res['status'], '; RESULT', out / 'RESULT.json', flush=True)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selftest', action='store_true'); g.add_argument('--preflight', action='store_true')
    g.add_argument('--gpu', action='store_true')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--no-gauge-channel', action='store_true')
    a = ap.parse_args()
    need(sys.flags.isolated, 'Run with python -I -u')
    selftest()
    if a.selftest:
        return 0
    root = a.root.resolve()
    T292 = import_pin(root / 'T292_SAME_CODE_MICRO_DONOR_FIBERS_GPU.py', 't292_pinned', PIN_T292)
    pre = T292.preflight(root)
    print('[r-sampled] PREFLIGHT PASS', flush=True)
    if a.preflight:
        return 0
    out = root / 'FAST_GPU_RESULTS' / NAME.format(w=a.world)
    if out.exists() and not a.resume:
        raise SystemExit('Output folder already exists: rerun with --resume, or move it: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'GPU_LIVE.log', 'a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(T292.Tee(sys.stdout, log)), contextlib.redirect_stderr(T292.Tee(sys.stderr, log)):
            try:
                run_gpu(root, out, T292, pre, a.resume, a.world, not a.no_gauge_channel); return 0
            except BaseException:
                traceback.print_exc(); print('[r-sampled] STOPPED: rerun with --gpu --resume', flush=True); return 2

if __name__ == '__main__':
    raise SystemExit(main())
