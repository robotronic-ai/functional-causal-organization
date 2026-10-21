#!/usr/bin/env python3
"""R, greedy transition (GPU): return L24(t) -> emitted token -> L24(t+1) on the C_L24 contract, world u26.

Lemma: at t+1, L24 receives the intervention made on its writes at t ONLY through the emitted token.
Stage 1 (identity gauge; option --gauges-stage1: the 10 replicas): for each of the 4096 C_L24 donor masks (same frozen
  donors, same background pairs, same active position as the gauge cube), FULL model forward at time t with the L24
  writes replaced; records the argmax token, the 20 best log-probabilities and the 12 L25 scores (check: bit for bit
  equal to the gauge cube of the same replica).
Stage 2 (10 replicas): for each distinct emitted token (and the clean token), forward at t+1 from the clean cache
  (token imposed); writes of the 12 L24 sites at t+1 (residual space).
Lemma checks (identity): on a sample of arms (all arms whose token changes, capped, plus 32 others),
  (a) arm cache + its token == clean cache + same token (bit for bit at L24(t+1));
  (b) arm cache + clean token == clean cache + clean token (bit for bit at L24(t+1)).
No value of R is computed here: RESULT gives the tokens, H(token) under q_donor, the token flip fraction per mask bit,
and the raw data for the analysis.
Run (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u r_greedy_return_gpu.py --preflight
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u r_greedy_return_gpu.py --gpu [--gauges-stage1]
  (resume: --gpu --resume)
Writes FAST_GPU_RESULTS/CLAUDE_S24_R_TOKEN_RETURN_WORLD{w}_GPU/{RESULT.json, S24_R_RAW_u{w}.npz}.
"""
import argparse, contextlib, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
NAME = 'CLAUDE_S24_R_TOKEN_RETURN_WORLD{w}_GPU'
S9FULL = 'FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
PIN_T292 = '64955123f179ff1fc9d798be0d7954a689ce1fadc933c372e14e4a4839423bc3'
N_ARMS = 4096
CHUNK = 256
TOPK = 20
N_LEMMA_MAX = 256
N_LEMMA_EXTRA = 32
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

def token_entropy(tokens):
    """H(token) under the uniform law over the 4096 masks, in bits (conditioning on the C_L24 context is done on CPU)."""
    _, cnt = np.unique(tokens, return_counts=True); p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())

def flip_witnesses(tokens):
    """For each mask bit i: fraction of the configurations x_-i where flipping i changes the token."""
    res = []
    for i in range(12):
        base = np.array([m for m in range(4096) if not (m >> i) & 1])
        res.append(float((tokens[base] != tokens[base | (1 << i)]).mean()))
    return res

def selftest():
    nkv, hd, dff = 8, 128, 12288
    f1 = gauge_factors('VO_DIAG', 1, 24, nkv, hd, dff)
    cand = {'kind': 'o', 'slices': [slice(h * hd, (h + 1) * hd) for h in (28, 29, 30, 31)]}
    v = [np.ones(hd, np.float32) for _ in range(4)]
    need(all(np.allclose(x, f1['o'][7]) for x in transport(v, cand, f1, hd, 4)), 'Transport VO_DIAG')
    t = np.zeros(4096, np.int64); t[np.arange(4096) & 32 > 0] = 7
    fw = flip_witnesses(t); need(fw[5] == 1.0 and sum(fw) == 1.0, 'Flip witnesses')
    need(abs(token_entropy(t) - 1.0) < 1e-12, 'Token entropy')
    need(rel(np.ones(4), np.ones(4)) == 0.0 and rel(np.ones(4), 2 * np.ones(4)) > 0.6, 'rel')
    print('[r-greedy] selftest OK (transport, flip witnesses, H(token), rel)')

def run_gpu(root, out, T292, pre, resume, world, gauges_stage1):
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
        print(f'[r-greedy] gauge cube u{world} present: bit for bit check of the L25 scores per replica', flush=True)
    t0 = time.monotonic()
    print('[r-greedy] Loading Qwen', flush=True)
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
        print(f'[r-greedy] u{world}: 10/10 frozen historical arms bit for bit (writes of the 12 sites)', flush=True)

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

    ck = out / 'S24_CHECKPOINT.npz'
    data = {}
    if resume and ck.exists():
        with np.load(ck, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        print(f'[r-greedy] resume: {len(data)} blocks already acquired', flush=True)
    def save_ck():
        tmp = ck.with_suffix('.tmp.npz')
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, ck)

    nblk = N_ARMS // CHUNK
    replicas = [('IDENTITY', 0)] + GAUGES
    stage1 = replicas if gauges_stage1 else replicas[:1]
    token = clean_by_u[world]
    with torch.inference_mode():
        prep_id = prep_of(world)
        replay_historical(prep_id)
        rng = np.random.default_rng(0)
        probe = rng.choice(N_ARMS, size=16, replace=False)
        diff_w = 0.0; eq_w = 0; eq_s = 0
        for m in probe:
            sw, _ = scores12(prep_id, edits_for(int(m), {}))
            _, _, sf = forward_t(prep_id, token, edits_for(int(m), {}))
            eq_w += int(np.array_equal(sw, sf)); diff_w = max(diff_w, float(np.abs(sw - sf).max()))
            if full is not None:
                eq_s += int(np.array_equal(sf, full['IDENTITY_0'][m]))
        print(f'[r-greedy] full forward == L24-L25 window (16 masks): {eq_w}/16 (max deviation {diff_w:.3g})'
              + (f' ; == gauge cube: {eq_s}/16' if full is not None else ''), flush=True)
        need(eq_w == 16 and (full is None or eq_s == 16), 'The full forward does not reproduce the C_L24 interventions bit for bit')

        for kind, seed in stage1:
            key = f'{kind}_{seed}'
            if all(f'{key}__tok__{c}' in data for c in range(nblk)):
                continue
            fac = {} if kind == 'IDENTITY' else gauge_factors(kind, seed, 24, nkv, hd, dff)
            cm = contextlib.nullcontext() if kind == 'IDENTITY' else Q.Gauged(model, 24, kind, seed)
            with cm:
                prep = prep_id if kind == 'IDENTITY' else prep_of(world)
                for c in range(nblk):
                    if f'{key}__tok__{c}' in data:
                        continue
                    toks = np.zeros(CHUNK, np.int64); tid = np.zeros((CHUNK, TOPK), np.int64)
                    tlp = np.zeros((CHUNK, TOPK), np.float32); sc = np.zeros((CHUNK, 12))
                    for i in range(CHUNK):
                        _, logits, s = forward_t(prep, token, edits_for(int(MASK12[c * CHUNK + i]), fac))
                        lp = torch.log_softmax(logits, -1); v, ix = lp.topk(TOPK)
                        toks[i] = int(ix[0]); tid[i] = ix.cpu().numpy(); tlp[i] = v.cpu().numpy(); sc[i] = s
                    data.update({f'{key}__tok__{c}': toks, f'{key}__topid__{c}': tid, f'{key}__toplp__{c}': tlp, f'{key}__sc__{c}': sc})
                    save_ck()
                    seen = np.unique(np.concatenate([data[f'{key}__tok__{b}'] for b in range(c + 1) if f'{key}__tok__{b}' in data]))
                    print(f'[r-greedy] stage 1 {key} block {c + 1}/{nblk} ; distinct tokens: {len(seen)} ; {time.monotonic() - t0:.0f} s', flush=True)
        tok_by = {f'{k}_{s}': np.concatenate([data[f'{k}_{s}__tok__{c}'] for c in range(nblk)]) for k, s in stage1}
        sc_by = {f'{k}_{s}': np.concatenate([data[f'{k}_{s}__sc__{c}'] for c in range(nblk)]) for k, s in stage1}
        tokens = tok_by['IDENTITY_0']
        cube_eq = {}
        if full is not None:
            for k in sc_by:
                cube_eq[k] = bool(np.array_equal(sc_by[k], full[k]))
            print(f'[r-greedy] L25 scores of the full forward == gauge cube, per replica: {cube_eq}', flush=True)

        c0, logits0, _ = forward_t(prep_id, token, None)
        tok_clean = int(logits0.argmax())
        distinct = sorted(set(np.concatenate(list(tok_by.values())).tolist()) | {tok_clean})
        print(f'[r-greedy] clean token {tok_clean} ; distinct tokens: {len(distinct)} -> {distinct[:20]}', flush=True)
        if len(distinct) > 64:
            print('[r-greedy] more than 64 distinct tokens: long stage 2 (one forward per token and per replica)', flush=True)
        if 'distinct_tokens' in data:
            need(data['distinct_tokens'].tolist() == distinct, 'Distinct tokens differ from the resumed run')
        data['distinct_tokens'] = np.array(distinct, np.int64)
        W1 = {}; clean_by_gauge = {}
        for kind, seed in replicas:
            key = f'{kind}_{seed}'
            if f'{key}__W1' in data:
                W1[key] = data[f'{key}__W1']; clean_by_gauge[key] = int(data[f'{key}__clean']); continue
            cm = contextlib.nullcontext() if kind == 'IDENTITY' else Q.Gauged(model, 24, kind, seed)
            with cm:
                prep = prep_id if kind == 'IDENTITY' else prep_of(world)
                cg, lg, _ = forward_t(prep, token, None)
                clean_by_gauge[key] = int(lg.argmax())
                W1[key] = np.stack([forward_t1(prep, cg, tk) for tk in distinct])
                del cg
            data[f'{key}__W1'] = W1[key]; data[f'{key}__clean'] = np.array(clean_by_gauge[key]); save_ck()
            print(f'[r-greedy] stage 2 {key}: L24(t+1) for {len(distinct)} tokens ; clean token {clean_by_gauge[key]} ; {time.monotonic() - t0:.0f} s', flush=True)

        changed = np.flatnonzero(tokens != tok_clean)[:N_LEMMA_MAX]
        same_tok = np.flatnonzero(tokens == tok_clean)
        others = rng.choice(same_tok, size=min(N_LEMMA_EXTRA, len(same_tok)), replace=False) if len(same_tok) else np.array([], int)
        idx_of = {tk: i for i, tk in enumerate(distinct)}
        lem_a = lem_b = n_lem = 0
        for m in np.concatenate([changed, others]).astype(int):
            cp, lg, _ = forward_t(prep_id, token, edits_for(int(m), {}))
            need(int(lg.argmax()) == int(tokens[m]), f'Token not reproducible for mask {m}')
            wa = forward_t1(prep_id, cp, int(tokens[m])); wb = forward_t1(prep_id, cp, tok_clean)
            del cp
            lem_a += int(np.array_equal(wa, W1['IDENTITY_0'][idx_of[int(tokens[m])]]))
            lem_b += int(np.array_equal(wb, W1['IDENTITY_0'][idx_of[tok_clean]]))
            n_lem += 1
        print(f'[r-greedy] lemma: (a) arm cache + its token == clean cache + same token: {lem_a}/{n_lem} ; '
              f'(b) arm cache + clean token == clean: {lem_b}/{n_lem}', flush=True)

    names_g = [f'{k}_{s}' for k, s in replicas]
    W1s = np.stack([W1[k] for k in names_g])
    d_t1 = np.array([max(rel(W1s[g, t, r], W1s[0, t, r]) for g in range(1, len(names_g)) for t in range(len(distinct))) for r in range(12)])
    sep = np.zeros((len(distinct), len(distinct), 12), bool)
    for i_ in range(len(distinct)):
        for j_ in range(len(distinct)):
            for r in range(12):
                sep[i_, j_, r] = rel(W1s[0, i_, r], W1s[0, j_, r]) > d_t1[r]
    final = out / f'S24_R_RAW_u{world}.npz'
    topid = np.concatenate([data[f'IDENTITY_0__topid__{c}'] for c in range(nblk)])
    toplp = np.concatenate([data[f'IDENTITY_0__toplp__{c}'] for c in range(nblk)])
    extra = {}
    if gauges_stage1:
        extra['tokens_by_gauge'] = np.stack([tok_by[k] for k in names_g])
        extra['scores_L25_by_gauge'] = np.stack([sc_by[k] for k in names_g])
    if final.exists():
        final.unlink()
    with open(final, 'xb') as f:
        np.savez_compressed(f, mask12=MASK12, tokens=tokens, top_ids=topid, top_logprobs=toplp, scores_L25=sc_by['IDENTITY_0'],
                            clean_token=np.array(tok_clean), distinct_tokens=np.array(distinct, np.int64), W1_L24_t1=W1s,
                            gauges=np.array(names_g), sites=np.asarray(T286.ALL12), delta_t1=d_t1, token_separation=sep, **extra)
    fw = flip_witnesses(tokens)
    lemma_ok = bool(n_lem > 0 and lem_a == n_lem and lem_b == n_lem)
    gauge_clean_ok = all(v == tok_clean for v in clean_by_gauge.values())
    bits = list(T286.NINE) + list(T286.BACKGROUND)
    res = {'status': 'PASS_R_GREEDY_RETURN_ACQUISITION' if lemma_ok else 'R_GREEDY_LEMMA_CHECK_FAILED',
           'world': world, 'masks': N_ARMS, 'active_token_t': token, 'clean_token_t_plus_1': tok_clean,
           'clean_token_per_gauge': clean_by_gauge, 'clean_token_identical_under_gauges': gauge_clean_ok,
           'distinct_tokens': distinct, 'token_counts': {str(t): int((tokens == t).sum()) for t in distinct},
           'masks_with_token_different_from_clean': int((tokens != tok_clean).sum()),
           'H_token_bits_uniform_4096_masks': token_entropy(tokens),
           'token_flip_fraction_per_mask_bit': dict(zip(bits, fw)),
           'delta_L24_t1_per_site': dict(zip(T286.ALL12, d_t1.tolist())),
           'token_pairs_separated_at_L24_t1_per_site': {T286.ALL12[r]: int(sep[:, :, r].sum() // 2) for r in range(12)},
           'lemma': {'sample': n_lem, 'a_arm_cache_plus_own_token_equals_clean_cache': lem_a,
                     'b_arm_cache_plus_clean_token_equals_clean': lem_b, 'valid': lemma_ok},
           'L25_scores_equal_gauge_cube_per_replica': cube_eq,
           'raw_sha256': sha(final), 'gpu_seconds': round(time.monotonic() - t0, 1), 'R_computed': False}
    with open(out / 'RESULT.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"[r-greedy] distinct tokens = {len(distinct)} ; masks whose token differs from the clean one = {res['masks_with_token_different_from_clean']} ; "
          f"H(token) = {res['H_token_bits_uniform_4096_masks']:.4f} bit", flush=True)
    print('[r-greedy] token flip fraction per mask bit:', {k: round(v, 4) for k, v in res['token_flip_fraction_per_mask_bit'].items()}, flush=True)
    print('[r-greedy] FINAL', res['status'], '; RESULT', out / 'RESULT.json', flush=True)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selftest', action='store_true'); g.add_argument('--preflight', action='store_true')
    g.add_argument('--gpu', action='store_true')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26); ap.add_argument('--gauges-stage1', action='store_true')
    a = ap.parse_args()
    need(sys.flags.isolated, 'Run with python -I -u')
    selftest()
    if a.selftest:
        return 0
    root = a.root.resolve()
    T292 = import_pin(root / 'T292_SAME_CODE_MICRO_DONOR_FIBERS_GPU.py', 't292_pinned', PIN_T292)
    pre = T292.preflight(root)
    print('[r-greedy] PREFLIGHT PASS', flush=True)
    if a.preflight:
        return 0
    out = root / 'FAST_GPU_RESULTS' / NAME.format(w=a.world)
    if out.exists() and not a.resume:
        raise SystemExit('Output folder already exists: rerun with --resume, or move it: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'GPU_LIVE.log', 'a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(T292.Tee(sys.stdout, log)), contextlib.redirect_stderr(T292.Tee(sys.stderr, log)):
            try:
                run_gpu(root, out, T292, pre, a.resume, a.world, a.gauges_stage1); return 0
            except BaseException:
                traceback.print_exc(); print('[r-greedy] STOPPED: rerun with --gpu --resume', flush=True); return 2

if __name__ == '__main__':
    raise SystemExit(main())
