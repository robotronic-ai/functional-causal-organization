#!/usr/bin/env python3
"""Gauge cube acquisition (GPU) for the C_L24 contract: 12 L24 donor sites x 4096 donor masks x 10 replicas.

Mask bits (12): 0..8 = NINE sites (heads[0-3], heads[4-7], heads[12-15], heads[20-23], heads[28-31], mlp0..mlp3),
9..11 = BACKGROUND sites (heads[8-11], heads[16-19], heads[24-27]). Each site takes one of its two frozen donor values
(frozen donor arrays for NINE, frozen D0 background pairs for BACKGROUND), transported into each exact Q1 recoding.
Readers: the 12 frozen L25 readers (frozen origins and directions; 'identifiable' flag recorded).
Replicas: identity + 9 exact Q1 recodings of layer 24 (UPDOWN_SCALE, VO_DIAG, QKNORM_RECIP; seeds 0, 1, 2).
Guards: the 10 frozen historical arms are replayed bit for bit (identity); in world u26 the identity sub-cube
{512 arms, 6 target readers} equals the pinned reference sub-cube bit for bit; if the reference gauge sub-cubes file is
present, every recoded replica is also checked bit for bit against it.
Checkpoint per (replica, block of 512 masks).

Run (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u acquire_gauge_cube_gpu.py --preflight
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u acquire_gauge_cube_gpu.py --gpu --world 26   (then --gpu --resume)
Writes FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU/{RESULT.json, S9_FULL12_GAUGE_CUBES_u{w}.npz}.
"""
import argparse, contextlib, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
NAME = 'CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU'
REF_GAUGE_NPZ = 'FAST_GPU_RESULTS/CLAUDE_S5_GAUGE_FIBER_COMMUTATION_GPU/S5_GAUGE_CUBES.npz'
REF_GAUGE_RES = 'FAST_GPU_RESULTS/CLAUDE_S5_GAUGE_FIBER_COMMUTATION_GPU/RESULT.json'
CHUNK = 512
PIN_T292 = '64955123f179ff1fc9d798be0d7954a689ce1fadc933c372e14e4a4839423bc3'

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
    """Regenerates EXACTLY the factors of apply_gauge_ (qwen_local_transition) for `layer` (first draw)."""
    import torch
    g = torch.Generator().manual_seed(1009 * seed + len(kind))
    if kind == 'UPDOWN_SCALE':
        return {'d': torch.exp(log_std * torch.randn(dff, generator=g, dtype=torch.float64)).numpy()}
    if kind == 'VO_DIAG':
        return {'o': [torch.exp(log_std * torch.randn(hd, generator=g, dtype=torch.float64)).numpy() for _ in range(nkv)]}
    return {}

def transport(values, cand, fac, hd, group):
    """values: list of arrays (one slice per head / block) of a site; returns the coordinates in the recoded replica."""
    out = []
    for arr, kind, sl in zip(values, [cand['kind']] * len(cand['slices']), cand['slices']):
        a = np.asarray(arr, dtype=np.float64)
        if kind == 'o' and 'o' in fac:
            gi = (sl.start // hd) // group; a = a * fac['o'][gi]
        elif kind == 'd' and 'd' in fac:
            a = a * fac['d'][sl]
        out.append(a.astype(np.float32))
    return out

def selftest():
    import torch
    nkv, hd, dff = 8, 128, 12288
    f1 = gauge_factors('VO_DIAG', 1, 24, nkv, hd, dff); f2 = gauge_factors('VO_DIAG', 1, 24, nkv, hd, dff)
    need(all(np.array_equal(a, b) for a, b in zip(f1['o'], f2['o'])), 'Non-deterministic factors')
    g = torch.Generator().manual_seed(1009 * 1 + len('VO_DIAG'))
    ref = torch.exp(0.3 * torch.randn(hd, generator=g, dtype=torch.float64)).numpy()
    need(np.array_equal(ref, f1['o'][0]), 'Draw order differs from apply_gauge_')
    cand = {'kind': 'o', 'slices': [slice(h * hd, (h + 1) * hd) for h in (28, 29, 30, 31)]}
    v = [np.ones(hd, np.float32) for _ in range(4)]
    t = transport(v, cand, f1, hd, 4)
    need(all(np.allclose(x, f1['o'][7]) for x in t), 'VO_DIAG transport: wrong KV group')
    candd = {'kind': 'd', 'slices': [slice(3072, 6144)]}
    fu = gauge_factors('UPDOWN_SCALE', 0, 24, nkv, hd, dff)
    need(np.allclose(transport([np.ones(3072, np.float32)], candd, fu, hd, 4)[0], fu['d'][3072:6144]), 'UPDOWN transport')
    need(transport(v, cand, {}, hd, 4)[0] is not v[0] and np.array_equal(transport(v, cand, {}, hd, 4)[0], v[0]), 'QKNORM')
    print('[gauge-cube] selftest OK (exact draws, VO_DIAG / UPDOWN / QKNORM transport)')

def run_gpu(root, out, T292, pre, resume, world):
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
    t0 = time.monotonic()
    print('[gauge-cube] Loading Qwen', flush=True)
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
    refg = None
    if world == 26 and (root / REF_GAUGE_RES).is_file():
        r5 = json.loads((root / REF_GAUGE_RES).read_text(encoding='utf-8'))
        need(sha(root / REF_GAUGE_NPZ) == r5['raw_sha256'], 'Reference gauge sub-cubes NPZ missing or SHA mismatch')
        with np.load(root / REF_GAUGE_NPZ, allow_pickle=False) as z:
            refg = {'IDENTITY_0': z['identity'].copy(), **{str(n): z['gauged'][i].copy() for i, n in enumerate(z['gauge_names'])}}

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
        print(f'[gauge-cube] u{world}: 10/10 frozen historical arms bit for bit (writes of the 12 sites)', flush=True)

    sub = {T286.lift_mask(m6) | (bg << 9): (m6, bg) for m6 in range(64) for bg in range(8)}
    ck = out / 'S9_CHECKPOINT.npz'
    data = {}
    if resume and ck.exists():
        with np.load(ck, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        print(f'[gauge-cube] resume: {len(data)} blocks already acquired', flush=True)
    def save_ck():
        tmp = ck.with_suffix('.tmp.npz')
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, ck)

    replicas = [('IDENTITY', 0)] + GAUGES
    with torch.inference_mode():
        for kind, seed in replicas:
            key = f'{kind}_{seed}'
            if all(f'{key}__{c}' in data for c in range(4096 // CHUNK)):
                continue
            fac = {} if kind == 'IDENTITY' else gauge_factors(kind, seed, 24, nkv, hd, dff)
            cm = contextlib.nullcontext() if kind == 'IDENTITY' else Q.Gauged(model, 24, kind, seed)
            with cm:
                prep = prep_of(world)
                if kind == 'IDENTITY':
                    replay_historical(prep)
                for c in range(4096 // CHUNK):
                    kc = f'{key}__{c}'
                    if kc in data:
                        continue
                    arr = np.zeros((CHUNK, 12))
                    for i in range(CHUNK):
                        arr[i], _ = scores12(prep, edits_for(c * CHUNK + i, fac))
                    data[kc] = arr; save_ck()
                    print(f'[gauge-cube] {key} block {c + 1}/{4096 // CHUNK} ; {time.monotonic() - t0:.0f} s', flush=True)
            cube = np.concatenate([data[f'{key}__{c}'] for c in range(4096 // CHUNK)])
            subc = np.zeros((64, 8, 6))
            for m12, (m6, bg) in sub.items():
                subc[m6, bg] = cube[m12, TI]
            if world == 26 and kind == 'IDENTITY':
                need(np.array_equal(subc.view(np.uint64), oldscore.view(np.uint64)), 'Identity sub-cube differs from the pinned reference')
                print('[gauge-cube] identity sub-cube (512 arms, 6 readers) bit for bit equal to the pinned reference', flush=True)
            if refg is not None:
                same = np.array_equal(subc.view(np.uint64), refg[key].view(np.uint64))
                print(f'[gauge-cube] {key}: sub-cube equals the reference gauge sub-cube bit for bit: {same}', flush=True)
                need(same, f'Sub-cube {key} differs from the reference gauge sub-cube')
    cubes = np.stack([np.concatenate([data[f'{k}_{s}__{c}'] for c in range(4096 // CHUNK)]) for k, s in replicas])
    dg = np.abs(cubes[1:] - cubes[0][None]).max((0, 1))
    final = out / f'S9_FULL12_GAUGE_CUBES_u{world}.npz'
    with open(final, 'xb') as f:
        np.savez_compressed(f, cubes=cubes, replica_names=np.array([f'{k}_{s}' for k, s in replicas]),
                            readers=np.asarray(T286.ALL12), identifiable=np.asarray(valid, dtype=bool),
                            mask_bits=np.asarray(list(T286.NINE) + list(T286.BACKGROUND)))
    res = {'status': 'PASS_GAUGE_CUBE_ACQUISITION', 'world': world, 'arms_per_replica': 4096,
           'gauge_delta_12_readers': dict(zip(T286.ALL12, dg.tolist())),
           'identifiable_readers': dict(zip(T286.ALL12, [bool(v) for v in valid])),
           'raw_sha256': sha(final), 'gpu_seconds': round(time.monotonic() - t0, 1), 'formal_Qwen_Gamma_certified': False}
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('[gauge-cube] gauge delta (12 readers):', {k: round(v, 4) for k, v in res['gauge_delta_12_readers'].items()}, flush=True)
    print('[gauge-cube] FINAL', res['status'], '; RESULT', out / 'RESULT.json', flush=True)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--selftest', action='store_true'); g.add_argument('--preflight', action='store_true')
    g.add_argument('--gpu', action='store_true')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--world', type=int, default=26)
    a = ap.parse_args()
    need(sys.flags.isolated, 'Run with python -I -u')
    selftest()
    if a.selftest:
        return 0
    root = a.root.resolve()
    T292 = import_pin(root / 'T292_SAME_CODE_MICRO_DONOR_FIBERS_GPU.py', 't292_pinned', PIN_T292)
    pre = T292.preflight(root)
    print('[gauge-cube] PREFLIGHT PASS (pinned preflight chain)', flush=True)
    if a.preflight:
        return 0
    out = root / 'FAST_GPU_RESULTS' / NAME.format(w=a.world)
    if out.exists() and not a.resume:
        raise SystemExit('Output folder already exists: rerun with --resume, or move it: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'GPU_LIVE.log', 'a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(T292.Tee(sys.stdout, log)), contextlib.redirect_stderr(T292.Tee(sys.stderr, log)):
            try:
                run_gpu(root, out, T292, pre, a.resume, a.world); return 0
            except BaseException:
                traceback.print_exc(); print('[gauge-cube] STOPPED: rerun with --gpu --resume', flush=True); return 2

if __name__ == '__main__':
    raise SystemExit(main())
