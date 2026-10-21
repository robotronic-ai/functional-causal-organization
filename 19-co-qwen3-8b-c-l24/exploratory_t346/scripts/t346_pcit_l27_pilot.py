#!/usr/bin/env python3
"""T346 pilot, PCI_T_DS_L27 branch: directional specificity at source layer 27 under an mlp0@L24 activation attenuation, five arms, one world (u26).

Arms: U26_BASELINE (no hook), U26_SHAM (same hook code path, dose 0), U26_ABLATE_025 / _050 / _100 (activation of down_proj input columns 0..3071 of layer 24
multiplied by 1 - dose). The instrument is the frozen PCI_T v0.20/v0.21 machinery reused as is (core runtime, tensor contract, ST scorer, direction bank and
directional metrics of the pinned X5.1 runner): source layer 27, amplitude 0.20, source-token fractions 0.20 and 0.40, 1 designated + 15 orthogonal RMS-matched
directions per site, horizon 32, K 8, D 64, 16 logical nulls. Deviations from the historical protocol (all recorded in RUN_HEADER.json): dtype is an argument
(default bfloat16, history is float16); the six episodes are new; the direction bank is derived once from the NO_HOOK baseline and held fixed in every arm;
seeds come from the T346 streams (directions 2026101803, bootstrap 2026101804).
Single file: the hook (copied unchanged from l24_mlp0_ablation_hook.py) and the six episodes are embedded; every input SHA is recorded in RUN_HEADER.json.
Modes:
  --selftest            CPU, tiny random 36-layer Qwen3, no GPU, no Qwen weights: full pipeline, SHAM identity, doses, resume, analysis, refusals
  --run                 GPU (needs --authorize-gpu-run): acquisition; --probe = 1 episode x {BASELINE, SHAM} timing probe; --arms/--episodes for a subset; --resume
  --analyze             CPU: analysis of an existing records.jsonl
Nothing here estimates a duration: every site records its own wall time. No inferential claim: one world, six episodes, descriptive paired differences.
"""
import argparse, copy, csv, hashlib, importlib.util, json, math, os, sys, tempfile, time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np

WORK = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
OUTNAME = 'FAST_GPU_RESULTS/CLAUDE_T346_PCIT_L27_PILOT'
V20_LAUNCHER_NAME = 'qwen3_8b_pcit_v020_r01.py'
V20_ZIP_REL = 'exports/pcit_qwen3_8b/R01_PCIT_V020/PCI_T_QWEN3_8B_RESULTS_v0.20.zip'
X1_NAME = 'qwen3_8b_pcit_multiamplitude_dev_v021x1.py'
X5_NAME = 'qwen3_8b_pcit_directional_dev_v021x5_1.py'
PINNED = {
    'v20_launcher': ('47def0be29eb761d2afc0a3c97faf8114fed4af3a01968803de38b94f4c6e2f7', None),
    'v20_result_zip': ('a1c805e8aa369c92ec670deb52290e3e7f61862d6d31f5277c66fcd7360ae237', None),
    'x1_script': ('3210bc513b7794e6bbe17455415124213b7ded800c13cc345664175a3dd98a8a', X1_NAME),
    'x5_1_script': ('b5072f85504a2ae1829b80e9dcfa719f7a44fbf006e4ffad95c572e9bd121238', X5_NAME),
    'st_module': ('7dff9b84ea8067085a9a33043cda04184e0c3e7fb492dee39387da455b9f927b', None),
}
MODEL_FINGERPRINT = 'e35bdbced08a69c1d7737f0a807b5866041329f9b3f2b462cdced438e2f4d708'
CONFIRM = 'I_AUTHORIZE_T346_PCIT_L27_PILOT_GPU_RUN'

AMPLITUDE, SOURCE_LAYER, FRACTIONS, N_DIRECTIONS = 0.20, 27, (0.20, 0.40), 16
HOOK_LAYER, HOOK_BLOCK, HOOK_POSITIONS = 24, 0, 'ALL_PROCESSED_TOKENS'
DIRECTION_SEED_BASE, BOOTSTRAP_SEED, BOOTSTRAP_REPS = 2026101803, 2026101804, 5000
SPLIT = 'T346_PILOT'
ARMS = (('U26_BASELINE', 'NO_HOOK', 0.0), ('U26_SHAM', 'SHAM', 0.0), ('U26_ABLATE_025', 'ATTENUATE', 0.25),
        ('U26_ABLATE_050', 'ATTENUATE', 0.50), ('U26_ABLATE_100', 'ATTENUATE', 1.0))
DELTA_BAND_PRIMARY = 0.10
SYNTH_METRIC = dict(h_step=32, k_depth=8, d_proj=64, projection_seed=1729, null_count_logical=16)


import torch
import math

MODES = ('NO_HOOK', 'SHAM', 'ATTENUATE')
POSITIONS = ('ALL_PROCESSED_TOKENS', 'LAST_TOKEN_OF_EACH_FORWARD')


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class BlockAttenuator:
    def __init__(self, model, layer=24, block=0, n_blocks=4, dose=0.0, mode='ATTENUATE', positions='ALL_PROCESSED_TOKENS', record=True):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        if positions not in POSITIONS:
            raise ValueError(f'positions must be one of {POSITIONS}')
        if mode == 'ATTENUATE' and not (0.0 < dose <= 1.0):
            raise ValueError('ATTENUATE needs 0 < dose <= 1')
        if mode in ('NO_HOOK', 'SHAM') and dose != 0.0:
            raise ValueError(f'{mode} requires dose = 0')
        dec = model.model if hasattr(model, 'model') else model
        if not 0 <= layer < len(dec.layers):
            raise ValueError(f'layer {layer} outside 0..{len(dec.layers) - 1}')
        self.down = dec.layers[layer].mlp.down_proj
        dff = self.down.weight.shape[1]
        if dff % n_blocks or not 0 <= block < n_blocks:
            raise ValueError('intermediate size must be divisible by n_blocks and block must be in range')
        self.bs = dff // n_blocks; self.lo, self.hi = block * self.bs, (block + 1) * self.bs
        self.layer, self.block, self.dose, self.mode, self.positions, self.record = layer, block, float(dose), mode, positions, record
        self.handle = None; self._w = None; self.calls = 0; self.removed_ss = 0.0; self.removed_n = 0; self.write_ss = 0.0; self.write_n = 0

    def _pre_hook(self, module, args):
        x = args[0]
        scale = 1.0 - self.dose
        y = x.clone()
        sl = (slice(None), slice(-1, None), slice(self.lo, self.hi)) if self.positions == 'LAST_TOKEN_OF_EACH_FORWARD' else (slice(None), slice(None), slice(self.lo, self.hi))
        blk = y[sl]
        removed = blk * self.dose
        if self.record:
            r = removed.detach().float()
            self.removed_ss += float((r * r).sum()); self.removed_n += r.numel()
            if self._w is None or self._w.device != r.device:
                self._w = module.weight[:, self.lo:self.hi].detach().float()
            w = torch.nn.functional.linear(r, self._w)
            self.write_ss += float((w * w).sum()); self.write_n += w.numel()
        y[sl] = blk * scale
        self.calls += 1
        return (y,) + tuple(args[1:])

    def __enter__(self):
        if self.mode != 'NO_HOOK':
            self.handle = self.down.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove(); self.handle = None
        return False

    def summary(self):
        rms = lambda ss, n: math.sqrt(ss / n) if n else None
        return dict(mode=self.mode, layer=self.layer, block=self.block, columns=[self.lo, self.hi], dose=self.dose, positions=self.positions, calls=self.calls,
                    removed_activation_rms=rms(self.removed_ss, self.removed_n), removed_residual_write_rms=rms(self.write_ss, self.write_n),
                    hook_source_sha256=source_sha256())


EPISODES = [
    ('T346P_001',
     'A community swimming pool performs a weekly check of its poolside rescue equipment. Each ring buoy is lifted from its wall bracket, the rope is inspected for fraying, and the reflective tape is wiped and checked. The attached whistle is tested once and the lifeguard initials the sheet. No rescue drill is carried out and no equipment enters the water. After the check, every buoy is hung back on the same bracket, a dated sticker is applied to the rope, and the checklist is filed in the pool office.'),
    ('T346P_002',
     'A bicycle rental shop prepares its helmets for the start of the season. Staff take each helmet from a numbered shelf, read the manufacturing date inside the shell, and press the chin strap buckle to confirm that it locks. Foam padding is examined for cracks but nothing is repaired during the check. Helmets that pass receive a green tag, and helmets that fail are placed in a separate crate. All results are typed into the shop register and the register is saved with the opening-day paperwork.'),
    ('T346P_003',
     'A ferry terminal office reviews the first-aid cabinets located along the passenger deck. An attendant opens each cabinet, compares its contents with a printed list, and records the expiry date printed on every sealed package. Scissors and gloves are counted but not used. If an item is missing, the attendant writes its name in the margin without ordering a replacement. The cabinet is closed, a numbered seal is fastened across the latch, and the completed list is handed to the terminal supervisor.'),
    ('T346P_004',
     'A public library inspects the microfilm readers in its local history room at the end of each month. A volunteer switches on every reader, confirms that the lamp lights, and turns the focus knob through its full range. A blank test reel is threaded once to check that the spool rotates freely. No archive material is viewed and no copies are made. Each reader is switched off, covered with a fabric sheet, and the volunteer records the serial number and the date on the maintenance card kept beside the door.'),
    ('T346P_005',
     'A school gymnasium undergoes an annual review of its wall-mounted fire extinguishers. A caretaker lifts each unit from its hook, reads the pressure gauge, and checks that the safety pin and tamper seal are intact. The hose is examined for cracks and the label is compared with the register of serial numbers. No extinguisher is discharged. Every unit is hung back at the marked height, the inspection tag is initialed, and the register page is returned to the safety folder in the administration office.'),
    ('T346P_006',
     'A regional bus garage conducts a morning audit of the destination display units installed in its parked vehicles. A technician steps aboard each bus, powers the display for a few seconds, and confirms that the stored route names scroll in the correct order. Brightness is compared against a printed reference card. The wiring cabinet is not opened and no route data is changed. The technician switches the unit off, records the bus number and the result on a clipboard form, and delivers the forms to the dispatch desk.'),
]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def load_module(name, path, expected=None):
    if expected and sha_file(path) != expected:
        raise SystemExit(f'SHA mismatch for {path}: restore it from the backup')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def load_instruments(work, runtime_dir=None):
    for rel in (V20_LAUNCHER_NAME, X1_NAME, X5_NAME):
        if not (work / rel).is_file():
            raise SystemExit(f'File missing: {work / rel} ; restore it from the backup')
    x1 = load_module(Path(X1_NAME).stem, work / X1_NAME, PINNED['x1_script'][0])
    x5 = load_module(Path(X5_NAME).stem, work / X5_NAME, PINNED['x5_1_script'][0])
    v20 = x1.import_v20_launcher(work / V20_LAUNCHER_NAME)
    if runtime_dir is not None:
        v20.RUNTIME_DIR = Path(runtime_dir)
    core, tc = v20.import_core_modules()
    st = Path(v20.RUNTIME_DIR) / 'pci_t_st_adapted_v0_7.py'
    if sha_file(st) != PINNED['st_module'][0]:
        raise SystemExit('ST module SHA differs from the frozen one')
    return dict(x1=x1, x5=x5, v20=v20, core=core, tc=tc, st_path=st)


def live_hashes(work, ins):
    d = dict(script=sha_file(Path(__file__)), v20_launcher=sha_file(work / V20_LAUNCHER_NAME), x1_script=sha_file(work / X1_NAME),
             x5_1_script=sha_file(work / X5_NAME), st_module=sha_file(ins['st_path']), episodes_embedded=sha_bytes(canon(EPISODES)))
    if (work / V20_ZIP_REL).is_file():
        d['v20_result_zip'] = sha_file(work / V20_ZIP_REL)
    return d


def site_keys(episode_ids):
    return [(eid, f) for eid in episode_ids for f in FRACTIONS]


def direction_seed(episode_index, fraction):
    return DIRECTION_SEED_BASE + 1000 * episode_index + 10 * FRACTIONS.index(fraction)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


class Pilot:
    def __init__(self, ins, model, engine, device, tokenized, metric, layer_rms, positions=HOOK_POSITIONS):
        self.i, self.model, self.engine, self.device = ins, model, engine, device
        self.metric, self.layer_rms, self.positions = metric, np.asarray(layer_rms, dtype=float), positions
        self.ids = ins['core'].tensorized_episode_map(tokenized, SPLIT, device)
        self.eids = [r['episode_id'] for r in tokenized['episodes'] if r['split'] == SPLIT]
        self.H, self.K, self.D = int(metric['h_step']), int(metric['k_depth']), int(metric['d_proj'])
        self.pseed, self.nnull = int(metric['projection_seed']), int(metric['null_count_logical'])
        if self.layer_rms.shape != (engine.n_layers,) or not np.all(np.isfinite(self.layer_rms)) or np.any(self.layer_rms <= 0):
            raise RuntimeError('layer RMS vector invalid')
        self.obs = ins['core'].causal_observation_depth_indices(engine.n_layers, SOURCE_LAYER, self.K)
        self.base = {}

    def geometry(self, eid, fraction):
        ids = self.ids[eid]; n = int(ids.shape[1]); pos = self.i['core'].source_token_index(n, fraction)
        if pos + self.H >= n:
            raise RuntimeError(f'horizon does not fit {eid} fraction={fraction}')
        return ids, pos

    def make_hook(self, mode, dose):
        return BlockAttenuator(self.model, layer=HOOK_LAYER, block=HOOK_BLOCK, dose=dose, mode=mode, positions=self.positions)

    def derive_directions(self):
        """Direction banks from the NO_HOOK baseline; reused unchanged in every arm."""
        core, x5 = self.i['core'], self.i['x5']
        for ei, eid in enumerate(self.eids):
            for f in FRACTIONS:
                ids, pos = self.geometry(eid, f)
                with self.make_hook('NO_HOOK', 0.0):
                    src, _ = core.baseline_site_capture(self.engine, ids, SOURCE_LAYER, pos, self.obs, self.H)
                s_rms, endo = x5.source_state_rms_unit(src)
                bank = x5.rms_orthogonal_bank(endo, direction_seed(ei, f), N_DIRECTIONS)
                self.base[(eid, f)] = dict(src=src.clone(), s_rms=s_rms, bank=bank, pos=int(pos), seed=direction_seed(ei, f),
                                           bank_sha256=sha_bytes(bank.numpy().tobytes()))
        return {f'{k[0]}|{k[1]:.2f}': dict(seed=v['seed'], source_position=v['pos'], source_rms=v['s_rms'], bank_sha256=v['bank_sha256']) for k, v in self.base.items()}

    def measure(self, arm, mode, dose, eid, f):
        core, tc, x5 = self.i['core'], self.i['tc'], self.i['x5']
        b = self.base[(eid, f)]; ids, pos = self.geometry(eid, f)
        t0 = time.perf_counter()
        with self.make_hook(mode, dose) as hk:
            src, panel = core.baseline_site_capture(self.engine, ids, SOURCE_LAYER, pos, self.obs, self.H)
            nullp = np.broadcast_to(panel, (self.nnull,) + panel.shape)
            scores, ret = [], []
            for d in range(N_DIRECTIONS):
                delta = float(AMPLITUDE) * b['s_rms'] * b['bank'][d]
                plus, minus = core.perturbed_pair_capture(self.engine, ids, SOURCE_LAYER, pos, self.obs, self.H, delta)
                bundle = tc.build_metric_ready_tensors(plus, minus, nullp, nullp, AMPLITUDE, self.layer_rms[list(self.obs)], self.obs, self.pseed, self.D)
                sc = core.score_metric_ready_tensors(bundle)
                scores.append(float(sc.pci_t_st_adapted)); ret.append(int(sc.n_retained_components))
            summ = hk.summary()
        if self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        valid = bool(np.all(np.isfinite(scores)))
        met = x5.directional_metrics(scores) if valid else {}
        diff = (src.double() - b['src'].double()).numpy()
        return dict(arm=arm, mode=mode, dose=dose, episode_id=eid, source_token_fraction=f, source_position=b['pos'], valid=valid,
                    direction_scores=scores, retained_components=ret, source_enrichment=met.get('source_enrichment'), source_score=met.get('source_score'),
                    control_median=met.get('control_median'), source_win_rate=met.get('source_win_rate'), bank_sha256=b['bank_sha256'],
                    panel_sha256=sha_bytes(np.ascontiguousarray(panel).tobytes()), source_rel_change_rms=float(rms(diff) / b['s_rms']),
                    hook_removed_activation_rms=summ['removed_activation_rms'], hook_removed_residual_write_rms=summ['removed_residual_write_rms'],
                    hook_calls=summ['calls'], seconds=round(sec, 3))


def read_records(path):
    out = []
    if Path(path).is_file():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def append_record(path, rec):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())


def run_pilot(p, out, header, arms=ARMS, n_episodes=None, resume=False, stop_if_sham_differs=True, log=print):
    out = Path(out); rec_path = out / 'records.jsonl'; hdr_path = out / 'RUN_HEADER.json'
    if out.exists() and not resume:
        raise SystemExit('Output folder already exists: move it or use --resume: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); dirs = p.derive_directions(); t_dir = time.perf_counter() - t0
    header = dict(header, directions=dirs, seconds_direction_derivation=round(t_dir, 2))
    if hdr_path.exists():
        old = json.loads(hdr_path.read_text(encoding='utf-8'))
        if old.get('directions') != dirs or old.get('provenance') != header.get('provenance'):
            raise SystemExit('RUN_HEADER differs from this session (directions or provenance): refuse to resume')
    else:
        with open(hdr_path, 'x', encoding='utf-8') as f:
            json.dump(header, f, indent=2, sort_keys=True)
    eids = p.eids[:n_episodes] if n_episodes else p.eids
    done = {(r['arm'], r['episode_id'], r['source_token_fraction']) for r in read_records(rec_path)}
    total = len(arms) * len(eids) * len(FRACTIONS); k = len(done)
    for arm, mode, dose in arms:
        for eid, f in site_keys(eids):
            if (arm, eid, f) in done:
                continue
            r = p.measure(arm, mode, dose, eid, f); append_record(rec_path, r); k += 1
            log(f'[T346-PCIT] {k}/{total} {arm} {eid} f={f:.2f} enrich={r["source_enrichment"]} src_rel_change={r["source_rel_change_rms"]:.4g} {r["seconds"]} s')
        if arm == 'U26_SHAM' and stop_if_sham_differs:
            cmp = sham_identity(read_records(rec_path), eids)
            if cmp is not None and not cmp['identical']:
                raise SystemExit('SHAM is not bit-identical to BASELINE (max score diff %g): stop; nothing further is acquired' % cmp['max_abs_score_diff'])
    return read_records(rec_path)


def sham_identity(records, eids=None):
    by = {(r['arm'], r['episode_id'], r['source_token_fraction']): r for r in records}
    pairs = [(by[k], by[('U26_SHAM',) + k[1:]]) for k in by if k[0] == 'U26_BASELINE' and ('U26_SHAM',) + k[1:] in by]
    if not pairs:
        return None
    md = max(max(abs(a - b) for a, b in zip(x['direction_scores'], y['direction_scores'])) for x, y in pairs)
    return dict(n_sites_compared=len(pairs), max_abs_score_diff=float(md), identical=bool(md == 0.0 and all(x['panel_sha256'] == y['panel_sha256'] for x, y in pairs)))


def boot_indices(n, reps):
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(reps, n))


def boot_ci(values, reps):
    v = np.asarray(values, dtype=float)
    if v.size < 2 or reps <= 0:
        return None
    m = np.median(v[boot_indices(v.size, reps)], axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975], method='linear')
    return [float(lo), float(hi)]


def episode_values(records, arm):
    ep = {}
    for r in records:
        if r['arm'] == arm and r['valid']:
            ep.setdefault(r['episode_id'], []).append(r)
    return {e: (float(np.median([x['source_enrichment'] for x in v])), float(np.median([x['source_score'] for x in v])), len(v)) for e, v in ep.items()}


def analyze(records, reps=BOOTSTRAP_REPS):
    arms = [a for a, _, _ in ARMS if any(r['arm'] == a for r in records)]
    dose = {a: d for a, _, d in ARMS}
    per = {a: episode_values(records, a) for a in arms}
    res = dict(use='EXPLORATORY_CALIBRATION_PILOT_u26_DESCRIPTIVE_NO_INFERENCE', instrument_matches_historical_protocol=False,
               deviations=['dtype is an argument (history float16)', 'new episodes', 'direction bank from NO_HOOK baseline fixed across arms', 'T346 seed streams'],
               n_invalid_sites={a: sum(1 for r in records if r['arm'] == a and not r['valid']) for a in arms},
               bootstrap=dict(unit='episode', reps=reps, seed=BOOTSTRAP_SEED, note='descriptive; six episodes of one world; not a significance test'),
               arms={}, paired_vs_sham={}, sham_vs_baseline=sham_identity(records), delta_reduction_valid='NOT_MEASURED_IN_THIS_SCRIPT',
               dose_response_numeric={}, descriptive_bands=dict(PCI_T_DS_L27_abs_ratio=DELTA_BAND_PRIMARY))
    for a in arms:
        prim = [v[0] for v in per[a].values()]; sec = [v[1] for v in per[a].values()]
        rs = [r for r in records if r['arm'] == a]
        res['arms'][a] = dict(dose=dose[a], n_episodes=len(prim), PCI_T_DS_L27=float(np.median(prim)) if prim else None, PCI_T_DS_L27_ci95_descriptive=boot_ci(prim, reps),
                              PCI_T_ST_L27_SOURCE=float(np.median(sec)) if sec else None, PCI_T_ST_L27_SOURCE_ci95_descriptive=boot_ci(sec, reps),
                              episode_values={e: dict(primary=v[0], secondary=v[1], n_fractions=v[2]) for e, v in per[a].items()},
                              median_source_rel_change_rms=float(np.median([r['source_rel_change_rms'] for r in rs])),
                              median_removed_activation_rms=float(np.median([r['hook_removed_activation_rms'] or 0.0 for r in rs])),
                              median_removed_residual_write_rms=float(np.median([r['hook_removed_residual_write_rms'] or 0.0 for r in rs])),
                              median_site_seconds=float(np.median([r['seconds'] for r in rs])), total_site_seconds=float(sum(r['seconds'] for r in rs)))
    if 'U26_SHAM' in per:
        for a in arms:
            if a == 'U26_SHAM':
                continue
            eids = sorted(set(per[a]) & set(per['U26_SHAM']))
            dp = [per[a][e][0] - per['U26_SHAM'][e][0] for e in eids]; ds = [per[a][e][1] - per['U26_SHAM'][e][1] for e in eids]
            if not dp:
                continue
            md = float(np.median(dp))
            res['paired_vs_sham'][a] = dict(n_pairs=len(dp), primary_diff_by_episode=dict(zip(eids, dp)), primary_median_diff=md,
                                            primary_median_diff_ci95_descriptive=boot_ci(dp, reps), n_episodes_diff_le_0=int(sum(1 for x in dp if x <= 0)),
                                            secondary_median_diff=float(np.median(ds)), secondary_median_diff_ci95_descriptive=boot_ci(ds, reps),
                                            abs_median_diff_exceeds_descriptive_band=bool(abs(md) > DELTA_BAND_PRIMARY))
            if a.startswith('U26_ABLATE'):
                res['dose_response_numeric'][f'{dose[a]:.2f}'] = md
    return res


def write_outputs(out, res):
    out = Path(out)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2, sort_keys=True)
    with open(out / 'pilot_table.csv', 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['arm', 'dose', 'episode_id', 'PCI_T_DS_L27_episode', 'PCI_T_ST_L27_SOURCE_episode', 'primary_diff_vs_sham'])
        for a, v in res['arms'].items():
            for e, x in sorted(v['episode_values'].items()):
                d = res['paired_vs_sham'].get(a, {}).get('primary_diff_by_episode', {}).get(e)
                w.writerow([a, v['dose'], e, x['primary'], x['secondary'], d])


def overlap_report(texts, other_files):
    def grams(t, n):
        w = [x.strip('.,;:()').lower() for x in t.split()]
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    hist = []
    for fp in other_files:
        try:
            obj = json.loads(Path(fp).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for e in obj.get('episodes', []) if isinstance(obj, dict) else []:
            if isinstance(e, dict) and isinstance(e.get('text'), str):
                hist.append((Path(fp).name, e.get('episode_id'), e['text']))
    rep = dict(n_historical_texts=len(hist), files=sorted({h[0] for h in hist}), per_episode={}, exact_duplicates=0)
    for eid, t in texts.items():
        g8, g4 = grams(t, 8), grams(t, 4); worst = (0, 0.0, None)
        for fn, hid, ht in hist:
            if ht.strip() == t.strip():
                rep['exact_duplicates'] += 1
            s8 = len(g8 & grams(ht, 8)); j4 = len(g4 & grams(ht, 4)) / max(1, len(g4 | grams(ht, 4)))
            if (s8, j4) > worst[:2]:
                worst = (s8, round(j4, 4), f'{fn}:{hid}')
        rep['per_episode'][eid] = dict(max_shared_8grams=worst[0], max_jaccard_4grams=worst[1], closest=worst[2])
    return rep


def episode_files(work):
    names = []
    for d in (work, work / '.pcit_runtime_v020'):
        if d.is_dir():
            names += [p for p in sorted(d.glob('PCI_T_*EPISODES*.json'))]
    return names


def tokenize_episodes(model_dir, episodes):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True, use_fast=True)
    rows, mx = [], -1
    for e in episodes:
        ids = [int(v) for v in tok(e['text'], add_special_tokens=True, return_attention_mask=False)['input_ids']]
        mx = max(mx, max(ids))
        rows.append(dict(episode_id=e['episode_id'], split=SPLIT, input_ids=ids, token_count=len(ids), text_sha256=sha_bytes(e['text'].encode('utf-8'))))
    return dict(artifact='T346_PCIT_PILOT_TOKENIZED_EPISODES', episodes=rows,
                tokenizer=dict(tokenizer_class=tok.__class__.__name__, vocab_size=int(getattr(tok, 'vocab_size', 0)), len_tokenizer=int(len(tok)), max_observed_token_id=int(mx)))


def load_metric_inputs(work, ins):
    src = ins['x1'].read_result_bundle(work / V20_ZIP_REL)
    camp, cal, frozen = src['PCI_T_QWEN3_8B_LOCAL_CAMPAIGN_v0.17.json'], src['PCI_T_QWEN3_8B_LOCAL_CALIBRATION_RESULT_v0.20.json'], src['PCI_T_QWEN3_8B_LOCAL_SCORE_v0.20.json']
    if frozen.get('model_content_fingerprint_sha256') != MODEL_FINGERPRINT:
        raise SystemExit('model fingerprint in the v0.20 result differs from the frozen one')
    return camp['metric'], np.asarray(cal['layer_rms'], dtype=float), src['PCI_T_QWEN3_8B_LOCAL_PREFLIGHT_v0.17.json']


def episode_dicts():
    return [dict(episode_id=i, text=t, text_sha256=sha_bytes(t.encode('utf-8'))) for i, t in EPISODES]


def do_run(a):
    if a.authorize_gpu_run != CONFIRM:
        raise SystemExit('GPU run refused: pass --authorize-gpu-run ' + CONFIRM)
    work = a.work; ins = load_instruments(work)
    prov = live_hashes(work, ins)
    for k, (exp, _) in PINNED.items():
        if k in prov and prov[k] != exp:
            raise SystemExit(f'{k} differs from its frozen SHA')
    metric, layer_rms, preflight = load_metric_inputs(work, ins)
    ov = overlap_report({i: t for i, t in EPISODES}, episode_files(work))
    if ov['exact_duplicates']:
        raise SystemExit('a pilot episode duplicates a historical episode')
    tokenized = tokenize_episodes(a.model_dir, episode_dicts())
    lock = ins['core'].verify_local_lock(a.model_dir, preflight)
    if lock.get('status') != 'PASS':
        raise SystemExit('local model content lock failed')
    ins['core'].configure_determinism()
    rt = ins['core'].load_campaign_runtime(a.model_dir, 'cuda', a.dtype)
    if rt['model_runtime']['dtype'] != a.dtype:
        raise SystemExit('runtime dtype differs from the requested one')
    p = Pilot(ins, rt['model'], rt['engine'], rt['device'], tokenized, metric, layer_rms)
    arms, n_ep = ARMS, a.episodes
    if a.probe:
        arms = tuple(x for x in ARMS if x[0] in ('U26_BASELINE', 'U26_SHAM')); n_ep = 1
    elif a.arms:
        want = a.arms.split(','); arms = tuple(x for x in ARMS if x[0] in want)
    out = work / (OUTNAME + ('_PROBE' if a.probe else '') + a.out_suffix)
    header = dict(provenance=prov, dtype=a.dtype, probe=bool(a.probe), arms=[x[0] for x in arms], n_episodes=n_ep or len(p.eids), metric=dict(metric),
                  token_counts={r['episode_id']: r['token_count'] for r in tokenized['episodes']}, episode_overlap_check=ov, tokenizer=tokenized['tokenizer'],
                  model_load_seconds=rt['model_runtime'].get('load_seconds'), device=str(rt['device']), hook=dict(layer=HOOK_LAYER, block=HOOK_BLOCK, positions=HOOK_POSITIONS),
                  deviations_from_historical_protocol=['dtype is an argument (history float16)', 'six new episodes', 'direction bank from NO_HOOK baseline fixed across arms',
                                                       'T346 seed streams', 'hook active on every processed token'])
    recs = run_pilot(p, out, header, arms, n_ep, a.resume, not a.continue_if_sham_differs)
    if not a.probe:
        write_outputs(out, analyze(recs, a.bootstrap_reps))
    print('[T346-PCIT] done ; records', out / 'records.jsonl')


def do_analyze(a):
    out = a.work / OUTNAME
    recs = read_records(out / 'records.jsonl')
    if not recs:
        raise SystemExit('No records in ' + str(out))
    write_outputs(out, analyze(recs, a.bootstrap_reps)); print('[T346-PCIT] RESULT', out / 'RESULT.json')


def tiny_model(dtype):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(5)
    cfg = Qwen3Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256, use_cache=False, attn_implementation='eager')
    return Qwen3ForCausalLM(cfg).to(dtype).eval()


def selftest(work):
    import torch
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        ins = load_instruments(work, Path(tmp) / 'rt')
        rng = np.random.default_rng(1)
        tok = dict(episodes=[dict(episode_id=f'S{i}', split=SPLIT, input_ids=[int(x) for x in rng.integers(0, 200, 130)]) for i in range(2)])
        layer_rms = np.ones(36)
        for dt in (torch.float32, torch.bfloat16):
            model = tiny_model(dt); eng = ins['core'].HFQwen3ResidualEngine(model, torch.device('cpu'))
            p = Pilot(ins, model, eng, torch.device('cpu'), tok, SYNTH_METRIC, layer_rms)
            if dt == torch.bfloat16:
                recs = run_pilot(p, Path(tmp) / 'bf16', dict(provenance='x'), ARMS[:2], 1, log=lambda *_: None)
                ident = sham_identity(recs); assert ident['identical'] and ident['n_sites_compared'] == 2, 'SHAM must be bit-identical in bf16'
                continue
            a_out = Path(tmp) / 'full'
            recs = run_pilot(p, a_out, dict(provenance='x'), log=lambda *_: None)
            assert len(recs) == 5 * 4 and all(r['valid'] for r in recs)
            ident = sham_identity(recs); assert ident['identical'] and ident['max_abs_score_diff'] == 0.0, ident
            by = {(r['arm'], r['episode_id'], r['source_token_fraction']): r for r in recs}
            assert len({r['bank_sha256'] for r in recs if r['episode_id'] == 'S0' and r['source_token_fraction'] == 0.2}) == 1, 'direction bank must be fixed across arms'
            chg = {a: np.median([by[(a, e, f)]['source_rel_change_rms'] for e in p.eids for f in FRACTIONS]) for a, _, _ in ARMS}
            assert chg['U26_BASELINE'] == 0.0 and chg['U26_SHAM'] == 0.0 and 0 < chg['U26_ABLATE_025'] < chg['U26_ABLATE_050'] < chg['U26_ABLATE_100'], chg
            assert any(by[('U26_ABLATE_100', e, f)]['direction_scores'] != by[('U26_SHAM', e, f)]['direction_scores'] for e in p.eids for f in FRACTIONS), 'dose 1 must change scores'
            assert all(by[('U26_ABLATE_100', e, f)]['hook_removed_activation_rms'] > by[('U26_ABLATE_050', e, f)]['hook_removed_activation_rms'] > 0 for e in p.eids for f in FRACTIONS)
            assert all(by[('U26_BASELINE', e, f)]['hook_calls'] == 0 and by[('U26_SHAM', e, f)]['hook_calls'] == 17 for e in p.eids for f in FRACTIONS)
            res = analyze(recs, 200); write_outputs(a_out, res)
            assert set(res['paired_vs_sham']) == {'U26_BASELINE', 'U26_ABLATE_025', 'U26_ABLATE_050', 'U26_ABLATE_100'} and res['sham_vs_baseline']['identical']
            assert res['paired_vs_sham']['U26_BASELINE']['primary_median_diff'] == 0.0 and res['arms']['U26_SHAM']['n_episodes'] == 2
            assert res['delta_reduction_valid'] == 'NOT_MEASURED_IN_THIS_SCRIPT' and sorted(res['dose_response_numeric']) == ['0.25', '0.50', '1.00']
            full = read_records(a_out / 'records.jsonl')
            # resume after an interruption reproduces the same records
            r_out = Path(tmp) / 'resume'; r_out.mkdir(); (r_out / 'RUN_HEADER.json').write_text((a_out / 'RUN_HEADER.json').read_text(encoding='utf-8'), encoding='utf-8')
            for r in full[:9]:
                append_record(r_out / 'records.jsonl', r)
            again = run_pilot(p, r_out, dict(provenance='x'), resume=True, log=lambda *_: None)
            strip = lambda rs: [{k: v for k, v in r.items() if k != 'seconds'} for r in rs]
            assert strip(again) == strip(full), 'resume must reproduce the uninterrupted run'
            try:
                run_pilot(p, r_out, dict(provenance='other'), resume=True, log=lambda *_: None); raise AssertionError('accepted a different provenance')
            except SystemExit:
                pass
            try:
                run_pilot(p, a_out, dict(provenance='x'), log=lambda *_: None); raise AssertionError('overwrote an existing run')
            except SystemExit:
                pass
            # a non-identical SHAM stops the run
            broken = copy.deepcopy(recs); [r.update(direction_scores=[s + 1e-9 for s in r['direction_scores']]) for r in broken if r['arm'] == 'U26_SHAM']
            assert not sham_identity(broken)['identical']
        ov = overlap_report({'A': 'x y z ' * 10, 'B': 'completely different words appear here for the check ok'}, [])
        assert ov['exact_duplicates'] == 0
        ov2 = overlap_report({i: t for i, t in EPISODES}, episode_files(work))
        assert ov2['exact_duplicates'] == 0
        if ov2['n_historical_texts'] == 0:
            print('[T346-PCIT] WARNING: no historical episode file found in the work folder or .pcit_runtime_v020: overlap check not performed')
        for k, v in ov2['per_episode'].items():
            print(f'[T346-PCIT] episode {k}: max shared 8-grams {v["max_shared_8grams"]}, max 4-gram Jaccard {v["max_jaccard_4grams"]} ({v["closest"]})')
    print(f'[T346-PCIT] selftest OK in {time.perf_counter() - t_start:.1f} s (CPU, tiny random model: SHAM bit-identical in fp32 and bf16, fixed direction bank across arms, '
          'source change increases with dose, resume reproduces the run, refusal of a different provenance / an existing run, analysis, historical-overlap check)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=WORK); ap.add_argument('--model-dir', type=Path, default=None)
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--run', action='store_true'); ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16'); ap.add_argument('--authorize-gpu-run')
    ap.add_argument('--probe', action='store_true'); ap.add_argument('--out-suffix', default=''); ap.add_argument('--arms'); ap.add_argument('--episodes', type=int); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--continue-if-sham-differs', action='store_true'); ap.add_argument('--bootstrap-reps', type=int, default=BOOTSTRAP_REPS)
    a = ap.parse_args()
    if a.model_dir is None:
        a.model_dir = a.work / 'Qwen3-8B'
    if a.selftest:
        selftest(a.work)
    elif a.run:
        do_run(a)
    elif a.analyze:
        do_analyze(a)
    else:
        ap.error('choose --selftest, --run or --analyze')


if __name__ == '__main__':
    main()
