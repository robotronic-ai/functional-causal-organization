#!/usr/bin/env python3
"""T346 pilot, FCA_E1 branch: counterfactual hard erasure (FCA-Bench v0.2.7, test E1) under an mlp0@L24 activation attenuation, five arms, one world (u26).

The FCA-Bench package is used unmodified (tasks, prompts, sampling seeds, E1 scoring). The model is loaded by its own HFLocalAdapter with the historical R6
configuration (Qwen3-8B, bitsandbytes 8-bit, float16 compute, sdpa, 20 draws per condition, micro-batch 4, 128 new tokens), which is the default here;
--quantization / --dtype override it and are recorded as deviations. Arms: U26_BASELINE (no hook), U26_SHAM (same hook code path, dose 0),
U26_ABLATE_025 / _050 / _100 (activation of down_proj input columns 0..3071 of layer 24 multiplied by 1 - dose, on every processed token: prefill and each decoding
step). The hook class is imported from t346_pcit_l27_pilot.py (same file, same behaviour) and wrapped so that it also works on int8 layers (no residual-write RMS then).
Task instances: seeds 2026101801 + k (k = 0..19), disjoint from the historical seeds 0..19; each instance is run in every arm with identical sampling seeds, so
SHAM must reproduce BASELINE exactly (checked after every instance; the run stops otherwise).
Modes:
  --selftest   CPU, tiny random 36-layer Qwen3 behind a fake adapter with the same session interface, no GPU, no Qwen: arms, SHAM identity, resume, analysis, refusals
  --run        GPU (needs --authorize-gpu-run): acquisition, instance-major order (each instance in all arms before the next), --resume supported;
               --probe = 1 instance x {BASELINE, SHAM} (timing probe); --arms / --instances for subsets; --out-suffix for a separate folder
  --analyze    CPU: analysis of an existing records.jsonl
Every generate call is timed: the duration is measured, never estimated. No inferential claim: one world, 20 instances, descriptive paired differences.
"""
import argparse, copy, csv, hashlib, importlib.util, json, math, os, sys, tempfile, time
from pathlib import Path

import numpy as np

WORK = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
OUTNAME = 'FAST_GPU_RESULTS/CLAUDE_T346_FCA_E1_PILOT'
PILOT_SCRIPT = 't346_pcit_l27_pilot.py'
CONFIRM = 'I_AUTHORIZE_T346_FCA_E1_PILOT_GPU_RUN'
TASK_SEED, BOOTSTRAP_SEED, BOOTSTRAP_REPS, N_INSTANCES = 2026101801, 2026101804, 5000, 20
HOOK_LAYER, HOOK_BLOCK, HOOK_POSITIONS = 24, 0, 'ALL_PROCESSED_TOKENS'
BAND_E1 = 0.05
ARMS = (('U26_BASELINE', 'NO_HOOK', 0.0), ('U26_SHAM', 'SHAM', 0.0), ('U26_ABLATE_025', 'ATTENUATE', 0.25),
        ('U26_ABLATE_050', 'ATTENUATE', 0.50), ('U26_ABLATE_100', 'ATTENUATE', 1.0))
DETAIL_KEYS = ('distributional_effect', 'residual_effect', 's_distributional', 's_direct', 'poison_direct_leak', 'erase_direct_leak', 'leak_reinjection_diagnostic')


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


HISTORICAL_R6_CONFIG = {"backend": "hf", "required_runner_id": "FCA027_GPU_R6", "system_id": "qwen3-8b-standard-bnb8-gpu-only",
                        "name": "Qwen3-8B standard Transformer (bnb8 GPU-only; FCA v0.2.7)", "model_path": "./Qwen3-8B", "device_map": None, "dtype": "float16",
                        "trust_remote_code": True, "max_new_tokens": 128, "temperature": 0.0, "sample_temperature": 0.8, "top_p": 0.95, "enable_thinking": False,
                        "seed": 1234, "e1_samples": 20, "e1_batch_size": 4, "lazy_context": True, "local_files_only": True, "progress": True, "strict_gpu": True,
                        "gpu_index": 0, "quantization": "bnb8", "attn_implementation": "sdpa"}
SKIP = {'.git', '__pycache__', 'node_modules', 'site-packages', '.idea', 'exports', 'FAST_GPU_RESULTS'}


def find_fca_dir(work, given):
    """Folder that contains the fca_bench package (the config file is optional: the historical R6 configuration is embedded)."""
    if given:
        c = Path(given)
        for d in (c, c / 'FCA_BENCH', c / '17-fca-bench' / 'FCA_BENCH'):
            if (d / 'fca_bench' / '__init__.py').is_file():
                return d.resolve()
        raise SystemExit(f'No fca_bench/__init__.py under {given}')
    found = []
    for root in (work, work.parent):
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                entries = sorted(d.iterdir())
            except OSError:
                continue
            if (d / 'fca_bench' / '__init__.py').is_file():
                found.append(d.resolve())
            if depth < 5:
                stack += [(e, depth + 1) for e in entries if e.is_dir() and e.name != 'fca_bench' and e.name not in SKIP and not e.name.startswith(('.venv', 'venv'))]
    found = list(dict.fromkeys(found))
    if not found:
        raise SystemExit('fca_bench/ package not found under ' + str(work) + ' or ' + str(work.parent) + ': pass --fca-dir <folder that contains fca_bench/>')
    info = [(d, fca_build_of(d)) for d in found]
    good = [d for d, (v, r) in info if v == '0.2.7' and r == 'FCA027_GPU_R6']
    print('[T346-FCA] fca_bench folders found:', *[f'{d} version={v} runner={r}' for d, (v, r) in info], sep='\n  ', flush=True)
    if not good:
        zips = find_fca_zips(work)
        msg = 'No fca_bench build 0.2.7 / FCA027_GPU_R6 among the folders above.'
        if zips:
            msg += '\nArchives holding that build were found. Run, from ' + str(work) + ':\n  python -I -u t346_restore_fca_v027.py --zip "' + str(zips[0]) + '"\nthen run this script again.'
            if len(zips) > 1:
                msg += '\nOther candidates: ' + '; '.join(map(str, zips[1:]))
        else:
            msg += ' No archive holding it was found near the working folder: restore it from the backup (t346_restore_fca_v027.py --zip PATH) or pass --fca-dir.'
        raise SystemExit(msg)
    return good[0]


def find_fca_zips(work):
    import re
    import zipfile
    dirs = [work, work.parent] + sorted(Path('/mnt/c/Users').glob('*/Downloads')) + sorted(Path('/mnt/c/Users').glob('*/Desktop')) + sorted(Path('/mnt/c/Users').glob('*/Documents'))
    out = []
    for d in dirs:
        try:
            zs = sorted(d.glob('*.zip'))
        except OSError:
            continue
        for zp in zs:
            try:
                with zipfile.ZipFile(zp) as z:
                    for n in z.namelist():
                        if n.endswith('fca_bench/__init__.py'):
                            t = z.read(n).decode('utf-8', 'replace')
                            if re.search(r'__runner_id__\s*=\s*[\'"]FCA027_GPU_R6', t):
                                out.append(zp.resolve()); break
            except Exception:
                continue
    return list(dict.fromkeys(out))


def fca_build_of(d):
    import re
    t = (d / 'fca_bench' / '__init__.py').read_text(encoding='utf-8', errors='replace')
    v = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', t); r = re.search(r'__runner_id__\s*=\s*[\'"]([^\'"]+)', t)
    return (v.group(1) if v else None, r.group(1) if r else None)


def fca_tree_sha(fca_dir):
    h = hashlib.sha256(); files = sorted([p for p in list((fca_dir / 'fca_bench').glob('*.py')) + [fca_dir / 'config.qwen3_8b_gpu8.json', fca_dir / 'run_qwen3_8b_gpu.py'] if p.is_file()])
    for p in files:
        h.update(p.name.encode()); h.update(sha_file(p).encode())
    return h.hexdigest(), {p.name: sha_file(p) for p in files}


def import_fca(fca_dir):
    if str(fca_dir) not in sys.path:
        sys.path.insert(0, str(fca_dir))
    import fca_bench
    from fca_bench import REGISTRY
    if getattr(fca_bench, '__version__', None) != '0.2.7' or getattr(fca_bench, '__runner_id__', None) != 'FCA027_GPU_R6':
        raise SystemExit(f'Wrong FCA build in {fca_dir}: {getattr(fca_bench, "__version__", None)} {getattr(fca_bench, "__runner_id__", None)}; expected 0.2.7 FCA027_GPU_R6')
    return fca_bench, REGISTRY


def make_attenuator_class(base):
    class FcaAttenuator(base):
        """Same hook, but the residual-write RMS is skipped when the layer is int8 (its weight slice is not a float matrix)."""

        def __init__(self, *a, record_write=True, **k):
            super().__init__(*a, **k); self.record_write = bool(record_write)

        def _pre_hook(self, module, args):
            if self.record_write or not self.record:
                return super()._pre_hook(module, args)
            x = args[0]; y = x.clone()
            sl = (slice(None), slice(-1, None), slice(self.lo, self.hi)) if self.positions == 'LAST_TOKEN_OF_EACH_FORWARD' else (slice(None), slice(None), slice(self.lo, self.hi))
            blk = y[sl]; removed = blk * self.dose
            r = removed.detach().float(); self.removed_ss += float((r * r).sum()); self.removed_n += r.numel()
            y[sl] = blk * (1.0 - self.dose); self.calls += 1
            return (y,) + tuple(args[1:])
    return FcaAttenuator


class GenStats:
    def __init__(self, model):
        self.calls = 0; self.seconds = 0.0; self.new_tokens = 0; self.model = model; self.orig = model.generate
        outer = self

        def timed(*a, **k):
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter(); out = outer.orig(*a, **k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outer.seconds += time.perf_counter() - t; outer.calls += 1
            try:
                outer.new_tokens += int(out.shape[0] * (out.shape[1] - a[0].shape[1]))
            except Exception:
                pass
            return out
        model.generate = timed

    def reset(self):
        self.calls = 0; self.seconds = 0.0; self.new_tokens = 0

    def snapshot(self):
        return dict(generate_calls=self.calls, generate_seconds=round(self.seconds, 3), generated_tokens=self.new_tokens)


def rec_to_dict(r):
    return dict(probe_id=r.probe_id, raw=r.raw, parsed=r.parsed, arm=r.arm, meta=r.meta)


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


def raws(rec):
    return [r['raw'] for r in rec['records']]


def sham_identity(records):
    by = {(r['arm'], r['instance_id']): r for r in records}
    pairs = [(by[k], by[('U26_SHAM', k[1])]) for k in by if k[0] == 'U26_BASELINE' and ('U26_SHAM', k[1]) in by]
    if not pairs:
        return None
    n_diff = sum(1 for x, y in pairs if raws(x) != raws(y) or x['result']['score'] != y['result']['score'])
    return dict(n_instances_compared=len(pairs), n_instances_different=n_diff, identical=n_diff == 0)


def run_pilot(rt, out, header, arms=ARMS, n_instances=N_INSTANCES, resume=False, stop_if_sham_differs=True, log=print):
    out = Path(out); rec_path = out / 'records.jsonl'; hdr_path = out / 'RUN_HEADER.json'
    if out.exists() and not resume:
        raise SystemExit('Output folder already exists: move it or use --resume: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    if hdr_path.exists():
        if json.loads(hdr_path.read_text(encoding='utf-8')).get('provenance') != header.get('provenance'):
            raise SystemExit('RUN_HEADER provenance differs from this session: refuse to resume')
    else:
        with open(hdr_path, 'x', encoding='utf-8') as f:
            json.dump(header, f, indent=2, sort_keys=True)
    adapter, test, Att, stats = rt['adapter'], rt['test'], rt['Att'], rt['stats']
    done = {(r['arm'], r['instance_id']) for r in read_records(rec_path)}
    total = len(arms) * n_instances; k = len(done); t_all = time.perf_counter()
    for i in range(n_instances):
        inst = test.generate(TASK_SEED + i)
        for arm, mode, dose in arms:
            if (arm, inst.id) in done:
                continue
            stats.reset(); t0 = time.perf_counter()
            with Att(adapter.model, layer=HOOK_LAYER, block=HOOK_BLOCK, dose=dose, mode=mode, positions=HOOK_POSITIONS, record_write=rt['record_write']) as hk:
                recs = test.run(inst, adapter)
                summ = hk.summary()
            res = test.score(inst, recs)
            rec = dict(arm=arm, mode=mode, dose=dose, instance_id=inst.id, seed=inst.seed, result=res.__dict__, records=[rec_to_dict(r) for r in recs],
                       gen=stats.snapshot(), seconds=round(time.perf_counter() - t0, 2), hook_calls=summ['calls'], hook_removed_activation_rms=summ['removed_activation_rms'],
                       hook_removed_residual_write_rms=summ['removed_residual_write_rms'])
            append_record(rec_path, rec); k += 1
            log(f'[T346-FCA] {k}/{total} {arm} {inst.id} score={res.score:.4f} valid={res.valid} gen_calls={rec["gen"]["generate_calls"]} {rec["seconds"]} s')
        if stop_if_sham_differs:
            ident = sham_identity([r for r in read_records(rec_path) if r['instance_id'] == inst.id])
            if ident is not None and not ident['identical']:
                raise SystemExit('SHAM output differs from BASELINE on ' + inst.id + ': stop; nothing further is acquired')
    log(f'[T346-FCA] acquisition loop finished in {time.perf_counter() - t_all:.1f} s')
    return read_records(rec_path)


def boot_ci(values, reps):
    v = np.asarray(values, dtype=float)
    if v.size < 2 or reps <= 0:
        return None
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, v.size, size=(reps, v.size))
    lo, hi = np.quantile(v[idx].mean(axis=1), [0.025, 0.975], method='linear')
    return [float(lo), float(hi)]


def analyze(records, reps=BOOTSTRAP_REPS):
    arms = [a for a, _, _ in ARMS if any(r['arm'] == a for r in records)]
    dose = {a: d for a, _, d in ARMS}
    by = {(r['arm'], r['instance_id']): r for r in records}
    res = dict(use='EXPLORATORY_CALIBRATION_PILOT_u26_DESCRIPTIVE_NO_INFERENCE', task_seed_base=TASK_SEED,
               bootstrap=dict(unit='instance', reps=reps, seed=BOOTSTRAP_SEED, statistic='mean', note='descriptive; 20 instances of one world; not a significance test'),
               descriptive_band_FCA_E1=BAND_E1, delta_reduction_valid='NOT_MEASURED_IN_THIS_SCRIPT', arms={}, paired_vs_sham={}, dose_response_numeric={},
               sham_vs_baseline=sham_identity(records))
    for a in arms:
        rs = [r for r in records if r['arm'] == a]; ok = [r for r in rs if r['result']['valid']]
        sc = [r['result']['score'] for r in ok]
        det = {k: (float(np.mean([r['result']['details'][k] for r in ok if k in r['result']['details']])) if any(k in r['result']['details'] for r in ok) else None) for k in DETAIL_KEYS}
        res['arms'][a] = dict(dose=dose[a], n_instances=len(rs), n_valid=len(ok), n_invalid=len(rs) - len(ok),
                              invalid_reasons=sorted({str(r['result']['details'].get('reason')) for r in rs if not r['result']['valid']}),
                              FCA_E1=float(np.mean(sc)) if sc else None, FCA_E1_ci95_descriptive=boot_ci(sc, reps), mean_details=det,
                              median_removed_activation_rms=float(np.median([r['hook_removed_activation_rms'] or 0.0 for r in rs])),
                              total_generate_calls=int(sum(r['gen']['generate_calls'] for r in rs)), total_generate_seconds=float(sum(r['gen']['generate_seconds'] for r in rs)),
                              median_instance_seconds=float(np.median([r['seconds'] for r in rs])), total_instance_seconds=float(sum(r['seconds'] for r in rs)))
    if 'U26_SHAM' in arms:
        for a in arms:
            if a == 'U26_SHAM':
                continue
            ids = sorted(i for (x, i), r in by.items() if x == a and r['result']['valid'] and by.get(('U26_SHAM', i), {}).get('result', {}).get('valid'))
            d = [by[(a, i)]['result']['score'] - by[('U26_SHAM', i)]['result']['score'] for i in ids]
            if not d:
                continue
            m = float(np.mean(d))
            res['paired_vs_sham'][a] = dict(n_pairs=len(d), diff_by_instance=dict(zip(ids, d)), mean_diff=m, median_diff=float(np.median(d)), mean_diff_ci95_descriptive=boot_ci(d, reps),
                                            n_instances_diff_le_0=int(sum(1 for x in d if x <= 0)), abs_mean_diff_exceeds_descriptive_band=bool(abs(m) > BAND_E1),
                                            n_excluded_not_valid_in_both=int(sum(1 for (x, i) in by if x == a) - len(d)))
            if a.startswith('U26_ABLATE'):
                res['dose_response_numeric'][f'{dose[a]:.2f}'] = m
    return res


def write_outputs(out, res):
    out = Path(out)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2, sort_keys=True)
    with open(out / 'pilot_table.csv', 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['arm', 'dose', 'instance_id', 'FCA_E1_score', 'valid', 'diff_vs_sham'])
        recs = read_records(out / 'records.jsonl')
        for r in recs:
            d = res['paired_vs_sham'].get(r['arm'], {}).get('diff_by_instance', {}).get(r['instance_id'])
            w.writerow([r['arm'], r['dose'], r['instance_id'], r['result']['score'], r['result']['valid'], d])


def build_runtime(a, work):
    fca_dir = find_fca_dir(work, a.fca_dir)
    fca, REG = import_fca(fca_dir)
    pilot = load_module('t346_pcit_l27_pilot', work / PILOT_SCRIPT)
    from fca_bench.adapters import adapter_from_config, load_config
    cfg_file = fca_dir / 'config.qwen3_8b_gpu8.json'
    cfg = load_config(str(cfg_file)) if cfg_file.is_file() else copy.deepcopy(HISTORICAL_R6_CONFIG)
    deviations = []
    cfg['model_path'] = str(a.model_dir); cfg['progress'] = False
    if a.quantization:
        cfg['quantization'] = a.quantization; deviations.append(f'quantization={a.quantization} (historical bnb8)')
    if a.dtype:
        cfg['dtype'] = a.dtype; deviations.append(f'dtype={a.dtype} (historical float16 compute)')
    adapter = adapter_from_config(cfg)
    tree, files = fca_tree_sha(fca_dir)
    return dict(adapter=adapter, test=REG['E1'], Att=make_attenuator_class(pilot.BlockAttenuator), stats=GenStats(adapter.model),
                record_write=str(cfg['quantization']).lower() in {'none', 'off', 'false', ''}, fca_dir=fca_dir, config=cfg, deviations=deviations,
                provenance=dict(script=sha_file(Path(__file__)), pilot_script=sha_file(work / PILOT_SCRIPT), fca_tree_sha256=tree, fca_files=files))


def do_run(a):
    if a.authorize_gpu_run != CONFIRM:
        raise SystemExit('GPU run refused: pass --authorize-gpu-run ' + CONFIRM)
    work = a.work; rt = build_runtime(a, work)
    arms, n_inst = ARMS, a.instances or N_INSTANCES
    if a.probe:
        arms = tuple(x for x in ARMS if x[0] in ('U26_BASELINE', 'U26_SHAM')); n_inst = 1
    elif a.arms:
        want = a.arms.split(','); arms = tuple(x for x in ARMS if x[0] in want)
    out = work / (OUTNAME + ('_PROBE' if a.probe else '') + a.out_suffix)
    header = dict(provenance=rt['provenance'], config=rt['config'], deviations_from_historical_R6=rt['deviations'], probe=bool(a.probe), arms=[x[0] for x in arms], n_instances=n_inst,
                  hook=dict(layer=HOOK_LAYER, block=HOOK_BLOCK, columns=[0, 3072], positions=HOOK_POSITIONS, record_write=rt['record_write']),
                  runtime=rt['adapter'].runtime_metadata(), task_seed_base=TASK_SEED)
    recs = run_pilot(rt, out, header, arms, n_inst, a.resume, not a.continue_if_sham_differs)
    if not a.probe:
        write_outputs(out, analyze(recs, a.bootstrap_reps))
    print('[T346-FCA] done ; records', out / 'records.jsonl')


def do_analyze(a):
    out = a.work / (OUTNAME + a.out_suffix)
    recs = read_records(out / 'records.jsonl')
    if not recs:
        raise SystemExit('No records in ' + str(out))
    write_outputs(out, analyze(recs, a.bootstrap_reps)); print('[T346-FCA] RESULT', out / 'RESULT.json')


class FakeAdapter:
    """Session-compatible stand-in around a tiny random Qwen3: the answers are derived from the model's generated ids, so the model (and the hook) drives them."""
    CODES = ['NIMBUS', 'CEDAR', 'ATLAS', 'MERIDIAN', 'ORBIT', 'FALCON', 'QUARTZ', 'EMBER']
    CATS = ['PLANT', 'PLACE', 'MATERIAL', 'ANIMAL', 'DEVICE', 'COLOR']

    def __init__(self, model, e1_samples=6, e1_batch_size=4, seed=1234):
        import torch
        self.model, self.torch, self.e1_samples, self.e1_batch_size, self.seed = model, torch, e1_samples, e1_batch_size, seed
        self.name = self.system_id = 'fake'

    def runtime_metadata(self):
        return dict(fake=True)

    def supports(self, _):
        return False

    def _stable_seed(self, instance, key):
        raw = f'{self.seed}|{instance.seed if instance is not None else 0}|{key}'.encode('utf-8')
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2 ** 31 - 1)

    def new_session(self, system=None, instance=None):
        outer = self

        class S:
            def __init__(self):
                self.msgs = [system or '']; self.pending = []; self.n = 0

            def _ids(self):
                txt = '\n'.join(self.msgs)[-200:]
                return outer.torch.tensor([[ord(c) % 200 for c in txt]], dtype=outer.torch.long)

            def send(self, text, channel='work', expect_reply=True, sample=False):
                self.msgs.append(('[CONTROL] ' if channel == 'control' else '[WORK] ') + text)
                if not expect_reply:
                    return ''
                return self._answer(self._ids(), text, outer._stable_seed(instance, f'single|{self.n}|{text[:40]}'))

            def _answer(self, ids, text, seed):
                outer.torch.manual_seed(seed)
                with outer.torch.inference_mode():
                    out = outer.model.generate(ids, attention_mask=outer.torch.ones_like(ids), max_new_tokens=5, do_sample=True, temperature=1.5, pad_token_id=0, use_cache=True)
                h = int(out[0, ids.shape[1]:].sum())
                return outer._json(text, h, '\n'.join(self.msgs))

            def send_many(self, text, n, channel='work', sample=True, seed_key='batch'):
                self.msgs.append(('[CONTROL] ' if channel == 'control' else '[WORK] ') + text)
                ids = self._ids(); outs = []
                for start in range(0, n, outer.e1_batch_size):
                    b = min(outer.e1_batch_size, n - start)
                    outer.torch.manual_seed(outer._stable_seed(instance, f'{seed_key}|{start}|{b}'))
                    with outer.torch.inference_mode():
                        g = outer.model.generate(ids.repeat(b, 1), attention_mask=outer.torch.ones_like(ids.repeat(b, 1)), max_new_tokens=5, do_sample=True, temperature=1.5, pad_token_id=0, use_cache=True)
                    for row in range(b):
                        outs.append(outer._json(text, int(g[row, ids.shape[1]:].sum()), '\n'.join(self.msgs)))
                return outs
        return S()

    def _json(self, prompt, h, context):
        if 'arbitrary code word' in prompt:
            ans = self.CODES[h % len(self.CODES)]
        elif 'arbitrary semantic category' in prompt:
            ans = self.CATS[h % len(self.CATS)]
        elif 'currently available' in prompt:
            tail = context.split('Auxiliary identifier for ')
            ans = tail[1].split(':')[1].split('.')[0].strip() if len(tail) > 1 and 'Revoke the auxiliary' not in context else 'NONE'
        else:
            ans = 'NONE'
        return json.dumps({'probe_id': 'x', 'answer': ans})


def selftest(a):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    t_start = time.perf_counter(); work = a.work
    fca_dir = find_fca_dir(work, a.fca_dir); fca, REG = import_fca(fca_dir)
    pilot = load_module('t346_pcit_l27_pilot', work / PILOT_SCRIPT)
    Att = make_attenuator_class(pilot.BlockAttenuator)
    torch.manual_seed(5)
    cfg = Qwen3Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, use_cache=True)
    model = Qwen3ForCausalLM(cfg).eval()
    adapter = FakeAdapter(model); stats = GenStats(model)
    rt = dict(adapter=adapter, test=REG['E1'], Att=Att, stats=stats, record_write=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'full'
        recs = run_pilot(rt, out, dict(provenance='x'), ARMS, 3, log=lambda *_: None)
        assert len(recs) == 15 and all(r['gen']['generate_calls'] == 25 for r in recs), [r['gen'] for r in recs][:2]
        ident = sham_identity(recs); assert ident['identical'] and ident['n_instances_compared'] == 3, ident
        by = {(r['arm'], r['instance_id']): r for r in recs}
        assert all(by[('U26_BASELINE', i)]['hook_calls'] == 0 and by[('U26_SHAM', i)]['hook_calls'] > 0 for _, i in by)
        assert any(raws(by[('U26_ABLATE_100', i)]) != raws(by[('U26_SHAM', i)]) for _, i in by), 'dose 1 must change some outputs'
        rms = [np.median([by[(a_, i)]['hook_removed_activation_rms'] for _, i in by if _ == a_]) for a_ in ('U26_ABLATE_025', 'U26_ABLATE_050', 'U26_ABLATE_100')]
        assert 0 < rms[0] < rms[1] < rms[2] and 3.5 < rms[2] / rms[0] < 4.5, rms
        res = analyze(recs, 200); write_outputs(out, res)
        assert set(res['paired_vs_sham']) == {'U26_BASELINE', 'U26_ABLATE_025', 'U26_ABLATE_050', 'U26_ABLATE_100'} and res['sham_vs_baseline']['identical']
        assert res['paired_vs_sham']['U26_BASELINE']['mean_diff'] == 0.0 and sorted(res['dose_response_numeric']) == ['0.25', '0.50', '1.00']
        assert all(v['n_valid'] + v['n_invalid'] == 3 for v in res['arms'].values()) and res['delta_reduction_valid'] == 'NOT_MEASURED_IN_THIS_SCRIPT'
        full = read_records(out / 'records.jsonl')
        r_out = Path(tmp) / 'resume'; r_out.mkdir(); (r_out / 'RUN_HEADER.json').write_text((out / 'RUN_HEADER.json').read_text(encoding='utf-8'), encoding='utf-8')
        for r in full[:7]:
            append_record(r_out / 'records.jsonl', r)
        again = run_pilot(rt, r_out, dict(provenance='x'), ARMS, 3, resume=True, log=lambda *_: None)
        strip = lambda rs: [{k: v for k, v in r.items() if k not in ('seconds', 'gen')} for r in rs]
        assert strip(again) == strip(full), 'resume must reproduce the uninterrupted run'
        for bad, kw in (('different provenance', dict(header=dict(provenance='other'), resume=True, out=r_out)), ('existing folder', dict(header=dict(provenance='x'), resume=False, out=out))):
            try:
                run_pilot(rt, kw['out'], kw['header'], ARMS, 3, resume=kw['resume'], log=lambda *_: None); raise AssertionError('accepted ' + bad)
            except SystemExit:
                pass
        # an int8-style layer (no float weight slice): the quantization-aware hook still records the activation RMS
        Att2 = make_attenuator_class(pilot.BlockAttenuator)
        with Att2(model, layer=24, block=0, dose=0.5, mode='ATTENUATE', positions=HOOK_POSITIONS, record_write=False) as hk:
            model(torch.randint(0, 200, (1, 12)))
        s = hk.summary(); assert s['removed_activation_rms'] > 0 and s['removed_residual_write_rms'] is None
        # the loop stops when SHAM differs from BASELINE
        class Poisoned(Att):
            def __init__(self, *aa, **kk):
                super().__init__(*aa, **kk)

            def _pre_hook(self, module, args):
                out_ = super()._pre_hook(module, args)
                if self.mode == 'SHAM':
                    out_ = (out_[0] * 1.5,) + out_[1:]
                return out_
        rt2 = dict(rt, Att=Poisoned)
        try:
            run_pilot(rt2, Path(tmp) / 'poison', dict(provenance='x'), ARMS, 1, log=lambda *_: None); raise AssertionError('did not stop on a non-identical SHAM')
        except SystemExit as e:
            assert 'SHAM output differs' in str(e), e
    print(f'[T346-FCA] selftest OK in {time.perf_counter() - t_start:.1f} s (CPU, tiny random model behind a fake adapter: 12 batched generate calls per micro-batch pair plus the diagnostic, as the real E1 (61 with 20 draws), SHAM identical, '
          'dose changes outputs, activation RMS scales with dose, int8-style recording, resume, refusals, stop on non-identical SHAM, analysis; FCA package imported unmodified)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=WORK); ap.add_argument('--model-dir', type=Path, default=None); ap.add_argument('--fca-dir', type=Path, default=None)
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--run', action='store_true'); ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--quantization', default=None, help='override the historical bnb8 (e.g. none)'); ap.add_argument('--dtype', default=None, help='override the historical float16')
    ap.add_argument('--authorize-gpu-run'); ap.add_argument('--probe', action='store_true'); ap.add_argument('--arms'); ap.add_argument('--instances', type=int)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--out-suffix', default=''); ap.add_argument('--continue-if-sham-differs', action='store_true')
    ap.add_argument('--bootstrap-reps', type=int, default=BOOTSTRAP_REPS)
    a = ap.parse_args()
    if a.model_dir is None:
        a.model_dir = a.work / 'Qwen3-8B'
    if a.selftest:
        selftest(a)
    elif a.run:
        do_run(a)
    elif a.analyze:
        do_analyze(a)
    else:
        ap.error('choose --selftest, --run or --analyze')


if __name__ == '__main__':
    main()
