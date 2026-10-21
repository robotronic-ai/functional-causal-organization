#!/usr/bin/env python3
"""Run the C_L24 numerical C_O chain (Gamma, Delta, R, Lambda0) on Qwen3-8B whose mlp0@L24 is attenuated by a dose, one isolated row per run.

Intervention (weights edited in memory after the model is loaded; no file on disk is changed):
  layers[LAYER].mlp.down_proj.weight[:, lo:hi] *= (1 - dose), columns of block BLOCK of N_BLOCKS (Qwen3-8B: mlp0 = columns 0..3071).
  For every forward pass this equals a forward pre-hook on down_proj that scales the same input channels at every processed token
  (l24_mlp0_ablation_hook.py, positions ALL_PROCESSED_TOKENS); dose 1 is the zero ablation. dose = 0 edits nothing.
What stays fixed: the frozen donor arrays, the 12 frozen readers (origins and directions), the donor masks, the world, the seeds, the dtype
  (bfloat16, as in the certified bf16 scripts). The game is the same function evaluated on the edited weights, and the scores are still
  measured relative to the readers' origins of the unmodified model. This is a stated design choice, not a result.
Row isolation: <rows-dir>/<row-id>/ (default <work>/T367_ROWS/<row-id>/) holds a symbolic link to every directory and a real copy of every file of
  the working folder, except FAST_GPU_RESULTS, which is a private folder with the same rule for the inputs and real directories for the outputs of the chain. Nothing of the base outputs is touched.
Guards: at dose 0 every guard of the legacy scripts stays strict (this reproduces the unmodified model bit for bit or the stage stops).
  At dose > 0 only the guards that compare an output with the unmodified model (frozen historical arms, identity sub-cube against its pinned
  reference, reference gauge sub-cubes, clean active token) are downgraded to a recorded deviation in ROW_DEVIATIONS.jsonl; the internal
  consistency guards (full forward equals windowed run bit for bit, token reproducibility, identity replay) stay strict.
Not measured by this script: FCA_E1, PCI_T_DS_L27 (use the T346 scripts with the same doses), the M0 and influence covariates.

Usage (from quantum/, PY = the project interpreter with -I -u):
  PY t367_intervened_c_o_gpu.py --selftest
  PY t367_intervened_c_o_gpu.py --stage preflight --row-id R000_DOSE000 --dose 0
  PY t367_intervened_c_o_gpu.py --stage plan --row-id R050_DOSE050 --dose 0.5
  PY t367_intervened_c_o_gpu.py --stage gauge --row-id R000_DOSE000 --dose 0 --authorize-gpu-run I_AUTHORIZE_T367_INTERVENED_C_O_GPU_RUN
  PY t367_intervened_c_o_gpu.py --stage chain --row-id R050_DOSE050 --dose 0.5 --resume --authorize-gpu-run I_AUTHORIZE_T367_INTERVENED_C_O_GPU_RUN
  PY t367_intervened_c_o_gpu.py --stage collect --row-id R050_DOSE050 --dose 0.5
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

AUTH = 'I_AUTHORIZE_T367_INTERVENED_C_O_GPU_RUN'
WORK_DEFAULT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
ROWS_DEFAULT = WORK_DEFAULT / 'T367_ROWS'
COPY_MAX = 256 << 20  # regular files up to this size are copied (the pinned loaders refuse symbolic links to files); larger ones are hard-linked

GPU = {  # stage -> (script names in order of preference, output directory template under FAST_GPU_RESULTS, measured seconds of the bf16 u26 run on the unmodified model)
    'gauge': (('claude_S9_full12_gauge_cube_gpu.py', 'acquire_gauge_cube_gpu.py'), 'CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD{w}_GPU', 1912.6),
    'r_greedy': (('claude_S24_R_token_return_gpu.py', 'r_greedy_return_gpu.py'), 'CLAUDE_S24_R_TOKEN_RETURN_WORLD{w}_GPU', 928.5),
    'r_sampled': (('claude_S26_R_sampled_token_gpu.py', 'r_sampled_return_gpu.py'), 'CLAUDE_S26_R_SAMPLED_TOKEN_WORLD{w}_GPU', 6005.0),
    'role_t1': (('role_identity_t1_gpu.py',), 'CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_GPU', 2062.0),
}
CPU = {  # stage -> (script names in order of preference, extra arguments, output directory template)
    'gamma': (('claude_S22_frozen_rule_gamma_table_cpu.py', 'gamma_frozen_rule_cpu.py'), [], 'CLAUDE_S22_FROZEN_RULE_GAMMA_WORLD{w}_CPU'),
    'delta': (('claude_S23_delta_C_L24_cpu.py', 'delta_joint_closure_cpu.py'), [], 'CLAUDE_S23_DELTA_C_L24_WORLD{w}_CPU'),
    'r_channel': (('claude_S25_R_token_channel_cpu.py', 'r_token_channel_cpu.py'), [], 'CLAUDE_S25_R_TOKEN_CHANNEL_WORLD{w}_CPU'),
    'r_export': (('claude_S26_export_T330_cpu.py', 'export_t330_schema_cpu.py'), [], 'S26_U26_FULLSOFTMAX'),
    'r_bound': (('claude_S27_R_return_bound_cpu.py', 'r_return_bound_cpu.py'), [], 'CLAUDE_S27_R_RETURN_BOUND_WORLD{w}_CPU'),
    'role_t1_cpu': (('role_identity_t1_cpu.py',), [], 'CLAUDE_S28_ROLE_IDENTITY_T1_WORLD{w}_CPU'),
    'role_sig': (('role_signature_t1_cpu.py',), [], 'CLAUDE_S28_ROLE_SIGNATURE_T1_WORLD{w}_CPU'),
    'lambda_rel': (('lambda_relation_cpu.py',), [], 'CLAUDE_LAMBDA_RELATION_C_L24_WORLD{w}_CPU'),
    'lambda_w1': (('extract_lambda_witnesses_cpu.py',), [], 'LAMBDA_WITNESS_EXTRACT_C_L24_u{w}'),
    'lambda_w2': (('extract_lambda_witnesses_cpu.py',), ['--factor', '2'], 'LAMBDA_WITNESS_EXTRACT_C_L24_u{w}_2delta'),
}
CHAIN = ['gauge', 'gamma', 'delta', 'r_greedy', 'r_channel', 'r_sampled', 'r_export', 'r_bound', 'role_t1', 'role_t1_cpu', 'role_sig',
         'lambda_rel', 'lambda_w1', 'lambda_w2']
OWN = re.compile(r'^(CLAUDE_S(9|22|23|24|25|26|27|28)_|CLAUDE_LAMBDA_RELATION|LAMBDA_WITNESS_EXTRACT|S26_U26_FULLSOFTMAX)')
NEED_DEF = 'def need(ok, msg):\n    if not ok:\n        raise RuntimeError(msg)\n'
SOFT = re.compile(r'^(Drift of frozen arm|Identity sub-cube differs|Sub-cube .* differs from the reference gauge sub-cube|Unexpected active token|'
                  r'Clean next token .* differs|Derive T276|Sous-cube identite|Sous-cube .* different de S5|Token actif inattendu)')
ROW_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')


def resolve(scripts_dir, names):
    for n in names:
        if (scripts_dir / n).is_file():
            return scripts_dir / n
    return None


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S')


# ------------------------------------------------------------------ intervention
def block_of(model, layer, block, n_blocks):
    dec = model.model if hasattr(model, 'model') else model
    if not 0 <= layer < len(dec.layers):
        raise ValueError(f'layer {layer} outside 0..{len(dec.layers) - 1}')
    w = dec.layers[layer].mlp.down_proj.weight
    dff = w.shape[1]
    if dff % n_blocks or not 0 <= block < n_blocks:
        raise ValueError('intermediate size must be divisible by n_blocks and block must be in range')
    bs = dff // n_blocks
    return w, block * bs, (block + 1) * bs


def attenuate(model, layer, block, n_blocks, dose):
    """Scales the columns of block `block` of layers[layer].mlp.down_proj.weight by (1 - dose) in place. Returns a record."""
    import torch
    w, lo, hi = block_of(model, layer, block, n_blocks)
    rec = dict(layer=layer, block=block, n_blocks=n_blocks, columns=[lo, hi], dose=float(dose), dtype=str(w.dtype), edited=False)
    with torch.no_grad():
        rec['block_norm_before'] = float(w[:, lo:hi].float().norm())
        if dose > 0:
            w[:, lo:hi] *= (1.0 - float(dose))
            rec['edited'] = True
        rec['block_norm_after'] = float(w[:, lo:hi].float().norm())
        rec['rest_norm'] = float(torch.cat([w[:, :lo], w[:, hi:]], dim=1).float().norm())
    return rec


class Switch:
    """Keeps the original columns of the edited block so that the edit can be switched off while the frozen donor values are captured."""
    def __init__(self, weight, lo, hi, dose):
        import torch
        self.w, self.lo, self.hi, self.dose = weight, lo, hi, float(dose)
        with torch.no_grad():
            self.saved = weight[:, lo:hi].detach().clone()
        self.active = False

    def set(self, active):
        import torch
        with torch.no_grad():
            if active:
                self.w[:, self.lo:self.hi] = self.saved * (1.0 - self.dose)
            else:
                self.w[:, self.lo:self.hi] = self.saved
        self.active = bool(active)


class Installed:
    """State of the load_model patch of one process."""
    def __init__(self):
        self.calls = 0
        self.records = []
        self.switch = None
        self.phases = []

    def donor_phase(self, donor):
        """Called by the transformed legacy script around the capture of the frozen D0 background values: the donors stay those of the
        unmodified model, so the edit is off during the capture and on afterwards."""
        self.phases.append('DONOR_CAPTURE_EDIT_OFF' if donor else 'GAME_EDIT_ON')
        if self.switch is not None:
            self.switch.set(not donor)


def install_load_patch(row, layer, block, n_blocks, dose, log):
    """Wraps gamma_local_qwen.load_model (module imported from <row>/CLAUDE_CO_Q2L1) so that the returned model carries the intervention."""
    src = row / 'CLAUDE_CO_Q2L1'
    if not src.is_dir():
        raise SystemExit('Folder missing: ' + str(src) + ' ; restore CLAUDE_CO_Q2L1 in the working folder')
    sys.path.insert(0, str(src))
    import gamma_local_qwen as G
    st = Installed()
    orig = G.load_model

    def patched(a, rt):
        out = orig(a, rt)
        model = out[0] if isinstance(out, tuple) else out
        if dose > 0:
            w, lo, hi = block_of(model, layer, block, n_blocks)
            st.switch = Switch(w, lo, hi, dose)
        rec = attenuate(model, layer, block, n_blocks, dose)
        if st.switch is not None:
            st.switch.active = True
        st.calls += 1
        st.records.append(rec)
        log(f'[t367] intervention applied after load: {json.dumps(rec, sort_keys=True)}')
        return out
    G.load_model = patched
    return st, G


# ------------------------------------------------------------------ guards of the legacy scripts
BG_OPEN = '    values_bg = []\n'
BG_CLOSE = '    pairs = T286.freeze_background_pair(values_bg)\n'


def transform_source(src, name):
    if src.count(NEED_DEF) != 1:
        raise SystemExit(f'{name}: the expected need() definition was not found exactly once; the script differs from the pinned version')
    src = src.replace(NEED_DEF, 'def need(ok, msg):\n    _t367_need(ok, msg)\n')
    if name.endswith('_gpu.py') and 'T286.D0' in src:
        if src.count(BG_OPEN) != 1 or src.count(BG_CLOSE) != 1:
            raise SystemExit(f'{name}: the capture of the frozen D0 background values was not found exactly once; the script differs from the pinned version')
        src = src.replace(BG_OPEN, '    _t367_donor_phase(True)\n' + BG_OPEN).replace(BG_CLOSE, BG_CLOSE + '    _t367_donor_phase(False)\n')
    return src


def make_need(dose, dev_path, stage, verbose=True):
    def _need(ok, msg):
        if ok:
            return
        if dose > 0 and SOFT.match(str(msg)):
            with open(dev_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(stage=stage, at=now(), dose=dose, guard=str(msg))) + '\n')
            if verbose:
                print(f'[t367] DEVIATION recorded (dose {dose} > 0): {msg}', flush=True)
            return
        raise RuntimeError(msg)
    return _need


def exec_script(path, argv, extra_globals):
    src = transform_source(Path(path).read_text(encoding='utf-8'), Path(path).name)
    g = {'__name__': '__main__', '__file__': str(path), '__builtins__': __builtins__}
    g.update(extra_globals)
    old = sys.argv
    sys.argv = [str(path)] + list(argv)
    try:
        exec(compile(src, str(path), 'exec'), g)
        return 0
    except SystemExit as e:
        if e.code is None or e.code == 0:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old


# ------------------------------------------------------------------ row sandbox
def place(e, link):
    """Directories become symbolic links; regular files become real copies (the pinned loaders refuse a symbolic link to a file)."""
    if e.is_dir():
        if link.is_symlink() and link.resolve() == e.resolve():
            return
        if link.exists() or link.is_symlink():
            return  # a private real folder of the row is never replaced
        os.symlink(e, link)
        return
    if link.is_symlink():
        link.unlink()
    st = e.stat()
    if link.is_file() and link.stat().st_size == st.st_size and int(link.stat().st_mtime) == int(st.st_mtime):
        return
    if st.st_size <= COPY_MAX:
        shutil.copy2(e, link)
    else:
        if link.exists():
            link.unlink()
        try:
            os.link(e, link)
        except OSError:
            raise SystemExit('Cannot copy or hard-link ' + str(e) + ' into the row folder (file larger than 256 MB)')


def build_row(work, rows_dir, row_id):
    row = rows_dir / row_id
    row.mkdir(parents=True, exist_ok=True)
    rows_real = rows_dir.resolve()
    try:
        for e in sorted(work.iterdir()):
            if e.name in ('FAST_GPU_RESULTS', '__pycache__') or e.name.startswith('.') or e.resolve() == rows_real:
                continue
            place(e, row / e.name)
        fr = row / 'FAST_GPU_RESULTS'
        fr.mkdir(exist_ok=True)
        base = work / 'FAST_GPU_RESULTS'
        if base.is_dir():
            for e in sorted(base.iterdir()):
                if OWN.match(e.name):
                    continue
                place(e, fr / e.name)
    except OSError as exc:
        raise SystemExit('Cannot build the row folder in ' + str(rows_dir) + ' (' + str(exc) + '); pass --rows-dir with another folder')
    return row


def state_path(row):
    return row / 'ROW_STATE.json'


def read_state(row):
    p = state_path(row)
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}


def write_state(row, stage, **kw):
    s = read_state(row)
    s[stage] = dict(kw, at=now())
    tmp = state_path(row).with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, state_path(row))


def write_meta(row, a, scripts_dir):
    p = row / 'ROW_META.json'
    meta = dict(schema='T367_ROW_META_V1', row_id=a.row_id, dose=a.dose, layer=a.layer, block=a.block, n_blocks=a.n_blocks, world=a.world, dtype='bfloat16',
                intervention='down_proj weight columns scaled by (1 - dose) in memory after load; equals ALL_PROCESSED_TOKENS attenuation',
                readers_and_origins='UNMODIFIED_C_L24_CONTRACT', launcher_sha256=sha_file(__file__), created_at=now(),
                scripts_sha256={resolve(scripts_dir, n).name: sha_file(resolve(scripts_dir, n)) for n, _, _ in GPU.values() if resolve(scripts_dir, n)})
    try:
        import torch, transformers
        meta['torch'], meta['transformers'] = torch.__version__, transformers.__version__
    except Exception:
        pass
    if p.is_file():
        old = json.loads(p.read_text(encoding='utf-8'))
        for k in ('row_id', 'dose', 'layer', 'block', 'n_blocks', 'world'):
            if old.get(k) != meta[k]:
                raise SystemExit(f'Row {a.row_id} already exists with {k}={old.get(k)}; refuse to mix settings. Use a new --row-id')
        return old
    p.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
    return meta


# ------------------------------------------------------------------ stages
def run_gpu_stage(a, row, scripts_dir):
    names, outname, _ = GPU[a.stage]
    path = resolve(scripts_dir, names)
    if path is None:
        raise SystemExit('Script missing: none of ' + ', '.join(names) + ' in ' + str(scripts_dir) + ' ; restore it from the backup')
    out = row / 'FAST_GPU_RESULTS' / outname.format(w=a.world)
    argv = ['--gpu', '--root', str(row), '--world', str(a.world)]
    if a.resume or out.exists():
        argv.append('--resume')
    dev = row / 'ROW_DEVIATIONS.jsonl'
    logf = open(row / 'T367_LAUNCHER.log', 'a', encoding='utf-8', buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + '\n')
    st, _ = install_load_patch(row, a.layer, a.block, a.n_blocks, a.dose, log)
    t0 = time.monotonic()
    write_state(row, a.stage, status='RUNNING')
    code = exec_script(path, argv, {'_t367_need': make_need(a.dose, dev, a.stage), '_t367_donor_phase': st.donor_phase})
    secs = round(time.monotonic() - t0, 1)
    if code == 0 and st.calls != 1:
        (out / 'T367_INVALID.txt').write_text(f'model loaded through G.load_model {st.calls} times; the intervention is not certain to be applied once\n')
        write_state(row, a.stage, status='INVALID_INTERVENTION_NOT_APPLIED_ONCE', seconds=secs, load_calls=st.calls)
        return 3
    if code == 0 and st.phases != ['DONOR_CAPTURE_EDIT_OFF', 'GAME_EDIT_ON']:
        (out / 'T367_INVALID.txt').write_text('the donor capture phases were not executed exactly once each: ' + json.dumps(st.phases) + '\n')
        write_state(row, a.stage, status='INVALID_DONOR_PHASES', seconds=secs, load_calls=st.calls, phases=st.phases)
        return 3
    write_state(row, a.stage, status='OK' if code == 0 else f'FAILED_{code}', seconds=secs, load_calls=st.calls, intervention=st.records, phases=st.phases,
                active_at_end=bool(st.switch.active) if st.switch else None)
    return code


def run_cpu_stage(a, row, scripts_dir):
    names, extra, outname = CPU[a.stage]
    path = resolve(scripts_dir, names)
    if path is None:
        raise SystemExit('Script missing: none of ' + ', '.join(names) + ' in ' + str(scripts_dir) + ' ; restore it from the backup')
    cmd = [sys.executable, '-I', '-u', str(path), '--root', str(row), '--world', str(a.world)] + extra
    print('[t367]', ' '.join(cmd), flush=True)
    t0 = time.monotonic()
    write_state(row, a.stage, status='RUNNING')
    r = subprocess.run(cmd, cwd=str(row))
    write_state(row, a.stage, status='OK' if r.returncode == 0 else f'FAILED_{r.returncode}', seconds=round(time.monotonic() - t0, 1))
    return r.returncode


def stage_cmd(a, stage, scripts_dir):
    c = [sys.executable, '-I', '-u', str(Path(__file__).resolve()), '--stage', stage, '--row-id', a.row_id, '--dose', repr(a.dose), '--world', str(a.world),
         '--layer', str(a.layer), '--block', str(a.block), '--n-blocks', str(a.n_blocks), '--work', str(a.work), '--rows-dir', str(a.rows_dir),
         '--scripts-dir', str(scripts_dir)]
    if a.resume:
        c.append('--resume')
    if stage in GPU:
        c += ['--authorize-gpu-run', a.authorize_gpu_run or '']
    return c


def run_chain(a, row, scripts_dir, stages):
    for s in stages:
        if read_state(row).get(s, {}).get('status') == 'OK':
            print(f'[t367] {s}: already OK, skipped', flush=True)
            continue
        c = stage_cmd(a, s, scripts_dir)
        print('[t367] chain ->', s, flush=True)
        r = subprocess.run(c)
        if r.returncode:
            print(f'[t367] chain stopped at {s} (code {r.returncode}); rerun the same command with --resume', flush=True)
            return r.returncode
    print('[t367] chain finished for', a.row_id, flush=True)
    return 0


def collect(a, row):
    dst = row / 'ROW_RESULTS'
    dst.mkdir(exist_ok=True)
    summary = dict(schema='T367_ROW_SUMMARY_V1', row_id=a.row_id, dose=a.dose, world=a.world, state=read_state(row), stages={}, npz_sha256={})
    names = [o.format(w=a.world) for _, o, _ in GPU.values()] + [o.format(w=a.world) for _, _, o in CPU.values()]
    for n in names:
        d = row / 'FAST_GPU_RESULTS' / n
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == '.npz':
                summary['npz_sha256'][f'{n}/{f.name}'] = sha_file(f)
            elif f.name in ('RESULT.json', 'summary.json') or f.name == 'RESULT.md':
                (dst / n).mkdir(exist_ok=True)
                shutil.copy(f, dst / n / f.name)
                summary['stages'].setdefault(n, []).append(f.name)
    for f in ('ROW_META.json', 'ROW_STATE.json', 'ROW_DEVIATIONS.jsonl', 'T367_LAUNCHER.log'):
        if (row / f).is_file():
            shutil.copy(row / f, dst / f)
    (dst / 'ROW_SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print('[t367] collected into', dst, '(no NPZ; their SHA-256 are in ROW_SUMMARY.json)')
    return 0


def plan(a, scripts_dir):
    tot = 0.0
    print(f'Row {a.row_id}: dose {a.dose}, layer {a.layer}, block {a.block}/{a.n_blocks}, world {a.world}')
    for s in CHAIN:
        if s in GPU:
            sec = GPU[s][2]
            tot += sec
            r = resolve(scripts_dir, GPU[s][0]); print(f'  {s:12s} GPU  {sec:7.1f} s  (measured, bf16 u26, unmodified model)  {r.name if r else "MISSING " + GPU[s][0][0]}')
        else:
            r = resolve(scripts_dir, CPU[s][0]); print(f'  {s:12s} CPU   (seconds to minutes, not measured)  {r.name if r else "MISSING " + CPU[s][0][0]} {" ".join(CPU[s][1])}')
    print(f'  GPU total {tot:.0f} s = {tot / 3600:.2f} h per row (assumes the same cost on an edited model; not measured). FCA_E1 and PCI_T are separate runs.')
    missing = [v[0][0] for v in list(GPU.values()) + list(CPU.values()) if resolve(scripts_dir, v[0]) is None]
    if missing:
        print('  MISSING in', scripts_dir, ':', ', '.join(sorted(missing)))
    return 0


# ------------------------------------------------------------------ self-test
def selftest():
    import tempfile
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    here = Path(__file__).resolve().parent
    # 1. attenuation
    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=64, use_cache=True)
    for dt in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        m0 = Qwen3ForCausalLM(cfg).to(dt).eval()
        ids = torch.randint(0, 100, (2, 9))
        ref = {k: v.clone() for k, v in m0.state_dict().items()}
        r0 = attenuate(m0, 2, 0, 4, 0.0)
        assert not r0['edited'] and all(torch.equal(ref[k], v) for k, v in m0.state_dict().items()), 'dose 0 must not change any weight'
        r5 = attenuate(m0, 2, 0, 4, 0.5)
        w = m0.model.layers[2].mlp.down_proj.weight
        assert r5['edited'] and torch.allclose(w[:, :32].float(), 0.5 * ref['model.layers.2.mlp.down_proj.weight'][:, :32].float(), rtol=1e-2)
        assert torch.equal(w[:, 32:], ref['model.layers.2.mlp.down_proj.weight'][:, 32:]), 'other columns must not change'
        assert abs(r5['block_norm_after'] / r5['block_norm_before'] - 0.5) < 1e-2 and abs(r5['rest_norm'] - float(torch.cat([ref['model.layers.2.mlp.down_proj.weight'][:, 32:]], 1).float().norm())) < 1e-3
        m1 = Qwen3ForCausalLM(cfg).to(dt).eval()
        m1.load_state_dict(ref)
        r1 = attenuate(m1, 2, 0, 4, 1.0)
        assert float(m1.model.layers[2].mlp.down_proj.weight[:, :32].detach().abs().max()) == 0.0 and r1['block_norm_after'] == 0.0
        for bad in (dict(layer=9), dict(block=4), dict(n_blocks=5)):
            try:
                attenuate(m1, **{**dict(layer=2, block=0, n_blocks=4, dose=0.5), **bad}); raise AssertionError(f'accepted {bad}')
            except ValueError:
                pass
    # donor-capture switch: off restores the original columns bit for bit, on reproduces the edited columns bit for bit
    for dt in (torch.float32, torch.bfloat16):
        torch.manual_seed(2)
        m = Qwen3ForCausalLM(cfg).to(dt).eval()
        w, lo, hi = block_of(m, 2, 0, 4)
        orig_cols = w[:, lo:hi].detach().clone()
        sw = Switch(w, lo, hi, 0.75)
        attenuate(m, 2, 0, 4, 0.75); sw.active = True
        edited_cols = w[:, lo:hi].detach().clone()
        assert not torch.equal(orig_cols, edited_cols)
        sw.set(False); assert torch.equal(w[:, lo:hi].detach(), orig_cols), 'switch off must restore the original columns exactly'
        sw.set(True); assert torch.equal(w[:, lo:hi].detach(), edited_cols), 'switch on must reproduce the edited columns exactly'
    # equivalence with the activation hook (float32)
    hook_file = here / 'l24_mlp0_ablation_hook.py'
    if hook_file.is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location('hk', str(hook_file)); hk = importlib.util.module_from_spec(spec); spec.loader.exec_module(hk)
        torch.manual_seed(1)
        a = Qwen3ForCausalLM(cfg).float().eval(); b = Qwen3ForCausalLM(cfg).float().eval(); b.load_state_dict(a.state_dict())
        ids = torch.randint(0, 100, (2, 9))
        with torch.no_grad():
            with hk.BlockAttenuator(a, layer=2, block=0, dose=0.3, mode='ATTENUATE') as h:
                la = a(ids).logits
            attenuate(b, 2, 0, 4, 0.3)
            lb = b(ids).logits
        assert torch.allclose(la, lb, atol=1e-5), 'weight edit must equal the activation hook'
        eqv = 'equals the activation hook (float32)'
    else:
        eqv = 'hook module absent, equivalence not checked'
    # 2. guards
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = td / 'fake.py'
        fake.write_text('import sys\n' + NEED_DEF + '\ndef main():\n    need(True, "ok")\n    need(False, "Drift of frozen arm u26 m9=3")\n    '
                        'need(sys.argv[1:] == ["--gpu"], "argv")\n    need(len(sys.argv) < 9, "Unexpected active token u26")\n    need(False, "full forward does not reproduce")\n'
                        '    return 0\nif __name__ == "__main__":\n    raise SystemExit(main())\n')
        dev = td / 'dev.jsonl'
        try:
            exec_script(fake, ['--gpu'], {'_t367_need': make_need(0.0, dev, 's')})
            raise AssertionError('dose 0 must keep the soft guard strict')
        except RuntimeError as e:
            assert 'Drift' in str(e)
        try:
            exec_script(fake, ['--gpu'], {'_t367_need': make_need(0.5, dev, 's', False)})
            raise AssertionError('hard guard must still raise at dose > 0')
        except RuntimeError as e:
            assert 'full forward' in str(e)
        lines = [json.loads(x) for x in dev.read_text().splitlines()]
        assert [x['guard'] for x in lines][-1] == 'Drift of frozen arm u26 m9=3' and all(x['dose'] in (0.5,) for x in lines)
        bad = td / 'bad.py'; bad.write_text('def need(ok, msg):\n    pass\n')
        try:
            transform_source(bad.read_text(), 'bad.py'); raise AssertionError('accepted a different need()')
        except SystemExit:
            pass
        # 3. the legacy scripts, when present, accept the transformation and compile
        found = 0
        for names, _, _ in GPU.values():
            for s in names:
                p = here / s
                if p.is_file():
                    ts = transform_source(p.read_text(encoding='utf-8'), s)
                    compile(ts, s, 'exec'); found += 1
                    assert 'T286.D0' in p.read_text(encoding='utf-8') and ts.count('_t367_donor_phase(True)') == 1 and ts.count('_t367_donor_phase(False)') == 1 \
                        and ts.index('_t367_donor_phase(True)') < ts.index('_t367_donor_phase(False)'), s
        assert SOFT.match('Derive T276 u26 m9=3') and SOFT.match('Sous-cube identite != T290') and SOFT.match('Sous-cube IDENTITY_0 different de S5') \
            and SOFT.match('Token actif inattendu u26') and not SOFT.match('La passe complete ne reproduit pas les interventions C_L24 bit a bit') \
            and not SOFT.match('Rejeu identite non conforme')
        # 4. load patch with a fake gamma_local_qwen
        row = td / 'row'; (row / 'CLAUDE_CO_Q2L1').mkdir(parents=True)
        (row / 'CLAUDE_CO_Q2L1' / 'gamma_local_qwen.py').write_text(
            'import torch\nfrom transformers import Qwen3Config, Qwen3ForCausalLM\n'
            'def load_model(a, rt):\n    torch.manual_seed(0)\n    c = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4, '
            'num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64)\n    return Qwen3ForCausalLM(c).eval(), None\n')
        msgs = []
        st, G = install_load_patch(row, 2, 0, 4, 1.0, msgs.append)
        model, tok = G.load_model(None, None)
        assert st.calls == 1 and float(model.model.layers[2].mlp.down_proj.weight[:, :32].detach().abs().max()) == 0.0 and msgs
        sys.modules.pop('gamma_local_qwen', None); sys.path.remove(str(row / 'CLAUDE_CO_Q2L1'))
        # 5. sandbox
        work = td / 'quantum'; (work / 'FAST_GPU_RESULTS' / 'CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU').mkdir(parents=True)
        (work / 'FAST_GPU_RESULTS' / 'CLAUDE_S5_GAUGE_FIBER_COMMUTATION_GPU').mkdir()
        (work / 'FAST_GPU_RESULTS' / 'T347_FROZEN_BACKGROUND_DONORS_BF16').mkdir()
        (work / 'FAST_GPU_RESULTS' / 'S26_U26_FULLSOFTMAX').mkdir()
        (work / 'Qwen3-8B').mkdir(); (work / 'a.py').write_text('x=1\n'); (work / '__pycache__').mkdir()
        rows = td / 'rows'
        r = build_row(work, rows, 'R1')
        fr = r / 'FAST_GPU_RESULTS'
        assert (r / 'Qwen3-8B').is_symlink() and (r / 'a.py').is_file() and not (r / 'a.py').is_symlink() and not (r / '__pycache__').exists()
        assert (fr / 'CLAUDE_S5_GAUGE_FIBER_COMMUTATION_GPU').is_symlink() and (fr / 'T347_FROZEN_BACKGROUND_DONORS_BF16').is_symlink()
        assert not (fr / 'CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU').exists() and not (fr / 'S26_U26_FULLSOFTMAX').exists() and not fr.is_symlink()
        (fr / 'CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU').mkdir()
        os.remove(r / 'a.py'); os.symlink(work / 'a.py', r / 'a.py')
        (work / 'b.json').write_text('{}')
        build_row(work, rows, 'R1')
        assert (r / 'a.py').is_file() and not (r / 'a.py').is_symlink() and (r / 'b.json').read_text() == '{}', 'files must be real copies, symlinked files replaced'
        assert not (fr / 'CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU').is_symlink(), 'second build must keep the private output folder'
        assert not any(x.is_symlink() for x in (work / 'FAST_GPU_RESULTS').iterdir()), 'the base outputs must be untouched'
        # 6. state and meta
        ns = argparse.Namespace(row_id='R1', dose=0.5, layer=24, block=0, n_blocks=4, world=26)
        write_meta(r, ns, here)
        write_meta(r, ns, here)
        ns2 = argparse.Namespace(**{**vars(ns), 'dose': 0.25})
        try:
            write_meta(r, ns2, here); raise AssertionError('mixed settings accepted')
        except SystemExit:
            pass
        write_state(r, 'gauge', status='OK'); assert read_state(r)['gauge']['status'] == 'OK'
        assert len(CHAIN) == len(set(CHAIN)) and all(s in GPU or s in CPU for s in CHAIN)
    print(f'[t367] selftest OK (dose 0 leaves every weight unchanged, dose 1 zeroes the block only, {eqv}, soft/hard guards, '
          f'{found} legacy GPU script(s) transformed and compiled, load patch, sandbox links and private outputs, refusal of mixed settings)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--stage', choices=['preflight', 'plan', 'chain', 'collect'] + list(GPU) + list(CPU))
    ap.add_argument('--stages', help='comma-separated subset of the chain, in chain order (default: all)')
    ap.add_argument('--row-id')
    ap.add_argument('--dose', type=float)
    ap.add_argument('--layer', type=int, default=24)
    ap.add_argument('--block', type=int, default=0)
    ap.add_argument('--n-blocks', type=int, default=4)
    ap.add_argument('--world', type=int, default=26)
    ap.add_argument('--work', type=Path, default=WORK_DEFAULT)
    ap.add_argument('--rows-dir', type=Path, default=ROWS_DEFAULT)
    ap.add_argument('--scripts-dir', type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--authorize-gpu-run')
    a = ap.parse_args()
    if not sys.flags.isolated:
        os.execv(sys.executable, [sys.executable, '-I', '-u', str(Path(__file__).resolve())] + sys.argv[1:])
    selftest()
    if a.selftest:
        return 0
    if not a.stage or a.row_id is None or a.dose is None:
        ap.error('--stage, --row-id and --dose are required')
    if not ROW_ID.match(a.row_id):
        ap.error('--row-id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
    if not 0.0 <= a.dose <= 1.0:
        ap.error('--dose must be in [0, 1]')
    scripts_dir = (a.scripts_dir or a.work).resolve()
    if a.stage == 'plan':
        return plan(a, scripts_dir)
    if not a.work.is_dir():
        raise SystemExit('Working folder missing: ' + str(a.work))
    row = build_row(a.work.resolve(), a.rows_dir.resolve(), a.row_id)
    write_meta(row, a, scripts_dir)
    if a.stage == 'collect':
        return collect(a, row)
    if a.stage == 'preflight':
        rc = 0
        for s, (names, _, _) in GPU.items():
            p = resolve(scripts_dir, names)
            if p is None:
                print('[t367] MISSING (one of)', ', '.join(names), 'in', scripts_dir); rc = 1; continue
            c = [sys.executable, '-I', '-u', str(p), '--preflight', '--root', str(row), '--world', str(a.world)]
            print('[t367]', ' '.join(c), flush=True)
            rc |= subprocess.run(c, cwd=str(row)).returncode
        for s, (names, _, _) in CPU.items():
            if resolve(scripts_dir, names) is None:
                print('[t367] MISSING (one of)', ', '.join(names), 'in', scripts_dir); rc = 1
        print('[t367] preflight', 'PASS' if rc == 0 else 'FAILED', '; row folder', row)
        return rc
    if a.stage == 'chain':
        stages = [s for s in CHAIN if not a.stages or s in a.stages.split(',')]
        if any(s in GPU for s in stages) and a.authorize_gpu_run != AUTH:
            raise SystemExit('GPU chain refused: pass --authorize-gpu-run ' + AUTH)
        return run_chain(a, row, scripts_dir, stages)
    if a.stage in GPU:
        if a.authorize_gpu_run != AUTH:
            raise SystemExit('GPU stage refused: pass --authorize-gpu-run ' + AUTH)
        return run_gpu_stage(a, row, scripts_dir)
    return run_cpu_stage(a, row, scripts_dir)


if __name__ == '__main__':
    raise SystemExit(main())
