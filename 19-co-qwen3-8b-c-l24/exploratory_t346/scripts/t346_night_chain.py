#!/usr/bin/env python3
"""Sequential overnight chain on one GPU: T347 float16 C_L24 gauge acquisition and core CPU stage, then FCA-E1 jobs (pilot, sensitivity calibration).

The two GPU jobs never overlap (one 16 GiB GPU). Every step is a separate process with its own log. A failed step is recorded and the chain goes on with
the steps that do not depend on it. Nothing here reads PCI_T. Durations are measured and written to CHAIN_SUMMARY.json; nothing is estimated by this script.

  PY t346_night_chain.py --selftest
  PY t346_night_chain.py --dry-run
  PY t346_night_chain.py --run --authorize-t347 I_AUTHORIZE_T347_CL24_FLOAT16_GPU_RUN --authorize-fca-cal I_AUTHORIZE_T346_FCA_CALIBRATION_GPU_RUN
Steps, always in this order: t347_gauge, t347_core, fca_e1, fca_cal; default t347_gauge,t347_core,fca_cal (the FCA pilot is complete).
Options: --steps (subset), --work DIR, --authorize-fca for the fca_e1 step.
"""
import argparse, datetime, hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path

WORK = Path(__file__).resolve().parent
PY = Path('/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python')
AUTH_T347 = 'I_AUTHORIZE_T347_CL24_FLOAT16_GPU_RUN'
AUTH_FCA = 'I_AUTHORIZE_T346_FCA_E1_PILOT_GPU_RUN'
ORDER = ('t347_gauge', 't347_core', 'fca_e1', 'fca_cal')
DEFAULT_STEPS = ('t347_gauge', 't347_core', 'fca_cal')
AUTH_CAL = 'I_AUTHORIZE_T346_FCA_CALIBRATION_GPU_RUN'
NEEDS = {'t347_core': 't347_gauge'}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def build_steps(work, py, a_t347, a_fca, a_cal='<cal>'):
    runner = work / 't347_run_cl24_float16_suite.py'
    fca = work / 't346_fca_e1_pilot.py'
    return {
        't347_gauge': dict(files=[runner, work / 'T347_CL24_F16_SCRIPTS'], gpu=True,
                           cmd=[str(py), '-I', '-u', str(runner), '--work', str(work), '--scripts-dir', str(work / 'T347_CL24_F16_SCRIPTS'), '--stage', 'gauge', '--resume', '--authorize-gpu-run', a_t347]),
        't347_core': dict(files=[runner], gpu=False,
                          cmd=[str(py), '-I', '-u', str(runner), '--work', str(work), '--scripts-dir', str(work / 'T347_CL24_F16_SCRIPTS'), '--stage', 'core']),
        'fca_e1': dict(files=[fca, work / 'FCA027' / 'fca_bench' / '__init__.py'], gpu=True,
                       cmd=[str(py), '-I', '-u', str(fca), '--run', '--resume', '--authorize-gpu-run', a_fca]),
        'fca_cal': dict(files=[work / 't346_fca_e1_calibration.py', fca, work / 'FCA027' / 'fca_bench' / '__init__.py'], gpu=True,
                        cmd=[str(py), '-I', '-u', str(work / 't346_fca_e1_calibration.py'), '--run', '--resume', '--authorize-gpu-run', a_cal]),
    }


def mask(cmd):
    return [('<authorization>' if x.startswith('I_AUTHORIZE_') else x) for x in cmd]


def run_chain(steps, selected, outdir, dry):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = dict(started=datetime.datetime.now().isoformat(timespec='seconds'), steps={}, note='durations are measured wall-clock seconds of each step')
    failed = set()
    for name in ORDER:
        if name not in selected:
            continue
        st = steps[name]; rec = dict(command=mask(st['cmd']))
        missing = [str(p) for p in st['files'] if not p.exists()]
        if missing:
            rec.update(status='SKIPPED_MISSING_FILES', missing=missing, action='restore these files from the backup')
        elif NEEDS.get(name) in failed or (NEEDS.get(name) in selected and NEEDS[name] not in summary['steps']):
            rec.update(status='SKIPPED_DEPENDENCY_NOT_OK', depends_on=NEEDS.get(name))
        elif NEEDS.get(name) and summary['steps'].get(NEEDS[name], {}).get('status') != 'OK' and NEEDS[name] in selected:
            rec.update(status='SKIPPED_DEPENDENCY_NOT_OK', depends_on=NEEDS[name])
        else:
            rec['script_sha256'] = {str(p.name): sha_file(p) for p in st['files'] if p.is_file()}
            print(f'[NIGHT] {name}: ' + ' '.join(mask(st['cmd'])), flush=True)
            if dry:
                rec.update(status='DRY_RUN')
            else:
                log = outdir / f'{name}.log'; t0 = time.monotonic()
                with open(log, 'w', encoding='utf-8') as f:
                    f.write('# started ' + datetime.datetime.now().isoformat(timespec='seconds') + '\n'); f.flush()
                    rc = subprocess.run(st['cmd'], stdout=f, stderr=subprocess.STDOUT).returncode
                rec.update(status='OK' if rc == 0 else 'FAILED', returncode=rc, seconds=round(time.monotonic() - t0, 1), log=str(log))
                print(f'[NIGHT] {name} -> {rec["status"]} in {rec["seconds"]} s', flush=True)
        if rec['status'] not in ('OK', 'DRY_RUN'):
            failed.add(name)
        summary['steps'][name] = rec
        (outdir / 'CHAIN_SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    summary['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
    (outdir / 'CHAIN_SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    return summary


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ok = td / 'ok.py'; ok.write_text('print("fine")\n'); bad = td / 'bad.py'; bad.write_text('import sys; sys.exit(3)\n')
        steps = {'t347_gauge': dict(files=[ok], gpu=True, cmd=[sys.executable, str(bad)]),
                 't347_core': dict(files=[ok], gpu=False, cmd=[sys.executable, str(ok)]),
                 'fca_e1': dict(files=[ok], gpu=True, cmd=[sys.executable, str(ok), 'I_AUTHORIZE_X']),
                 'fca_cal': dict(files=[ok], gpu=True, cmd=[sys.executable, str(ok)])}
        s = run_chain(steps, set(ORDER), td / 'o1', False)
        assert s['steps']['t347_gauge']['status'] == 'FAILED' and s['steps']['t347_gauge']['returncode'] == 3
        assert s['steps']['t347_core']['status'] == 'SKIPPED_DEPENDENCY_NOT_OK'
        assert s['steps']['fca_e1']['status'] == 'OK' and '<authorization>' in s['steps']['fca_e1']['command']
        assert (td / 'o1' / 'fca_e1.log').read_text().strip().endswith('fine')
        steps['t347_gauge']['cmd'] = [sys.executable, str(ok)]
        s = run_chain(steps, set(ORDER), td / 'o2', False)
        assert [s['steps'][k]['status'] for k in ORDER] == ['OK', 'OK', 'OK', 'OK']
        steps['fca_e1']['files'] = [td / 'nope.py']
        s = run_chain(steps, {'fca_e1'}, td / 'o3', False)
        assert s['steps']['fca_e1']['status'] == 'SKIPPED_MISSING_FILES'
        s = run_chain(steps, {'t347_core'}, td / 'o4', True)
        assert s['steps']['t347_core']['status'] == 'DRY_RUN'
        real = build_steps(WORK, PY, '<a>', '<b>', '<c>')
        assert list(real) == list(ORDER) and 'gauge' in real['t347_gauge']['cmd'] and '--resume' in real['fca_e1']['cmd'] and '<c>' in real['fca_cal']['cmd']
    print('[NIGHT] selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--run', action='store_true')
    ap.add_argument('--work', type=Path, default=WORK); ap.add_argument('--steps', default=','.join(DEFAULT_STEPS))
    ap.add_argument('--authorize-t347'); ap.add_argument('--authorize-fca'); ap.add_argument('--authorize-fca-cal')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    selected = [s for s in a.steps.split(',') if s]
    bad = [s for s in selected if s not in ORDER]
    if bad:
        raise SystemExit('unknown steps: ' + ','.join(bad))
    if not (a.run or a.dry_run):
        ap.error('pass --run or --dry-run (or --selftest)')
    if a.run:
        need = []
        if any(s.startswith('t347') and s == 't347_gauge' for s in selected) and a.authorize_t347 != AUTH_T347:
            need.append('--authorize-t347 ' + AUTH_T347)
        if 'fca_e1' in selected and a.authorize_fca != AUTH_FCA:
            need.append('--authorize-fca ' + AUTH_FCA)
        if 'fca_cal' in selected and a.authorize_fca_cal != AUTH_CAL:
            need.append('--authorize-fca-cal ' + AUTH_CAL)
        if need:
            raise SystemExit('GPU run refused, pass: ' + ' '.join(need))
    py = PY if PY.is_file() else Path(sys.executable)
    steps = build_steps(a.work, py, a.authorize_t347 or AUTH_T347, a.authorize_fca or AUTH_FCA, a.authorize_fca_cal or AUTH_CAL)
    outdir = a.work / 'FAST_GPU_RESULTS' / ('NIGHT_CHAIN_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    s = run_chain(steps, set(selected), outdir, a.dry_run)
    print('[NIGHT] summary', outdir / 'CHAIN_SUMMARY.json')
    print(json.dumps({k: v['status'] for k, v in s['steps'].items()}, indent=1))


if __name__ == '__main__':
    main()
