#!/usr/bin/env python3
"""Launcher (CPU): chain run after the sampled-return acquisition (r_sampled_return_gpu.py), in this order:
  1. adapter to the T330 schema   export_t330_schema_cpu.py
  2. T330 verifier (independent)  T330_S27_DETERMINISTIC_R_BOUNDS_CPU.py --selftest, then on the export
  3. R lower bounds               r_return_bound_cpu.py
Each step is skipped when its output already exists.
Usage (from quantum/):
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u run_r_chain_cpu.py
"""
import subprocess, sys
from pathlib import Path

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
PY = '/mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python'
EXPORT = ROOT / 'FAST_GPU_RESULTS/S26_U26_FULLSOFTMAX/S26_U26_FULLSOFTMAX_EXPORT.npz'
T330_OUT = ROOT / 'FAST_GPU_RESULTS/T330_S27_U26_CPU/RESULT.json'
S27_OUT = ROOT / 'FAST_GPU_RESULTS/CLAUDE_S27_R_RETURN_BOUND_WORLD26_CPU/RESULT.json'

def run(args):
    print('[run-r] ' + ' '.join(str(a) for a in args), flush=True)
    r = subprocess.run([PY, '-I', '-u'] + [str(a) for a in args], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f'[run-r] step failed (code {r.returncode})')

def need_script(*names):
    for n in names:
        if (ROOT / n).is_file():
            return n
    raise SystemExit('[run-r] script missing in quantum/: ' + ' or '.join(names))

def main():
    if not EXPORT.is_file():
        run([need_script('export_t330_schema_cpu.py'), '--world', '26'])
    if not T330_OUT.is_file():
        t330 = need_script('T330_S27_DETERMINISTIC_R_BOUNDS_CPU.py')
        run([t330, '--selftest'])
        run([t330, '--input', EXPORT, '--output', T330_OUT])
    if not S27_OUT.is_file():
        run([need_script('r_return_bound_cpu.py'), '--world', '26'])
    print('[run-r] done:', T330_OUT, ';', S27_OUT, flush=True)

if __name__ == '__main__':
    sys.exit(main())
