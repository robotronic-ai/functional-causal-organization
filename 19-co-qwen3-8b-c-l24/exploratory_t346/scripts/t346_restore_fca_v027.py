#!/usr/bin/env python3
"""Restore the FCA-Bench v0.2.7 package from a zip archive into <work>/FCA027 (CPU, no GPU, no model).

The zip may be the stand-alone FCA-Bench archive (17-fca-bench.zip) or any archive that contains a folder
FCA_BENCH/ with fca_bench/__init__.py (for example the handoff archive). The package is extracted unmodified,
checked against its own MANIFEST.json, and the version and runner identifier are verified.

Usage: PY t346_restore_fca_v027.py --zip PATH_TO_ZIP [--work DIR] [--dest NAME]
       PY t346_restore_fca_v027.py --selftest
"""
import argparse, hashlib, json, re, shutil, sys, tempfile, zipfile
from pathlib import Path

WORK = Path(__file__).resolve().parent
WANT_VERSION = '0.2.7'
WANT_RUNNER = 'FCA027_GPU_R6'


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def find_roots(z):
    roots = []
    for n in z.namelist():
        if n.endswith('/fca_bench/__init__.py') or n == 'fca_bench/__init__.py':
            root = n[:-len('fca_bench/__init__.py')]
            t = z.read(n).decode('utf-8', 'replace')
            v = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', t); r = re.search(r'__runner_id__\s*=\s*[\'"]([^\'"]+)', t)
            roots.append((root, v.group(1) if v else None, r.group(1) if r else None))
    return roots


def restore(zip_path, work, dest):
    out = work / dest
    if out.exists():
        raise SystemExit(f'{out} already exists: refusing to overwrite (rename or remove it first)')
    with zipfile.ZipFile(zip_path) as z:
        roots = find_roots(z)
        print('[T346-RESTORE] fca_bench packages in the archive:', *[f'{r or "."} version={v} runner={x}' for r, v, x in roots], sep='\n  ')
        good = [r for r, v, x in roots if v == WANT_VERSION and x == WANT_RUNNER]
        if not good:
            raise SystemExit(f'No fca_bench {WANT_VERSION} / {WANT_RUNNER} in {zip_path}')
        root = good[0]
        members = [n for n in z.namelist() if n.startswith(root) and not n.endswith('/') and '__pycache__' not in n and not n.endswith('.pyc')
                   and not n[len(root):].startswith('results/')]
        out.mkdir(parents=True)
        for n in members:
            target = out / n[len(root):]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))
    man = out / 'MANIFEST.json'
    bad = missing = checked = 0
    if man.is_file():
        files = json.loads(man.read_text(encoding='utf-8')).get('files', {})
        for rel, h in files.items():
            p = out / rel
            if not p.is_file():
                missing += 1; continue
            checked += 1
            if sha_bytes(p.read_bytes()) != h:
                bad += 1; print('[T346-RESTORE] MISMATCH', rel)
    print(f'[T346-RESTORE] extracted {len(members)} files to {out} ; manifest checked {checked}, mismatches {bad}, missing {missing}')
    if bad or missing:
        print('[T346-RESTORE] WARNING: package differs from its MANIFEST.json; do not use it for a run')
        return 1
    print('[T346-RESTORE] OK ; package folder =', out)
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = {'FCA_BENCH/fca_bench/__init__.py': b'__version__ = "0.2.7"\n__runner_id__ = "FCA027_GPU_R6"\n',
               'FCA_BENCH/README.md': b'x\n', 'FCA_BENCH/results/big.json': b'{}',
               'old/fca_bench/__init__.py': b'__version__ = "0.2.2"\n'}
        man = {'protocol_version': WANT_VERSION, 'files': {'README.md': sha_bytes(b'x\n'), 'fca_bench/__init__.py': sha_bytes(pkg['FCA_BENCH/fca_bench/__init__.py'])}}
        zp = td / 't.zip'
        with zipfile.ZipFile(zp, 'w') as z:
            for k, v in pkg.items(): z.writestr(k, v)
            z.writestr('FCA_BENCH/MANIFEST.json', json.dumps(man))
        assert restore(zp, td, 'FCA027') == 0
        assert (td / 'FCA027' / 'fca_bench' / '__init__.py').is_file() and not (td / 'FCA027' / 'results').exists()
        try:
            restore(zp, td, 'FCA027'); raise AssertionError('overwrite must be refused')
        except SystemExit:
            pass
        (td / 'FCA027' / 'README.md').write_bytes(b'changed')
        shutil.rmtree(td / 'FCA028', ignore_errors=True)
        zbad = td / 'bad.zip'
        with zipfile.ZipFile(zbad, 'w') as z:
            z.writestr('old/fca_bench/__init__.py', b'__version__ = "0.2.2"\n')
        try:
            restore(zbad, td, 'FCA028'); raise AssertionError('wrong build must be refused')
        except SystemExit:
            pass
    print('[T346-RESTORE] selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', type=Path); ap.add_argument('--work', type=Path, default=WORK); ap.add_argument('--dest', default='FCA027')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.zip or not a.zip.is_file():
        raise SystemExit('pass --zip PATH (the FCA-Bench v0.2.7 archive, e.g. 17-fca-bench.zip or the handoff archive)')
    sys.exit(restore(a.zip, a.work, a.dest))


if __name__ == '__main__':
    main()
