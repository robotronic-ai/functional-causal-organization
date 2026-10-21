#!/usr/bin/env python3
"""Install the TURN_053 T347 suite (frozen-donor transport hotfix) from the handoff archive into the working folder (CPU, no GPU, no model).

Every file of <archive>/06_CURRENT_T347/ except README.md is written to <work>/<same relative path>. A file that already exists with different content is
first renamed to <name>.bak_turn052 (never deleted, never overwritten twice). The installed files are checked against TURN_053_FILE_MANIFEST.json.

  PY t347_deploy_turn053.py --zip PATH_TO_HANDOFF_ZIP [--work DIR] [--dry-run]
  PY t347_deploy_turn053.py --selftest
"""
import argparse, hashlib, json, sys, tempfile, zipfile
from pathlib import Path

WORK = Path(__file__).resolve().parent
SUBDIR = '06_CURRENT_T347/'
SKIP = {'README.md'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def deploy(zip_path, work, dry):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        root = next((n[:n.index(SUBDIR) + len(SUBDIR)] for n in names if SUBDIR in n and n.endswith('TURN_053_FILE_MANIFEST.json')), None)
        if root is None:
            raise SystemExit(f'{zip_path}: no {SUBDIR}TURN_053_FILE_MANIFEST.json (this is not the TURN_053 handoff archive)')
        manifest = json.loads(z.read(root + 'TURN_053_FILE_MANIFEST.json').decode('utf-8'))
        want = {e['path']: e['sha256'] for e in manifest['files']}
        members = {n[len(root):]: n for n in names if n.startswith(root)}
        bad = [p for p, h in want.items() if p not in members or sha(z.read(members[p])) != h]
        if bad:
            raise SystemExit('archive content differs from TURN_053_FILE_MANIFEST.json: ' + ', '.join(bad))
        new = same = backed = 0
        for rel, n in sorted(members.items()):
            if rel in SKIP:
                continue
            data = z.read(n); dst = work / rel
            if dst.is_file():
                if sha(dst.read_bytes()) == sha(data):
                    same += 1; continue
                bak = dst.with_name(dst.name + '.bak_turn052')
                if bak.exists():
                    raise SystemExit(f'{bak} already exists: refusing to overwrite a backup (move it first)')
                print(('[DEPLOY] would back up ' if dry else '[DEPLOY] backup ') + f'{dst.name} -> {bak.name}')
                if not dry:
                    dst.rename(bak)
                backed += 1
            else:
                new += 1
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
    if dry:
        print(f'[DEPLOY] dry run: {new} new, {same} identical, {backed} would be backed up')
        return 0
    wrong = [p for p, h in want.items() if p not in SKIP and (not (work / p).is_file() or sha((work / p).read_bytes()) != h)]
    print(f'[DEPLOY] installed: {new} new, {same} already identical, {backed} replaced (old copy kept as .bak_turn052)')
    if wrong:
        print('[DEPLOY] MISMATCH after install:', ', '.join(wrong)); return 1
    print('[DEPLOY] OK ; all manifest files match TURN_053 ;', work)
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        files = {'t347_run.py': b'new runner\n', 'T347_CL24_F16_SCRIPTS/a.py': b'new a\n', 'README.md': b'readme\n'}
        man = {'files': [{'path': k, 'sha256': sha(v)} for k, v in files.items()]}
        zp = td / 'h.zip'
        with zipfile.ZipFile(zp, 'w') as z:
            for k, v in files.items(): z.writestr('x/06_CURRENT_T347/' + k, v)
            z.writestr('x/06_CURRENT_T347/TURN_053_FILE_MANIFEST.json', json.dumps(man))
        w = td / 'w'; (w / 'T347_CL24_F16_SCRIPTS').mkdir(parents=True)
        (w / 't347_run.py').write_bytes(b'old runner\n'); (w / 'README.md').write_bytes(b'mine\n')
        assert deploy(zp, w, True) == 0 and (w / 't347_run.py').read_bytes() == b'old runner\n'
        assert deploy(zp, w, False) == 0
        assert (w / 't347_run.py').read_bytes() == b'new runner\n' and (w / 't347_run.py.bak_turn052').read_bytes() == b'old runner\n'
        assert (w / 'T347_CL24_F16_SCRIPTS' / 'a.py').read_bytes() == b'new a\n' and (w / 'README.md').read_bytes() == b'mine\n'
        assert deploy(zp, w, False) == 0
        bad = td / 'b.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('x/06_CURRENT_T347/t347_run.py', b'tampered'); z.writestr('x/06_CURRENT_T347/TURN_053_FILE_MANIFEST.json', json.dumps({'files': [{'path': 't347_run.py', 'sha256': sha(b'other')}]}))
        try:
            deploy(bad, td / 'w2', False); raise AssertionError('tampered archive must be refused')
        except SystemExit:
            pass
        try:
            deploy(td / 'h.zip', td / 'w', True) if False else None
        except SystemExit:
            pass
    print('[DEPLOY] selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', type=Path); ap.add_argument('--work', type=Path, default=WORK); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.zip or not a.zip.is_file():
        raise SystemExit('pass --zip PATH (the CO_PCIT_CLAUDE_HANDOFF_THROUGH_TURN_053 archive)')
    sys.exit(deploy(a.zip, a.work, a.dry_run))


if __name__ == '__main__':
    main()
