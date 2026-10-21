#!/usr/bin/env python3
"""Checks the raw arrays on disk (NPZ, and the .pt raw role writes of the L1 runs) against the SHA-256 digests recorded in this bundle, and optionally the bundle
manifest.

Usage:
  /mnt/c/Users/Merien/pycharm/llama/.venv_linux/bin/python -I -u VERIFY_NPZ_SHA.py [--root QUANTUM_DIR] [--manifest]
Exit code 0 when every present file matches; 1 if any file is missing or differs.
"""
import argparse, hashlib, sys
from pathlib import Path

ROOT = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
EXPECTED = [
    ('FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU/S9_FULL12_GAUGE_CUBES_u26.npz',
     '8d421b29a240c61ba10ae005674bc27b2b4d4fcc4e575aa055d32f32d4da3cc8', 'gauge cube u26'),
    ('FAST_GPU_RESULTS/CLAUDE_S22_FROZEN_RULE_GAMMA_WORLD26_CPU/S22_CUT_TABLE_u26.npz',
     '919d99328921b61c26f6712a873b0ca79fef6fb844a7f466baae4f52d7851b4c', 'Gamma cut table u26'),
    ('FAST_GPU_RESULTS/CLAUDE_S23_DELTA_C_L24_WORLD26_CPU/S23_GAME_TABLE_u26.npz',
     'c34a1532118d328f55322d9bf184685eedc0f2d72f1f330b9632e730260eeccc', 'Delta joint-closure game table u26'),
    ('FAST_GPU_RESULTS/T325_DELTA_FULL11_U26_CPU/DELTA_FULL11_GAME.npz',
     'bf2ac04c78c0d8b8321de8c1bc4e6e6909af0ca1c8c463403381571e15897368', 'Delta per-reader game u26'),
    ('FAST_GPU_RESULTS/CLAUDE_S24_R_TOKEN_RETURN_WORLD26_GPU/S24_R_RAW_u26.npz',
     'dc30fa3587ff8b25596f28addddaf7cdf8321dc8cd6b9c8c3f5e645de6a84b6e', 'R greedy return u26'),
    ('FAST_GPU_RESULTS/CLAUDE_S26_R_SAMPLED_TOKEN_WORLD26_GPU/S26_R_SAMPLED_RAW_u26.npz',
     'b6b5a312bad8a36e9e796b937993ec850a8919ace0fc3aa61d4c38c33adafcc2', 'R sampled return u26'),
    ('FAST_GPU_RESULTS/S26_U26_FULLSOFTMAX/S26_U26_FULLSOFTMAX_EXPORT.npz',
     '14f4a3221dffb48e251a87c5b1953b18db4510032e3993f7c6556dc784ecee9a', 'R export for the independent verifier u26'),
    ('FAST_GPU_RESULTS/CLAUDE_S28_ROLE_IDENTITY_T1_WORLD26_GPU/S28_T1_GAUGE_CUBES_u26.npz',
     '51b9d5d7edeb5d5ff2f65f9b16b5fa02071168cb2b1a8760cfce3cbb318347cf', 'gauge cube at t+1 u26 (role identity)'),
    ('FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD25_GPU/S9_FULL12_GAUGE_CUBES_u25.npz',
     'c393dad78d7c7a1fa909a205dfb6bec4ccac82283c2fc53a52e85e2d775a3745', 'gauge cube u25 (replication)'),
    ('FAST_GPU_RESULTS/CLAUDE_S22_FROZEN_RULE_GAMMA_WORLD25_CPU/S22_CUT_TABLE_u25.npz',
     '213e8fa86c0fb799a6090083b56b19e7ec80d5be38d6b10d828aba6fb2edd844', 'Gamma cut table u25 (replication)'),
    ('NUIT_T127/T135/ck_L1_L23/rep0_IDENTITY_0.pt',
     '9e65daa8975fdf287798ace0e5c29628f340f436e8e7860ed456fa678342751f', 'L1 raw role writes L1_L23 rep0_IDENTITY_0'),
    ('NUIT_T127/T135/ck_L1_L23/rep1_UPDOWN_SCALE_0.pt',
     'fac7a75104ede91c4970b5e3a1780b7935072c3300643df6321a34d73d1e4b45', 'L1 raw role writes L1_L23 rep1_UPDOWN_SCALE_0'),
    ('NUIT_T127/T135/ck_L1_L23/rep2_UPDOWN_SCALE_1.pt',
     '5456f4aa2bf0be533e98a90279c4bc09df6f96715a673ba4f500b5c8d346cc6f', 'L1 raw role writes L1_L23 rep2_UPDOWN_SCALE_1'),
    ('NUIT_T127/T135/ck_L1_L23/rep3_UPDOWN_SCALE_2.pt',
     'bcc0035af7e8e5f5cea578f37cc9f48f03175dbfdb4e689dd5e4490d11828512', 'L1 raw role writes L1_L23 rep3_UPDOWN_SCALE_2'),
    ('NUIT_T127/T135/ck_L1_L23/rep4_VO_DIAG_0.pt',
     '4445dc123236e31c1e692ba1e4236b9d7b024128f22b77da952eab7e9477dcd7', 'L1 raw role writes L1_L23 rep4_VO_DIAG_0'),
    ('NUIT_T127/T135/ck_L1_L23/rep5_VO_DIAG_1.pt',
     'f523798681791f74dfb50e52c46faeed28dc928b5cfd902348d94a220414cdcf', 'L1 raw role writes L1_L23 rep5_VO_DIAG_1'),
    ('NUIT_T127/T135/ck_L1_L23/rep6_VO_DIAG_2.pt',
     '30012d54b6043f5314dbf1d0dfce1ff356b9e35082c8cde16a068b4d114e8c41', 'L1 raw role writes L1_L23 rep6_VO_DIAG_2'),
    ('NUIT_T127/T135/ck_L1_L23/rep7_QKNORM_RECIP_0.pt',
     '11b92690173c77f5e6808ec3c4a62c5e021970be28bef81f7a08520f1af63abd', 'L1 raw role writes L1_L23 rep7_QKNORM_RECIP_0'),
    ('NUIT_T127/T135/ck_L1_L23/rep8_QKNORM_RECIP_1.pt',
     'a311e48ad825e3309ee37319a5fff7a8fce3d26e4c0201b1d80cfbc2e140005b', 'L1 raw role writes L1_L23 rep8_QKNORM_RECIP_1'),
    ('NUIT_T127/T135/ck_L1_L23/rep9_QKNORM_RECIP_2.pt',
     '003057818e130fd6f0363ef984b7249316c0cf861fcfad4604e72ab305623a27', 'L1 raw role writes L1_L23 rep9_QKNORM_RECIP_2'),
    ('NUIT_T127/T135/ck_L1_L24/rep0_IDENTITY_0.pt',
     '86b4ac1aa3f22360f603614bc71ad71117d8e7934ef60090f3c74ad83835c466', 'L1 raw role writes L1_L24 rep0_IDENTITY_0'),
    ('NUIT_T127/T135/ck_L1_L24/rep1_UPDOWN_SCALE_0.pt',
     '85043d60add31cece968e6356a949113a66bbe3ed9cebf9dccf09729bea83ed9', 'L1 raw role writes L1_L24 rep1_UPDOWN_SCALE_0'),
    ('NUIT_T127/T135/ck_L1_L24/rep2_UPDOWN_SCALE_1.pt',
     'fb5ffb66e084fd048b37362d6398f0199a74defd30e0365210b79e5e4b3a31e1', 'L1 raw role writes L1_L24 rep2_UPDOWN_SCALE_1'),
    ('NUIT_T127/T135/ck_L1_L24/rep3_UPDOWN_SCALE_2.pt',
     '9ed0ffc8f82d1b51e093e0211cc2c61cbd9860e04d36537b446b65a03cd53150', 'L1 raw role writes L1_L24 rep3_UPDOWN_SCALE_2'),
    ('NUIT_T127/T135/ck_L1_L24/rep4_VO_DIAG_0.pt',
     'aeae35d6b96b5630534df75dc17f608d7cd96c7e32e216f42022f81f62ba424e', 'L1 raw role writes L1_L24 rep4_VO_DIAG_0'),
    ('NUIT_T127/T135/ck_L1_L24/rep5_VO_DIAG_1.pt',
     '33d1757827463ed434d262654c4478c520fb90207eacaf54c5b7dc48d5fe5dd6', 'L1 raw role writes L1_L24 rep5_VO_DIAG_1'),
    ('NUIT_T127/T135/ck_L1_L24/rep6_VO_DIAG_2.pt',
     '25f529021a3218fe7a1a163174060f354d4f7a78e6e2b3a55872e7139ea48075', 'L1 raw role writes L1_L24 rep6_VO_DIAG_2'),
    ('NUIT_T127/T135/ck_L1_L24/rep7_QKNORM_RECIP_0.pt',
     'b4bb369fa4e3bdb40bdced19b2194a857be20f7d6114a7e5b7eab2dd7c91f709', 'L1 raw role writes L1_L24 rep7_QKNORM_RECIP_0'),
    ('NUIT_T127/T135/ck_L1_L24/rep8_QKNORM_RECIP_1.pt',
     '7bcf7b607b60032a570397fd3011c4feffd550bcebd615596984888c372745c5', 'L1 raw role writes L1_L24 rep8_QKNORM_RECIP_1'),
    ('NUIT_T127/T135/ck_L1_L24/rep9_QKNORM_RECIP_2.pt',
     '8b0c55ab17051e5044d7437a7e99c797ef42ace13b122fd948d714a7b1a82258', 'L1 raw role writes L1_L24 rep9_QKNORM_RECIP_2'),
]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def check_manifest(here):
    bad = 0
    for line in (here / 'MANIFEST_SHA256.txt').read_text(encoding='utf-8').splitlines():
        digest, rel = line.split(None, 1)
        p = here / rel
        ok = p.is_file() and sha(p) == digest
        bad += not ok
        if not ok:
            print('MANIFEST MISMATCH', rel)
    print(f'[verify] manifest: {"OK" if not bad else f"{bad} mismatch(es)"}')
    return bad

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=ROOT); ap.add_argument('--manifest', action='store_true')
    a = ap.parse_args()
    bad = 0
    for rel, digest, label in EXPECTED:
        p = a.root / rel
        if not p.is_file():
            print(f'MISSING   {label}: {rel}'); bad += 1; continue
        ok = sha(p) == digest
        print(f'{"OK      " if ok else "MISMATCH"}  {label}: {rel}')
        bad += not ok
    if a.manifest:
        bad += check_manifest(Path(__file__).resolve().parent)
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main())
