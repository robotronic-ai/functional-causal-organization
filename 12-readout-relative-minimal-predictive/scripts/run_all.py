#!/usr/bin/env python3
"""Run all protocol verification scripts."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "verify_all_readouts_orbit_corollary.py",
    "verify_finite_predictive_theorem.py",
    "verify_u0_linear_corollary.py",
    "verify_controlled_finite_theorem.py",
    "verify_markov_expectation_corollary.py",
]

for name in SCRIPTS:
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=str(HERE))

print("\nALL VERIFIERS PASSED")
