import json
import subprocess
import sys

scripts = [
    "verify_discounted_bernoulli.py",
    "verify_epsilon_nontransitivity.py",
]

results = {}
ok = True

for script in scripts:
    proc = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        ok = False
        results[script] = {
            "returncode": proc.returncode,
            "stderr": proc.stderr,
        }
        continue
    data = json.loads(proc.stdout)
    results[script] = data
    if data.get("status") != "PASS":
        ok = False

print(json.dumps({
    "status": "PASS" if ok else "FAIL",
    "results": results,
}, indent=2, sort_keys=True))
