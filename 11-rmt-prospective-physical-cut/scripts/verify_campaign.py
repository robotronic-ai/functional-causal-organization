#!/usr/bin/env python3
import json, subprocess, sys, tempfile, pathlib, os
HERE=pathlib.Path(__file__).resolve().parent
subprocess.run([sys.executable,str(HERE/'generate_synthetic_controls.py')],cwd=HERE,check=True)
expected={'ideal':'QUALIFIED_PASS','degraded':'ACCESS_FAIL','bypass':'BYPASS_FAIL'}
results={}
ok=True
for kind,status in expected.items():
    p=subprocess.run([sys.executable,str(HERE/'analyze_rmt_cut.py'),str(HERE/f'synthetic_{kind}.json')],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout)
    results[kind]=data
    if data['status'] != status:
        ok=False
# Ideal receiver must saturate D=p on every p.
for x in results['ideal']['frontier']:
    if abs(x['distortion_frontier_gap']) > 1e-12 or abs(x['information_gap_bits']) > 1e-12:
        ok=False
# Bypass must have positive fixed-T source dependence.
if results['bypass']['max_fixed_T_source_TV'] <= 0:
    ok=False
print(json.dumps({'status':'PASS' if ok else 'FAIL','expected_statuses':expected,'results':results},indent=2,sort_keys=True))
