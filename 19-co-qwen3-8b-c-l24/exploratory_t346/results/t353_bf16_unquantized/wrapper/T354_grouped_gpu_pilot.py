#!/usr/bin/env python3
"""T354 HOTFIX: T353 single-command GPU acquisition using the EXISTING frozen T346 FCA-E1 and PCI_T L27 pilots.

This program does not rewrite the scientific instruments. It verifies content hashes,
runs each existing pilot in its own process (FCA followed by PCI_T), preserves original
per-instrument outputs, and creates a single return ZIP. It never extracts ZIP files.

Modes: --preflight (CPU only), --selftest (CPU only), --run (GPU, explicit approval
string required), --collect (CPU, combine existing completed records and results).

One u26 realization family only: descriptive paired performance and dose response;
not confirmatory CO-vs-rivals, not functional closure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

NAME='T354_T353_GROUPED_GPU_VENV_SYMLINK_HOTFIX_V1'
AUTH='I_AUTHORIZE_T353_GROUPED_GPU_RUN'
ARMS=['U26_BASELINE','U26_SHAM','U26_ABLATE_025','U26_ABLATE_050','U26_ABLATE_100']
BASE_FILES={
 't346_fca_e1_pilot.py':'c0eb6944ab6801ca9a0fce37c9ede08dbb664c783cc983d741957261020ae893',
 't346_pcit_l27_pilot.py':'49c6a7f33e27b5d746b8b3c7c4955afe9d1e2c82b5f6114fd82a050865459901',
 'qwen3_8b_pcit_v020_r01.py':'47def0be29eb761d2afc0a3c97faf8114fed4af3a01968803de38b94f4c6e2f7',
 'qwen3_8b_pcit_directional_dev_v021x5_1.py':'b5072f85504a2ae1829b80e9dcfa719f7a44fbf006e4ffad95c572e9bd121238',
 'qwen3_8b_pcit_multiamplitude_dev_v021x1.py':'3210bc513b7794e6bbe17455415124213b7ded800c13cc345664175a3dd98a8a',
}
EXTRA_FILES={'FCA027/fca_bench/__init__.py': 'cb31f384bc3219b7b902be401615d82deb8c5804f0527ab43523b2db115e16fc',
 'FCA027/fca_bench/adapters.py': '501ceebf6c679375cadf4e571cb88ce55087fdce447e2e89027aab27dd85b009',
 'FCA027/fca_bench/analyze_specificity.py': '636712bd4ffdc850426e66345577e8938da3b7bd4dce4f96d9a1aec2feb04d88',
 'FCA027/fca_bench/audit_b1.py': '743c1c94cfbe3473df407072598dfdc1c22f28901166e487453a870e11feba1e',
 'FCA027/fca_bench/core.py': '6cf59e6d017f8a217d3812c4aaeebabd709009ee8533c1d72208aafa59c51f12',
 'FCA027/fca_bench/harness.py': '20a92c00a343b29cb6f8150e3298b5d93869f11102b54a9f1bf25212c8ed4da1',
 'FCA027/fca_bench/mechanistic.py': '79f30d41f8583c687d3e08d9a8f2edfc7d534ce3efdfc5c796fa166dae640fb4',
 'FCA027/fca_bench/merge_results.py': 'c8a5ebcf2abeac4a0d66346c371bcc2b48a0a12188948df4ce32b3bc9093d6db',
 'FCA027/fca_bench/metrics.py': 'ae72eb0e42b918c1fc69caa24e41c4fceee377d0bc66c13a648a2596e817b7d3',
 'FCA027/fca_bench/preflight.py': '261fcc94959ae36fbcdfef9416039d68d11a1ee462bc8daf284fb153278dcf0e',
 'FCA027/fca_bench/progress.py': '4c00185fb75debe4e41d5c65e22614bd44a4c6e8f61b48c36ac5f2ea8bf313f0',
 'FCA027/fca_bench/run.py': '84a92977dd1f745c0dc7a89e43505c33d29fbf0fd6b6345017fb53ef0f295ca4',
 'FCA027/fca_bench/sealing.py': 'a359b27e74eea14aac73b7df6a65b9bc45c0b5b4632e3fb1be2012919f6748bc',
 'FCA027/fca_bench/tests_agency.py': '22a1c968663968aa7f6196d190c9ddc51e4843daa4d5c9c0a4833e4283995787',
 'FCA027/fca_bench/tests_attention_schema.py': '3da1424f2ef12c3511c62b07e6938c1b3e38f115c3fca55b68d6fe100d0d2452',
 'FCA027/fca_bench/tests_boundary.py': 'fbcc2f79c9363abd6c533ee2a29034b6d55494b0d73e22ad8aa15b8379d13710',
 'FCA027/fca_bench/tests_control.py': 'f2e3f37ed6c41b225e7316a4d3943fa8aac6c105bb8d02411f104eb1843e7dbd',
 'FCA027/fca_bench/tests_erasure.py': 'c533146055c20087e74a5853e44d6631632f2d3cb6e5e5144735dfc0e3c56ad0',
 'FCA027/fca_bench/tests_interrupt.py': '9514c5e273afa2468c70ff9241fe7528523e6ff41f88dc62e0d113264fff4c12',
 'FCA027/fca_bench/tests_latent_state.py': '4b2ae8e6c650c042b19d6bd90c1a87fb29a335faeb1a37936f7a729a423ae1e2',
 'FCA027/fca_bench/tests_meta.py': '539346852f074f3dec2cc50c90d1c1705d027eb90232a5af3ef5bf4a625c3ae4',
 'FCA027/fca_bench/tests_mutable_state.py': '5a265217f7ec410ba1fb97f98287c21dfe0e1110e6326a61f596c757fad92617',
 'FCA027/fca_bench/tests_recurrence.py': '50e8b7cd7f06d2d831895b094d222e6fd5dae748a3b8ab21a669e3bf5d874aa5',
 'FCA027/fca_bench/tests_selfpred.py': '95b57906f9de0c432613f556e113105fe75b49b610ec6309a98dde450e28aeee',
 'FCA027/fca_bench/tests_temporal_dynamics.py': 'b39bce3971b79b854b64b3078c3c97f73ef188fe32a4b5ab663064c9d1515022',
 'FCA027/fca_bench/tests_time.py': '1c32837941d585b41e6d38374229fdf289f82598039eb862698aa7b6d966fc9a',
 'FCA027/fca_bench/tests_workspace.py': '5160496ca9a01832285df939b53d954c231e1c8caba62b459a0b1602e2364622',
 'FCA027/fca_bench/validate.py': '95994b6659f699de5d04831f73ef5f37f3cf48fdd7c8363ff9576edbeb9bea28'}
PCI_ZIP='exports/pcit_qwen3_8b/R01_PCIT_V020/PCI_T_QWEN3_8B_RESULTS_v0.20.zip'
PCI_ZIP_SHA='a1c805e8aa369c92ec670deb52290e3e7f61862d6d31f5277c66fcd7360ae237'
OUT_FCA='FAST_GPU_RESULTS/CLAUDE_T346_FCA_E1_PILOT_T353'
OUT_PCI='FAST_GPU_RESULTS/CLAUDE_T346_PCIT_L27_PILOT_T353'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def check_environment(repo,model,fast=False):
 errors=[];checks=[]; hashes={}; required={**BASE_FILES,**EXTRA_FILES}
 for rel,expected in required.items():
  p=repo/rel
  if not p.is_file():errors.append(f'MISSING:{rel}');continue
  got=sha(p);hashes[rel]=got
  if got!=expected:errors.append(f'SHA_MISMATCH:{rel}:expected={expected}:actual={got}')
  else:checks.append('HASH_OK:'+rel)
 # Explicitly avoid fallback to quantized model or another model family.
 if not model.is_dir():errors.append('MODEL_DIR_MISSING:'+str(model))
 else:
  if not (model/'config.json').is_file():errors.append('MODEL_CONFIG_MISSING:'+str(model/'config.json'))
  else:
   cfg=load(model/'config.json')
   arches=' '.join(cfg.get('architectures',[])).lower()
   if 'qwen3' not in arches and cfg.get('model_type')!='qwen3':errors.append('MODEL_NOT_QWEN3')
   hashes['MODEL/config.json']=sha(model/'config.json')
 if not (repo/PCI_ZIP).is_file():errors.append('PCI_V020_RESULT_MISSING:'+PCI_ZIP)
 else:
  v=sha(repo/PCI_ZIP);hashes[PCI_ZIP]=v
  if v!=PCI_ZIP_SHA:errors.append('PCI_V020_RESULT_HASH_MISMATCH')
  else:checks.append('HASH_OK:'+PCI_ZIP)
 ini=repo/'FCA027'/'fca_bench'/'__init__.py'
 if ini.is_file():
  s=ini.read_text(encoding='utf-8')
  if '__version__ = "0.2.7"' not in s or '__runner_id__ = "FCA027_GPU_R6"' not in s:
   errors.append('FCA_BUILD_NOT_V027_R6')
 # Exact frozen contract source files are present in the original T346/T347 package; do not silently reinterpret.
 if (repo/'T346_PILOT_ACQUISITION_LOCK.json').is_file():
  checks.append('T346_PILOT_LOCK_PRESENT')
 report=dict(schema='T353_GROUPED_GPU_PREFLIGHT_V1',ok=not errors,
             gpu_invoked=False,eligible_for_confirmatory_rival_test=False,
             run_type='EXPLORATORY_ONE_U26_FAMILY',dtype='bfloat16',quantization='none',
             stages=['FCA_E1_20_INSTANCES_5_ARMS','PCI_T_L27_6_EPISODES_5_ARMS'],
             model_dir=str(model),errors=errors,checks=checks,sha256=hashes)
 return report

def commands(repo,model,python, resume=False):
 c=[python,'-I','-u']
 fca=c+[str(repo/'t346_fca_e1_pilot.py'),'--run','--work',str(repo),'--model-dir',str(model),
        '--fca-dir',str(repo/'FCA027'),'--quantization','none','--dtype','bfloat16',
        '--instances','20','--out-suffix','_T353','--authorize-gpu-run',
        'I_AUTHORIZE_T346_FCA_E1_PILOT_GPU_RUN']
 pci=c+[str(repo/'t346_pcit_l27_pilot.py'),'--run','--work',str(repo),'--model-dir',str(model),
        '--dtype','bfloat16','--episodes','6','--out-suffix','_T353',
        '--authorize-gpu-run','I_AUTHORIZE_T346_PCIT_L27_PILOT_GPU_RUN']
 if resume:
  fca.append('--resume');pci.append('--resume')
 return [('FCA_E1',fca),('PCI_T_L27',pci)]

def stream_run(cmd, logfile, env=None):
 with Path(logfile).open('a',encoding='utf-8') as f:
  p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,
                     bufsize=1,env=env)
  for line in p.stdout:
   f.write(line);f.flush();print(line,end='',flush=True)
  return p.wait()

def already_complete(repo,stage):
 """Resume guard: do not rerun an instrument whose RESULT and all records are complete.
 T346 uses exclusive RESULT.json creation; repeating an already finished stage would fail.
 """
 base=repo/(OUT_FCA if stage=='FCA_E1' else OUT_PCI)
 result=base/'RESULT.json';recs=base/'records.jsonl';hdr=base/'RUN_HEADER.json'
 if not (result.exists() and recs.exists() and hdr.exists()):return False
 try:
  v=load(result);h=load(hdr)
  if v.get('sham_vs_baseline',{}).get('identical') is not True:return False
  if h.get('arms')!=ARMS:return False
  expected=100 if stage=='FCA_E1' else 60
  rows=[json.loads(line) for line in recs.read_text(encoding='utf-8').splitlines() if line.strip()]
  if len(rows)!=expected:return False
  names={(r['arm'],r.get('instance_id',r.get('episode_id')),r.get('source_token_fraction')) for r in rows}
  if len(names)!=expected:return False
  return set(v.get('arms',{}))==set(ARMS)
 except (ValueError,KeyError,TypeError,OSError):return False

def summary_for(repo):
 files={'FCA_E1':repo/OUT_FCA/'RESULT.json','PCI_T_DS_L27':repo/OUT_PCI/'RESULT.json'}
 missing=[k for k,v in files.items() if not v.is_file()]
 data={k:load(p) for k,p in files.items() if p.is_file()}
 scores={};pair={}; sham={}
 for k, v in data.items():
  field='FCA_E1' if k=='FCA_E1' else 'PCI_T_DS_L27'
  scores[k]={arm:{'dose':x.get('dose'),'value':x.get(field),
                      'n_valid':x.get('n_valid',x.get('n_episodes')),
                      'interval95_descriptive':x.get(field+'_ci95_descriptive')} for arm,x in v.get('arms',{}).items()}
  pair[k]=v.get('paired_vs_sham',{}); sham[k]=v.get('sham_vs_baseline',{})
  if sham[k].get('identical') is not True:missing.append(k+'_SHAM_NOT_IDENTICAL')
 return dict(schema='T353_EXPLORATORY_GROUPED_SCORECARD_V1',
             status='COMPLETE_DESCRIPTIVE_ONLY' if not missing else 'INCOMPLETE_OR_SHAM_FAILED',
             missing_or_failed=missing,model='Qwen3-8B',quantization='none',dtype='bfloat16',
             organization_component_certifications=['Gamma','Delta','R','Lambda0'],
             co_component_changes_measured=False,
             cannot_claim_CO_mediation_without_matched_numeric_CO=True,
             cannot_claim_rival_superiority_one_family=True,
             independent_u26_families=1,
             scores_by_arm=scores,paired_vs_sham=pair,sham_identity=sham,
             source_results={k:str(p) for k,p in files.items()})

def collect(repo,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 result=summary_for(repo);dump(out/'T353_GROUPED_SCORECARD.json',result)
 included=[]
 zip_out=out/'T353_GPU_RETURN.zip'
 with zipfile.ZipFile(zip_out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
  for p in [out/'T353_GROUPED_SCORECARD.json',out/'T353_PREFLIGHT.json',out/'T353_RUN_LOG.json']:
   if p.is_file():z.write(p,arcname=p.name);included.append(p.name)
  for key, rel in [('FCA_E1',OUT_FCA),('PCI_T_L27',OUT_PCI)]:
   base=repo/rel
   for fname in ('RUN_HEADER.json','RESULT.json','pilot_table.csv','records.jsonl'):
    p=base/fname
    if p.is_file():
     name=f'{key}/{fname}';z.write(p,arcname=name);included.append(name)
  for pname in ('T353_FCA_E1_GPU.log','T353_PCI_T_L27_GPU.log'):
   p=out/pname
   if p.is_file():z.write(p,arcname=pname);included.append(pname)
 with zipfile.ZipFile(zip_out) as z:
  assert z.testzip() is None
 return zip_out,result,included

def selftest():
 import tempfile
 with tempfile.TemporaryDirectory() as td:
  r=Path(td)
  cmds=commands(r,r/'Qwen3-8B','python',resume=True)
  a,b=cmds[0][1],cmds[1][1]
  assert '--quantization' in a and a[a.index('--quantization')+1]=='none'
  assert '--dtype' in a and a[a.index('--dtype')+1]=='bfloat16'
  assert b[b.index('--dtype')+1]=='bfloat16'
  assert '--resume' in a and '--resume' in b
  assert '--episodes' in b and b[b.index('--episodes')+1]=='6'
  assert a[a.index('--instances')+1]=='20'
  assert sha(Path(__file__))
  import os
  fake=r/'venv'/'bin'/'python';fake.parent.mkdir(parents=True)
  fake.symlink_to(Path(sys.executable).resolve())
  assert str(fake)==os.path.abspath(str(fake))
  assert os.path.abspath(str(fake))!=str(fake.resolve()), 'symlink regression'
  check=check_python_runtime(str(fake))
  assert check['ok'] or check['error'].startswith('MISSING_PACKAGES_')
  for v in ('FCA_E1','PCI_T_L27'):
   p=r/(OUT_FCA if v=='FCA_E1' else OUT_PCI);p.mkdir(parents=True)
   field='FCA_E1' if v=='FCA_E1' else 'PCI_T_DS_L27'
   (p/'RESULT.json').write_text(json.dumps({'arms':{'U26_SHAM':{field:0.7,'dose':0}},'sham_vs_baseline':{'identical':True},'paired_vs_sham':{}}))
  package,score,files=collect(r,r/'out')
  assert package.is_file() and score['status']=='COMPLETE_DESCRIPTIVE_ONLY'
  (r/OUT_PCI/'RESULT.json').unlink()
  assert summary_for(r)['status']=='INCOMPLETE_OR_SHAM_FAILED'
 print('T353 SELFTEST PASS: frozen 20/6 design, bf16 no quantization, resume, independent scores, missing-result refusal, return ZIP; CPU only')

def check_python_runtime(python_command):
 """Inspect the actual subprocess interpreter, without loading the model or invoking CUDA.

 Passing /path/to/venv/bin/python directly (possibly a symlink) preserves the
 virtual environment. Never call Path.resolve() on it.
 """
 import textwrap
 report={'schema':'T354_PYTHON_RUNTIME_V1','command':python_command,'ok':False,
         'packages':{},'error':None}
 if not Path(python_command).is_file():
  report['error']='INTERPRETER_NOT_FOUND';return report
 snippet=("import sys, json, importlib\n"
          "mods = ('numpy','torch','transformers')\n"
          "present = {}\n"
          "for name in mods:\n"
          " try: importlib.import_module(name); present[name] = True\n"
          " except Exception as e: present[name] = False\n"
          "print(json.dumps({'sys_executable':sys.executable,'sys_prefix':sys.prefix,'sys_base_prefix':sys.base_prefix,'modules':present},sort_keys=True))\n")
 try:
  p=subprocess.run([python_command,'-I','-c',snippet],text=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
  if p.returncode!=0:
   report['error']='PYTHON_IMPORT_PROBE_FAILED: '+p.stderr[-1000:]
  else:
   info=json.loads(p.stdout.strip())
   report['sys_executable']=info['sys_executable']
   report['sys_prefix']=info['sys_prefix']
   report['sys_base_prefix']=info['sys_base_prefix']
   report['packages']=info['modules']
   missing=[k for k,v in report['packages'].items() if not v]
   report['ok']=not missing
   if missing:report['error']='MISSING_PACKAGES_IN_SELECTED_PYTHON: '+','.join(missing)
 except (OSError,ValueError,subprocess.TimeoutExpired) as e:
  report['error']=str(e)
 return report

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 modes=ap.add_mutually_exclusive_group(required=True)
 for k in ('preflight','selftest','run','collect'):modes.add_argument('--'+k,action='store_true')
 ap.add_argument('--repo',type=Path,default=Path.cwd());ap.add_argument('--model-dir',type=Path)
 ap.add_argument('--out',type=Path,default=None);ap.add_argument('--resume',action='store_true')
 ap.add_argument('--approve',default='');ap.add_argument('--python',type=Path,default=Path(sys.executable))
 args=ap.parse_args()
 if args.selftest:selftest();return
 repo=args.repo.resolve();model=(args.model_dir or repo/'Qwen3-8B').resolve()
 out=(args.out or repo/'FAST_GPU_RESULTS'/'T353_GROUPED_PILOT').resolve()
 if args.preflight:
  pre=check_environment(repo,model);dump(out/'T353_PREFLIGHT.json',pre)
  print(json.dumps({'status':'PASS_PREFLIGHT_CPU_ONLY' if pre['ok'] else 'BLOCKED_NO_GPU',
                    'issues':pre['errors'],'report':str(out/'T353_PREFLIGHT.json')},indent=2))
  if not pre['ok']:sys.exit(2)
  return
 if args.collect:
  f,s,files=collect(repo,out)
  print(json.dumps({'status':s['status'],'return_zip':str(f),'files':files},indent=2));return
 if args.approve!=AUTH:
  raise SystemExit('GPU launch explicitly refused. Supply --approve '+AUTH+' only when you authorize the real acquisition.')
 pre=check_environment(repo,model);dump(out/'T353_PREFLIGHT.json',pre)
 if not pre['ok']:
  print('GPU NOT STARTED:',*pre['errors'],sep='\n- ');sys.exit(2)
 log={'schema':'T353_LAUNCH_V1','approved_by_cli':True,'gpu_acquisition_attempted':True,
      'run_kind':'EXPLORATORY_ONE_U26_FAMILY','out':str(out),'stages':[],
      'hashes':pre['sha256'],'FCA_quantization':'none','dtype':'bfloat16'}
 dump(out/'T353_RUN_LOG.json',log)
 # Run each existing instrument once, separately, to preserve unmodified official scoring.
 # Retain the venv entry-point path (often a symlink to /usr/bin/python3.12).
 # pathlib.Path.resolve() dereferences that symlink and loses venv site-packages.
 python_command=os.path.abspath(os.fspath(args.python))
 runtime=check_python_runtime(python_command)
 dump(out/'T354_RUNTIME_PREFLIGHT.json',runtime)
 if not runtime['ok']:
  raise SystemExit('PYTHON RUNTIME BLOCKED BEFORE GPU: '+str(runtime['error']))
 print('T354 interpreter:',python_command,'| runtime packages:',runtime['packages'],flush=True)
 log['python_command']=python_command
 dump(out/'T353_RUN_LOG.json',log)
 for name,cmd in commands(repo,model,python_command,args.resume):
  current={'stage':name,'started_epoch':time.time(),'command':cmd,'status':'RUNNING'}
  log['stages'].append(current);dump(out/'T353_RUN_LOG.json',log)
  if args.resume and already_complete(repo,name):
   print('SKIP COMPLETED STAGE '+name+' (resume, complete records and results verified)',flush=True)
   current.update(return_code=0,finished_epoch=time.time(),status='SKIPPED_ALREADY_COMPLETE')
   dump(out/'T353_RUN_LOG.json',log)
   continue
  print('RUNNING '+name,flush=True)
  rc=stream_run(cmd,out/f'T353_{name}_GPU.log')
  current.update(return_code=rc,finished_epoch=time.time(),status='DONE' if rc==0 else 'FAILED')
  dump(out/'T353_RUN_LOG.json',log)
  if rc!=0:
   p,sc,fl=collect(repo,out)
   raise SystemExit(f'{name} FAILED with exit code {rc}. Other stage not run. Upload {p}; see log. To resume later, pass --resume.')
 p,sc,fl=collect(repo,out)
 print(json.dumps({'status':sc['status'],'return_zip':str(p),'gpu_stages':[r['stage'] for r in log['stages']],
                   'claim':'EXPLORATORY ONLY / no CO superiority or closure certification'},indent=2))
 if sc['status']!='COMPLETE_DESCRIPTIVE_ONLY':sys.exit(3)

if __name__=='__main__':main()
