#!/usr/bin/env python3
"""Structural L24 cache-bypass test in a standard cached Qwen3 causal decoder.

This is NOT an R>0 certificate, nor a reproduction of a u26 frozen donor
intervention. It tests the proposed conditional lemma with a post-L24-write
surrogate intervention, identical history, and the SAME forced next token.
It can optionally repeat in all nine locally implemented Q1 gauge recodings.
Only direct L24 return is tested; upper-layer persistent caches can differ.
"""
import argparse, contextlib, json, time
from pathlib import Path

DEFAULT_ROOT=Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
GAUGES=[('IDENTITY',0)]+[(k,i) for k in ('UPDOWN_SCALE','VO_DIAG','QKNORM_RECIP') for i in range(3)]

def selftest():
 # Toy structural DAG: the intervention is after self K/V and before upper layers.
 def step(x, intervention):
  kv_pre=x*7;write=x*3+intervention;upper=write*11
  return kv_pre,write,upper
 def next_l24(cache_pre,token):return cache_pre*5+token*13
 k0,w0,u0=step(3,0);k1,w1,u1=step(3,17)
 assert k0==k1 and u0!=u1 and next_l24(k0,4)==next_l24(k1,4)
 assert next_l24(k0,4)!=next_l24(k1,5)
 print('[T325-R] SELFTEST PASS: no direct L24 bypass, token-clamp invariant')

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
 ap.add_argument('--prompt',default='The capital of France is')
 ap.add_argument('--token-id',type=int,default=None)
 ap.add_argument('--strength',type=float,default=100.)
 ap.add_argument('--all-gauges',action='store_true')
 ap.add_argument('--selftest',action='store_true');ap.add_argument('--overwrite',action='store_true')
 a=ap.parse_args();selftest()
 if a.selftest:return
 import torch
 from transformers import AutoModelForCausalLM, AutoTokenizer
 if not torch.cuda.is_available():raise SystemExit('GPU required: CUDA unavailable')
 model_path=a.root/'Qwen3-8B';t0=time.monotonic()
 tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
 try:
  model=AutoModelForCausalLM.from_pretrained(str(model_path),dtype=torch.bfloat16,device_map='cuda',local_files_only=True)
 except TypeError:
  model=AutoModelForCausalLM.from_pretrained(str(model_path),torch_dtype=torch.bfloat16,device_map='cuda',local_files_only=True)
 model.eval();layers=model.model.layers;L=layers[24]
 ids=tokenizer(a.prompt,return_tensors='pt')['input_ids'].to(next(model.parameters()).device)
 if ids.shape[1]<2:raise SystemExit('Prompt should have at least 2 tokens')
 Q=None
 if a.all_gauges:
  import sys
  sys.path.insert(0,str(a.root/'CLAUDE_CO_Q2L1'))
  import qwen_local_transition as Q
  expected=[(kind,seed) for kind,seed in GAUGES[1:]]
  if [tuple(x) for x in Q.GAUGE_SET]!=expected:raise SystemExit('Q1 gauge set mismatch')

 def one(prefix_intervention, next_token, capture=True):
  """First pass with optional L24 MLP post-projection edit, then fixed token."""
  state={'phase':'t'}; seen={}
  def edit(mod,inputs,out):
   if state['phase']=='t' and prefix_intervention:
    perturb=torch.zeros_like(out);perturb[...,0]=prefix_intervention
    return out+perturb
   return None
  def record(key):
   def fn(mod,inputs):
    if state['phase']=='t1':seen[key]=inputs[0].detach().clone()
   return fn
  def capture_output(mod,inputs,output):
   if state['phase']=='t1':
    x=output[0] if isinstance(output,(tuple,list)) else output
    seen['layer24_output']=x.detach().clone()
  handles=[L.mlp.down_proj.register_forward_hook(edit)]
  try:
   with torch.inference_mode():
    t_out=model(input_ids=ids,use_cache=True,return_dict=True)
    next_logits=t_out.logits[0,-1].detach()
    auto_token=int(next_logits.argmax())
    if next_token is None:next_token=auto_token
    handles.extend([L.register_forward_pre_hook(record('layer24_input')),
      L.self_attn.o_proj.register_forward_pre_hook(record('o_proj_input')),
      L.mlp.down_proj.register_forward_pre_hook(record('down_proj_input')),
      L.register_forward_hook(capture_output)])
    state['phase']='t1'
    input_next=torch.tensor([[next_token]],dtype=ids.dtype,device=ids.device)
    next_out=model(input_ids=input_next,past_key_values=t_out.past_key_values,use_cache=True,return_dict=True)
    torch.cuda.synchronize()
    return {'readers':{k:v.cpu() for k,v in seen.items()},'next_logits':next_logits.cpu(),
            'auto_token':auto_token,'fixed_token':next_token,
            't1_next_logits':next_out.logits[0,-1].detach().cpu()}
  finally:
   for h in handles:h.remove()

 reports=[]
 gauges=GAUGES if a.all_gauges else GAUGES[:1]
 for kind,seed in gauges:
  cm=contextlib.nullcontext() if kind=='IDENTITY' else Q.Gauged(model,24,kind,seed)
  with cm:
   baseline=one(0,a.token_id)
   token=int(a.token_id if a.token_id is not None else baseline['auto_token'])
   perturbed=one(a.strength,token)
   comparison={}
   for key,base in baseline['readers'].items():
    other=perturbed['readers'][key]
    comparison[key]={'bitwise_equal':bool(torch.equal(base,other)),
        'max_abs_difference':float((base.float()-other.float()).abs().max())}
   if not comparison:raise SystemExit('No L24 readers captured')
   ok=all(v['bitwise_equal'] for v in comparison.values())
   rec=dict(gauge=f'{kind}_{seed}',fixed_next_token=token,
      unforced_argmax_token_control=baseline['auto_token'],
      unforced_argmax_token_perturbed=perturbed['auto_token'],
      all_L24_t1_readers_bitwise_equal=ok,comparisons=comparison,
      t1_final_logits_bitwise_equal=bool(torch.equal(baseline['t1_next_logits'],perturbed['t1_next_logits'])))
   reports.append(rec)
   print(f'[T325-R] {rec["gauge"]}: L24(t+1) bitwise_equal={ok}, '
         f'tokens {rec["unforced_argmax_token_control"]}/{rec["unforced_argmax_token_perturbed"]}',flush=True)
 out=a.root/'FAST_GPU_RESULTS'/'T325_R_L24_CACHE_BYPASS_GPU'
 out.mkdir(parents=True,exist_ok=True);p=out/'RESULT.json'
 if p.exists() and not a.overwrite:raise SystemExit('Output exists. Pass --overwrite to update.')
 data=dict(status='STRUCTURAL_CACHE_BYPASS_TEST_ONLY',world='generic_prompt_not_u26',
  tested_intervention='surrogate L24 mlp.down_proj forward-output perturbation at t',
  token_clamp='fixed identical token at t+1', source_role='L24',target_role='L24 t+1',
  cache_bypass_l24_supported=all(r['all_L24_t1_readers_bitwise_equal'] for r in reports),
  q1_gauges_tested=len(reports),reentry_positive_certified=False,
  R_intrinsic_certified=False,notes=[
   'The same-role recurrent R requires frozen-donor u26 intervention and positive free-token effect',
   'This test does not establish temporal Q1 causal-role identity or K_eff state sufficiency',
   'Caches of upper layers may change and later layers may depend on them; claim is specifically about L24'],
  reports=reports,elapsed_seconds=round(time.monotonic()-t0,3))
 p.write_text(json.dumps(data,indent=2)+'\n')
 print('[T325-R] RESULT',p)
 if not data['cache_bypass_l24_supported']:raise SystemExit('Clamped-token L24 bypass test FAILED')
if __name__=='__main__':main()
