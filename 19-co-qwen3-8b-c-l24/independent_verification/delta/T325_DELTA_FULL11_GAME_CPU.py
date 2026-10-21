#!/usr/bin/env python3
"""Exact 11-player, 2048-coalition finite Delta game on the u26 FULL12 cube.

P: 11 retained independent Bernoulli donor bits; C: excluded donor bit 11.
q_donor: 2^-12 on 12 donor masks, NOT q_bal.
For each gauge/context independently, scalar outputs form connected delta components.
Y_S is the T229-style TUPLE of these fixed rolewise component labels; no
cross-gauge decoder and no post-hoc policy-bit selection.
v(S) = E_C I(P;Y_S | C) = E_C H(Y_S | C) for deterministic cubes.
An independent rolewise closure is NOT joint 11-dimensional graph closure;
this explicit reader convention is needed for T229's coordinate projections.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
SOURCE='FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU/S9_FULL12_GAUGE_CUBES_u26.npz'
SOURCE_SHA='8d421b29a240c61ba10ae005674bc27b2b4d4fcc4e575aa055d32f32d4da3cc8'
OUT='FAST_GPU_RESULTS/T325_DELTA_FULL11_U26_CPU'
RETAINED=tuple(range(11)); CONTEXT=11; M=11; N=1<<M
GAUGES=['IDENTITY_0']+[f'{kind}_{i}' for kind in ['UPDOWN_SCALE','VO_DIAG','QKNORM_RECIP'] for i in range(3)]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def labels_of_scalar(y,delta):
 """Components of the one-dimensional <=delta graph, in input order."""
 order=np.argsort(y,kind='mergesort')
 splits=np.concatenate(([1],(np.diff(y[order])>delta+1e-15).astype(np.int32)))
 ranks=np.cumsum(splits,dtype=np.int32)-1
 out=np.empty(len(y),dtype=np.int32);out[order]=ranks
 return out

def entropy_of_counts(counts):
 p=counts/counts.sum()
 return float(-np.dot(p,np.log2(p)))

def compress_pair(old,new,n=N):
 """Exact intersection of TWO ALREADY CLOSURED coordinate partitions."""
 key=old.astype(np.int64)*n+new.astype(np.int64)
 _,idx,cnt=np.unique(key,return_inverse=True,return_counts=True)
 return idx.astype(np.int32),entropy_of_counts(cnt)

def evaluate_gauge(y,delta):
 """y shape (2,2048,12), delta shape (12)."""
 if y.shape!=(2,N,12):raise ValueError('wrong input shape')
 labels=np.empty((2,12,N),dtype=np.int32)
 for c in range(2):
  for r in range(12):labels[c,r]=labels_of_scalar(y[c,:,r],float(delta[r]))
 parts=np.zeros((2,1<<M,N),dtype=np.int32)
 vals=np.zeros(1<<M,dtype=np.float64)
 for mask in range(1,1<<M):
  bit=mask & -mask;r=bit.bit_length()-1;parent=mask^bit
  for c in range(2):
   parts[c,mask],e=compress_pair(parts[c,parent],labels[c,r]);vals[mask]+=e/2
 phi=np.zeros(M)
 for mask in range((1<<M)-1):
  k=mask.bit_count();w=1/(M*math.comb(M-1,k))
  for r in range(M):
   if not (mask>>r)&1:
    gain=vals[mask|(1<<r)]-vals[mask]
    if gain < -1e-9: raise AssertionError('MI coalitional monotonicity failed')
    phi[r]+=w*max(0,gain)
 if abs(phi.sum()-vals[-1])>1e-8:raise AssertionError('Shapley efficiency failed')
 # P* operational signatures, measured on ALL 12 readers at BOTH contexts.
 all0,_=compress_pair(parts[0,-1],labels[0,11]);all1,_=compress_pair(parts[1,-1],labels[1,11]);
 pstar,m_dummy=compress_pair(all0,all1)
 nstar=int(np.unique(pstar).size)
 d=math.log2(nstar) if nstar>1 else 0
 cap=float(vals[-1]/d) if d>0 else 0.
 ps=phi/phi.sum() if phi.sum()>0 else np.zeros(M)
 H=-sum(float(p)*math.log2(float(p)) for p in ps if p>0)
 pi_h=float((2**H-1)/(M-1)) if phi.sum()>0 else 0.
 pi_s=float((1/(ps@ps)-1)/(M-1)) if phi.sum()>0 else 0.
 return vals,phi,dict(Pstar_operational_count=nstar,Pstar_log2=d,game_total_bits=float(vals[-1]),
     delta_cap=cap,pi_H=pi_h,pi_S=pi_s,delta_H=cap*pi_h,delta_S=cap*pi_s,
     positive_shapley_roles=int(np.sum(phi>1e-12)),
     role_shares=ps.tolist()),labels

def selftest():
 d=np.ones(12)*.1
 y=np.zeros((2,N,12),dtype=np.float64)
 bits=np.arange(N,dtype=np.int32)
 y[:,:,0]=((bits&1)!=0)[None,:].astype(float)
 y[:,:,5]=((bits&2)!=0)[None,:].astype(float)
 # All other bits are distinguishable in one additional reader, so P*=2048.
 y[:,:,11]=bits[None,:].astype(float)*10
 v,p,r,_=evaluate_gauge(y,d)
 assert v[0]==0 and v[-1]==2 and abs(p.sum()-2)<1e-10
 assert abs(p[0]-1)<1e-10 and abs(p[5]-1)<1e-10 and r['Pstar_operational_count']==2048
 assert r['delta_H']>0 and r['delta_S']>0
 # Isolated single informational reader => zero inter-role propagation.
 y[:,:,5]=0
 v,p,r,_=evaluate_gauge(y,d)
 assert abs(r['pi_H'])<1e-12 and abs(r['pi_S'])<1e-12 and v[-1]==1
 print('[T325-Delta] PASS: exact coalitions/Shapley, 11-bit P*, two contexts, one-role null')

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT)
 a.add_argument('--selftest',action='store_true');a.add_argument('--overwrite',action='store_true')
 args=a.parse_args();selftest()
 if args.selftest:return
 source=args.root/SOURCE
 if not source.is_file():raise SystemExit(f'NPZ not found: {source}')
 sourcehash=sha(source)
 if sourcehash!=SOURCE_SHA:raise SystemExit(f'SOURCE_SHA mismatch: {sourcehash}')
 with np.load(source,allow_pickle=False) as z:
  cubes=z['cubes'].astype(np.float64);gauges=list(map(str,z['replica_names']))
  readers=list(map(str,z['readers']));bits=list(map(str,z['mask_bits']))
 assert cubes.shape==(10,4096,12) and gauges==GAUGES
 perm=[readers.index(role) for role in bits];cubes=cubes[:,:,perm]
 if not np.isfinite(cubes).all():raise SystemExit('Nonfinite cubes')
 delta=np.abs(cubes[1:]-cubes[:1]).max(axis=(0,1))
 policy=np.arange(N,dtype=np.int64)
 fullmask=np.stack((policy,policy|(1<<CONTEXT)),axis=0)
 vtable=np.empty((10,1<<M),dtype=np.float64);ptable=np.empty((10,M));records=[]
 start=time.monotonic()
 for g,name in enumerate(gauges):
  vals,phi,stats,_=evaluate_gauge(cubes[g,fullmask,:],delta)
  vtable[g]=vals;ptable[g]=phi
  records.append({'gauge':name,**stats,'shapley_bits_by_role':dict(zip(bits[:M],map(float,phi)))})
  print(f'[T325-Delta] {name}: |P*|={stats["Pstar_operational_count"]} '
        f'Delta_H={stats["delta_H"]:.8f} Delta_S={stats["delta_S"]:.8f}',flush=True)
 out=args.root/OUT;out.mkdir(parents=True,exist_ok=True)
 report=out/'RESULT.json';table=out/'DELTA_FULL11_GAME.npz'
 if (report.exists() or table.exists()) and not args.overwrite:raise SystemExit('Outputs exist; use --overwrite')
 with table.open('wb') as f:
  np.savez_compressed(f,v=vtable,shapley=ptable,gauge_names=np.array(gauges),reader_names=np.array(bits[:M]),
                      delta_abs=delta,source_sha256=np.array(sourcehash))
 doc={'status':'FINITE_DELTA_C_L24_U26_EVALUATED__NO_GLOBAL_PSTAR_CLAIM','world':26,'q':'q_donor',
      'source_sha256':sourcehash,'n_gauges':10,'n_coalitions':1<<M,'n_donor_masks':4096,'n_contexts':2,
      'role_count':11,'context_role':bits[CONTEXT],'pstar_definition':
       'Distinct causal response signatures over BOTH contexts and ALL 12 individual delta-closure readers; operational one-layer estimate, not all-future P*',
      'reader_equivalence':'per-reader scalar <=delta connected components; Y_S is ordered tuple of these fixed reader classes',
      'shapley_game':'v(S)=E_C I(P;Y_S|C), P=11 independent donor bits, C=last donor bit',
      'denominator':'log2(number of operational P* response signatures), per gauge; not automatically 11',
      'source_policy':'NOT the historical T324 two-bit battery; all 2048 donor policies are tested',
      'per_gauge':records,'min_delta_H':min(r['delta_H'] for r in records),
      'min_delta_S':min(r['delta_S'] for r in records),
      'min_delta_cap':min(r['delta_cap'] for r in records),
      'all_gauges_positive':all(r['delta_H']>0 and r['delta_S']>0 for r in records),
      'npz_filename':table.name,'npz_sha256':sha(table),'elapsed_seconds':round(time.monotonic()-start,3),
      'intrinsic_global_delta_certified':False}
 report.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print('[T325-Delta] RESULT:',report,'NPZ:',table)
if __name__=='__main__':main()
