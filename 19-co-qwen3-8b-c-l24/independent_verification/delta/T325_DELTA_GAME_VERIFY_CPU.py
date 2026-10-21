#!/usr/bin/env python3
"""Independent accounting and partial raw replay of T325's exact 11-role MI game.

This verifier never imports the producer; it audits all Shapley subsets and
recomputes selected coalition entropies from the original FULL12 cubes.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
OUT='FAST_GPU_RESULTS/T325_DELTA_FULL11_U26_CPU'
SOURCE='FAST_GPU_RESULTS/CLAUDE_S9_FULL12_GAUGE_CUBE_WORLD26_GPU/S9_FULL12_GAUGE_CUBES_u26.npz'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def ent(counts):
 probs=counts/counts.sum();return float(-sum(x*math.log2(x) for x in probs if x>0))

def scalar_partition(values,threshold):
 # Independent closure implementation, using sorted consecutive gap breaks.
 z=np.asarray(values,dtype=np.float64);order=sorted(range(len(z)),key=lambda k:(z[k],k))
 lab=np.zeros(len(z),np.int32);curr=0
 for pos in range(1,len(z)):
  if z[order[pos]]-z[order[pos-1]]>threshold+1e-15:curr+=1
  lab[order[pos]]=curr
 return lab

def raw_entropy(scores,delta,role_indices):
 if not role_indices:return 0.
 sig=np.stack([scalar_partition(scores[:,r],float(delta[r])) for r in role_indices],axis=1)
 _,counts=np.unique(sig,axis=0,return_counts=True)
 return ent(counts)

def replay(root,rep,vals,phi,delta):
 src=root/SOURCE
 if not src.is_file():raise FileNotFoundError(src)
 if sha(src)!=rep['source_sha256']:raise AssertionError('raw source sha mismatch')
 with np.load(src,allow_pickle=False) as f:
  cube=f['cubes'].astype(np.float64);reads=list(map(str,f['readers']));maskbits=list(map(str,f['mask_bits']))
  gauges=list(map(str,f['replica_names']))
 perm=[reads.index(s) for s in maskbits];cube=cube[:,:,perm]
 dd=np.max(np.abs(cube[1:]-cube[:1]),axis=(0,1))
 if not np.array_equal(dd,delta):raise AssertionError('delta recomputation disagrees')
 selected=(0,1,2,3,7,15,32,1023,1365,2047)
 policies=np.arange(2048,dtype=np.int64)
 n_checked=0
 for g in range(10):
  for mask in selected:
   readers=[r for r in range(11) if mask>>r&1]
   avg=sum(raw_entropy(cube[g,policies|(ctx<<11),:],delta,readers) for ctx in (0,1))/2
   if abs(avg-float(vals[g,mask]))>1e-9:
    raise AssertionError(f'RAW mismatch g={g} mask={mask} {avg} != {vals[g,mask]}')
   n_checked+=1
 return n_checked

def verify_game(v,phi):
 if v.shape!=(10,2048) or phi.shape!=(10,11):raise AssertionError('bad shapes')
 if not np.isfinite(v).all() or not np.isfinite(phi).all():raise AssertionError('nonfinite')
 if np.max(np.abs(v[:,0]))>1e-12:raise AssertionError('nonzero null coalition')
 for g in range(10):
  exact=np.zeros(11,dtype=np.float64)
  for mask in range(2047):
   k=mask.bit_count();w=1/(11*math.comb(10,k))
   for r in range(11):
    if (mask>>r)&1:continue
    gain=v[g,mask|(1<<r)]-v[g,mask]
    if gain<-1e-9:raise AssertionError('nonmonotone subset game')
    exact[r]+=max(0,float(gain))*w
  if abs(float(v[g,-1])-sum(exact))>1e-8 or not np.allclose(exact,phi[g],atol=1e-9,rtol=0):
   raise AssertionError('independent Shapley mismatch')
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
 p.add_argument('--selftest',action='store_true');p.add_argument('--no-raw',action='store_true')
 a=p.parse_args()
 if a.selftest:
  v=np.zeros((10,2048));phi=np.zeros((10,11));
  for m in range(2048):v[:,m]=float((m&1)!=0)+float((m&2)!=0)
  phi[:,0]=1;phi[:,1]=1
  verify_game(v,phi)
  print('[T325-Delta-Verify] SELFTEST PASS');return
 out=a.root/OUT;rep=json.loads((out/'RESULT.json').read_text())
 table=out/rep['npz_filename'];assert sha(table)==rep['npz_sha256']
 with np.load(table,allow_pickle=False) as z:
  v=z['v'].copy();phi=z['shapley'].copy();delta=z['delta_abs'].copy()
 verify_game(v,phi)
 n=0 if a.no_raw else replay(a.root,rep,v,phi,delta)
 result=dict(status='FULL_GAME_ARITHMETIC_VERIFIED' if a.no_raw else 'FULL_GAME_AND_SELECTED_RAW_REPLAY_VERIFIED',
    world=26,n_gauges=10,n_coalitions=2048,checked_raw_gauge_coalitions=n,
    source_sha256=rep['source_sha256'],npz_sha256=rep['npz_sha256'],
    all_gauges_positive=rep['all_gauges_positive'],min_delta_H=rep['min_delta_H'],min_delta_S=rep['min_delta_S'],
    warning='Selected raw coalition replay is not exhaustive reproduction of the entire raw game.')
 path=out/'INDEPENDENT_VERIFY.json';path.write_text(json.dumps(result,indent=2)+'\n')
 print('[T325-Delta-Verify]',result['status'],'RESULT',path)
if __name__=='__main__':main()
