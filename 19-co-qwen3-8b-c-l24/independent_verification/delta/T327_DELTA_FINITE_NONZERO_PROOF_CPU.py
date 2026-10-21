#!/usr/bin/env python3
"""T327: independent finite-donor Delta nonzero certificate and refinement lemma.

Reads the already acquired exact 11-player T325 Shapley game. No GPU or new
interventions. Uses only q_donor, 11 retained roles and the one-layer L25
frozen role readouts. A richer future may refine those readouts, but this lemma
DOES NOT cover role re-identification, a larger policy alphabet or a changed
system boundary. Does not claim global, substrate-free Delta certification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
GAME=Path('FAST_GPU_RESULTS/T325_DELTA_FULL11_U26_CPU/DELTA_FULL11_GAME.npz')
OUTPUT=Path('FAST_GPU_RESULTS/T327_DELTA_FINITE_NONZERO_U26_CPU/RESULT.json')
GAME_SHA='bf2ac04c78c0d8b8321de8c1bc4e6e6909af0ca1c8c463403381571e15897368'
ROLE_COUNT=11
INFORMATIVE=(1,5,6,7,8)
GAUGES=['IDENTITY_0']+[f'{kind}_{i}' for kind in ('UPDOWN_SCALE','VO_DIAG','QKNORM_RECIP') for i in range(3)]


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            d.update(block)
    return d.hexdigest()


def ratio_lower_bound(n,k,minimum_singleton_bits=1):
    """Prove positive H/Simpson propagation under ANY readout refinement.

    If k distinct readers have v({r}) >= a, Shapley_i >= a/n from its
    empty-coalition term. Since v(N)<=H(P)<=n, each normalized share
    p_i>=a/n**2. Therefore sum p_i**2 <= (1-(k-1)e)^2+(k-1)e**2.
    Shannon effective diversity is >= inverse Simpson.
    Delta_cap >= a/n because |P*|<=2**n on this finite policy domain.
    """
    if n<2 or not (2<=k<=n):raise ValueError('Invalid role count')
    a=Fraction(minimum_singleton_bits)
    epsilon=a/(n*n)
    if k*epsilon>1:raise ValueError('Invalid informative role lower bound')
    simpson_upper=(1-(k-1)*epsilon)**2+(k-1)*epsilon**2
    spread=(1/simpson_upper-1)/(n-1)
    cap_lb=a/n
    return {'n_roles':n,'n_informative_readers':k,
       'minimum_normalized_share':str(epsilon),'simpson_index_upper':str(simpson_upper),
       'pi_S_lower_bound':str(spread),'pi_H_lower_bound':str(spread),
       'delta_cap_lower_bound':str(cap_lb),
       'delta_H_lower_bound':str(spread*cap_lb),
       'delta_S_lower_bound':str(spread*cap_lb),
       'both_lower_bound_float':float(spread*cap_lb)}


def exact_or_game(n,informative):
    ids=np.arange(1<<n,dtype=np.int64)
    informative_mask=sum(1<<i for i in informative)
    return (np.bitwise_and(ids,informative_mask)!=0).astype(np.float64)


def verify_game(game,phi):
    n=ROLE_COUNT
    if game.shape!=(10,1<<n) or phi.shape!=(10,n):
        raise ValueError('Expected 10 x 2048 exact game and 10 x 11 Shapley')
    if not np.isfinite(game).all() or not np.isfinite(phi).all():
        raise ValueError('Nonfinite evidence')
    exact=exact_or_game(n,INFORMATIVE)
    if not np.array_equal(game,np.broadcast_to(exact,game.shape)):
        raise ValueError('Reported game is not the exact five-reader OR game')
    target=np.zeros(n);target[list(INFORMATIVE)]=1/len(INFORMATIVE)
    if not np.allclose(phi,np.broadcast_to(target,phi.shape),atol=1e-12,rtol=0):
        raise ValueError('Shapley does not match exact combinatorial expectation')
    if not np.allclose(phi.sum(axis=1),game[:,-1],atol=1e-12,rtol=0):
        raise ValueError('Shapley efficiency failure')
    return exact,target


def selftest():
    oracle=exact_or_game(ROLE_COUNT,INFORMATIVE)
    phi=np.zeros(ROLE_COUNT);phi[list(INFORMATIVE)]=1/5
    verify_game(np.tile(oracle,(10,1)),np.tile(phi,(10,1)))
    try:
        bad=oracle.copy();bad[1]=1-bad[1]
        verify_game(np.tile(bad,(10,1)),np.tile(phi,(10,1)))
        raise AssertionError('Corrupted coalition was accepted')
    except ValueError:pass
    bound=ratio_lower_bound(11,5)
    assert bound['delta_S_lower_bound']=='474/753115',bound
    assert abs(bound['both_lower_bound_float']-.0006293859503528678)<1e-17
    assert ratio_lower_bound(3,2)['both_lower_bound_float']>0
    print('[T327-Delta] SELFTEST PASS: exhaustive OR game, Shapley, refinement bound, corruption rejection')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--game',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    a=parser.parse_args()
    selftest()
    if a.selftest:return
    path=a.game if a.game else a.root/GAME
    if not path.is_file():raise SystemExit('Game NPZ unavailable: '+str(path))
    source_hash=sha(path)
    if source_hash!=GAME_SHA:raise SystemExit('Pinned exact game SHA mismatch: '+source_hash)
    with np.load(path,allow_pickle=False) as z:
        v=np.array(z['v'],dtype=np.float64)
        phi=np.array(z['shapley'],dtype=np.float64)
        gauge_names=list(map(str,z['gauge_names']))
        reader_names=list(map(str,z['reader_names']))
    if gauge_names!=GAUGES or len(reader_names)!=11:
        raise SystemExit('Unexpected gauge or reader order')
    verify_game(v,phi)
    theorem=ratio_lower_bound(11,5)
    report={
       'status':'NONZERO_CERTIFIED_DELTA_H_AND_DELTA_S_ON_U26_FINITE_DONOR_CONTRACT',
       'world':26,'q':'q_donor = Bernoulli(0.5)^12, one context bit',
       'n_roles':11,'n_contexts':2,'gauges_checked':10,
       'coalitions_per_gauge':2048,'game_entries_verified':int(v.size),
       'full_game_exact':'v(S)=1 bit if and only if S intersects {1,5,6,7,8}; otherwise 0',
       'five_informative_reader_indices':list(INFORMATIVE),
       'five_informative_reader_names':[reader_names[r] for r in INFORMATIVE],
       'delta_cap_operational_reported':1.0,
       'delta_H_on_measured_one_layer_interface':0.4,
       'delta_S_on_measured_one_layer_interface':0.4,
       'shapley_per_informative_role':0.2,
       'finite_alphabet_refinement_theorem':theorem,
       'refinement_conditions':[
           'The intervention alphabet remains eleven independent binary donor roles, with the twelfth site treated as context.',
           'Each future role readout includes or refines its original L25 scalar delta class.',
           'The same eleven distinct certified causal roles, declared boundaries and q_donor are preserved.',
           'The Shapley game remains monotone v(S)=I(P;Y_S|C); no unobserved policy classes enter the denominator.'
       ],
       'excluded_claims':[
           'No proof for a changed physical or causal boundary.',
           'No certification for continuous interventions outside the frozen donor alphabet.',
           'No assertion that the operational two-class P* equals the complete Qwen macro quotient.'
       ],
       'source_npz_sha256':source_hash,
       'original_full12_npz_sha256':'8d421b29a240c61ba10ae005674bc27b2b4d4fcc4e575aa055d32f32d4da3cc8'
    }
    out=a.output if a.output else a.root/OUTPUT
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not a.overwrite:raise SystemExit('Output exists: '+str(out))
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('[T327-Delta] RESULT:',out)
    print('[T327-Delta] measured Delta_H=Delta_S=0.4; refinement lower bound',theorem['both_lower_bound_float'])

if __name__=='__main__':main()
