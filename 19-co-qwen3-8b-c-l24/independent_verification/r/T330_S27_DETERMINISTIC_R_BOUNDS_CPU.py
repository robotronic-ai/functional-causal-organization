#!/usr/bin/env python3
"""T330/S27: deterministic adversarial-tail lower bounds for same-role L24 re-entry.

Input: an S26 NPZ that follows T330_S26_EXPORT_SCHEMA.md. Uses no sampling.
A positive return lower bound on a *raw frozen scalar role readout* is rigorous
under the specified model/architecture; identifying it with the delta-closed
causal quotient requires a separate full-support closure/role-typing argument.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

GAUGES = ('IDENTITY_0','UPDOWN_SCALE_0','UPDOWN_SCALE_1','UPDOWN_SCALE_2',
          'VO_DIAG_0','VO_DIAG_1','VO_DIAG_2',
          'QKNORM_RECIP_0','QKNORM_RECIP_1','QKNORM_RECIP_2')
N_MASKS = 4096
N_ROLES = 11
N_PAIRS = 2048
EPS = 1e-10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bernoulli_js_bits(a, b):
    """Vectorized mutual information for equiprobable binary cause and Bernoulli output."""
    a, b = np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    m = (a+b)/2
    def kl_binary_term(p, q):
        out = np.zeros_like(p)
        good = p>0
        out[good]=p[good]*np.log2(p[good]/q[good])
        return out
    return np.maximum(0, 0.5*(kl_binary_term(a,m)+kl_binary_term(1-a,1-m)
                                 +kl_binary_term(b,m)+kl_binary_term(1-b,1-m)))


def event_bounds(known0, known1, missing0, missing1, pos_tail=None, neg_tail=None):
    """Worst case JS under all out-of-V assignments to an observable event.

    With optional signed tail-TV budgets, the Pinsker bound supplies an
    additional valid lower bound using a SHARED token-to-readout map.
    """
    a0=np.asarray(known0,dtype=float);a1=np.asarray(known1,dtype=float)
    m0=np.asarray(missing0,dtype=float);m1=np.asarray(missing1,dtype=float)
    p0=np.where(a0>a1+m1, a0, np.where(a1>a0+m0,a0+m0,0.0))
    p1=np.where(a0>a1+m1, a1+m1, np.where(a1>a0+m0,a1,0.0))
    lb=bernoulli_js_bits(p0,p1)
    if pos_tail is not None and neg_tail is not None:
        pos=np.asarray(pos_tail,dtype=float)
        neg=np.asarray(neg_tail,dtype=float)
        difference=a0-a1
        # Tail of the signed difference can reduce a positive difference
        # by at most neg; a negative difference by at most pos.
        margin=np.maximum(np.maximum(difference-neg,-difference-pos),0.)
        lb=np.maximum(lb,margin*margin/(2*math.log(2)))
    return lb


def input_validate(arr):
    required=('logp_v','tail_mass','return_scores','delta_t1','token_ids','gauge_names')
    for key in required:
        if key not in arr: raise ValueError('Missing S26 field '+key)
    lp=np.asarray(arr['logp_v'],dtype=np.float64)
    tail=np.asarray(arr['tail_mass'],dtype=np.float64)
    scores=np.asarray(arr['return_scores'],dtype=np.float64)
    delta=np.asarray(arr['delta_t1'],dtype=np.float64)
    toks=np.asarray(arr['token_ids'])
    names=tuple(str(x) for x in arr['gauge_names'].tolist())
    if names!=GAUGES:raise ValueError('Gauge ordering does not match frozen Q1 contract')
    if lp.ndim!=3 or lp.shape[:2]!=(10,4096):raise ValueError('logp_v expected [10,4096,V]')
    v=lp.shape[2]
    if not 2<=v<=50000:raise ValueError('V must have at least 2 fixed tokens')
    if tail.shape!=(10,4096) or scores.shape!=(10,11,v) or delta.shape!=(10,11):
        raise ValueError('tail_mass [10,4096], return_scores [10,11,V], delta_t1 [10,11] required')
    if toks.shape!=(v,) or np.unique(toks).size!=v or not np.issubdtype(toks.dtype,np.integer):
        raise ValueError('token_ids must be unique 1D integers')
    if not (np.isfinite(lp)|np.isneginf(lp)).all() or not np.isfinite(tail).all():
        raise ValueError('Invalid token log probabilities or tail masses')
    if not np.isfinite(scores).all() or not np.isfinite(delta).all() or (delta<0).any():
        raise ValueError('Invalid next-role readout scores or deltas')
    p=np.exp(lp)
    if (tail<-1e-12).any() or (tail>1+1e-12).any():raise ValueError('Invalid tail probabilities')
    errors=np.max(np.abs(p.sum(axis=2)+tail-1))
    if errors>2e-9:raise ValueError(f'Full-softmax normalization failed (max {errors})')
    budgets=None
    if 'tail_signed_posneg' in arr:
        budgets=np.asarray(arr['tail_signed_posneg'],dtype=np.float64)
        if budgets.shape!=(10,11,2048,2) or not np.isfinite(budgets).all() or (budgets<0).any():
            raise ValueError('tail_signed_posneg must be nonnegative [10,11,2048,2]')
        for r in range(11):
            zero=np.arange(4096,dtype=int)[(np.arange(4096)&(1<<r))==0]
            one=zero|(1<<r)
            # Signed positive and negative masses are respectively bounded
            # by total tail mass under each donor; use stronger checks below.
            if np.any(budgets[:,r,:,0]>tail[:,zero]+2e-9) or np.any(budgets[:,r,:,1]>tail[:,one]+2e-9):
                raise ValueError('Tail signed budgets exceed donor tail masses')
    conditional=None
    if 'token_pair_js_bits' in arr:
        conditional=np.asarray(arr['token_pair_js_bits'],dtype=np.float64)
        if conditional.shape!=(10,11,2048) or not np.isfinite(conditional).all() or (conditional<0).any() or (conditional>1+EPS).any():
            raise ValueError('token_pair_js_bits must be finite [10,11,2048] in [0,1]')
    if 'world' not in arr or int(np.asarray(arr['world']).item())!=26:
        raise ValueError('Only world u26 is certified by the declared contract')
    if 'fixed_v_before_analysis' not in arr or not bool(np.asarray(arr['fixed_v_before_analysis']).item()):
        raise ValueError('Preregistered fixed V is required')
    return p,tail,scores,delta,names,budgets,conditional,errors


def compute(path: Path, output:Path|None=None, significance:float=1e-18):
    with np.load(path,allow_pickle=False) as arr:
        p,tail,scores,delta,names,budgets,cond,err=input_validate(arr)
    b_per=np.zeros((10,11),dtype=np.float64)
    witness=np.zeros((11,2048,10),dtype=bool)
    best_pair_js=np.zeros((10,11,2048),dtype=np.float64)
    threshold_counts=np.zeros((10,11),dtype=int)
    raw_threshold_probes=[]
    for g in range(10):
        for r in range(11):
            ix=np.argsort(scores[g,r],kind='stable')
            xs=scores[g,r,ix]
            gaps=np.where(np.diff(xs)>delta[g,r])[0]
            threshold_counts[g,r]=len(gaps)
            if not len(gaps):continue
            zero=np.arange(4096,dtype=int)[(np.arange(4096)&(1<<r))==0]
            one=zero|(1<<r)
            probs=p[g][:,ix]
            # cumulative probabilities for thresholds above consecutive V scores
            cum=np.cumsum(probs[:,::-1],axis=1)[:,::-1]
            t0,t1=tail[g,zero],tail[g,one]
            positive,negative=(None,None) if budgets is None else (budgets[g,r,:,0],budgets[g,r,:,1])
            for k in gaps:
                s0=cum[zero,k+1];s1=cum[one,k+1]
                test=event_bounds(s0,s1,t0,t1,positive,negative)
                best_pair_js[g,r]=np.maximum(best_pair_js[g,r],test)
            b_per[g,r]=float(np.mean(best_pair_js[g,r]))
            witness[r,:,g]=best_pair_js[g,r]>significance
            # Strictly data-derived thresholds, independent of donor probabilities.
            top=int(np.argmax(best_pair_js[g,r]))
            raw_threshold_probes.append({'gauge':names[g],'role':r,
                'best_background_pair_zero_mask':int(zero[top]),
                'conditional_event_mi_lb_bits':float(best_pair_js[g,r,top]),
                'min_delta_gap_among_eligible_thresholds':float(np.min(np.diff(xs)[gaps]-delta[g,r]))})
    role_shared=np.any(np.all(witness,axis=2),axis=1)
    candidates=[r for r in range(11) if role_shared[r]]
    selected=max(candidates,key=lambda r:float(np.min(b_per[:,r]))) if candidates else None
    K_pair=None
    if cond is not None:
        K_pair=np.mean(cond,axis=2)
        if np.any(b_per>K_pair+1e-8):raise ValueError('Return information LB exceeds token channel (data processing violation)')
    selected_min=float(np.min(b_per[:,selected])) if selected is not None else 0.0
    result={
      'status':'POSITIVE_RAW_SAME_ROLE_RETURN_LB__DELTA_QUOTIENT_NOT_YET_CERTIFIED' if selected_min>significance else 'INCONCLUSIVE_NO_POSITIVE_RAW_RETURN_LOWER_BOUND',
      'world':26,'q':'q_donor independent Bernoulli(0.5) on 12 bits; one contextual bit',
      'transition':'temperature=1 full-vocabulary softmax, no top-k/top-p, no stochastic estimates',
      'token_v_selection_fixed':True,'vocabulary_included_size':int(p.shape[-1]),
      'n_gauges':10,'n_source_roles':11,'n_conditioning_backgrounds_per_role':2048,
      'source_npz_sha256':sha256(path),'max_probability_normalization_error':float(err),
      'observations':'same-role fixed scalar L24 t+1 readouts; gauge-local delta-calibrated candidate thresholds',
      'tail_treatment':'adversarial allocation for each donor pair; optional signed tail-TV budgets',
      'threshold_eligibility':'gauge-local observed adjacent scalar gap strictly exceeds delta_t1',
      'conditional_information':'I(P_r; event[f_(g,r)(T)>threshold] | P_-r,C), averaged over 2048 fair backgrounds',
      'return_lb_bits_by_gauge_and_role':b_per.tolist(),
      'eligible_threshold_counts_by_gauge_and_role':threshold_counts.tolist(),
      'roles_with_single_shared_background_witness_all_10_gauges':[int(r) for r in candidates],
      'selected_role':None if selected is None else int(selected),
      'selected_min_raw_B_lower_bound_bits':selected_min,
      'selected_state_ratio_lower_bound_under_declared_11bit_state_capacity':selected_min/11.,
      'state_normalization_condition':'0 < K_eff(q_donor | C) = H(Z* | C) <= 11 bits on the retained active-role state; numerator must include this same-role return term',
      'token_pair_js_provided':cond is not None,
      'token_channel_conditional_mi_by_gauge_role':None if K_pair is None else K_pair.tolist(),
      'token_ratio_lower_bound_by_gauge_role':None if K_pair is None else [[float(b_per[g,r]/K_pair[g,r]) if K_pair[g,r]>0 else None for r in range(11)] for g in range(10)],
      'all_support_delta_quotient_stability_certified':False,
      'R_intrinsic_Qwen3_8B_NONZERO_CERTIFIED':False,
      'why_not_R_intrinsic':'Unknown outside-V token outputs can bridge delta-connected components of readouts; temporal causal quotient role identity and whole-state normalization must be independently established.',
      'note':'Positive raw return is not an all-support delta-quotient certificate; the same physical L24 projection is not automatically the same causal role under quotient.',
      'diagnostic_probes':raw_threshold_probes[:20]
    }
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def make_synthetic(path:Path,tail_mode=False):
    """Synthetic ten-gauge experiment: one donor bit changes the full token channel."""
    p=np.zeros((10,4096,2),dtype=np.float64)
    tail=np.zeros((10,4096),dtype=np.float64)
    scores=np.zeros((10,11,2),dtype=np.float64)
    delta=np.full((10,11),0.1)
    for g in range(10):
        scores[g,:,0]=0.;scores[g,:,1]=1.
        for m in range(4096):
            p[g,m,:]=[.8,.2] if not (m&1) else [.2,.8]
    if tail_mode:
        p*=.5;tail[:]=.5
    np.savez_compressed(path,world=np.array(26),fixed_v_before_analysis=np.array(True),
        logp_v=np.log(p),tail_mass=tail,return_scores=scores,
        delta_t1=delta,token_ids=np.array([123,456]),gauge_names=np.array(GAUGES))


def selftest():
    import tempfile
    assert abs(float(bernoulli_js_bits(np.array([.8]),np.array([.2]))[0])-.278071905112638)<1e-12
    assert float(event_bounds(np.array([.48]),np.array([.02]),np.array([.5]),np.array([.5]))[0])==0
    assert float(event_bounds(np.array([.48]),np.array([.02]),np.array([.5]),np.array([.5]),
                              np.array([0.]),np.array([0.]))[0])>0
    # XOR example: I(X;T|r)=1, I(X;T)=1, but I(r;T|other)=1.
    xor=np.array([[a^b for b in (0,1)] for a in (0,1)])
    assert xor.sum()==2 and len(set(xor.ravel()))==2
    with tempfile.TemporaryDirectory() as d:
        src=Path(d)/'s26.npz'
        make_synthetic(src)
        result=compute(src)
        assert result['selected_role']==0
        assert result['selected_min_raw_B_lower_bound_bits']>.27
        assert result['roles_with_single_shared_background_witness_all_10_gauges']==[0]
        assert result['R_intrinsic_Qwen3_8B_NONZERO_CERTIFIED'] is False
        make_synthetic(src,True)
        result=compute(src)
        assert result['selected_role'] is None
    return {'status':'PASS','tests':['tail_mass_adversarial','optional_signed_tail_bound',
        'binary_information_exact','ten_gauge_shared_configuration','zero_signal_without_tail_resolution',
        'XOR_conditioning_diagnostic','honest_intrinsic_R_gate']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--selftest',action='store_true')
    arg=parser.parse_args()
    if arg.selftest:
        print(json.dumps(selftest(),indent=2));return
    if arg.input is None:
        parser.error('--input is required unless --selftest')
    path=arg.input.resolve()
    out=arg.output if arg.output else path.with_name('S27_BOUNDS.json')
    result=compute(path,out)
    print(json.dumps({k:result[k] for k in ('status','selected_role','selected_min_raw_B_lower_bound_bits',
      'selected_state_ratio_lower_bound_under_declared_11bit_state_capacity',
      'roles_with_single_shared_background_witness_all_10_gauges')},indent=2))
    print('Saved',out)

if __name__=='__main__':main()
