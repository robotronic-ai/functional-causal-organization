#!/usr/bin/env python3
import argparse, json, math
from collections import defaultdict

CATS = ['BIT0','BIT1','EMPTY','OTHER']
P_GRID = [0.0,0.1,0.2,0.3,0.4,0.5]
QUAL = {0,2,4,6}
CONF = {1,3,5,7}

def h2(x):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x*math.log2(x)-(1-x)*math.log2(1-x)

def rate_dist_binary(D):
    return 0.0 if D >= 0.5 else 1.0-h2(D)

def tv(a,b):
    return 0.5*sum(abs(x-y) for x,y in zip(a,b))

def probs(row):
    return [float(row[f'response_prob_{k}']) for k in CATS]

def correct_prob(row, c):
    return float(row['response_prob_BIT1' if c == 1 else 'response_prob_BIT0'])

def analyze(rows, split='confirmatory', horizon=2, logits_tolerance=0.01):
    allowed = CONF if split == 'confirmatory' else QUAL
    rows = [r for r in rows if int(r['context_index']) in allowed and int(r['horizon']) == horizon]
    idx = {(int(r['context_index']), int(r['source_c']), int(r['forced_t'])): r for r in rows}
    expected = len(allowed)*4
    if len(idx) != expected:
        raise ValueError(f'Expected {expected} unique rows, got {len(idx)}')

    # Bypass audit.
    bypass = []
    for y in sorted(allowed):
        for t in [0,1]:
            r0 = idx[(y,0,t)]
            r1 = idx[(y,1,t)]
            bypass.append({'context':y,'t':t,'tv':tv(probs(r0), probs(r1))})
    beta = max(x['tv'] for x in bypass)

    out = []
    for p in P_GRID:
        error = 0.0
        n_terms = 0
        # Exact average over contexts and balanced C, with exact BSC branch weights.
        for y in sorted(allowed):
            for c in [0,1]:
                for t in [0,1]:
                    w = (1-p) if t == c else p
                    r = idx[(y,c,t)]
                    error += (1/len(allowed))*0.5*w*(1.0-correct_prob(r,c))
                    n_terms += 1
        D = error
        I = 1.0-h2(p)
        R = rate_dist_binary(D)
        out.append({
            'p':p,
            'D':D,
            'I_C_T_given_Y_bits':I,
            'R_Z_given_Y_bits':R,
            'information_gap_bits':I-R,
            'distortion_frontier_gap':D-p,
        })

    D0 = next(x['D'] for x in out if x['p'] == 0.0)
    access_pass = D0 <= 0.10 + 1e-12
    # Auto-detect one-hot table.
    onehot = all(sum(abs(v-round(v)) for v in probs(r)) < 1e-9 for r in rows)
    bypass_tol = 0.0 if onehot else logits_tolerance
    bypass_pass = beta <= bypass_tol + 1e-12
    theorem_pass = all(x['information_gap_bits'] >= -1e-9 and x['distortion_frontier_gap'] >= -1e-9 for x in out)

    if not access_pass:
        status = 'ACCESS_FAIL'
    elif not bypass_pass:
        status = 'BYPASS_FAIL'
    elif not theorem_pass:
        status = 'THEOREM_VIOLATION'
    else:
        status = 'QUALIFIED_PASS'

    return {
        'status':status,
        'split':split,
        'horizon':horizon,
        'contexts':sorted(allowed),
        'onehot_detected':onehot,
        'bypass_tolerance':bypass_tol,
        'max_fixed_T_source_TV':beta,
        'access_gate_D_p0':D0,
        'bypass_details':bypass,
        'frontier':out,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('response_table_json')
    ap.add_argument('--split', choices=['qualification','confirmatory'], default='confirmatory')
    ap.add_argument('--horizon', type=int, default=2)
    ap.add_argument('--output')
    args = ap.parse_args()
    rows = json.load(open(args.response_table_json))
    result = analyze(rows,args.split,args.horizon)
    text = json.dumps(result,indent=2,sort_keys=True)
    if args.output:
        open(args.output,'w').write(text+'\n')
    print(text)

if __name__ == '__main__':
    main()
