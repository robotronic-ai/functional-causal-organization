#!/usr/bin/env python3
import json
CATS=['BIT0','BIT1','EMPTY','OTHER']

def dist(label):
    return {f'response_prob_{k}':1.0 if k==label else 0.0 for k in CATS}

def make(kind):
    rows=[]
    for y in range(8):
        for h in [2]:
            for c in [0,1]:
                for t in [0,1]:
                    if kind == 'ideal':
                        label = 'BIT1' if t else 'BIT0'
                    elif kind == 'degraded':
                        # Ignores T and always predicts 0; balanced C gives D=0.5.
                        label = 'BIT0'
                    elif kind == 'bypass':
                        # Illegal direct access to source C.
                        label = 'BIT1' if c else 'BIT0'
                    else:
                        raise ValueError(kind)
                    row={'context_index':y,'horizon':h,'source_c':c,'forced_t':t,
                         'query_text_hash':'SYNTHETIC','role_b_state_id':'FIXED','cache_reset_confirmed':True}
                    row.update(dist(label))
                    rows.append(row)
    return rows

for kind in ['ideal','degraded','bypass']:
    with open(f'synthetic_{kind}.json','w') as f:
        json.dump(make(kind),f,indent=2,sort_keys=True)
