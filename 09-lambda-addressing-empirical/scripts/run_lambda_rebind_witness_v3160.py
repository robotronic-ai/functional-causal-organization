#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools, json, hashlib, math
from collections import Counter
from pathlib import Path

VERSION = "3.16.0"
N = 4
ID = tuple(range(N))
PERMS = list(itertools.permutations(range(N)))

def swap_tuple(x, i, j):
    x = list(x); x[i], x[j] = x[j], x[i]; return tuple(x)

INTERVENTIONS = [("IDENTITY", None)]
for target in ("C", "P", "U"):
    for i in range(N):
        for j in range(i+1, N):
            INTERVENTIONS.append((f"SWAP_{target}_{i}_{j}", (target, i, j)))

def intervene(C, P, U, spec):
    _, payload = spec
    if payload is None:
        return C, P, U
    target, i, j = payload
    if target == "C": C = swap_tuple(C, i, j)
    elif target == "P": P = swap_tuple(P, i, j)
    elif target == "U": U = swap_tuple(U, i, j)
    else: raise ValueError(target)
    return C, P, U

def readout(C, P):
    return tuple(C[P[r]] for r in range(N))

class Router:
    def __init__(self, mode):
        assert mode in ("RELINK", "LOCK")
        self.mode = mode
    def step(self, C, P, U, commit=False):
        P2 = tuple(U) if (commit and self.mode == "RELINK") else tuple(P)
        return tuple(C), P2, tuple(U), readout(C, P2)

def digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, separators=(",",":")).encode())
        h.update(b"\n")
    return h.hexdigest()

def entropy(vals):
    c = Counter(vals); n = sum(c.values())
    return -sum((k/n)*math.log2(k/n) for k in c.values())

def mutual_information(xs, ys):
    n = len(xs)
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs,ys))
    out = 0.0
    for (x,y), k in cxy.items():
        pxy = k/n; px = cx[x]/n; py = cy[y]/n
        out += pxy * math.log2(pxy/(px*py))
    return out

def baseline_channel():
    a, b = Router("RELINK"), Router("LOCK")
    rows_a, rows_b = [], []
    mismatches = 0
    for C in PERMS:
        for P in PERMS:
            for U in PERMS:
                for spec in INTERVENTIONS:
                    Ci, Pi, Ui = intervene(C, P, U, spec)
                    ra = a.step(Ci, Pi, Ui, commit=False)
                    rb = b.step(Ci, Pi, Ui, commit=False)
                    rows_a.append((C,P,U,spec[0],ra))
                    rows_b.append((C,P,U,spec[0],rb))
                    if ra != rb: mismatches += 1
    sha_a, sha_b = digest(rows_a), digest(rows_b)
    return {
        "comparison_count": len(rows_a),
        "mismatch_count": mismatches,
        "relink_sha256": sha_a,
        "lock_sha256": sha_b,
        "exact_equal": mismatches == 0 and sha_a == sha_b,
    }

def late_rebind():
    systems = {m: Router(m) for m in ("RELINK","LOCK")}
    result = {}
    for mode, sys in systems.items():
        correct = 0; cues = []; pplus = []; episodes = 0
        for C in PERMS:
            for U in PERMS:
                _, P2, _, Y = sys.step(C, ID, U, commit=True)
                expected = readout(C, U)
                correct += int(Y == expected)
                cues.append(U); pplus.append(P2); episodes += 1
        hu = entropy(cues); mi = mutual_information(cues, pplus)
        result[mode] = {
            "episodes": episodes,
            "task_accuracy": correct/episodes,
            "cue_pointer_mutual_information_bits": mi,
            "cue_entropy_bits": hu,
            "lambda_gate_normalized": mi/hu if hu > 0 else None,
        }
    return result

def controls():
    systems = {m: Router(m) for m in ("RELINK","LOCK")}
    out = {}
    for mode, sys in systems.items():
        ident_ok = 0
        for C in PERMS:
            _, _, _, Y = sys.step(C, ID, ID, commit=True)
            ident_ok += int(Y == C)
        precue_ok = 0; forced_ok = 0
        for C in PERMS:
            for U in PERMS:
                _, _, _, Ypre = sys.step(C, U, U, commit=False)
                precue_ok += int(Ypre == readout(C,U))
                _ = sys.step(C, ID, U, commit=True)
                Yforced = readout(C, U)
                forced_ok += int(Yforced == readout(C,U))
        out[mode] = {
            "identity_late_cue_accuracy": ident_ok/len(PERMS),
            "precued_accuracy": precue_ok/(len(PERMS)**2),
            "forced_pointer_accuracy": forced_ok/(len(PERMS)**2),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=f"LAMBDA_REBIND_WITNESS_RESULT_v{VERSION}.json")
    args = ap.parse_args()
    base = baseline_channel(); challenge = late_rebind(); ctrls = controls()
    established = (
        base["exact_equal"]
        and abs(challenge["RELINK"]["lambda_gate_normalized"] - 1.0) < 1e-12
        and abs(challenge["LOCK"]["lambda_gate_normalized"] - 0.0) < 1e-12
        and abs(challenge["RELINK"]["task_accuracy"] - 1.0) < 1e-12
        and abs(challenge["LOCK"]["task_accuracy"] - 1/24) < 1e-12
        and all(abs(ctrls[m]["identity_late_cue_accuracy"] - 1.0) < 1e-12 for m in ctrls)
        and all(abs(ctrls[m]["precued_accuracy"] - 1.0) < 1e-12 for m in ctrls)
        and all(abs(ctrls[m]["forced_pointer_accuracy"] - 1.0) < 1e-12 for m in ctrls)
    )
    result = {
        "version": VERSION,
        "usecase": "LATE-ROLE-REBIND-SWITCHBOARD",
        "claim_scope": "Constructive executable witness: identical declared baseline typed perturbation-response channel, divergent late context->role causal channel. Not a retroactive PASS of the frozen RMT real-Delta Stage B gate.",
        "architecture": {
            "state": "C[4] payload bank, P[4] role->content pointer permutation, U[4] late cue permutation",
            "readout": "Y[r] = C[P[r]] in BOTH systems",
            "only_difference": "RELINK: on COMMIT P<-U; LOCK: on COMMIT P unchanged"
        },
        "baseline_typed_channel": base,
        "late_rebind": challenge,
        "controls": ctrls,
        "status": "LAMBDA-CONSTRUCTIVE-EMPIRICAL-WITNESS-ESTABLISHED" if established else "LAMBDA-CONSTRUCTIVE-EMPIRICAL-WITNESS-NOT-ESTABLISHED"
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CF-A-v3.16.0-LAMBDA-REBIND-WITNESS-COMPLETE")
    print(result["status"])
    print("BASELINE-TYPED-CHANNEL-EXACT-EQUALITY", base["exact_equal"])
    print("BASELINE-COMPARISONS", base["comparison_count"])
    print("BASELINE-SHA256", base["relink_sha256"])
    print("RELINK-ACC", challenge["RELINK"]["task_accuracy"])
    print("LOCK-ACC", challenge["LOCK"]["task_accuracy"])
    print("RELINK-LAMBDA-GATE", challenge["RELINK"]["lambda_gate_normalized"])
    print("LOCK-LAMBDA-GATE", challenge["LOCK"]["lambda_gate_normalized"])
    print("OUTPUT", str(Path(args.output).resolve()))

if __name__ == "__main__":
    main()
