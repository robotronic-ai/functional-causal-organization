import json

# One-step terminating predictive processes:
# future-law TV is simply |p-q| for Bernoulli(p) vs Bernoulli(q).
p = 0.0
q = 0.4
r = 0.8
epsilon = 0.5

d_pq = abs(p - q)
d_qr = abs(q - r)
d_pr = abs(p - r)

result = {
    "epsilon": epsilon,
    "d_pq": d_pq,
    "d_qr": d_qr,
    "d_pr": d_pr,
    "p_related_q": d_pq <= epsilon,
    "q_related_r": d_qr <= epsilon,
    "p_related_r": d_pr <= epsilon,
    "status": "PASS" if (d_pq <= epsilon and d_qr <= epsilon and d_pr > epsilon) else "FAIL",
}
print(json.dumps(result, indent=2, sort_keys=True))
