import math
import json
from math import comb

def tv_bernoulli_product(p, q, m):
    total = 0.0
    for k in range(m + 1):
        a = comb(m, k) * (p ** k) * ((1.0 - p) ** (m - k))
        b = comb(m, k) * (q ** k) * ((1.0 - q) ** (m - k))
        total += abs(a - b)
    return 0.5 * total

def discounted_partial(p, q, gamma, H):
    return sum(
        (1.0 - gamma) * (gamma ** (m - 1)) * tv_bernoulli_product(p, q, m)
        for m in range(1, H + 1)
    )

gamma = 0.8
grid = [i / 10.0 for i in range(11)]
H_long = 80
tol = 1e-10

monotonic_failures = 0
truncation_failures = 0
triangle_failures = 0

dist = {}
for p in grid:
    for q in grid:
        seq = [tv_bernoulli_product(p, q, m) for m in range(1, 31)]
        if any(seq[i + 1] + tol < seq[i] for i in range(len(seq) - 1)):
            monotonic_failures += 1

        long_partial = discounted_partial(p, q, gamma, H_long)
        dist[(p, q)] = long_partial

        for H in (1, 2, 5, 10, 20):
            partial = discounted_partial(p, q, gamma, H)
            if long_partial + tol < partial:
                truncation_failures += 1
            if long_partial > partial + gamma ** H + tol:
                truncation_failures += 1

for p in grid:
    for q in grid:
        for r in grid:
            if dist[(p, r)] > dist[(p, q)] + dist[(q, r)] + 1e-8:
                triangle_failures += 1

illustrative = {}
for p, q in [(0.4, 0.6), (0.49, 0.51), (0.2, 0.8)]:
    partial = discounted_partial(p, q, gamma, 100)
    illustrative[f"{p:.2f},{q:.2f}"] = {
        "discounted_partial_H100": partial,
        "tail_bound_gamma_H": gamma ** 100,
        "finite_horizon_tv": {
            str(m): tv_bernoulli_product(p, q, m)
            for m in (1, 5, 20, 100, 200)
        },
    }

result = {
    "gamma": gamma,
    "grid_size": len(grid),
    "monotonic_failures": monotonic_failures,
    "truncation_failures": truncation_failures,
    "triangle_failures": triangle_failures,
    "illustrative_pairs": illustrative,
    "status": "PASS" if monotonic_failures == 0 and truncation_failures == 0 and triangle_failures == 0 else "FAIL",
}
print(json.dumps(result, indent=2, sort_keys=True))
