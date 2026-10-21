# Why undiscounted infinite-path total variation is too strong

Source material: `WHY_NOT_INFINITE_TV.md`. This is the motivating example for why the
campaign discounts finite-horizon discrepancies instead of using the raw infinite-path
total-variation (TV) metric.

## The example

Consider a continuous family of latent states indexed by `p in (0, 1)`. From state `p`,
observations are i.i.d. `Y_t ~ Bernoulli(p)`. For two distinct parameters `p != q`, the
infinite product measures `P_p^inf` and `P_q^inf` are **mutually singular**: by the
strong law of large numbers, the empirical mean `(1/n) sum_{t<=n} Y_t` converges to `p`
almost surely under `P_p` and to `q` almost surely under `P_q`. So there exist disjoint
measurable sets of infinite sequences, each carrying probability one under one of the
two laws. Hence

```
TV(P_p^inf, P_q^inf) = 1   for every p != q.
```

The exact infinite-path TV metric is therefore **discrete** on this continuous
parametric family — it does not distinguish "close" parameters from "far" parameters at
all. This does not make exact path-law equivalence wrong as a notion; it shows that
undiscounted infinite-path TV is often unsuitable as an *approximate* geometry, because
it collapses every pair of distinct parameters to maximal distance.

## Why finite horizons and discounting fix this

For every finite horizon `m`, `TV(P_p^m, P_q^m) -> 0` as `q -> p` (finite-dimensional
Bernoulli-product distributions are continuous in the parameter). The discounted metric

```
d_gamma(p, q) = (1 - gamma) * sum_{m>=1} gamma^(m-1) * TV(P_p^m, P_q^m)
```

therefore satisfies `d_gamma(p, q) -> 0` as `q -> p`, by dominated convergence (each
term is continuous and uniformly bounded by the summable weight
`(1-gamma) gamma^(m-1)`). Discounting preserves exact predictive equivalence at zero
distance (Section 2–3 of `docs/theorem-and-proof-sketch.md`) while recovering a useful,
continuous approximate geometry — which is exactly what the raw infinite-horizon metric
fails to give on this family.

## Numerical confirmation

`scripts/verify_discounted_bernoulli.py` checks this directly on an 11-point grid of
Bernoulli parameters `p, q in {0.0, 0.1, ..., 1.0}` with `gamma = 0.8`: as horizon `m`
grows, the finite-horizon TV climbs toward `1` (e.g. for `p=0.4, q=0.6`:
`TV_1 = 0.20`, `TV_20 = 0.63`, `TV_100 = 0.956`, `TV_200 = 0.996`), while the discounted
partial sum through horizon `100` stays finite and moderate (`0.322` for that pair).
Full table in `results/discounted_bernoulli_results.json`; status `PASS`, zero
monotonicity/truncation/triangle-inequality failures on the grid.
