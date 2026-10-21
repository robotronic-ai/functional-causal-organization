# Theorem and proof sketch: the discounted predictive pseudometric

This is the formal core of the campaign: the definitions, the main theorem statements,
and their proof sketches. Source material: `THEOREM_SPEC.md` and `PROOF_SKETCH.md`.

## 1. Controlled future-law distances

Let `X_R` be the reachable state or history set. For each horizon `m >= 1`, let `Pi_m`
be a declared class of future control/intervention policies of length `m`, and let
`P_{x,m}^pi` be the length-`m` future observation law from state `x` under policy
`pi`. Let `D_m` be a discrepancy on length-`m` laws satisfying: `0 <= D_m <= 1`;
symmetry; the triangle inequality; and `D_m(P, Q) = 0` iff `P = Q` when exact law
equality is required. Define

```
d_m(x, y) = sup_{pi in Pi_m} D_m(P_{x,m}^pi, P_{y,m}^pi).
```

Every `d_m` is a pseudometric. For total variation and policy classes closed under
compatible restriction/extension, `d_m` is nondecreasing in `m`.

## 2. The discounted predictive pseudometric theorem

Fix `0 < gamma < 1` and define

```
d_gamma(x, y) = (1 - gamma) * sum_{m=1}^inf gamma^(m-1) * d_m(x, y).
```

Then:

1. `d_gamma` is a pseudometric — a nonnegative weighted sum of pseudometrics is a
   pseudometric, and the weights `(1-gamma) gamma^(m-1)` sum to one.
2. `0 <= d_gamma <= 1`.
3. `d_gamma(x, y) = 0` iff `d_m(x, y) = 0` for every `m` (all weights are strictly
   positive).
4. If the finite-dimensional future laws determine the complete path law, then
   `d_gamma(x, y) = 0` iff `x` and `y` are exactly predictively equivalent for the
   declared experiment class.

### Universal truncation certificate

Define the partial sum `d_{gamma,H}(x, y) = (1-gamma) * sum_{m=1}^H gamma^(m-1) d_m(x, y)`.
Then, with **no mixing or contraction assumption required**:

```
d_{gamma,H}(x, y)  <=  d_gamma(x, y)  <=  d_{gamma,H}(x, y) + gamma^H.
```

*Proof.* Because every `d_m <= 1`,

```
d_gamma - d_{gamma,H} = (1-gamma) sum_{m>H} gamma^(m-1) d_m
                      <= (1-gamma) sum_{m>H} gamma^(m-1) = gamma^H.
```

If certified upper bounds `dbar_m(x,y) >= d_m(x,y)` are available (e.g. from a
statistical estimate), the same argument gives
`d_gamma(x,y) <= (1-gamma) sum_{m=1}^H gamma^(m-1) dbar_m(x,y) + gamma^H`, separating
finite-horizon estimation error from the unobserved temporal tail.

## 3. Exact quotient at zero distance

Define `x ~_0 y` iff `d_gamma(x, y) = 0`. Because zero distance in a pseudometric is
transitive, `~_0` is an equivalence relation, and the quotient `X_R / ~_0` is the
**exact** predictive quotient for the declared experiment class — the same notion of
exact predictive equivalence used elsewhere in this project's finite theory (see
`12-readout-relative-minimal-predictive/`).

## 4. No automatic epsilon quotient

For `epsilon > 0`, the relation `d_gamma(x, y) <= epsilon` need not be transitive, so
there is no canonical epsilon quotient without an additional clustering rule. Valid
approximate objects instead include: epsilon-covers; partitions with predictive
diameter at most epsilon; metric-state abstractions that retain the pseudometric; or
learned embeddings with a certified distortion bound.

**Explicit counterexample (exactly verified).** Take one-step terminating Bernoulli
processes, where future-law TV between Bernoulli(`p`) and Bernoulli(`q`) is `|p-q|`.
With `p=0, q=0.4, r=0.8` and `epsilon=0.5`:

```
d(p,q) = 0.4 <= 0.5   -> p ~ q
d(q,r) = 0.4 <= 0.5   -> q ~ r
d(p,r) = 0.8 >  0.5   -> p NOT~ r
```

`p ~ q` and `q ~ r` but `p` is not related to `r`: thresholding is not transitive.
Verified by `scripts/verify_epsilon_nontransitivity.py`, status `PASS`
(`results/epsilon_nontransitivity_results.json`).

## 5. Compactness and finite predictive covers

Assume `X_R` is compact in a reference topology and each `d_m(x,y)` is continuous on
`X_R x X_R`. Because `(1-gamma) gamma^(m-1) d_m(x,y) <= (1-gamma) gamma^(m-1)`, and the
right-hand series converges, the Weierstrass M-test gives uniform convergence of the
defining series, hence `d_gamma` is continuous. Consequently the predictive metric
quotient is compact and totally bounded, so for every `epsilon > 0` there exists a
finite `epsilon`-cover. This defines the predictive covering complexity
`N_gamma(epsilon) = N(epsilon; X_R/~_0, d_gamma)`, a behavioral approximate-complexity
measure (see `docs/complexity-notions-and-guardrails.md` for how this relates to, and
must be kept separate from, linear predictive rank).

## 6. Representative approximation theorem

Given an `epsilon`-cover `{c_1, ..., c_N}` in `d_gamma` with one representative `r_i`
per cover element, encoding a state `x` by a representative `r_{q(x)}` satisfying
`d_gamma(x, r_{q(x)}) <= epsilon` changes the declared discounted future-law signature
by at most `epsilon`. This is a **one-state** predictive guarantee; it does not by
itself guarantee accurate multi-step rollout of a learned abstract transition model
(see next section).

## 7. Dynamic rollout theorem under approximate homomorphism

Consider a deterministic controlled system `x_{t+1} = F_u(x_t)` and an abstract model
`c_{t+1} = Fhat_u(c_t)` with representatives `r(c)`. Assume a Lipschitz/contraction
bound `d_gamma(F_u x, F_u y) <= L * d_gamma(x, y)` for every declared control `u`, and a
one-step model defect `d_gamma(F_u r(c), r(Fhat_u(c))) <= delta`. Let
`e_t = d_gamma(x_t, r(c_t))`. Then

```
e_{t+1} <= L * e_t + delta.
```

*Proof.* `e_{t+1} = d_gamma(F_u x_t, r(Fhat_u c_t))`, and by the triangle inequality
this is bounded by `d_gamma(F_u x_t, F_u r(c_t)) + d_gamma(F_u r(c_t), r(Fhat_u c_t))
<= L e_t + delta`. Iterating the scalar recurrence gives, for `L != 1`,
`e_t <= L^t e_0 + delta*(1-L^t)/(1-L)`, and for `L = 1`, `e_t <= e_0 + t*delta`. This
cleanly separates state-quantization error (`e_0`, from the representative-approximation
theorem above) from transition-model defect (`delta`, from the abstract model's own
fit). No dynamic model is trained or rolled out numerically in this bundle; this is a
proved theorem awaiting an experimental target (see
`docs/rmt-bridge-status-and-open-questions.md`).

### Stochastic analogue

For stochastic latent transitions, replacing pointwise propagation by a
Wasserstein/Kantorovich lifting of `d_gamma` gives the same recurrence for
distributional rollout error, under the corresponding lifted Lipschitz and one-step
defect bounds. Stated but not numerically exercised in this bundle.
