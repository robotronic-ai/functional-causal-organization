# Complexity notions and methodology guardrails

Source material: `APPROXIMATE_COMPLEXITY.md` and `METHODOLOGY_GUARDRAILS.md`.

## Two different complexity notions — do not identify them

**Behavioral complexity.** The primary nonlinear approximate complexity is the
covering number `N_gamma(epsilon)` from `docs/theorem-and-proof-sketch.md` Section 5: how
many predictive representatives are needed so that every reachable state is within
predictive distance `epsilon` of at least one representative. This is invariant to
arbitrary reparameterization of the latent coordinates when the predictive law is
unchanged.

**Linear predictive complexity.** A separate object is the approximate rank of a
predictive Hankel matrix or operator. Given a finite empirical Hankel matrix `H`,
define an effective rank only relative to a declared norm, a declared weighting of
histories/tests, and a declared approximation tolerance — e.g.
`r_eta = min{ r : ||H - H_r|| <= eta }`. This is a linear-representation complexity.

**A system can have many behaviorally distinct predictive states and low linear
predictive dimension at the same time.** The exact finite stochastic protocol used
elsewhere in this project already exhibits this phenomenon (see
`12-readout-relative-minimal-predictive/docs/finite-verification-results.md`, where the
behavioral quotient and the minimal linear predictive dimension are shown to be
genuinely different quantities on small finite systems). The same separation must be
retained in continuous systems:

```
predictive covering complexity  !=  effective Hankel rank.
```

**Norm warning.** Small Frobenius error in an empirical Hankel matrix does not
automatically imply small total-variation error for future path laws. A direct
future-law metric must be evaluated on held-out tests when the scientific claim
concerns predictive distributional accuracy.

## Methodology guardrails

1. **No epsilon-equivalence without transitivity.** `d(x,y) <= epsilon` is generally
   not transitive (proved by explicit counterexample in
   `docs/theorem-and-proof-sketch.md`, Section 4). Use covers, bounded-diameter
   partitions, or retain the predictive metric directly.
2. **No finite-horizon claim without a tail statement.** A finite horizon does not
   certify an undiscounted infinite path law unless an additional mixing/contraction/
   tail theorem is given. For the discounted metric, always report the universal tail
   term `gamma^H`.
3. **No latent Euclidean proxy without validation.** Do not replace predictive distance
   by raw hidden-state Euclidean distance unless a distortion bound has been
   demonstrated.
4. **No low-rank-to-TV shortcut.** Low-rank Hankel reconstruction error in a generic
   matrix norm is not automatically a bound on future-law total variation. Evaluate the
   declared future-law discrepancy directly.
5. **No target-index recoding.** A representation that merely stores the future targets
   in another coordinate system is not a generative predictive result. Strict held-out
   transport (`docs/protocol-and-held-out-transport.md`) is required.
6. **Distinguish three error sources**, always: predictive abstraction error,
   statistical estimation error, and temporal truncation error.
7. **Capacity is not realized causal use.** Predictive decodability or predictive
   compression is not by itself evidence that the system uses the represented variable
   causally.
