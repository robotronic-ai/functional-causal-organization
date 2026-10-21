# Certified-interval Γ protocol for the recurrent-memory target transformer

## Purpose

This package replaces the manuscript's spectral surrogate for Γ, on the frozen recurrent-memory
transformer used in the manuscript's external causal-validation campaign, with a **statistically
certified interval** `L_Γ ≤ Γ_q ≤ U_Γ`, rather than a point estimate. The certificate code
(`scripts/gamma_certificate_from_trials.py`) is architecture-independent: it consumes only a
trial table (randomized macro-role interventions and their next-cycle responses) and never
inspects model weights.

## Primary estimand

For an admissible bipartition `A|B` of the causal minimal macrostate,

\[
J_q(A\to B) = \mathbb E_{b\sim q_B}\, I_q(A;B'\mid do(B=b)),
\qquad
\Gamma_q = \min_{A|B} \frac{J_q(A\to B) + J_q(B\to A)}{2\min(\log|A|,\log|B|)}.
\]

The primary target experiment uses binary macro-role intervention classes, so
`log|A| = |A|` and `log|B| = |B|`.

## Non-negotiable quotient gate

Γ is never computed on raw activation coordinates. Before certification, a causal minimal-state
quotient must be proposed on discovery data and frozen, then shown on independent holdout
interventions to satisfy:

1. **causal sufficiency** — microstates mapped to one macrostate cannot yield distinguishable
   future macrostates under the declared intervention algebra;
2. **irreducibility** — every retained macro distinction changes a declared future response in
   at least one intervention context;
3. **intervention closure** — every intervention used by the Γ battery has a well-defined
   transported intervention on the quotient.

If this gate fails, the result is `NA`. It is never replaced by a raw activation-space score.
(See `gamma-cert-causal-quotient/` in this same folder for a dedicated stress test of exactly
this gate under representational redundancy.)

## Intervention battery and four independent statistical streams

The primary battery `q_bal` sets every declared binary macro-role intervention class to
probability 1/2, independently of prompt or context, with deterministic inference (dropout
disabled, no stochastic decoding). Four independent streams are required and no decoder, edge,
quotient, cut, or hyperparameter may be selected on the holdout stream used to certify it:

| Split | Role |
|---|---|
| 0 | lower-bound discovery |
| 1 | lower-bound holdout |
| 2 | upper-bound cut discovery |
| 3 | upper-bound holdout |

## Certified lower bound (decoder + Fano + spectral)

For each directed pair `i→j`, a binary decoder (selected on discovery, retaining only the
`top_k_per_source` strongest candidates per source) predicts source intervention `X_i` from
next-state response `Y'_j` and context. On holdout, its error receives a simultaneous one-sided
Clopper–Pearson upper bound `p^U_ij`, and Fano's inequality gives

\[
d^L_{ij} = 1 - h_2(\min(p^U_{ij}, 1/2)).
\]

Each source distributes its certified information over its retained positive targets (no
double counting), giving a symmetric weighted graph that is a *proven* causal lower graph. With
`δ_min` its minimum weighted degree and `λ_2` the second eigenvalue of its normalized Laplacian,

\[
L_\Gamma = \frac{\delta_{\min}\,\lambda_2}{4} \le \Gamma_q.
\]

A disconnected certified graph yields `L_Γ = 0`, never a fabricated positive estimate.

## Certified upper bound (cross-entropy)

The lower graph supplies only candidate Fiedler orderings; the actual candidate cut is selected
on an independent discovery stream. For a deterministic frozen transformer,
`J_q(A→B) = H(Y_B∣X_B,U)`, and cross-entropy gives
`H(Y_B∣X_B,U) ≤ 𝔼[-log₂ p̂(Y_B∣X_B,U)]` for predictors frozen before upper holdout. An
empirical-Bernstein upper confidence bound on held-out log loss yields the simultaneous upper
bound `Γ_q ≤ U_Γ`.

## Statistical confidence and precision status

Preregistered error budget `δ_L = δ_U = 0.025`, so `P(L_Γ ≤ Γ_q ≤ U_Γ) ≥ 0.95`. The primary
precision band is frozen before the target run: `U_Γ - L_Γ ≤ 0.10`.

- If the quotient gate passes **and** the interval width is `≤ 0.10`: status **`V1`**.
- If both bounds are valid but the width exceeds `0.10`: status **`V3`**.
- If the quotient is not causally certified: status **`NA`**.

The `0.10` band is a precision criterion, not a substantive threshold for integration.

## What the model adapter must supply

`scripts/adapter_contract.py` defines the strict, architecture-independent interface the
certifier requires: the frozen post-memory-training checkpoint and a local loader; the exact
tokenizer/context generator from the existing external validation campaign; the exact
recurrent-memory intervention hook; the candidate micro-to-macro quotient map (or its discovery
procedure); the matched intervention transport for binary macro-role classes; and deterministic
forward inference for every trial. The certificate code must not inspect model weights after
trials are written. See `docs/execution-status.md` for which of these are currently available.
