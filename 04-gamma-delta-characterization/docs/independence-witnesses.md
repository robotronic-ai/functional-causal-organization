# Γ / Δ independence witnesses

## Question

Given that Γ (integration) and Δ (differentiation) are both derived from causal-intervention
statistics of the same system, is either one a universal function of the other on any
reasonable finite domain — i.e. could the manuscript, in principle, drop one of the two
scalars without loss?

The protocol (`Protocole_GAMMA_DELTA_AX_v0.1_FROZEN.md`, §4) requires two exact finite
witnesses, built from the exact formulas (not from oracle labels):

- **IND-1**: two contexts with equal Γ but different Δ.
- **IND-2**: two contexts with equal Δ but different Γ.

## IND-1 — equal Γ, unequal Δ

Two contexts `C1`, `C2` share the same causal integration:

\[
\Gamma(C_1) = \Gamma(C_2) = 1,
\]

but differ in differentiation:

\[
\Delta(C_1) = 1, \qquad \Delta(C_2) = 0.
\]

Concretely: both contexts use a perfect bidirectional directional channel (`Γ = 1`); `C1`'s
Δ-channel is the perfect deterministic channel (`Δ_cap = 1`), `C2`'s Δ-channel is the fully
independent channel (`Δ_cap = 0`). So **no universal function `Δ = f(Γ)` exists.**

## IND-2 — equal Δ, unequal Γ

Two contexts `C3`, `C4` share the same differentiation:

\[
\Delta(C_3) = \Delta(C_4) = 1,
\]

but differ in causal integration:

\[
\Gamma(C_3) = 1, \qquad \Gamma(C_4) = 1/2.
\]

Concretely: both contexts use the perfect deterministic Δ-channel (`Δ_cap = 1`); `C3`'s Γ-cut
is perfectly bidirectional (`x=y=1`, so `Γ=1`), `C4`'s Γ-cut is purely one-directional
(`x=1, y=0`, so `Γ=(1+0)/2=1/2`). So **no universal function `Γ = g(Δ)` exists.**

## Verdict

\[
\boxed{\text{GAMMA/DELTA: INDEPENDENCE-WITNESS}}
\]

`gamma_delta_axiom_audit_v0_1.py` confirms both witnesses exactly:

```
IND-1: Gamma 1.0 = 1.0   Delta 1.0 != 0.0
IND-2: Delta 1.0 = 1.0   Gamma 1.0 != 0.5
Gamma->Delta functional dependence: REFUTED on finite domain
Delta->Gamma functional dependence: REFUTED on finite domain
```

This independence result is a load-bearing input to the closure status
(`docs/closure-status.md`): it is what licenses reporting Γ and Δ (and their auxiliary
profiles β and `(κ_q, δ_q)`) as separate, non-redundant quantities in the manuscript, rather
than collapsing them into one another.

## Script

- `scripts/gamma_delta_axiom_audit_v0_1.py` — constructs and checks IND-1 and IND-2.
