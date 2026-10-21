# Finite exact audit (K0–K7)

## Verdict

\[
\boxed{\text{KEFF-NORM finite exact audit PASS}}
\]

All eight mandatory finite controls declared in the frozen protocol pass. This campaign is
purely analytic plus finite audit — there is no external estimator, no generator/harness pair,
and no private oracle file involved, unlike the GD-SEM campaign in
`04-gamma-delta-characterization/`. The audit is a set of small, explicit finite
distributions/kernels, each checked directly against the theorem in
`r-normalization-theorem.md`.

## The eight controls

| Control | Construction | What it checks |
|---|---|---|
| **K0** | four uniform causal classes, perfect return | calibration: `R = 1` when the return channel bijectively recovers `C` (K5) |
| **K1** | four non-uniform causal classes, perfect return | `K_eff = H(C\|U)` still correctly reflects a non-uniform class distribution while `R` stays `1` under perfect return |
| **K2** | independent return channel | absence-of-return calibration: `R = 0` when `C ⟂ Y^ret \| U` (K6) |
| **K3** | an erasure channel on the return side | a partial, non-degenerate `R` strictly between 0 and 1 |
| **K4** | duplicated intervention labels, causal class unchanged | `K_eff` is unaffected by subdividing labels within a class (the duplication axiom, K3) |
| **K5** | bijective recoding of labels / classes / outputs | `K_eff` and `R` are invariant under relabeling (K2) |
| **K6** | context `U` with different causal-class distributions per context | `K_eff(q)` correctly tracks the *conditional* entropy `H(C\|U)`, not a context-independent quantity |
| **K7** | two substrates with **different nominal dimensions but the same causal quotient** | the decisive control: `K_eff` and `R` must come out **exactly identical** across substrates once the causal quotient is held fixed, regardless of nominal dimensionality |

K7 is the control that most directly exercises the campaign's central claim — that `K_eff` is
a property of the *causal quotient* of the intervention battery, not of the raw substrate
dimensionality. Two systems can differ arbitrarily in nominal state size (activation width, KV
cache size, etc.) and still be normalized identically by `K_eff`/`R`, provided their causal
quotients coincide.

## How to recompute a control by hand

Each control is a finite discrete-variable computation of the same two quantities used
throughout this repository's Γ/Δ/K_eff/R campaigns:

\[
K_{\rm eff} = H(C\mid U) = -\sum_{u} q(u) \sum_{c} q(c\mid u)\log_2 q(c\mid u),
\]

\[
B_{\rm reentry} = I(C;Y^{ret}\mid U) = H(C\mid U) - H(C\mid Y^{ret},U),
\]

\[
R = B_{\rm reentry} / K_{\rm eff} \quad (\text{NA if } K_{\rm eff}=0).
\]

For example, K0's construction (four uniform causal classes, perfect return) gives
`H(C|U) = log₂4 = 2` bits and, since the return is perfect, `H(C|Y^ret,U) = 0`, so
`I(C;Y^ret|U) = 2` bits and `R = 2/2 = 1`, matching the K5 calibration axiom exactly. K2
(independent return) gives `I(C;Y^ret|U) = 0` regardless of `H(C|U)`, so `R = 0`, matching K6.
K7 requires constructing two label alphabets of different nominal cardinality that collapse to
the *same* causal-class distribution `q(C|U)` under `χ`; by the definition in
`keff-definition-and-invariance.md`, `H(C|U)` depends only on that class distribution, so both
substrates necessarily produce the same `K_eff` and, for a matched return channel, the same
`R`. The `entropy`/`mi` helper functions in
`04-gamma-delta-characterization/scripts/gamma_delta_axiom_audit_v0_1.py` compute exactly these
same two quantities (conditional entropy and mutual information over finite discrete
distributions) and can be reused directly to recompute any of the eight controls above.

## Scope

This audit fixes the **normalization**. It does not measure anything on a real transformer.
The next campaign (folder `06-transformer-measurement/`, out of scope here) must construct,
empirically, a source intervention algebra, the causal quotient `C`, a return channel `Y^ret`
consistent with the neutralized/cut model, and `I(C;Y^ret|U)` — activation patching can supply
the intervention mechanism but does not by itself define the causal quotient or the battery.
