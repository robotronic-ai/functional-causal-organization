# The R ∈ [0, 1] theorem

## Statement

Let `C, U, Y` be finite discrete random variables. Define

\[
K_{\rm eff} = H(C \mid U), \qquad B = I(C; Y \mid U).
\]

Then

\[
\boxed{0 \le B \le K_{\rm eff}.}
\]

## Proof

By the definition of conditional mutual information,

\[
I(C;Y\mid U) = H(C\mid U) - H(C\mid Y,U).
\]

Since conditional entropy is non-negative, `H(C|Y,U) ≥ 0`, so

\[
I(C;Y\mid U) \le H(C\mid U).
\]

Conditional mutual information is itself non-negative, so `I(C;Y|U) ≥ 0`. Combining the two
bounds gives `0 ≤ B ≤ K_eff`. Consequently, whenever `H(C|U) > 0`,

\[
\boxed{0 \le R = \dfrac{I(C;Y\mid U)}{H(C\mid U)} \le 1.}
\]

∎

## Calibration at both ends

**Perfect calibration.** If `C` is determined by `(Y,U)` (the return channel bijectively
recovers the causal quotient given the context), then `H(C|Y,U) = 0`, so
`I(C;Y|U) = H(C|U)` and

\[
\boxed{R = 1.}
\]

**Absence of return.** If `C ⟂ Y | U` (the return is conditionally independent of the causal
quotient given the context), then `I(C;Y|U) = 0` and

\[
\boxed{R = 0.}
\]

## Invariance to label duplication

Let `Q` be an intervention label and `C = χ(Q)` its causal quotient. Any operation that
subdivides labels *within* a single causal class without changing `q(C|U)` leaves `H(C|U)`
unchanged — the quotient entropy only depends on the class-level distribution, not on how
finely the labels inside a class are subdivided. So `K_eff` cannot be inflated artificially by
multiplying causally-redundant intervention labels.

## Coverage / re-entry decomposition

When `|C*| > 1` (the causal quotient has a non-trivial effective support), define

\[
\kappa_R = \frac{H(C\mid U)}{\log_2|C^*|}, \qquad R_{\rm dec} = \frac{I(C;Y\mid U)}{H(C\mid U)}.
\]

Then, algebraically,

\[
\kappa_R \cdot R_{\rm dec} = \frac{I(C;Y\mid U)}{\log_2|C^*|} =: R_{\rm cap}.
\]

So

\[
\boxed{R_{\rm cap} = \kappa_R \cdot R_{\rm dec}.}
\]

This is exactly analogous to the Δ decomposition in `04-gamma-delta-characterization/`:
`Δ_cap = κ_q · Δ_dec`. The recommended manuscript choice for the primary re-entry scalar is
`R = R_dec` — i.e. the decodability-normalized quantity, not the support-capacity-normalized
one — reported alongside the coverage factor `κ_R` as the primary profile `(κ_R, R)`.
