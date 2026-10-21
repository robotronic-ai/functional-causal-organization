# Δ decomposition

## The formula under test

The manuscript's differentiation score for a perturbation battery `P` with declared effective
support `P*` and readout `Y` is

\[
\Delta_q = \frac{I(P;Y)}{\log_2|P^*|}.
\]

This campaign asks what this formula actually measures, and whether it decomposes into
independently interpretable and independently characterizable factors.

## Exact decomposition: Δ = κ_q · δ_q

Define:

\[
\kappa_q = \frac{H(P)}{\log_2|P^*|}
\qquad\text{(entropic coverage of the declared battery)},
\]

\[
\delta_q = \frac{I(P;Y)}{H(P)}
\qquad\text{(discriminability conditional on that coverage, for } H(P)>0\text{)}.
\]

Then, algebraically, for any `q` with `H(P) > 0`:

\[
\boxed{\Delta_q = \kappa_q \cdot \delta_q.}
\]

This is an **analytic identity** — it holds by construction, not merely on the tested
fixtures. `gamma_delta_axiom_audit_v0_1.py` confirms it numerically on three channels (uniform
perfect, uniform independent, uniform noisy) crossed with two battery distributions (uniform,
non-uniform `q=(3/4,1/4)`).

## Worked numbers

| Case | `κ_q` | `δ_q` (dec) | `Δ_cap` |
|---|---|---|---|
| uniform battery, perfect decoding | 1.0 | 1.0 | 1.0 |
| non-uniform battery `q=(3/4,1/4)`, perfect decoding | 0.811278124459 | 1.0 | 0.811278124459 |
| uniform battery, independent channel | 1.0 | 0.0 | 0.0 |
| uniform battery, noisy channel (`3/4`/`1/4` confusion) | 1.0 | 0.188721875541 | 0.188721875541 |

The non-uniform, perfect-decoding row is the decisive case: decoding is perfect
(`δ_q = 1`), yet `Δ_cap = κ_q ≈ 0.8113 < 1`. For `q = (3/4, 1/4)`:

\[
\Delta_q = \kappa_q = H_2(1/4) \approx 0.811278124459
\]

(the binary entropy of `1/4`). The drop below 1 reflects **incomplete/non-uniform coverage of
the battery**, not a loss of discriminability in the system — exactly the distinction the
`(κ_q, δ_q)` profile is meant to make visible where the scalar `Δ` alone would hide it.

**Verdict:** `DELTA actuelle = D-CAP, pas pure D-DEC` — the manuscript's current `Δ` formula
realizes the **D-CAP** semantics (fraction of the *nominal battery capacity* transmitted), not
the **D-DEC** semantics (fraction of the *actually injected uncertainty* that is decodable):

\[
D_{\rm cap} = \frac{I(P;Y)}{\log_2|P^*|}
\qquad\text{(current formula, = }\Delta\text{)},
\qquad
D_{\rm dec} = \frac{I(P;Y)}{H(P)}
\qquad\text{(}=\delta_q\text{)}.
\]

D-DEC requires that a perfectly decodable `P` always score 1 regardless of the (non-degenerate)
weighting `q`; D-CAP does not.

## Two restricted-but-exact characterizations (v0.2)

The characterization audit (`gamma_delta_characterization_audit_v0_2.py`) pins down each
normalization inside its own function class:

- **Support-only class** `D = c(|P*|) · I(P;Y)`. Calibrating `c` so that a uniform,
  perfectly-decodable channel scores 1 forces `c(n) = 1/log₂(n)` — exactly D-CAP:
  `D = I(P;Y)/log₂|P*|`. Checked for `n ∈ {2,3,4,8}` (`c(n) ≈ 1, 0.630929753571, 0.5,
  0.333333333333`).
- **q-adaptive class** `D_q = c(q) · I(P;Y)`. Under the axiom "every perfectly decodable
  channel scores 1 for every non-degenerate `q`", `c(q)` is forced to `1/H(P)` — exactly
  D-DEC: `D_q = I(P;Y)/H(P)`. Checked for `q ∈ {(1/2,1/2), (3/4,1/4), (7/8,1/8)}`
  (`H(P) ≈ 1, 0.811278124459, 0.543564443200`; `c_dec = 1/H(P)`; `κ = H(P)`).

The difference between the two normalizations is therefore **entirely semantic** (which
quantity you calibrate against), not algebraic — both are exact within their declared class.

**Verdict:** `Gamma-arithmetic uniquely characterized by G1–G4` (see
`gamma-axiom-characterization.md`); `Delta-cap uniquely characterized within the support-only
MI-linear class`; `Delta-dec uniquely characterized within the q-adaptive MI-linear class`.
The semantic choice between D-CAP and D-DEC as the manuscript's primary scalar remains
explicit, not mathematically forced — which is why the GD-SEM campaign (see
`gd-sem-prospective-validation.md`) promotes the pair `(κ_q, δ_q)` to the primary reported
profile, with `Δ = κ_q·δ_q` kept only as a historical scalar synthesis.

## Scripts

- `scripts/gamma_delta_axiom_audit_v0_1.py` — decomposition identity, D-CAP/D-DEC numeric
  distinction.
- `scripts/gamma_delta_characterization_audit_v0_2.py` — support-only and q-adaptive
  characterizations.
