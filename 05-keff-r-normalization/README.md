# 05 — K_eff / R normalization

Clean edition of the campaign answering manuscript §12.5: how should the denominator of the
re-entry ratio `R = B_reentry / K_eff` be defined so that `R` is a well-behaved, bounded,
cross-system-comparable quantity — as opposed to being defined by a raw, substrate-specific
dimensionality (residual width, neuron count, nominal KV-cache size, covariance rank,
participation ratio, floating-point precision), none of which is axiomatically tied to the
information-theoretic numerator `B_reentry`.

## Reproduction discipline

This campaign follows the same discipline as every other campaign in this repository (see the
root `README.md` for the full statement): a protocol is frozen and hashed before any estimator
is built; definitions are proved analytically wherever possible; and every claim is backed by
an exact finite audit before it is allowed to apply to real, empirical measurement. Verdicts
are reported as `V1/POINT` (a unique value), `V3/PARTIAL` (several values remain compatible,
reported as a set) or `V4/NA` (the intervention algebra abstains) — no result here silently
turns a partial finding into a single number. This folder's scope is the **definition and
finite audit only**; the empirical transformer measurement that applies this normalization is
folder `06-transformer-measurement/` and is out of scope here.

## Main results

### K_eff is defined as a causal-equivalence quotient, not a raw dimensionality

For a source intervention battery with labels `Q` under context `U`, define the causal
quotient `C = χ(Q)` — two labels are equivalent iff they realize the same causal
interventional state, as declared *before* the return channel is examined (this ordering is
what prevents a circular normalization). Then:

\[
\boxed{K_{\rm eff}(q) = H_q(C \mid U)}
\qquad\text{(the conditional entropy of the causal quotient, not of the raw substrate).}
\]

The matching numerator is `B_reentry,q = I_q(C; Y^ret | U)`, so

\[
\boxed{R_q = \frac{I_q(C; Y^{ret} \mid U)}{H_q(C \mid U)}} \qquad (\mathrm{NA \ if}\ H_q(C\mid U)=0).
\]

Because `K_eff` depends only on the causal quotient of the source battery, two systems with
different *nominal* substrate dimensions but the same causal quotient get the same `K_eff` and
the same `R` — this is what makes `R` comparable across systems. See
`docs/keff-definition-and-invariance.md`.

### R is proved to lie in [0, 1], with exact calibration at both ends

The accompanying theorem proves, directly from the conditional-mutual-information chain rule
`I(C;Y|U) = H(C|U) − H(C|Y,U)` and the non-negativity of conditional entropy:

\[
\boxed{0 \le B_{\rm reentry} \le K_{\rm eff}} \qquad\Longrightarrow\qquad \boxed{0 \le R \le 1}.
\]

`R = 1` exactly when the return channel determines the causal quotient class bijectively
(conditional on `U`); `R = 0` exactly when the return is conditionally independent of the
causal quotient given `U`. The same conditional-entropy argument shows `K_eff` is unaffected
by relabeling or duplicating intervention labels that do not create a new causal class. See
`docs/r-normalization-theorem.md`.

### 8-control finite exact audit: PASS

Eight controls (K0–K7) exercise uniform/non-uniform battery distributions, an independent
return channel, an erasure channel, label duplication without new causal classes, bijective
relabeling, context-dependent class distributions, and — the decisive control, K7 — two
substrates with **different nominal dimensions but the same causal quotient**, required to
produce exactly the same `K_eff` and the same `R`. All eight controls pass exactly.

**Verdict:** `KEFF-NORM finite exact audit PASS`. See `docs/finite-audit-results.md`.

## How to verify

This campaign is purely analytic plus a finite exact audit; there is no empirical estimator or
external dependency to run. To check the claims:

1. Read the proof in `docs/r-normalization-theorem.md` — it is a four-line consequence of the
   conditional-mutual-information chain rule and can be checked by hand.
2. Re-derive the eight K0–K7 control values in `docs/finite-audit-results.md` — each control is
   a small, explicit finite distribution/kernel; the expected `K_eff`, `B_reentry` and `R` for
   each one are given so they can be recomputed independently (e.g. with the same
   `entropy`/`mi` helpers used in `04-gamma-delta-characterization/scripts/`, which compute
   exactly the same quantities — conditional entropy and mutual information of finite discrete
   variables).

See the root `README.md` for the full protocol → manuscript-section table.
