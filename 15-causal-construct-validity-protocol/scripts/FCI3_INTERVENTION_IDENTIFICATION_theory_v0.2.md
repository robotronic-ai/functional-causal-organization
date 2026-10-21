# FCI-3 INTERVENTION IDENTIFICATION v0.2

**Date:** 2026-09-14  
**Status:** theory frozen before synthetic identification fixtures.

## 1. Why architecture intervention is not enough

The construct-validity target is not merely:

\[
Z_{\rm arch}\rightarrow \Delta C_F
\]

and:

\[
Z_{\rm arch}\rightarrow \Delta\mathcal A_F.
\]

The same architecture intervention may affect functional-access outcomes through
other pathways:

\[
Z_{\rm arch}
\rightarrow
U
\rightarrow
\mathcal A_F
\]

where \(U\) may include memory capacity, optimization ease, bandwidth, compute,
depth, routing, or another unmeasured causal feature.

Therefore:

\[
Z_{\rm arch}\to C_F
\quad\text{and}\quad
Z_{\rm arch}\to \mathcal A_F
\]

does not by itself establish that \(C_F\) mediates the access effect.

## 2. Target causal object

Let:

\[
\mathbf c
=
(\Gamma,\Delta_{\rm prop},R)
\]

denote the compact causal profile.

Let:

\[
\mathbf a
=
(a_{\rm flex},a_{\rm delay},a_{\rm update},a_{\rm status},\ldots)
\]

denote independently measured functional-access consequences.

The ideal target is a profile-response surface:

\[
\boxed{
m(\mathbf c,\mathbf r)
=
E[
\mathbf a
\mid
do(\mathbf c),
\mathbf r
]
}
\]

where \(\mathbf r\) denotes matched rival variables.

Direct \(do(\mathbf c)\) may be physically unavailable.  FCI-3 therefore uses
architecture interventions as candidate instruments.

## 3. Instrument conditions

For an architectural manipulation \(Z_j\) to identify a component-level causal
link, require:

### I1 — relevance

\[
Z_j
\not\!\perp
c_j.
\]

The manipulation must move the intended \(C_F\) component by at least a frozen
minimum amount.

### I2 — rival equivalence

Measured rival variables remain within preregistered equivalence margins.

### I3 — exclusion, when claimed

Conditional on the realized causal profile and frozen rivals, the intervention
has no additional direct path to the primary access endpoint:

\[
\mathbf a
\perp
Z_j
\mid
\mathbf c,\mathbf r.
\]

This is a substantive identification assumption, not a consequence of matching.

### I4 — stable measurement

The \(C_F\) and AF measurement protocols are unchanged across instrument arms.

### I5 — target semantics

The access task and its target variable are identical across intervention arms.

## 4. Why exclusion is difficult

Architectural manipulations rarely satisfy I3 by inspection.

Adding a latent recurrent block, for example, may simultaneously alter:

- optimization landscape;
- effective depth;
- compute allocation;
- memory capacity;
- routing geometry;
- representational bottlenecks.

Therefore FCI-3 must not rely on a single architecture manipulation.

## 5. Convergent-instrument principle

Use at least two mechanistically distinct interventions:

\[
Z_1,Z_2
\]

that produce matched changes in the target profile coordinate while differing
in implementation details.

If:

\[
\Delta\mathbf c(Z_1)
\simeq
\Delta\mathbf c(Z_2)
\]

but:

\[
\Delta\mathbf a(Z_1)
\not\simeq
\Delta\mathbf a(Z_2),
\]

then at least one of the following holds:

1. the compact \(C_F\) profile is incomplete;
2. one or both interventions violate exclusion;
3. an uncontrolled rival variable remains.

Thus same-profile / different-outcome pairs are direct construct-validity
stress tests.

## 6. Overidentification criterion

For a target component \(c_j\), freeze \(K\ge2\) distinct intervention families.

Each valid family estimates a local response:

\[
\beta_{jk}
=
\frac{\Delta a_j^{(k)}}{\Delta c_j^{(k)}}.
\]

The compact-profile hypothesis predicts transportability of these effects
within a declared local region:

\[
\boxed{
\beta_{j1}
\simeq
\beta_{j2}
\simeq
\cdots
\simeq
\beta_{jK}.
}
\]

Large preregistered disagreement is evidence against a single compact
component-level causal map.

## 7. Same-profile invariance test

The strongest empirical sufficiency stress test is not monotonicity.

It is:

\[
\boxed{
\mathbf c(e_1)
\simeq
\mathbf c(e_2),
\quad
\mathbf r(e_1)
\simeq
\mathbf r(e_2)
\Rightarrow
\mathbf a(e_1)
\simeq
\mathbf a(e_2).
}
\]

This is the empirical analogue of fiber constancy.

A replicated violation is evidence that the compact profile omits an
access-relevant organizational variable.

## 8. Directional hypotheses remain secondary to fiber tests

Positive directional effects such as:

\[
\Delta R>0
\Rightarrow
E[\Delta a_{\rm delay}]>0
\]

are useful but can be mimicked by correlated architectural changes.

A same-profile violation is stronger because it directly attacks sufficiency.

Therefore FCI-3 should report, in order:

1. manipulation validity;
2. same-profile invariance;
3. convergent-instrument agreement;
4. directional dose response.

## 9. Identification statuses

Every empirical contrast receives one of:

- `IDENTIFIED-VALID`;
- `EXCLUSION-UNSUPPORTED`;
- `RIVAL-MISMATCH`;
- `PROFILE-MISMATCH`;
- `UNRESOLVED`.

No causal construct-validity claim is made from contrasts outside
`IDENTIFIED-VALID`.

## 10. Scientific interpretation

A successful FCI-3 campaign supports:

> Within the tested architecture class and local profile region, independently
> implemented interventions that induce the same \(C_F\) changes induce the
> same frozen functional-access consequences after rival matching.

This is stronger than correlation and weaker than a universal theorem.

It directly targets the construct-validity objection.
